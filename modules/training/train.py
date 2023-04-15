import os
import traceback
from random import shuffle
from typing import *

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..inference import commons
from ..inference.models import (
    MultiPeriodDiscriminator,
    SynthesizerTrnMs256NSFSid,
    SynthesizerTrnMs256NSFSidNono,
)
from ..models import MODELS_DIR
from ..utils import find_empty_port
from . import utils
from .checkpoints import save
from .data_utils import (
    DistributedBucketSampler,
    TextAudioCollate,
    TextAudioCollateMultiNSFsid,
    TextAudioLoader,
    TextAudioLoaderMultiNSFsid,
)
from .losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from .mel_processing import mel_spectrogram_torch, spec_to_mel_torch


def run_training(
    gpus: List[int],
    model_name: str,
    sample_rate: int,
    f0: int,
    batch_size: int,
    total_epoch: int,
    save_every_epoch: int,
    pretrain_g: str,
    pretrain_d: str,
    save_only_last: bool = False,
    cache_in_gpu: bool = False,
):
    training_dir = os.path.join(MODELS_DIR, "training", "models", model_name)
    hps = utils.get_hparams(
        model_name,
        training_dir,
        gpus,
        sample_rate,
        f0,
        batch_size,
        total_epoch,
        save_every_epoch,
        pretrain_g,
        pretrain_d,
        save_only_last,
        cache_in_gpu,
    )

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(find_empty_port())

    deterministic = torch.backends.cudnn.deterministic
    benchmark = torch.backends.cudnn.benchmark
    PREV_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", None)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu) for gpu in gpus])

    mp.spawn(
        run,
        nprocs=len(gpus),
        args=(
            len(gpus),
            hps,
        ),
    )

    if PREV_CUDA_VISIBLE_DEVICES is None:
        del os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = PREV_CUDA_VISIBLE_DEVICES

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark


def run(
    rank: int,
    world_size: List[int],
    hps: utils.HParams,
):
    global_step = 0
    is_main_process = rank == 0
    if is_main_process:
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)

    dist.init_process_group(
        backend="gloo", init_method="env://", rank=rank, world_size=world_size
    )
    torch.manual_seed(hps.train.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    if hps.if_f0 == 1:
        train_dataset = TextAudioLoaderMultiNSFsid(hps.data.training_files, hps.data)
    else:
        train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size * world_size,
        # [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200,1400],  # 16s
        [100, 200, 300, 400, 500, 600, 700, 800, 900],  # 16s
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    # It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit.
    # num_workers=8 -> num_workers=4
    if hps.if_f0 == 1:
        collate_fn = TextAudioCollateMultiNSFsid()
    else:
        collate_fn = TextAudioCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=8,
    )
    if hps.if_f0 == 1:
        net_g = SynthesizerTrnMs256NSFSid(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            is_half=hps.train.fp16_run,
            sr=hps.sample_rate,
        )
    else:
        net_g = SynthesizerTrnMs256NSFSidNono(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            is_half=hps.train.fp16_run,
        )
    if torch.cuda.is_available():
        net_g = net_g.cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)
    if torch.cuda.is_available():
        net_d = net_d.cuda(rank)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    # net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
    # net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
    if torch.cuda.is_available():
        net_g = DDP(net_g, device_ids=[rank])
        net_d = DDP(net_d, device_ids=[rank])
    else:
        net_g = DDP(net_g)
        net_d = DDP(net_d)

    try:  # 如果能加载自动resume
        _, _, _, epoch = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"),
            net_d,
            optim_d,
        )  # D多半加载没事
        if is_main_process:
            print(f'loaded D {utils.latest_checkpoint_path(hps.model_dir, "D_*.pth")}')
        # _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g,load_opt=0)
        _, _, _, epoch = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"),
            net_g,
            optim_g,
        )
        global_step = (epoch - 1) * len(train_loader)
    except:  # 如果首次不能加载，加载pretrain
        traceback.print_exc()
        epoch = 1
        global_step = 0
        if is_main_process:
            print(f"loaded pretrained {hps.pretrainG} {hps.pretrainD}")

        print(
            net_g.module.load_state_dict(
                torch.load(hps.pretrainG, map_location="cpu")["model"]
            )
        )  ##测试不加载优化器
        print(
            net_d.module.load_state_dict(
                torch.load(hps.pretrainD, map_location="cpu")["model"]
            )
        )

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch - 2
    )

    scaler = GradScaler(enabled=hps.train.fp16_run)

    cache = []
    progress_bar = tqdm.tqdm(range((hps.total_epoch - epoch) * len(train_loader)))
    progress_bar.set_postfix(epoch=epoch)
    for epoch in range(epoch, hps.total_epoch + 1):
        train_loader.batch_sampler.set_epoch(epoch)

        net_g.train()
        net_d.train()
        if cache == [] or hps.if_cache_data_in_gpu == False:
            for batch_idx, info in enumerate(train_loader):
                progress_bar.update(1)
                if hps.if_f0 == 1:
                    (
                        phone,
                        phone_lengths,
                        pitch,
                        pitchf,
                        spec,
                        spec_lengths,
                        wave,
                        wave_lengths,
                        sid,
                    ) = info
                else:
                    (
                        phone,
                        phone_lengths,
                        spec,
                        spec_lengths,
                        wave,
                        wave_lengths,
                        sid,
                    ) = info

                if torch.cuda.is_available():
                    phone, phone_lengths = phone.cuda(
                        rank, non_blocking=True
                    ), phone_lengths.cuda(rank, non_blocking=True)
                    if hps.if_f0 == 1:
                        pitch, pitchf = pitch.cuda(
                            rank, non_blocking=True
                        ), pitchf.cuda(rank, non_blocking=True)
                    sid = sid.cuda(rank, non_blocking=True)
                    spec, spec_lengths = spec.cuda(
                        rank, non_blocking=True
                    ), spec_lengths.cuda(rank, non_blocking=True)
                    wave, wave_lengths = wave.cuda(
                        rank, non_blocking=True
                    ), wave_lengths.cuda(rank, non_blocking=True)
                if hps.if_cache_data_in_gpu == True:
                    if hps.if_f0 == 1:
                        cache.append(
                            (
                                batch_idx,
                                (
                                    phone,
                                    phone_lengths,
                                    pitch,
                                    pitchf,
                                    spec,
                                    spec_lengths,
                                    wave,
                                    wave_lengths,
                                    sid,
                                ),
                            )
                        )
                    else:
                        cache.append(
                            (
                                batch_idx,
                                (
                                    phone,
                                    phone_lengths,
                                    spec,
                                    spec_lengths,
                                    wave,
                                    wave_lengths,
                                    sid,
                                ),
                            )
                        )
                with autocast(enabled=hps.train.fp16_run):
                    if hps.if_f0 == 1:
                        (
                            y_hat,
                            ids_slice,
                            x_mask,
                            z_mask,
                            (z, z_p, m_p, logs_p, m_q, logs_q),
                        ) = net_g(
                            phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid
                        )
                    else:
                        (
                            y_hat,
                            ids_slice,
                            x_mask,
                            z_mask,
                            (z, z_p, m_p, logs_p, m_q, logs_q),
                        ) = net_g(phone, phone_lengths, spec, spec_lengths, sid)
                    mel = spec_to_mel_torch(
                        spec,
                        hps.data.filter_length,
                        hps.data.n_mel_channels,
                        hps.data.sampling_rate,
                        hps.data.mel_fmin,
                        hps.data.mel_fmax,
                    )
                    y_mel = commons.slice_segments(
                        mel, ids_slice, hps.train.segment_size // hps.data.hop_length
                    )
                    with autocast(enabled=False):
                        y_hat_mel = mel_spectrogram_torch(
                            y_hat.float().squeeze(1),
                            hps.data.filter_length,
                            hps.data.n_mel_channels,
                            hps.data.sampling_rate,
                            hps.data.hop_length,
                            hps.data.win_length,
                            hps.data.mel_fmin,
                            hps.data.mel_fmax,
                        )
                    if hps.train.fp16_run == True:
                        y_hat_mel = y_hat_mel.half()
                    wave = commons.slice_segments(
                        wave, ids_slice * hps.data.hop_length, hps.train.segment_size
                    )  # slice

                    # Discriminator
                    y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
                    with autocast(enabled=False):
                        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                            y_d_hat_r, y_d_hat_g
                        )
                optim_d.zero_grad()
                scaler.scale(loss_disc).backward()
                scaler.unscale_(optim_d)
                grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
                scaler.step(optim_d)

                with autocast(enabled=hps.train.fp16_run):
                    # Generator
                    y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
                    with autocast(enabled=False):
                        loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                        loss_kl = (
                            kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                        )
                        loss_fm = feature_loss(fmap_r, fmap_g)
                        loss_gen, losses_gen = generator_loss(y_d_hat_g)
                        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
                optim_g.zero_grad()
                scaler.scale(loss_gen_all).backward()
                scaler.unscale_(optim_g)
                grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
                scaler.step(optim_g)
                scaler.update()

                if is_main_process:
                    if global_step % hps.train.log_interval == 0:
                        lr = optim_g.param_groups[0]["lr"]
                        print(
                            "Train Epoch: {} [{:.0f}%]".format(
                                epoch, 100.0 * batch_idx / len(train_loader)
                            )
                        )
                        # Amor For Tensorboard display
                        if loss_mel > 50:
                            loss_mel = 50
                        if loss_kl > 5:
                            loss_kl = 5

                        print(
                            f"loss_disc={loss_disc:.3f}, loss_gen={loss_gen:.3f}, loss_fm={loss_fm:.3f},loss_mel={loss_mel:.3f}, loss_kl={loss_kl:.3f}"
                        )
                        scalar_dict = {
                            "loss/g/total": loss_gen_all,
                            "loss/d/total": loss_disc,
                            "learning_rate": lr,
                            "grad_norm_d": grad_norm_d,
                            "grad_norm_g": grad_norm_g,
                        }
                        scalar_dict.update(
                            {
                                "loss/g/fm": loss_fm,
                                "loss/g/mel": loss_mel,
                                "loss/g/kl": loss_kl,
                            }
                        )

                        scalar_dict.update(
                            {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
                        )
                        scalar_dict.update(
                            {
                                "loss/d_r/{}".format(i): v
                                for i, v in enumerate(losses_disc_r)
                            }
                        )
                        scalar_dict.update(
                            {
                                "loss/d_g/{}".format(i): v
                                for i, v in enumerate(losses_disc_g)
                            }
                        )
                        image_dict = {
                            "slice/mel_org": utils.plot_spectrogram_to_numpy(
                                y_mel[0].data.cpu().numpy()
                            ),
                            "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                                y_hat_mel[0].data.cpu().numpy()
                            ),
                            "all/mel": utils.plot_spectrogram_to_numpy(
                                mel[0].data.cpu().numpy()
                            ),
                        }
                        utils.summarize(
                            writer=writer,
                            global_step=global_step,
                            images=image_dict,
                            scalars=scalar_dict,
                        )
                global_step += 1
            if epoch % hps.save_every_epoch == 0 and is_main_process:
                if hps.if_latest == 0:
                    utils.save_checkpoint(
                        net_g,
                        optim_g,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
                    )
                    utils.save_checkpoint(
                        net_d,
                        optim_d,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
                    )
                else:
                    utils.save_checkpoint(
                        net_g,
                        optim_g,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(hps.model_dir, "G_{}.pth".format(2333333)),
                    )
                    utils.save_checkpoint(
                        net_d,
                        optim_d,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(hps.model_dir, "D_{}.pth".format(2333333)),
                    )

        else:  # 后续的epoch直接使用打乱的cache
            shuffle(cache)
            # print("using cache")
            for batch_idx, info in cache:
                progress_bar.update(1)
                if hps.if_f0 == 1:
                    (
                        phone,
                        phone_lengths,
                        pitch,
                        pitchf,
                        spec,
                        spec_lengths,
                        wave,
                        wave_lengths,
                        sid,
                    ) = info
                else:
                    (
                        phone,
                        phone_lengths,
                        spec,
                        spec_lengths,
                        wave,
                        wave_lengths,
                        sid,
                    ) = info
                with autocast(enabled=hps.train.fp16_run):
                    if hps.if_f0 == 1:
                        (
                            y_hat,
                            ids_slice,
                            x_mask,
                            z_mask,
                            (z, z_p, m_p, logs_p, m_q, logs_q),
                        ) = net_g(
                            phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid
                        )
                    else:
                        (
                            y_hat,
                            ids_slice,
                            x_mask,
                            z_mask,
                            (z, z_p, m_p, logs_p, m_q, logs_q),
                        ) = net_g(phone, phone_lengths, spec, spec_lengths, sid)
                    mel = spec_to_mel_torch(
                        spec,
                        hps.data.filter_length,
                        hps.data.n_mel_channels,
                        hps.data.sampling_rate,
                        hps.data.mel_fmin,
                        hps.data.mel_fmax,
                    )
                    y_mel = commons.slice_segments(
                        mel, ids_slice, hps.train.segment_size // hps.data.hop_length
                    )
                    with autocast(enabled=False):
                        y_hat_mel = mel_spectrogram_torch(
                            y_hat.float().squeeze(1),
                            hps.data.filter_length,
                            hps.data.n_mel_channels,
                            hps.data.sampling_rate,
                            hps.data.hop_length,
                            hps.data.win_length,
                            hps.data.mel_fmin,
                            hps.data.mel_fmax,
                        )
                    if hps.train.fp16_run == True:
                        y_hat_mel = y_hat_mel.half()
                    wave = commons.slice_segments(
                        wave, ids_slice * hps.data.hop_length, hps.train.segment_size
                    )  # slice

                    # Discriminator
                    y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
                    with autocast(enabled=False):
                        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                            y_d_hat_r, y_d_hat_g
                        )
                optim_d.zero_grad()
                scaler.scale(loss_disc).backward()
                scaler.unscale_(optim_d)
                grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
                scaler.step(optim_d)

                with autocast(enabled=hps.train.fp16_run):
                    # Generator
                    y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
                    with autocast(enabled=False):
                        loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                        loss_kl = (
                            kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                        )

                        loss_fm = feature_loss(fmap_r, fmap_g)
                        loss_gen, losses_gen = generator_loss(y_d_hat_g)
                        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
                optim_g.zero_grad()
                scaler.scale(loss_gen_all).backward()
                scaler.unscale_(optim_g)
                grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
                scaler.step(optim_g)
                scaler.update()

                if is_main_process:
                    if global_step % hps.train.log_interval == 0:
                        lr = optim_g.param_groups[0]["lr"]
                        # Amor For Tensorboard display
                        if loss_mel > 50:
                            loss_mel = 50
                        if loss_kl > 5:
                            loss_kl = 5
                        scalar_dict = {
                            "loss/g/total": loss_gen_all,
                            "loss/d/total": loss_disc,
                            "learning_rate": lr,
                            "grad_norm_d": grad_norm_d,
                            "grad_norm_g": grad_norm_g,
                        }
                        scalar_dict.update(
                            {
                                "loss/g/fm": loss_fm,
                                "loss/g/mel": loss_mel,
                                "loss/g/kl": loss_kl,
                            }
                        )

                        scalar_dict.update(
                            {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
                        )
                        scalar_dict.update(
                            {
                                "loss/d_r/{}".format(i): v
                                for i, v in enumerate(losses_disc_r)
                            }
                        )
                        scalar_dict.update(
                            {
                                "loss/d_g/{}".format(i): v
                                for i, v in enumerate(losses_disc_g)
                            }
                        )
                        image_dict = {
                            "slice/mel_org": utils.plot_spectrogram_to_numpy(
                                y_mel[0].data.cpu().numpy()
                            ),
                            "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                                y_hat_mel[0].data.cpu().numpy()
                            ),
                            "all/mel": utils.plot_spectrogram_to_numpy(
                                mel[0].data.cpu().numpy()
                            ),
                        }
                        utils.summarize(
                            writer=writer,
                            global_step=global_step,
                            images=image_dict,
                            scalars=scalar_dict,
                        )
                global_step += 1
            # if global_step % hps.train.eval_interval == 0:
            if epoch % hps.save_every_epoch == 0 and is_main_process:
                if hps.if_latest == 0:
                    utils.save_checkpoint(
                        net_g,
                        optim_g,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
                    )
                    utils.save_checkpoint(
                        net_d,
                        optim_d,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
                    )
                else:
                    utils.save_checkpoint(
                        net_g,
                        optim_g,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(hps.model_dir, "G_{}.pth".format(2333333)),
                    )
                    utils.save_checkpoint(
                        net_d,
                        optim_d,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(hps.model_dir, "D_{}.pth".format(2333333)),
                    )
        if is_main_process:
            progress_bar.set_postfix(epoch=epoch)

        scheduler_g.step()
        scheduler_d.step()

    if is_main_process:
        print("Training is done. The program is closed.")
        if hasattr(net_g, "module"):
            ckpt = net_g.module.state_dict()
        else:
            ckpt = net_g.state_dict()
        save(ckpt, hps.sample_rate, hps.if_f0, hps.name, epoch)
