import os
import time
from random import shuffle
from typing import *

import faiss
import numpy as np
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
from ..utils import find_empty_port
from . import utils
from .checkpoints import save
from .config import TrainConfig
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
    training_dir: str,
    model_name: str,
    sample_rate: int,
    f0: int,
    batch_size: int,
    total_epoch: int,
    save_every_epoch: int,
    pretrain_g: str,
    pretrain_d: str,
    save_only_last: bool = False,
    cache_in_gpu: bool = True,
):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(find_empty_port())

    deterministic = torch.backends.cudnn.deterministic
    benchmark = torch.backends.cudnn.benchmark
    PREV_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", None)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu) for gpu in gpus])

    config = utils.load_config(training_dir, sample_rate)

    start = time.perf_counter()

    mp.spawn(
        run,
        nprocs=len(gpus),
        args=(
            len(gpus),
            config,
            training_dir,
            model_name,
            sample_rate,
            f0,
            batch_size,
            total_epoch,
            save_every_epoch,
            pretrain_g,
            pretrain_d,
            save_only_last,
            cache_in_gpu,
        ),
    )

    end = time.perf_counter()

    print(f"Time: {end - start}")

    if PREV_CUDA_VISIBLE_DEVICES is None:
        del os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = PREV_CUDA_VISIBLE_DEVICES

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark


def run(
    rank: int,
    world_size: List[int],
    config: TrainConfig,
    training_dir: str,
    model_name: str,
    sample_rate: int,
    f0: int,
    batch_size: int,
    total_epoch: int,
    save_every_epoch: int,
    pretrain_g: str,
    pretrain_d: str,
    save_only_last: bool = False,
    cache_in_gpu: bool = True,
):
    config.train.batch_size = batch_size
    log_dir = os.path.join(training_dir, "logs")
    state_dir = os.path.join(training_dir, "state")
    training_files_path = os.path.join(training_dir, "filelist.txt")

    global_step = 0
    is_main_process = rank == 0

    if is_main_process:
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(state_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

    dist.init_process_group(
        backend="gloo", init_method="env://", rank=rank, world_size=world_size
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    torch.manual_seed(config.train.seed)

    if f0 == 1:
        train_dataset = TextAudioLoaderMultiNSFsid(training_files_path, config.data)
    else:
        train_dataset = TextAudioLoader(training_files_path, config.data)

    train_sampler = DistributedBucketSampler(
        train_dataset,
        config.train.batch_size * world_size,
        [100, 200, 300, 400, 500, 600, 700, 800, 900],
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    if f0 == 1:
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

    if f0 == 1:
        net_g = SynthesizerTrnMs256NSFSid(
            config.data.filter_length // 2 + 1,
            config.train.segment_size // config.data.hop_length,
            **config.model.dict(),
            is_half=config.train.fp16_run,
            sr=sample_rate,
        )
    else:
        net_g = SynthesizerTrnMs256NSFSidNono(
            config.data.filter_length // 2 + 1,
            config.train.segment_size // config.data.hop_length,
            **config.model.dict(),
            is_half=config.train.fp16_run,
        )
    if torch.cuda.is_available():
        net_g = net_g.cuda(rank)

    net_d = MultiPeriodDiscriminator(config.model.use_spectral_norm)
    if torch.cuda.is_available():
        net_d = net_d.cuda(rank)

    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        config.train.learning_rate,
        betas=config.train.betas,
        eps=config.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        config.train.learning_rate,
        betas=config.train.betas,
        eps=config.train.eps,
    )

    if torch.cuda.is_available():
        net_g = DDP(net_g, device_ids=[rank])
        net_d = DDP(net_d, device_ids=[rank])
    else:
        net_g = DDP(net_g)
        net_d = DDP(net_d)

    last_d_state = utils.latest_checkpoint_path(state_dir, "D_*.pth")
    last_g_state = utils.latest_checkpoint_path(state_dir, "G_*.pth")

    if last_d_state is None or last_g_state is None:
        epoch = 1
        global_step = 0
        net_g.module.load_state_dict(
            torch.load(pretrain_g, map_location="cpu")["model"]
        )
        net_d.module.load_state_dict(
            torch.load(pretrain_d, map_location="cpu")["model"]
        )
        if is_main_process:
            print(f"loaded pretrained {pretrain_g} {pretrain_d}")
    else:
        _, _, _, epoch = utils.load_checkpoint(last_d_state, net_d, optim_d)
        _, _, _, epoch = utils.load_checkpoint(last_g_state, net_g, optim_g)
        if is_main_process:
            print(f"loaded last state {last_d_state} {last_g_state}")
        global_step = (epoch - 1) * len(train_loader)

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=config.train.lr_decay, last_epoch=epoch - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=config.train.lr_decay, last_epoch=epoch - 2
    )

    scaler = GradScaler(enabled=config.train.fp16_run)

    cache = []
    progress_bar = tqdm.tqdm(range((total_epoch - epoch + 1) * len(train_loader)))
    progress_bar.set_postfix(epoch=epoch)
    for epoch in range(epoch, total_epoch + 1):
        train_loader.batch_sampler.set_epoch(epoch)

        net_g.train()
        net_d.train()

        use_cache = len(cache) == len(train_loader)
        data = cache if use_cache else enumerate(train_loader)

        if use_cache:
            shuffle(cache)

        for batch_idx, batch in data:
            progress_bar.update(1)
            if f0 == 1:
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
                ) = batch
            else:
                (
                    phone,
                    phone_lengths,
                    spec,
                    spec_lengths,
                    wave,
                    wave_lengths,
                    sid,
                ) = batch

            if not use_cache:
                if torch.cuda.is_available():
                    phone, phone_lengths = phone.cuda(
                        rank, non_blocking=True
                    ), phone_lengths.cuda(rank, non_blocking=True)
                    if f0 == 1:
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
                if cache_in_gpu == True:
                    if f0 == 1:
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

            with autocast(enabled=config.train.fp16_run):
                if f0 == 1:
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
                    config.data.filter_length,
                    config.data.n_mel_channels,
                    config.data.sampling_rate,
                    config.data.mel_fmin,
                    config.data.mel_fmax,
                )
                y_mel = commons.slice_segments(
                    mel, ids_slice, config.train.segment_size // config.data.hop_length
                )
                with autocast(enabled=False):
                    y_hat_mel = mel_spectrogram_torch(
                        y_hat.float().squeeze(1),
                        config.data.filter_length,
                        config.data.n_mel_channels,
                        config.data.sampling_rate,
                        config.data.hop_length,
                        config.data.win_length,
                        config.data.mel_fmin,
                        config.data.mel_fmax,
                    )
                if config.train.fp16_run == True:
                    y_hat_mel = y_hat_mel.half()
                wave = commons.slice_segments(
                    wave, ids_slice * config.data.hop_length, config.train.segment_size
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

            with autocast(enabled=config.train.fp16_run):
                # Generator
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
                with autocast(enabled=False):
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * config.train.c_mel
                    loss_kl = (
                        kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config.train.c_kl
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
                if global_step % config.train.log_interval == 0:
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
        if epoch % save_every_epoch == 0 and is_main_process:
            if save_only_last:
                old_g_path = os.path.join(
                    state_dir, f"G_{epoch - save_every_epoch}.pth"
                )
                old_d_path = os.path.join(
                    state_dir, f"D_{epoch - save_every_epoch}.pth"
                )
                if os.path.exists(old_g_path):
                    os.remove(old_g_path)
                if os.path.exists(old_d_path):
                    os.remove(old_d_path)
            utils.save_state(
                net_g,
                optim_g,
                config.train.learning_rate,
                epoch,
                os.path.join(state_dir, f"G_{epoch}.pth"),
            )
            utils.save_state(
                net_d,
                optim_d,
                config.train.learning_rate,
                epoch,
                os.path.join(state_dir, f"D_{epoch}.pth"),
            )

        if is_main_process:
            progress_bar.set_postfix(
                epoch=epoch,
                loss_g=float(loss_gen_all),
                loss_d=float(loss_disc),
                lr=float(lr),
            )

        scheduler_g.step()
        scheduler_d.step()

    if is_main_process:
        print("Training is done. The program is closed.")
        checkpoint_path = save(net_g, sample_rate, f0, model_name, epoch)

        feature_256_dir = os.path.join(training_dir, "3_feature256")
        npys = []
        for name in os.listdir(feature_256_dir):
            if name.endswith(".npy"):
                phone = np.load(f"{feature_256_dir}/{name}")
                npys.append(phone)

        big_npy = np.concatenate(npys, 0)
        n_ivf = big_npy.shape[0] // 39
        index = faiss.index_factory(256, f"IVF{n_ivf},Flat")
        index_ivf = faiss.extract_index_ivf(index)
        index_ivf.nprobe = int(np.power(n_ivf, 0.3))
        index.train(big_npy)
        index.add(big_npy)
        np.save(
            os.path.join(os.path.dirname(checkpoint_path), f"{model_name}.big.npy"),
            big_npy,
        )
        faiss.write_index(
            index,
            os.path.join(os.path.dirname(checkpoint_path), f"{model_name}.index"),
        )
