import glob
import json
import os
import re
import time
import operator
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
from ..shared import MODELS_DIR
from ..utils import find_empty_port
from . import utils
from .checkpoints import save
from .config import DatasetMetadata, TrainConfig
from .data_utils import (
    DistributedBucketSampler,
    TextAudioCollate,
    TextAudioCollateMultiNSFsid,
    TextAudioLoader,
    TextAudioLoaderMultiNSFsid,
)
from .losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from .mel_processing import mel_spectrogram_torch, spec_to_mel_torch


def glob_dataset(glob_str: str, speaker_id: int):
    datasets_speakers = []
    if os.path.isdir(glob_str):
        files = os.listdir(glob_str)
        # pattern: {glob_str}/{decimal}[_]* and isdir
        dirs = [(os.path.join(glob_str, f), int(f.split("_")[0])) for f in files if os.path.isdir(os.path.join(glob_str, f)) and f.split("_")[0].isdecimal()]
        
        if len(dirs) > 0:
            # multi speakers at once train
            match_files_re = re.compile(r'.+\.(wav|flac)')  # matches .wav and .flac
            datasets_speakers = [(file, dir[1]) for dir in dirs for file in glob.iglob(os.path.join(dir[0], "*"), recursive=True) if match_files_re.search(file)]
            # for dir in dirs:
            #     for file in glob.iglob(dir[0], recursive=True):
            #         if match_files_re.search(file):
            #             datasets_speakers.append((file, dirs[1]))
            # return sorted(datasets_speakers, key=operator.itemgetter(0))
        
        glob_str = os.path.join(glob_str, "*.wav")
        
    datasets_speakers.extend([(file, speaker_id) for file in glob.iglob(glob_str, recursive=True)])
    
    return sorted(datasets_speakers, key=operator.itemgetter(0))


def create_dataset_meta(training_dir: str, sr: str, f0: bool):
    gt_wavs_dir = os.path.join(training_dir, "0_gt_wavs")
    co256_dir = os.path.join(training_dir, "3_feature256")

    names = set([
        os.path.join(dir, name.split(".")[0]) for dir in os.listdir(gt_wavs_dir)
        for name in os.listdir(os.path.join(gt_wavs_dir, dir))
    ]) & set([
        os.path.join(dir, name.split(".")[0]) for dir in os.listdir(co256_dir)
        for name in os.listdir(os.path.join(co256_dir, dir))
    ])

    if f0:
        f0_dir = os.path.join(training_dir, "2a_f0")
        f0nsf_dir = os.path.join(training_dir, "2b_f0nsf")
        names = (
            names
            & set([
                os.path.join(dir, name.split(".")[0]) for dir in os.listdir(f0_dir)
                for name in os.listdir(os.path.join(f0_dir, dir))
            ])
            & set([
                os.path.join(dir, name.split(".")[0]) for dir in os.listdir(f0nsf_dir)
                for name in os.listdir(os.path.join(f0nsf_dir, dir))
            ])
        )

    meta = {
        "files": {},
    }

    for name in names:
        speaker_id = os.path.dirname(name).split("_")[0]
        speaker_id = int(speaker_id) if speaker_id.isdecimal() else 0
        if f0:
            gt_wav_path = os.path.join(gt_wavs_dir, f"{name}.wav")
            co256_path = os.path.join(co256_dir, f"{name}.npy")
            f0_path = os.path.join(f0_dir, f"{name}.wav.npy")
            f0nsf_path = os.path.join(f0nsf_dir, f"{name}.wav.npy")
            meta["files"][name] = {
                "gt_wav": gt_wav_path,
                "co256": co256_path,
                "f0": f0_path,
                "f0nsf": f0nsf_path,
                "speaker_id": speaker_id,
            }
        else:
            gt_wav_path = os.path.join(gt_wavs_dir, f"{name}.wav")
            co256_path = os.path.join(co256_dir, f"{name}.npy")
            meta["files"][name] = {
                "gt_wav": gt_wav_path,
                "co256": co256_path,
                "speaker_id": speaker_id,
            }

    # if f0:
    #     mute_gt_wav = os.path.join(
    #         MODELS_DIR, "training", "mute", "0_gt_wavs", f"mute{sr}.wav"
    #     )
    #     mute_co256 = os.path.join(
    #         co256_dir, "mute.npy"
    #     )
    #     mute_f0 = os.path.join(MODELS_DIR, "training", "mute", "2a_f0", f"mute.wav.npy")
    #     mute_f0nsf = os.path.join(
    #         MODELS_DIR, "training", "mute", "2b_f0nsf", f"mute.wav.npy"
    #     )
    #     meta["mute"] = {
    #         "gt_wav": mute_gt_wav,
    #         "co256": mute_co256,
    #         "f0": mute_f0,
    #         "f0nsf": mute_f0nsf,
    #         "speaker_id": speaker_id,
    #     }
    # else:
    #     mute_gt_wav = os.path.join(
    #         MODELS_DIR, "training", "mute", "0_gt_wavs", f"mute{sr}.wav"
    #     )
    #     mute_co256 = os.path.join(
    #         co256_dir, "mute.npy"
    #     )
    #     meta["mute"] = {
    #         "gt_wav": mute_gt_wav,
    #         "co256": mute_co256,
    #         "speaker_id": speaker_id,
    #     }

    with open(os.path.join(training_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def train_index(training_dir: str, model_name: str, emb_ch: int):
    checkpoint_path = os.path.join(MODELS_DIR, "checkpoints", model_name)
    feature_256_dir = os.path.join(training_dir, "3_feature256")
    npys = []
    for name in [
        os.path.join(dir, file) for dir in os.listdir(feature_256_dir)
        for file in os.listdir(os.path.join(feature_256_dir, dir))
        ]:
        if name.endswith(".npy"):
            phone = np.load(os.path.join(feature_256_dir, name))
            npys.append(phone)

    big_npy = np.concatenate(npys, 0)
    n_ivf = big_npy.shape[0] // 39
    index = faiss.index_factory(emb_ch, f"IVF{n_ivf},Flat")
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


def train_model(
    gpus: List[int],
    training_dir: str,
    model_name: str,
    sample_rate: int,
    f0: int,
    batch_size: int,
    cache_batch: bool,
    total_epoch: int,
    save_every_epoch: int,
    pretrain_g: str,
    pretrain_d: str,
    embedder_name: str,
    save_only_last: bool = False,
):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(find_empty_port())

    deterministic = torch.backends.cudnn.deterministic
    benchmark = torch.backends.cudnn.benchmark
    PREV_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", None)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu) for gpu in gpus])
    
    config = utils.load_config(training_dir, sample_rate, 768 if embedder_name.endswith("768") else 256)

    start = time.perf_counter()

    mp.spawn(
        training_runner,
        nprocs=len(gpus),
        args=(
            len(gpus),
            config,
            training_dir,
            model_name,
            sample_rate,
            f0,
            batch_size,
            cache_batch,
            total_epoch,
            save_every_epoch,
            pretrain_g,
            pretrain_d,
            embedder_name,
            save_only_last,
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


def training_runner(
    rank: int,
    world_size: List[int],
    config: TrainConfig,
    training_dir: str,
    model_name: str,
    sample_rate: int,
    f0: int,
    batch_size: int,
    cache_in_gpu: bool,
    total_epoch: int,
    save_every_epoch: int,
    pretrain_g: str,
    pretrain_d: str,
    embedder_name: str,
    save_only_last: bool = False,
):
    config.train.batch_size = batch_size
    log_dir = os.path.join(training_dir, "logs")
    state_dir = os.path.join(training_dir, "state")
    training_files_path = os.path.join(training_dir, "meta.json")
    training_meta = DatasetMetadata.parse_file(training_files_path)
    embedder_out_channels = config.model.emb_channels

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
        train_dataset = TextAudioLoaderMultiNSFsid(training_meta, config.data)
    else:
        train_dataset = TextAudioLoader(training_meta, config.data)

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
        net_g_state = torch.load(pretrain_g, map_location="cpu")["model"]
        emb_phone_size = (config.model.hidden_channels, config.model.emb_channels)
        if emb_phone_size != net_g_state["enc_p.emb_phone.weight"].size():
            # interpolate
            orig_shape = net_g_state["enc_p.emb_phone.weight"].size()
            if net_g_state["enc_p.emb_phone.weight"].dtype == torch.half:
                net_g_state["enc_p.emb_phone.weight"] = F.interpolate(net_g_state["enc_p.emb_phone.weight"].float().unsqueeze(0).unsqueeze(0), size=emb_phone_size, mode="bilinear").half().squeeze(0).squeeze(0)
            else:
                net_g_state["enc_p.emb_phone.weight"] = F.interpolate(net_g_state["enc_p.emb_phone.weight"].unsqueeze(0).unsqueeze(0), size=emb_phone_size, mode="bilinear").squeeze(0).squeeze(0)
            print(
                "interpolated pretrained state enc_p.emb_phone from",
                orig_shape,
                "to",
                emb_phone_size,
            )
        net_g.module.load_state_dict(
            net_g_state
        )
        del net_g_state
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

        epoch += 1
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

        if is_main_process:
            lr = optim_g.param_groups[0]["lr"]

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
                if cache_in_gpu:
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
        if is_main_process and save_every_epoch != 0 and epoch % save_every_epoch == 0:
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

            save(
                net_g,
                sample_rate,
                f0,
                embedder_name,
                embedder_out_channels,
                os.path.join(training_dir, "checkpoints", f"{model_name}-{epoch}.pth"),
                epoch,
            )

        if is_main_process:
            progress_bar.set_postfix(
                epoch=epoch,
                loss_g=float(loss_gen_all),
                loss_d=float(loss_disc),
                lr=float(lr),
                use_cache=use_cache,
            )

        scheduler_g.step()
        scheduler_d.step()

    if is_main_process:
        print("Training is done. The program is closed.")
        save(
            net_g,
            sample_rate,
            f0,
            embedder_name,
            embedder_out_channels,
            os.path.join(MODELS_DIR, "checkpoints", f"{model_name}.pth"),
            epoch,
        )
