import glob
import json
import operator
import os
import shutil
import time
from random import shuffle
from typing import *

import faiss
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchaudio
import tqdm
from sklearn.cluster import MiniBatchKMeans
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from . import commons, utils
from .checkpoints import save
from .config import DatasetMetadata, TrainConfig
from .data_utils import (DistributedBucketSampler, TextAudioCollate,
                         TextAudioCollateMultiNSFsid, TextAudioLoader,
                         TextAudioLoaderMultiNSFsid)
from .losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from .mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from .models import (MultiPeriodDiscriminator, SynthesizerTrnMs256NSFSid,
                     SynthesizerTrnMs256NSFSidNono)
from .preprocessing.extract_feature import (MODELS_DIR, get_embedder,
                                            load_embedder)


def is_audio_file(file: str):
    if "." not in file:
        return False
    ext = os.path.splitext(file)[1]
    return ext.lower() in [
        ".wav",
        ".flac",
        ".ogg",
        ".mp3",
        ".m4a",
        ".wma",
        ".aiff",
    ]


def glob_dataset(
    glob_str: str,
    speaker_id: int,
    multiple_speakers: bool = False,
    recursive: bool = True,
    training_dir: str = ".",
):
    globs = glob_str.split(",")
    speaker_count = 0
    datasets_speakers = []
    speaker_to_id_mapping = {}
    for glob_str in globs:
        if os.path.isdir(glob_str):
            if multiple_speakers:
                # Multispeaker format:
                # dataset_path/
                # - speakername/
                #     - {wav name here}.wav
                #     - ...
                # - next_speakername/
                #     - {wav name here}.wav
                #     - ...
                # - ...
                print("Multispeaker dataset enabled; Processing speakers.")
                for dir in tqdm.tqdm(os.listdir(glob_str)):
                    print("Speaker ID " + str(speaker_count) + ": " + dir)
                    speaker_to_id_mapping[dir] = speaker_count
                    speaker_path = glob_str + "/" + dir
                    for audio in tqdm.tqdm(os.listdir(speaker_path)):
                        if is_audio_file(glob_str + "/" + dir + "/" + audio):
                            datasets_speakers.append((glob_str + "/" + dir + "/" + audio, speaker_count))
                    speaker_count += 1
                with open(os.path.join(training_dir, "speaker_info.json"), "w") as outfile:
                    print("Dumped speaker info to {}".format(os.path.join(training_dir, "speaker_info.json")))
                    json.dump(speaker_to_id_mapping, outfile)
                continue # Skip the normal speaker extend

            glob_str = os.path.join(glob_str, "**", "*")
        print("Single speaker dataset enabled; Processing speaker as ID " + str(speaker_id) + ".")
        datasets_speakers.extend(
            [
                (file, speaker_id)
                for file in glob.iglob(glob_str, recursive=recursive)
                if is_audio_file(file)
            ]
        )

    return sorted(datasets_speakers)


def create_dataset_meta(training_dir: str, f0: bool):
    gt_wavs_dir = os.path.join(training_dir, "0_gt_wavs")
    co256_dir = os.path.join(training_dir, "3_feature256")

    def list_data(dir: str):
        files = []
        for subdir in os.listdir(dir):
            speaker_dir = os.path.join(dir, subdir)
            for name in os.listdir(speaker_dir):
                files.append(os.path.join(subdir, name.split(".")[0]))
        return files

    names = set(list_data(gt_wavs_dir)) & set(list_data(co256_dir))

    if f0:
        f0_dir = os.path.join(training_dir, "2a_f0")
        f0nsf_dir = os.path.join(training_dir, "2b_f0nsf")
        names = names & set(list_data(f0_dir)) & set(list_data(f0nsf_dir))

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

    with open(os.path.join(training_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def change_speaker(net_g, speaker_info, embedder, embedding_output_layer, phone, phone_lengths, pitch, pitchf, spec_lengths):
    """
    random change formant
    inspired by https://github.com/auspicious3000/contentvec/blob/d746688a32940f4bee410ed7c87ec9cf8ff04f74/contentvec/data/audio/audio_utils_1.py#L179
    """
    N = phone.shape[0]
    device = phone.device
    dtype = phone.dtype

    f0_bin = 256
    f0_max = 1100.0
    f0_min = 50.0
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)

    pitch_median = torch.median(pitchf, 1).values
    lo = 75. + 25. * (pitch_median >= 200).to(dtype=dtype)
    hi = 250. + 150. * (pitch_median >= 200).to(dtype=dtype)
    pitch_median = torch.clip(pitch_median, lo, hi).unsqueeze(1)

    shift_pitch = torch.exp2((1. - 2. * torch.rand(N)) / 4).unsqueeze(1).to(device, dtype)   # ピッチを半オクターブの範囲でずらす

    new_sid = np.random.choice(np.arange(len(speaker_info))[speaker_info > 0], size=N)
    rel_pitch = pitchf / pitch_median
    new_pitch_median = torch.from_numpy(speaker_info[new_sid]).to(device, dtype).unsqueeze(1) * shift_pitch
    new_pitchf = new_pitch_median * rel_pitch
    new_sid = torch.from_numpy(new_sid).to(device)

    new_pitch = 1127. * torch.log(1. + new_pitchf / 700.)
    new_pitch = (pitch - f0_mel_min) * (f0_bin - 2.) / (f0_mel_max - f0_mel_min) + 1.
    new_pitch = torch.clip(new_pitch, 1, f0_bin - 1).to(dtype=torch.int)

    new_wave = net_g.infer(phone, phone_lengths, new_pitch, new_pitchf, new_sid)[0]
    new_wave_16k = torchaudio.functional.resample(new_wave, net_g.sr, 16000, rolloff=0.99).squeeze(1)
    padding_mask = torch.arange(new_wave_16k.shape[1]).unsqueeze(0).to(device) > (spec_lengths.unsqueeze(1) * 160).to(device)

    inputs = {
        "source": new_wave_16k.to(device, dtype),
        "padding_mask": padding_mask.to(device),
        "output_layer": embedding_output_layer
    }
    logits = embedder.extract_features(**inputs)
    if phone.shape[-1] == 768:
        feats = logits[0]
    else:
        feats = embedder.final_proj(logits[0])
    feats = torch.repeat_interleave(feats, 2, 1)
    new_phone = torch.zeros(phone.shape).to(device, dtype)
    new_phone[:, :feats.shape[1]] = feats[:, :phone.shape[1]]
    return new_phone.to(device)


def change_speaker_nono(net_g, embedder, embedding_output_layer, phone, phone_lengths, spec_lengths):
    """
    random change formant
    inspired by https://github.com/auspicious3000/contentvec/blob/d746688a32940f4bee410ed7c87ec9cf8ff04f74/contentvec/data/audio/audio_utils_1.py#L179
    """
    N = phone.shape[0]
    device = phone.device
    dtype = phone.dtype

    new_sid = np.random.randint(net_g.spk_embed_dim, size=N)
    new_sid = torch.from_numpy(new_sid).to(device)

    new_wave = net_g.infer(phone, phone_lengths, new_sid)[0]
    new_wave_16k = torchaudio.functional.resample(new_wave, net_g.sr, 16000, rolloff=0.99).squeeze(1)
    padding_mask = torch.arange(new_wave_16k.shape[1]).unsqueeze(0).to(device) > (spec_lengths.unsqueeze(1) * 160).to(device)

    inputs = {
        "source": new_wave_16k.to(device, dtype),
        "padding_mask": padding_mask.to(device),
        "output_layer": embedding_output_layer
    }

    logits = embedder.extract_features(**inputs)
    if phone.shape[-1] == 768:
        feats = logits[0]
    else:
        feats = embedder.final_proj(logits[0])
    feats = torch.repeat_interleave(feats, 2, 1)
    new_phone = torch.zeros(phone.shape).to(device, dtype)
    new_phone[:, :feats.shape[1]] = feats[:, :phone.shape[1]]
    return new_phone.to(device)


def train_index(
    training_dir: str,
    model_name: str,
    out_dir: str,
    emb_ch: int,
    num_cpu_process: int,
    maximum_index_size: Optional[int],
):
    checkpoint_path = os.path.join(out_dir, model_name)
    feature_256_dir = os.path.join(training_dir, "3_feature256")
    index_dir = os.path.join(os.path.dirname(checkpoint_path), f"{model_name}_index")
    os.makedirs(index_dir, exist_ok=True)
    for speaker_id in tqdm.tqdm(
        sorted([dir for dir in os.listdir(feature_256_dir) if dir.isdecimal()])
    ):
        feature_256_spk_dir = os.path.join(feature_256_dir, speaker_id)
        speaker_id = int(speaker_id)
        npys = []
        for name in [
            os.path.join(feature_256_spk_dir, file)
            for file in os.listdir(feature_256_spk_dir)
            if file.endswith(".npy")
        ]:
            phone = np.load(os.path.join(feature_256_spk_dir, name))
            npys.append(phone)

        # shuffle big_npy to prevent reproducing the sound source
        big_npy = np.concatenate(npys, 0)
        big_npy_idx = np.arange(big_npy.shape[0])
        np.random.shuffle(big_npy_idx)
        big_npy = big_npy[big_npy_idx]

        if not maximum_index_size is None and big_npy.shape[0] > maximum_index_size:
            kmeans = MiniBatchKMeans(
                n_clusters=maximum_index_size,
                batch_size=256 * num_cpu_process,
                init="random",
                compute_labels=False,
            )
            kmeans.fit(big_npy)
            big_npy = kmeans.cluster_centers_

        # recommend parameter in https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
        emb_ch = big_npy.shape[1]
        emb_ch_half = emb_ch // 2
        n_ivf = int(8 * np.sqrt(big_npy.shape[0]))
        if big_npy.shape[0] >= 1_000_000:
            index = faiss.index_factory(
                emb_ch, f"IVF{n_ivf},PQ{emb_ch_half}x4fsr,RFlat"
            )
        else:
            index = faiss.index_factory(emb_ch, f"IVF{n_ivf},Flat")

        index.train(big_npy)
        batch_size_add = 8192
        for i in range(0, big_npy.shape[0], batch_size_add):
            index.add(big_npy[i : i + batch_size_add])
        np.save(
            os.path.join(index_dir, f"{model_name}.{speaker_id}.big.npy"),
            big_npy,
        )
        faiss.write_index(
            index,
            os.path.join(index_dir, f"{model_name}.{speaker_id}.index"),
        )


def train_model(
    gpus: List[int],
    config: TrainConfig,
    training_dir: str,
    model_name: str,
    out_dir: str,
    sample_rate: int,
    f0: bool,
    batch_size: int,
    augment: bool,
    augment_path: Optional[str],
    speaker_info_path: Optional[str],
    cache_batch: bool,
    total_epoch: int,
    save_every_epoch: int,
    save_wav_with_checkpoint: bool,
    pretrain_g: str,
    pretrain_d: str,
    embedder_name: str,
    embedding_output_layer: int,
    save_only_last: bool = False,
    device: Optional[Union[str, torch.device]] = None,
):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(utils.find_empty_port())

    deterministic = torch.backends.cudnn.deterministic
    benchmark = torch.backends.cudnn.benchmark
    PREV_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", None)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu) for gpu in gpus])

    start = time.perf_counter()

    # Mac(MPS)でやると、mp.spawnでなんかトラブルが出るので普通にtraining_runnerを呼び出す。
    if device is not None:
        training_runner(
            0,  # rank
            1,  # world size
            config,
            training_dir,
            model_name,
            out_dir,
            sample_rate,
            f0,
            batch_size,
            augment,
            augment_path,
            speaker_info_path,
            cache_batch,
            total_epoch,
            save_every_epoch,
            save_wav_with_checkpoint,
            pretrain_g,
            pretrain_d,
            embedder_name,
            embedding_output_layer,
            save_only_last,
            device,
        )
    else:
        mp.spawn(
            training_runner,
            nprocs=len(gpus),
            args=(
                len(gpus),
                config,
                training_dir,
                model_name,
                out_dir,
                sample_rate,
                f0,
                batch_size,
                augment,
                augment_path,
                speaker_info_path,
                cache_batch,
                total_epoch,
                save_every_epoch,
                save_wav_with_checkpoint,
                pretrain_g,
                pretrain_d,
                embedder_name,
                embedding_output_layer,
                save_only_last,
                device,
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
    out_dir: str,
    sample_rate: int,
    f0: bool,
    batch_size: int,
    augment: bool,
    augment_path: Optional[str],
    speaker_info_path: Optional[str],
    cache_in_gpu: bool,
    total_epoch: int,
    save_every_epoch: int,
    save_wav_with_checkpoint: bool,
    pretrain_g: str,
    pretrain_d: str,
    embedder_name: str,
    embedding_output_layer: int,
    save_only_last: bool = False,
    device: Optional[Union[str, torch.device]] = None,
):
    config.train.batch_size = batch_size
    log_dir = os.path.join(training_dir, "logs")
    state_dir = os.path.join(training_dir, "state")
    training_files_path = os.path.join(training_dir, "meta.json")
    training_meta = DatasetMetadata.parse_file(training_files_path)
    embedder_out_channels = config.model.emb_channels

    is_multi_process = world_size > 1

    if device is not None:
        if type(device) == str:
            device = torch.device(device)

    global_step = 0
    is_main_process = rank == 0

    if is_main_process:
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(state_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo", init_method="env://", rank=rank, world_size=world_size
        )

    if is_multi_process:
        torch.cuda.set_device(rank)

    torch.manual_seed(config.train.seed)

    if f0:
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

    if f0:
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
    speaker_info = None
    if os.path.exists(os.path.join(training_dir, "speaker_info.json")):
        with open(os.path.join(training_dir, "speaker_info.json"), "r") as f:
            speaker_info = json.load(f)
            config.model.spk_embed_dim = len(speaker_info)
    if f0:
        net_g = SynthesizerTrnMs256NSFSid(
            config.data.filter_length // 2 + 1,
            config.train.segment_size // config.data.hop_length,
            **config.model.dict(),
            is_half=False, # config.train.fp16_run,
            sr=int(sample_rate[:-1] + "000"),
        )
    else:
        net_g = SynthesizerTrnMs256NSFSidNono(
            config.data.filter_length // 2 + 1,
            config.train.segment_size // config.data.hop_length,
            **config.model.dict(),
            is_half=False, # config.train.fp16_run,
            sr=int(sample_rate[:-1] + "000"),
        )

    if is_multi_process:
        net_g = net_g.cuda(rank)
    else:
        net_g = net_g.to(device=device)

    if config.version == "v1":
        periods = [2, 3, 5, 7, 11, 17]
    elif config.version == "v2":
        periods = [2, 3, 5, 7, 11, 17, 23, 37]
    net_d = MultiPeriodDiscriminator(config.model.use_spectral_norm, periods=periods)
    if is_multi_process:
        net_d = net_d.cuda(rank)
    else:
        net_d = net_d.to(device=device)

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

    if is_multi_process:
        net_g = DDP(net_g, device_ids=[rank])
        net_d = DDP(net_d, device_ids=[rank])

    last_d_state = utils.latest_checkpoint_path(state_dir, "D_*.pth")
    last_g_state = utils.latest_checkpoint_path(state_dir, "G_*.pth")

    if augment:
        # load embedder
        embedder_filepath, _, embedder_load_from = get_embedder(embedder_name)

        if embedder_load_from == "local":
            embedder_filepath = os.path.join(
                MODELS_DIR, "embeddings", embedder_filepath
            )
        embedder, _ = load_embedder(embedder_filepath, device)
        if not config.train.fp16_run:
            embedder = embedder.float()

        if (augment_path is not None):
            state_dict = torch.load(augment_path, map_location="cpu")
            if state_dict["f0"] == 1:
                augment_net_g = SynthesizerTrnMs256NSFSid(
                    **state_dict["params"], is_half=config.train.fp16_run
                )
                augment_speaker_info = np.load(speaker_info_path)
            else:
                augment_net_g = SynthesizerTrnMs256NSFSidNono(
                    **state_dict["params"], is_half=config.train.fp16_run
                )

            augment_net_g.load_state_dict(state_dict["weight"], strict=False)
            augment_net_g.eval().to(device)

            if config.train.fp16_run:
                augment_net_g = augment_net_g.half()
            else:
                augment_net_g= augment_net_g.float()
        else:
            augment_net_g = net_g
            if f0:
                medians = [[] for _ in range(augment_net_g.spk_embed_dim)]
                for file in training_meta.files.values():
                    f0f = np.load(file.f0nsf)
                    if np.any(f0f > 0):
                        medians[file.speaker_id].append(np.median(f0f[f0f > 0]))
                augment_speaker_info = np.array([np.median(x) if len(x) else 0. for x in medians])
                np.save(os.path.join(training_dir, "speaker_info.npy"), augment_speaker_info)

    if last_d_state is None or last_g_state is None:
        epoch = 1
        global_step = 0
        if os.path.exists(pretrain_g) and os.path.exists(pretrain_d):
            net_g_state = torch.load(pretrain_g, map_location="cpu")["model"]
            emb_spk_size = (config.model.spk_embed_dim, config.model.gin_channels)
            emb_phone_size = (config.model.hidden_channels, config.model.emb_channels)
            if emb_spk_size != net_g_state["emb_g.weight"].size():
                original_weight = net_g_state["emb_g.weight"]
                net_g_state["emb_g.weight"] = original_weight.mean(dim=0, keepdims=True) * torch.ones(emb_spk_size, device=original_weight.device, dtype=original_weight.dtype)
            if emb_phone_size != net_g_state["enc_p.emb_phone.weight"].size():
                # interpolate
                orig_shape = net_g_state["enc_p.emb_phone.weight"].size()
                if net_g_state["enc_p.emb_phone.weight"].dtype == torch.half:
                    net_g_state["enc_p.emb_phone.weight"] = (
                        F.interpolate(
                            net_g_state["enc_p.emb_phone.weight"]
                            .float()
                            .unsqueeze(0)
                            .unsqueeze(0),
                            size=emb_phone_size,
                            mode="bilinear",
                        )
                        .half()
                        .squeeze(0)
                        .squeeze(0)
                    )
                else:
                    net_g_state["enc_p.emb_phone.weight"] = (
                        F.interpolate(
                            net_g_state["enc_p.emb_phone.weight"]
                            .unsqueeze(0)
                            .unsqueeze(0),
                            size=emb_phone_size,
                            mode="bilinear",
                        )
                        .squeeze(0)
                        .squeeze(0)
                    )
                print(
                    "interpolated pretrained state enc_p.emb_phone from",
                    orig_shape,
                    "to",
                    emb_phone_size,
                )
            if is_multi_process:
                net_g.module.load_state_dict(net_g_state)
            else:
                net_g.load_state_dict(net_g_state)

            del net_g_state

            if is_multi_process:
                net_d.module.load_state_dict(
                    torch.load(pretrain_d, map_location="cpu")["model"]
                )
            else:
                net_d.load_state_dict(
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
    step = -1
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
            step += 1
            progress_bar.update(1)
            if f0:
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
                phone, phone_lengths = (
                    phone.to(device=device, non_blocking=True),
                    phone_lengths.to(device=device, non_blocking=True),
                )
                if f0:
                    pitch, pitchf = (
                        pitch.to(device=device, non_blocking=True),
                        pitchf.to(device=device, non_blocking=True),
                    )
                sid = sid.to(device=device, non_blocking=True)
                spec, spec_lengths = (
                    spec.to(device=device, non_blocking=True),
                    spec_lengths.to(device=device, non_blocking=True),
                )
                wave, wave_lengths = (
                    wave.to(device=device, non_blocking=True),
                    wave_lengths.to(device=device, non_blocking=True),
                )
                if cache_in_gpu:
                    if f0:
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
                if augment:
                    with torch.no_grad():
                        if type(augment_net_g) == SynthesizerTrnMs256NSFSid:
                            new_phone = change_speaker(augment_net_g, augment_speaker_info, embedder, embedding_output_layer, phone, phone_lengths, pitch, pitchf, spec_lengths)
                        else:
                            new_phone = change_speaker_nono(augment_net_g, embedder, embedding_output_layer, phone, phone_lengths, spec_lengths)
                        weight = np.power(.5, step / len(train_dataset))  # 学習の初期はそのままのphone embeddingを使う
                        phone = phone * weight + new_phone * (1. - weight)

                if f0:
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
                if config.train.fp16_run == True and device != torch.device("mps"):
                    y_hat_mel = y_hat_mel.half()
                wave_slice = commons.slice_segments(
                    wave, ids_slice * config.data.hop_length, config.train.segment_size
                )  # slice

                # Discriminator
                y_d_hat_r, y_d_hat_g, _, _ = net_d(wave_slice, y_hat.detach())
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
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave_slice, y_hat)
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
                progress_bar.set_postfix(
                    epoch=epoch,
                    loss_g=float(loss_gen_all) if loss_gen_all is not None else 0.0,
                    loss_d=float(loss_disc) if loss_disc is not None else 0.0,
                    lr=float(lr) if lr is not None else 0.0,
                    use_cache=use_cache,
                )
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
                old_wav_path = os.path.join(
                    state_dir, f"wav_sample_{epoch - save_every_epoch}"
                )
                if os.path.exists(old_g_path):
                    os.remove(old_g_path)
                if os.path.exists(old_d_path):
                    os.remove(old_d_path)
                if os.path.exists(old_wav_path):
                    shutil.rmtree(old_wav_path)

            if save_wav_with_checkpoint:
                with autocast(enabled=config.train.fp16_run):
                    with torch.no_grad():
                        if f0:
                            new_wave = net_g.infer(phone, phone_lengths, pitch, pitchf, sid)[0]
                        else:
                            new_wave = net_g.infer(phone, phone_lengths, sid)[0]
                os.makedirs(os.path.join(state_dir, f"wav_sample_{epoch}"), exist_ok=True)
                for i in range(new_wave.shape[0]):
                    torchaudio.save(filepath=os.path.join(state_dir, f"wav_sample_{epoch}", f"{i:02}_y_true.wav"), src=wave[i].detach().cpu().float(), sample_rate=int(sample_rate[:-1] + "000"))
                    torchaudio.save(filepath=os.path.join(state_dir, f"wav_sample_{epoch}", f"{i:02}_y_pred.wav"), src=new_wave[i].detach().cpu().float(), sample_rate=int(sample_rate[:-1] + "000"))

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
                config.version,
                sample_rate,
                f0,
                embedder_name,
                embedder_out_channels,
                embedding_output_layer,
                os.path.join(training_dir, "checkpoints", f"{model_name}-{epoch}.pth"),
                epoch,
                speaker_info
            )

        scheduler_g.step()
        scheduler_d.step()

    if is_main_process:
        print("Training is done. The program is closed.")
        save(
            net_g,
            config.version,
            sample_rate,
            f0,
            embedder_name,
            embedder_out_channels,
            embedding_output_layer,
            os.path.join(out_dir, f"{model_name}.pth"),
            epoch,
            speaker_info
        )
