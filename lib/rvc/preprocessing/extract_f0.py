import os
import traceback
from concurrent.futures import ProcessPoolExecutor
from typing import *

import numpy as np
import pyworld
import torch
import torchcrepe
from torch import Tensor
from tqdm import tqdm

from lib.rvc.utils import load_audio

def get_optimal_torch_device(index: int = 0) -> torch.device:
    # Get cuda device
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index % torch.cuda.device_count()}") # Very fast
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    # Insert an else here to grab "xla" devices if available. TO DO later. Requires the torch_xla.core.xla_model library
    # Else wise return the "cpu" as a torch device, 
    return torch.device("cpu")

def get_f0_official_crepe_computation(
        x,
        sr,
        f0_min,
        f0_max,
        model="full",
):
    batch_size = 512
    torch_device = get_optimal_torch_device()
    audio = torch.tensor(np.copy(x))[None].float()
    f0, pd = torchcrepe.predict(
        audio,
        sr,
        160,
        f0_min,
        f0_max,
        model,
        batch_size=batch_size,
        device=torch_device,
        return_periodicity=True,
    )
    pd = torchcrepe.filter.median(pd, 3)
    f0 = torchcrepe.filter.mean(f0, 3)
    f0[pd < 0.1] = 0
    f0 = f0[0].cpu().numpy()
    f0 = f0[1:] # Get rid of extra first frame
    return f0

def get_f0_crepe_computation(
        x, 
        sr,
        f0_min,
        f0_max,
        hop_length=160, # 512 before. Hop length changes the speed that the voice jumps to a different dramatic pitch. Lower hop lengths means more pitch accuracy but longer inference time.
        model="full", # Either use crepe-tiny "tiny" or crepe "full". Default is full
):
    x = x.astype(np.float32) # fixes the F.conv2D exception. We needed to convert double to float.
    x /= np.quantile(np.abs(x), 0.999)
    torch_device = get_optimal_torch_device()
    audio = torch.from_numpy(x).to(torch_device, copy=True)
    audio = torch.unsqueeze(audio, dim=0)
    if audio.ndim == 2 and audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True).detach()
    audio = audio.detach()
    print("Initiating prediction with a crepe_hop_length of: " + str(hop_length))
    pitch: Tensor = torchcrepe.predict(
        audio,
        sr,
        hop_length,
        f0_min,
        f0_max,
        model,
        batch_size=hop_length * 2,
        device=torch_device,
        pad=True
    )
    p_len = x.shape[0] // hop_length
    # Resize the pitch for final f0
    source = np.array(pitch.squeeze(0).cpu().float().numpy())
    source[source < 0.001] = np.nan
    target = np.interp(
        np.arange(0, len(source) * p_len, len(source)) / p_len,
        np.arange(0, len(source)),
        source
    )
    f0 = np.nan_to_num(target)
    f0 = f0[1:] # Get rid of extra first frame
    return f0 # Resized f0


def compute_f0(
    path: str,
    f0_method: str,
    fs: int,
    hop: int,
    f0_max: float,
    f0_min: float,
):
    x = load_audio(path, fs)
    if f0_method == "harvest":
        f0, t = pyworld.harvest(
            x.astype(np.double),
            fs=fs,
            f0_ceil=f0_max,
            f0_floor=f0_min,
            frame_period=1000 * hop / fs,
        )
        f0 = pyworld.stonemask(x.astype(np.double), f0, t, fs)
    elif f0_method == "dio":
        f0, t = pyworld.dio(
            x.astype(np.double),
            fs=fs,
            f0_ceil=f0_max,
            f0_floor=f0_min,
            frame_period=1000 * hop / fs,
        )
        f0 = pyworld.stonemask(x.astype(np.double), f0, t, fs)
    elif f0_method == "mangio-crepe":
        f0 = get_f0_crepe_computation(x, fs, f0_min, f0_max, 160, "full")
    elif f0_method == "crepe":
        f0 = get_f0_official_crepe_computation(x.astype(np.double), fs, f0_min, f0_max, "full")
    return f0


def coarse_f0(f0, f0_bin, f0_mel_min, f0_mel_max):
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (
        f0_mel_max - f0_mel_min
    ) + 1

    # use 0 or 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = np.rint(f0_mel).astype(np.int)
    assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
        f0_coarse.max(),
        f0_coarse.min(),
    )
    return f0_coarse


def processor(paths, f0_method, samplerate=16000, hop_size=160, process_id=0):
    fs = samplerate
    hop = hop_size

    f0_bin = 256
    f0_max = 1100.0
    f0_min = 50.0
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    if len(paths) != 0:
        for idx, (inp_path, opt_path1, opt_path2) in enumerate(
            tqdm(paths, position=1 + process_id)
        ):
            try:
                if (
                    os.path.exists(opt_path1 + ".npy") == True
                    and os.path.exists(opt_path2 + ".npy") == True
                ):
                    continue
                featur_pit = compute_f0(inp_path, f0_method, fs, hop, f0_max, f0_min)
                np.save(
                    opt_path2,
                    featur_pit,
                    allow_pickle=False,
                )  # nsf
                coarse_pit = coarse_f0(featur_pit, f0_bin, f0_mel_min, f0_mel_max)
                np.save(
                    opt_path1,
                    coarse_pit,
                    allow_pickle=False,
                )  # ori
            except:
                print(f"f0 failed {idx}: {inp_path} {traceback.format_exc()}")


def run(training_dir: str, num_processes: int, f0_method: str):
    paths = []
    dataset_dir = os.path.join(training_dir, "1_16k_wavs")
    opt_dir_f0 = os.path.join(training_dir, "2a_f0")
    opt_dir_f0_nsf = os.path.join(training_dir, "2b_f0nsf")

    if os.path.exists(opt_dir_f0) and os.path.exists(opt_dir_f0_nsf):
        return

    os.makedirs(opt_dir_f0, exist_ok=True)
    os.makedirs(opt_dir_f0_nsf, exist_ok=True)

    names = []

    for pathname in sorted(list(os.listdir(dataset_dir))):
        if os.path.isdir(os.path.join(dataset_dir, pathname)):
            for f in sorted(list(os.listdir(os.path.join(dataset_dir, pathname)))):
                if "spec" in f:
                    continue
                names.append(os.path.join(pathname, f))
        else:
            names.append(pathname)

    for name in names:  # dataset_dir/{05d}/file.ext
        filepath = os.path.join(dataset_dir, name)
        if "spec" in filepath:
            continue
        opt_filepath_f0 = os.path.join(opt_dir_f0, name)
        opt_filepath_f0_nsf = os.path.join(opt_dir_f0_nsf, name)
        paths.append([filepath, opt_filepath_f0, opt_filepath_f0_nsf])

    for dir in set([(os.path.dirname(p[1]), os.path.dirname(p[2])) for p in paths]):
        os.makedirs(dir[0], exist_ok=True)
        os.makedirs(dir[1], exist_ok=True)

    with ProcessPoolExecutor() as executer:
        for i in range(num_processes):
            executer.submit(processor, paths[i::num_processes], f0_method, process_id=i)

    processor(paths, f0_method)
