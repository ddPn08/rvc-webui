import os
import traceback
from concurrent.futures import ProcessPoolExecutor
from typing import *

import librosa
import numpy as np
import parselmouth
import pyworld
from tqdm import tqdm


def compute_f0(
    path: str,
    f0_method: str,
    fs: int,
    hop: int,
    f0_max: float,
    f0_min: float,
):
    x, sr = librosa.load(path, sr=fs, res_type="soxr_vhq")
    p_len = x.shape[0] // hop
    assert sr == fs
    if f0_method == "pm":
        time_step = 160 / 16000 * 1000
        f0_max = 1100
        f0_min = 50
        f0 = (
            parselmouth.Sound(x, sr)
            .to_pitch_ac(
                time_step=time_step / 1000,
                voicing_threshold=0.6,
                pitch_floor=f0_min,
                pitch_ceiling=f0_max,
            )
            .selected_array["frequency"]
        )
        pad_size = (p_len - len(f0) + 1) // 2
        if pad_size > 0 or p_len - len(f0) - pad_size > 0:
            f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant")
    elif f0_method == "harvest":
        f0, t = pyworld.harvest(
            x.astype(np.double),
            fs=sr,
            f0_ceil=f0_max,
            f0_floor=f0_min,
            frame_period=1000 * hop / sr,
        )
        f0 = pyworld.stonemask(x.astype(np.double), f0, t, fs)
    elif f0_method == "dio":
        f0, t = pyworld.dio(
            x.astype(np.double),
            fs=sr,
            f0_ceil=f0_max,
            f0_floor=f0_min,
            frame_period=1000 * hop / sr,
        )
        f0 = pyworld.stonemask(x.astype(np.double), f0, t, fs)
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
