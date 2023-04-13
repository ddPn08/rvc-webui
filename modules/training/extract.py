import asyncio
import os
import traceback
from multiprocessing import Process
from typing import *

import librosa
import numpy as np
import parselmouth
import pyworld
import soundfile as sf
import torch
import torch.nn.functional as F
from fairseq import checkpoint_utils

from modules.models import MODELS_DIR
from modules.shared import device


class FeatureInput(object):
    def __init__(self, samplerate=16000, hop_size=160):
        self.fs = samplerate
        self.hop = hop_size

        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    def compute_f0(self, path, f0_method):
        x, sr = librosa.load(path, self.fs)
        p_len = x.shape[0] // self.hop
        assert sr == self.fs
        if f0_method == "pm":
            time_step = 160 / 16000 * 1000
            f0_min = 50
            f0_max = 1100
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
                f0 = np.pad(
                    f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
                )
        elif f0_method == "harvest":
            f0, t = pyworld.harvest(
                x.astype(np.double),
                fs=sr,
                f0_ceil=1100,
                frame_period=1000 * self.hop / sr,
            )
            f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.fs)
        elif f0_method == "dio":
            f0, t = pyworld.dio(
                x.astype(np.double),
                fs=sr,
                f0_ceil=1100,
                frame_period=1000 * self.hop / sr,
            )
            f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.fs)
        return f0

    def coarse_f0(self, f0):
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
            self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1

        # use 0 or 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(np.int)
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
            f0_coarse.max(),
            f0_coarse.min(),
        )
        return f0_coarse

    def go(self, paths, f0_method):
        if len(paths) != 0:
            for idx, (inp_path, opt_path1, opt_path2) in enumerate(paths):
                try:
                    if (
                        os.path.exists(opt_path1 + ".npy") == True
                        and os.path.exists(opt_path2 + ".npy") == True
                    ):
                        continue
                    featur_pit = self.compute_f0(inp_path, f0_method)
                    np.save(
                        opt_path2,
                        featur_pit,
                        allow_pickle=False,
                    )  # nsf
                    coarse_pit = self.coarse_f0(featur_pit)
                    np.save(
                        opt_path1,
                        coarse_pit,
                        allow_pickle=False,
                    )  # ori
                except:
                    print(f"f0 failed {idx}: {inp_path} {traceback.format_exc()}")


def extract_f0(training_dir: str, num_processes: int, f0_method: str):
    feature_input = FeatureInput()
    paths = []
    dataset_dir = os.path.join(training_dir, "1_16k_wavs")
    opt_dir_f0 = os.path.join(training_dir, "2a_f0")
    opt_dir_f0_nsf = os.path.join(training_dir, "2b_f0nsf")

    if os.path.exists(opt_dir_f0) and os.path.exists(opt_dir_f0_nsf):
        return

    os.makedirs(opt_dir_f0, exist_ok=True)
    os.makedirs(opt_dir_f0_nsf, exist_ok=True)

    for name in sorted(list(os.listdir(dataset_dir))):
        dir = os.path.join(dataset_dir, name)
        if "spec" in dir:
            continue
        opt_filepath_f0 = os.path.join(opt_dir_f0, name)
        opt_filepath_f0_nsf = os.path.join(opt_dir_f0_nsf, name)
        paths.append([dir, opt_filepath_f0, opt_filepath_f0_nsf])

    ps = []
    for i in range(num_processes):
        p = Process(
            target=feature_input.go,
            args=(
                paths[i::num_processes],
                f0_method,
            ),
        )
        p.start()
        ps.append(p)
    for p in ps:
        p.join()


def extract_feature(training_dir: str):
    wav_dir = os.path.join(training_dir, "1_16k_wavs")
    out_dir = os.path.join(training_dir, "3_feature256")

    if os.path.exists(out_dir):
        return

    os.makedirs(out_dir, exist_ok=True)

    # wave must be 16k, hop_size=320
    def readwave(wav_path, normalize=False):
        wav, sr = sf.read(wav_path)
        assert sr == 16000
        feats = torch.from_numpy(wav).float()
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        if normalize:
            with torch.no_grad():
                feats = F.layer_norm(feats, feats.shape)
        feats = feats.view(1, -1)
        return feats

    # HuBERT model
    models, cfg, _ = checkpoint_utils.load_model_ensemble_and_task(
        [os.path.join(MODELS_DIR, "hubert_base.pt")],
        suffix="",
    )
    model = models[0]
    model = model.to(device)
    if torch.cuda.is_available():
        model = model.half()
    model.eval()

    num_gpus = torch.cuda.device_count()

    def process(todo: List[str], device: torch.device):
        for file in todo:
            try:
                if file.endswith(".wav"):
                    wav_filepath = os.path.join(wav_dir, file)
                    out_filepath = os.path.join(out_dir, file.replace("wav", "npy"))

                    if os.path.exists(out_filepath):
                        continue

                    feats = readwave(wav_filepath, normalize=cfg.task.normalize)
                    padding_mask = torch.BoolTensor(feats.shape).fill_(False)
                    inputs = {
                        "source": feats.half().to(device)
                        if torch.cuda.is_available()
                        else feats.to(device),
                        "padding_mask": padding_mask.to(device),
                        "output_layer": 9,  # layer 9
                    }
                    with torch.no_grad():
                        logits = model.extract_features(**inputs)
                        feats = model.final_proj(logits[0])

                    feats = feats.squeeze(0).float().cpu().numpy()
                    if np.isnan(feats).sum() == 0:
                        np.save(out_filepath, feats, allow_pickle=False)
                    else:
                        print(f"{file} contains nan")
            except Exception as e:
                print(f"Error: {e} {file}")
                traceback.print_exc()

    async def run_tasks():
        todo = sorted(list(os.listdir(wav_dir)))
        loop = asyncio.get_event_loop()
        await asyncio.gather(
            *[
                loop.run_in_executor(
                    None, process, todo[i::num_gpus], torch.device(f"cuda:{i}")
                )
                for i in range(num_gpus)
            ]
        )

    loop = asyncio.new_event_loop()
    loop.run_until_complete(run_tasks())
