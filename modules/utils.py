import os

import ffmpeg
import numpy as np
import requests
import torch
from tqdm import tqdm

from lib.rvc.config import TrainConfig
from modules.shared import ROOT_DIR


def load_audio(file: str, sr):
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = (
            file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )  # Prevent small white copy path head and tail with spaces and " and return
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

    return np.frombuffer(out, np.float32).flatten()


def get_gpus():
    num_gpus = torch.cuda.device_count()
    return [torch.device(f"cuda:{i}") for i in range(num_gpus)]



def download_file(url: str, out: str, position: int = 0, show: bool = True):
    req = requests.get(url, stream=True, allow_redirects=True)
    content_length = req.headers.get("content-length")
    if show:
        progress_bar = tqdm(
            total=int(content_length) if content_length is not None else None,
            leave=False,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            position=position,
        )

    # with tqdm
    with open(out, "wb") as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                if show:
                    progress_bar.update(len(chunk))
                f.write(chunk)


def load_config(training_dir: str, sample_rate: str, emb_channels: int, fp16: bool):
    if emb_channels == 256:
        config_path = os.path.join(ROOT_DIR, "configs", f"{sample_rate}.json")
    else:
        config_path = os.path.join(
            ROOT_DIR, "configs", f"{sample_rate}-{emb_channels}.json"
        )

    config = TrainConfig.parse_file(config_path)
    config.train.fp16_run = fp16

    config_save_path = os.path.join(training_dir, "config.json")

    with open(config_save_path, "w") as f:
        f.write(config.json())

    return config
