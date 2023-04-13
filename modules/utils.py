import ffmpeg
import numpy as np
import requests
import torch


def load_audio(file, sr):
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = (
            file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )  # Prevent small white copy path head and tail with spaces and " and return
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def get_gpus():
    num_gpus = torch.cuda.device_count()
    return [torch.device(f"cuda:{i}") for i in range(num_gpus)]


def donwload_file(url, out):
    req = requests.get(url, allow_redirects=True)
    with open(out, "wb") as f:
        f.write(req.content)
