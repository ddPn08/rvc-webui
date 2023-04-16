import glob
import logging
import os
import shutil
import sys

import matplotlib
import matplotlib.pylab as plt
import numpy as np
import torch
from scipy.io.wavfile import read

from modules.shared import ROOT_DIR

from .config import TrainConfig

matplotlib.use("Agg")
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging


def load_checkpoint(checkpoint_path, model, optimizer=None, load_opt=1):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")

    saved_state_dict = checkpoint_dict["model"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():  # 模型需要的shape
        try:
            new_state_dict[k] = saved_state_dict[k]
            if saved_state_dict[k].shape != state_dict[k].shape:
                print(
                    f"shape-{k}-mismatch|need-{state_dict[k].shape}|get-{saved_state_dict[k].shape}"
                )
                raise KeyError
        except:
            # print(traceback.format_exc())
            print(f"{k} is not in the checkpoint")
            new_state_dict[k] = v  # 模型自带的随机值
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)
    print("Loaded model weights")

    epoch = checkpoint_dict["epoch"]
    learning_rate = checkpoint_dict["learning_rate"]
    if optimizer is not None and load_opt == 1:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    print("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, epoch))
    return model, optimizer, learning_rate, epoch


def save_state(model, optimizer, learning_rate, epoch, checkpoint_path):
    print(
        "Saving model and optimizer state at epoch {} to {}".format(
            epoch, checkpoint_path
        )
    )
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(
        {
            "model": state_dict,
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


def summarize(
    writer,
    global_step,
    scalars={},
    histograms={},
    images={},
    audios={},
    audio_sampling_rate=22050,
):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    filelist = glob.glob(os.path.join(dir_path, regex))
    if len(filelist) == 0:
        return None
    filelist.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    filepath = filelist[-1]
    return filepath


def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot_alignment_to_numpy(alignment, info=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        alignment.transpose(), aspect="auto", origin="lower", interpolation="none"
    )
    fig.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_config(training_dir: str, sample_rate: int):
    config_path = os.path.join(ROOT_DIR, "configs", f"{sample_rate}.json")
    config_save_path = os.path.join(training_dir, "config.json")

    shutil.copyfile(config_path, config_save_path)

    return TrainConfig.parse_file(config_save_path)
