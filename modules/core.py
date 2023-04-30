import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from modules.models import MODELS_DIR
from modules.shared import ROOT_DIR
from modules.utils import download_file


def download_models():
    tasks = []
    for template in [
        "D{}k{}",
        "G{}k{}",
        "f0D{}k{}",
        "f0G{}k{}",
    ]:
        for sr in ["32", "40", "48"]:
            for emb_channels in ["256", "768"]:
                basename = template.format(sr, emb_channels)
                filename = f"https://huggingface.co/ddPn08/rvc-webui-models/resolve/main/pretrained/{basename}.pth"
                out = os.path.join(MODELS_DIR, "pretrained", f"{basename}.pth")
                if os.path.exists(out):
                    # TODO: Hash check
                    continue
                tasks.append((filename, out))

    for filename in [
        "hubert_base.pt",
        "checkpoint_best_legacy_500.pt",
    ]:
        out = os.path.join(MODELS_DIR, "embeddings", filename)
        if not os.path.exists(out):
            tasks.append(
                (
                    f"https://huggingface.co/ddPn08/rvc-webui-models/resolve/main/embeddings/{filename}",
                    out,
                )
            )

    # japanese-hubert-base (Fairseq)
    # from official repo
    # NOTE: change filename?
    out = os.path.join(MODELS_DIR, "embeddings", "model.pt")
    if not os.path.exists(out):
        tasks.append(
            (
                f"https://huggingface.co/rinna/japanese-hubert-base/resolve/main/fairseq/model.pt",
                out,
            )
        )

    if len(tasks) < 1:
        return

    with ProcessPoolExecutor() as pool:
        with tqdm(total=len(tasks)) as progress:
            for _ in pool.map(
                download_file,
                *zip(*[(filename, out, 0, False) for filename, out in tasks]),
            ):
                progress.update()


def install_ffmpeg():
    if os.path.exists(os.path.join(ROOT_DIR, "bin", "ffmpeg.exe")):
        return
    tmpdir = os.path.join(ROOT_DIR, "tmp")
    url = (
        "https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-5.1.2-essentials_build.zip"
    )
    out = os.path.join(tmpdir, "ffmpeg.zip")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    download_file(url, out)
    shutil.unpack_archive(out, os.path.join(tmpdir, "ffmpeg"))
    shutil.copyfile(
        os.path.join(
            tmpdir, "ffmpeg", "ffmpeg-5.1.2-essentials_build", "bin", "ffmpeg.exe"
        ),
        os.path.join(ROOT_DIR, "bin", "ffmpeg.exe"),
    )
    os.remove(os.path.join(tmpdir, "ffmpeg.zip"))
    shutil.rmtree(os.path.join(tmpdir, "ffmpeg"))


def update_modelnames():
    for sr in ["32k", "40k", "48k"]:
        files = [
            f"f0G{sr}",
            f"f0D{sr}",
            f"G{sr}",
            f"D{sr}",
        ]
        for file in files:
            filepath = os.path.join(MODELS_DIR, "pretrained", f"{file}.pth")
            if os.path.exists(filepath):
                os.rename(
                    filepath,
                    os.path.join(MODELS_DIR, "pretrained", f"{file}256.pth"),
                )

    if not os.path.exists(os.path.join(MODELS_DIR, "embeddings")):
        os.makedirs(os.path.join(MODELS_DIR, "embeddings"))

    if os.path.exists(os.path.join(MODELS_DIR, "hubert_base.pt")):
        os.rename(
            os.path.join(MODELS_DIR, "hubert_base.pt"),
            os.path.join(MODELS_DIR, "embeddings", "hubert_base.pt"),
        )
    if os.path.exists(os.path.join(MODELS_DIR, "checkpoint_best_legacy_500.pt")):
        os.rename(
            os.path.join(MODELS_DIR, "checkpoint_best_legacy_500.pt"),
            os.path.join(MODELS_DIR, "embeddings", "checkpoint_best_legacy_500.pt"),
        )


def preload():
    update_modelnames()
    download_models()
    if sys.platform == "win32":
        install_ffmpeg()
