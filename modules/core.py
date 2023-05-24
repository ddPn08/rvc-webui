import hashlib
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

from modules.models import MODELS_DIR
from modules.shared import ROOT_DIR
from modules.utils import download_file


def get_hf_etag(url: str):
    r = requests.head(url)

    etag = r.headers["X-Linked-ETag"] if "X-Linked-ETag" in r.headers else ""

    if etag.startswith('"') and etag.endswith('"'):
        etag = etag[1:-1]

    return etag


def calc_sha256(filepath: str):
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_models():
    def hash_check(url: str, out: str):
        if not os.path.exists(out):
            return False
        etag = get_hf_etag(url)
        hash = calc_sha256(out)
        return etag == hash

    os.makedirs(os.path.join(MODELS_DIR, "pretrained", "v2"), exist_ok=True)

    tasks = []
    for basename in [
        f"{f0}{net}{rate}k{channels}"
        for f0 in ["", "f0"]
        for net in ["D", "G"]
        for rate in [32, 40, 48]
        for channels in [256, 768]
    ]:

        url = f"https://huggingface.co/ddPn08/rvc-webui-models/resolve/main/pretrained/v1/{basename}.pth"
        out = os.path.join(MODELS_DIR, "pretrained", f"{basename}.pth")

        if hash_check(url, out):
            continue

        tasks.append((url, out))

    for template in [
        "D{}k",
        "G{}k",
        "f0D{}k",
        "f0G{}k",
    ]:
        basename = template.format("40")

        url = f"https://huggingface.co/ddPn08/rvc-webui-models/resolve/main/pretrained/v2/{basename}.pth"
        out = os.path.join(MODELS_DIR, "pretrained", "v2", f"{basename}.pth")

        if hash_check(url, out):
            continue

        tasks.append((url, out))

    for filename in [
        "checkpoint_best_legacy_500.pt",
    ]:
        out = os.path.join(MODELS_DIR, "embeddings", filename)
        url = f"https://huggingface.co/ddPn08/rvc-webui-models/resolve/main/embeddings/{filename}"

        if hash_check(url, out):
            continue

        tasks.append(
            (
                f"https://huggingface.co/ddPn08/rvc-webui-models/resolve/main/embeddings/{filename}",
                out,
            )
        )

    # japanese-hubert-base (Fairseq)
    # from official repo
    # NOTE: change filename?
    hubert_jp_url = f"https://huggingface.co/rinna/japanese-hubert-base/resolve/main/fairseq/model.pt"
    out = os.path.join(MODELS_DIR, "embeddings", "rinna_hubert_base_jp.pt")
    if not hash_check(hubert_jp_url, out):
        tasks.append(
            (
                hubert_jp_url,
                out,
            )
        )

    if len(tasks) < 1:
        return

    with ThreadPoolExecutor() as pool:
        pool.map(
            download_file,
            *zip(
                *[(filename, out, i, True) for i, (filename, out) in enumerate(tasks)]
            ),
        )


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
