import os
import sys
import shutil

from modules import models
from modules.utils import donwload_file
from modules.shared import ROOT_DIR


def install_ffmpeg():
    if os.path.exists(os.path.join(ROOT_DIR, "bin", "ffmpeg.exe")):
        return
    tmpdir = os.path.join(ROOT_DIR, "tmp")
    url = (
        "https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-5.1.2-essentials_build.zip"
    )
    out = os.path.join(tmpdir, "ffmpeg.zip")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    donwload_file(url, out)
    shutil.unpack_archive(out, os.path.join(tmpdir, "ffmpeg"))
    shutil.copyfile(
        os.path.join(
            tmpdir, "ffmpeg", "ffmpeg-5.1.2-essentials_build", "bin", "ffmpeg.exe"
        ),
        os.path.join(ROOT_DIR, "bin", "ffmpeg.exe"),
    )
    os.remove(os.path.join(tmpdir, "ffmpeg.zip"))
    shutil.rmtree(os.path.join(tmpdir, "ffmpeg"))


def preload():
    models.download_models()
    if sys.platform == "win32":
        install_ffmpeg()
