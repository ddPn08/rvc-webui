import operator
import os
from concurrent.futures import ProcessPoolExecutor
from typing import *

import librosa
import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
from tqdm import tqdm

from modules.models import MODELS_DIR
from modules.utils import load_audio

from .slicer import Slicer

SR_K_DICT = {
    32000: "32k",
    40000: "40k",
    48000: "48k",
}


def norm_write(
    tmp_audio: np.ndarray,
    idx0: int,
    idx1: int,
    speaker_id: int,
    outdir: str,
    outdir_16k: str,
    sampling_rate: int,
    max: float,
    alpha: float,
    is_normalize: bool,
):
    if is_normalize:
        tmp_audio = (tmp_audio / np.abs(tmp_audio).max() * (max * alpha)) + (
            1 - alpha
        ) * tmp_audio
    else:
        # clip level to max (cause sometimes when floating point decoding)
        audio_min = np.min(tmp_audio)
        if audio_min < -max:
            tmp_audio = tmp_audio / -audio_min * max
        audio_max = np.max(tmp_audio)
        if audio_max > max:
            tmp_audio = tmp_audio / audio_max * max

    wavfile.write(
        os.path.join(outdir, f"{speaker_id:05}", f"{idx0}_{idx1}.wav"),
        sampling_rate,
        tmp_audio.astype(np.float32),
    )

    tmp_audio = librosa.resample(
        tmp_audio, orig_sr=sampling_rate, target_sr=16000, res_type="soxr_vhq"
    )
    wavfile.write(
        os.path.join(outdir_16k, f"{speaker_id:05}", f"{idx0}_{idx1}.wav"),
        16000,
        tmp_audio.astype(np.float32),
    )


def write_mute(
    mute_wave_filename: str,
    speaker_id: int,
    outdir: str,
    outdir_16k: str,
    sampling_rate: int,
):
    tmp_audio = load_audio(mute_wave_filename, sampling_rate)
    wavfile.write(
        os.path.join(outdir, f"{speaker_id:05}", "mute.wav"),
        sampling_rate,
        tmp_audio.astype(np.float32),
    )
    tmp_audio = librosa.resample(
        tmp_audio, orig_sr=sampling_rate, target_sr=16000, res_type="soxr_vhq"
    )
    wavfile.write(
        os.path.join(outdir_16k, f"{speaker_id:05}", "mute.wav"),
        16000,
        tmp_audio.astype(np.float32),
    )


def pipeline(
    slicer: Slicer,
    datasets: List[Tuple[str, int]],  # List[(path, speaker_id)]
    outdir: str,
    outdir_16k: str,
    sampling_rate: int,
    is_normalize: bool,
    process_id: int = 0,
):
    per = 3.7
    overlap = 0.3
    tail = per + overlap
    max = 0.95
    alpha = 0.8

    bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=sampling_rate)

    for index, (wave_filename, speaker_id) in tqdm(datasets, position=1 + process_id):
        audio = load_audio(wave_filename, sampling_rate)
        audio = signal.lfilter(bh, ah, audio)

        idx1 = 0
        for audio in slicer.slice(audio):
            i = 0
            while 1:
                start = int(sampling_rate * (per - overlap) * i)
                i += 1
                if len(audio[start:]) > tail * sampling_rate:
                    tmp_audio = audio[start : start + int(per * sampling_rate)]
                    norm_write(
                        tmp_audio,
                        index,
                        idx1,
                        speaker_id,
                        outdir,
                        outdir_16k,
                        sampling_rate,
                        max,
                        alpha,
                        is_normalize,
                    )
                    idx1 += 1
                else:
                    tmp_audio = audio[start:]
                    break
            norm_write(
                tmp_audio,
                index,
                idx1,
                speaker_id,
                outdir,
                outdir_16k,
                sampling_rate,
                max,
                alpha,
                is_normalize,
            )


def preprocess_audio(
    datasets: List[Tuple[str, int]],  # List[(path, speaker_id)]
    sampling_rate: int,
    num_processes: int,
    training_dir: str,
    is_normalize: bool,
):
    waves_dir = os.path.join(training_dir, "0_gt_wavs")
    waves16k_dir = os.path.join(training_dir, "1_16k_wavs")
    if os.path.exists(waves_dir) and os.path.exists(waves16k_dir):
        return

    for speaker_id in set([spk for _, spk in datasets]):
        os.makedirs(os.path.join(waves_dir, f"{speaker_id:05}"), exist_ok=True)
        os.makedirs(os.path.join(waves16k_dir, f"{speaker_id:05}"), exist_ok=True)

    all = [(i, x) for i, x in enumerate(sorted(datasets, key=operator.itemgetter(0)))]

    # n of datasets per process
    process_all_nums = [len(all) // num_processes] * num_processes
    # add residual datasets
    for i in range(len(all) % num_processes):
        process_all_nums[i] += 1

    assert len(all) == sum(process_all_nums), print(
        f"len(all): {len(all)}, sum(process_all_nums): {sum(process_all_nums)}"
    )

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        all_index = 0
        for i in range(num_processes):
            data = all[all_index : all_index + process_all_nums[i]]
            slicer = Slicer(
                sr=sampling_rate,
                threshold=-40,
                min_length=800,
                min_interval=400,
                hop_size=15,
                max_sil_kept=150,
            )
            executor.submit(
                pipeline,
                slicer,
                data,
                waves_dir,
                waves16k_dir,
                sampling_rate,
                is_normalize,
                process_id=i,
            )
            all_index += process_all_nums[i]

    mute_wav = os.path.join(
        MODELS_DIR,
        "training",
        "mute",
        "0_gt_wavs",
        f"mute{SR_K_DICT[sampling_rate]}.wav",
    )
    for speaker_id in set([spk for _, spk in datasets]):
        write_mute(mute_wav, speaker_id, waves_dir, waves16k_dir, sampling_rate)
