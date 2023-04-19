import operator
import os
import traceback
from typing import *

import librosa
import numpy as np
import tqdm
import scipy.signal as signal
from scipy.io import wavfile

from modules.models import MODELS_DIR
from modules.utils import load_audio

from .slicer import Slicer

SR_K_DICT = {
    32000: "32k",
    40000: "40k",
    48000: "48k",
}


class PreProcess:
    def __init__(self, sampling_rate, training_dir):
        self.slicer = Slicer(
            sr=sampling_rate,
            threshold=-32,
            min_length=800,
            min_interval=400,
            hop_size=15,
            max_sil_kept=150,
        )
        self.sr = sampling_rate
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr)
        self.per = 3.7
        self.overlap = 0.3
        self.tail = self.per + self.overlap
        self.max = 0.95
        self.alpha = 0.8

        self.gt_wavs_dir = os.path.join(training_dir, "0_gt_wavs")
        self.wavs16k_dir = os.path.join(training_dir, "1_16k_wavs")

    def norm_write(self, tmp_audio, idx0, idx1, speaker_id, is_normalize):
        if is_normalize:
            tmp_audio = (
                tmp_audio / np.abs(tmp_audio).max() * (self.max * self.alpha)
            ) + (1 - self.alpha) * tmp_audio

        wavfile.write(
            os.path.join(self.gt_wavs_dir, f"{speaker_id:05}", f"{idx0}_{idx1}.wav"),
            self.sr,
            tmp_audio.astype(np.float32),
        )

        tmp_audio = librosa.resample(
            tmp_audio, orig_sr=self.sr, target_sr=16000, res_type="soxr_vhq"
        )
        wavfile.write(
            os.path.join(self.wavs16k_dir, f"{speaker_id:05}", f"{idx0}_{idx1}.wav"),
            16000,
            tmp_audio.astype(np.float32),
        )

    def write_mute(self, mute_wave_filename, speaker_id):
        tmp_audio = load_audio(mute_wave_filename, self.sr)
        wavfile.write(
            os.path.join(self.gt_wavs_dir, f"{speaker_id:05}", "mute.wav"),
            self.sr,
            tmp_audio.astype(np.float32),
        )
        tmp_audio = librosa.resample(
            tmp_audio, orig_sr=self.sr, target_sr=16000, res_type="soxr_vhq"
        )
        wavfile.write(
            os.path.join(self.wavs16k_dir, f"{speaker_id:05}", "mute.wav"),
            16000,
            tmp_audio.astype(np.float32),
        )

    def pipeline(self, speaker_id: int, path: str, index: int, is_normalize: bool):
        try:
            audio = load_audio(path, self.sr)
            audio = signal.lfilter(self.bh, self.ah, audio)

            idx1 = 0
            for audio in self.slicer.slice(audio):
                i = 0
                while 1:
                    start = int(self.sr * (self.per - self.overlap) * i)
                    i += 1
                    if len(audio[start:]) > self.tail * self.sr:
                        tmp_audio = audio[start : start + int(self.per * self.sr)]
                        self.norm_write(
                            tmp_audio, index, idx1, speaker_id, is_normalize
                        )
                        idx1 += 1
                    else:
                        tmp_audio = audio[start:]
                        break
                self.norm_write(tmp_audio, index, idx1, speaker_id, is_normalize)
        except:
            traceback.print_exc()

    def pipeline_mapping(
        self, datasets: List[Tuple[str, int]], num_processes: int, is_normalize: bool
    ):
        for speaker_id in set([spk for _, spk in datasets]):
            os.makedirs(
                os.path.join(self.gt_wavs_dir, f"{speaker_id:05}"), exist_ok=True
            )
            os.makedirs(
                os.path.join(self.wavs16k_dir, f"{speaker_id:05}"), exist_ok=True
            )
        for index, path_spk in enumerate(
            tqdm.tqdm(sorted(datasets, key=operator.itemgetter(0)))
        ):
            self.pipeline(path_spk[1], path_spk[0], index, is_normalize)

        # def task(infos):
        #     for path, index in tqdm.tqdm(infos):
        #         self.pipeline(path, index)

        # with ProcessPoolExecutor() as executor:
        #     for i in range(num_processes):
        #         executor.submit(task, infos[i::num_processes])


def preprocess_dataset(
    datasets: List[Tuple[str, int]],  # List[(path, speaker_id)]
    sampling_rate: int,
    num_processes: int,
    training_dir: str,
    is_normalize: bool,
):
    pp = PreProcess(sampling_rate, training_dir)
    if os.path.exists(pp.gt_wavs_dir) and os.path.exists(pp.wavs16k_dir):
        return
    pp.pipeline_mapping(datasets, num_processes, is_normalize)

    # process mute file
    mute_wav = os.path.join(
        MODELS_DIR,
        "training",
        "mute",
        "0_gt_wavs",
        f"mute{SR_K_DICT[sampling_rate]}.wav",
    )
    for speaker_id in set([spk for _, spk in datasets]):
        pp.write_mute(mute_wav, speaker_id)
