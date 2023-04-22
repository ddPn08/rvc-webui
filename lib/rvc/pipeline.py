import os
import traceback
from typing import *

import faiss
import numpy as np
import parselmouth
import pyworld
import scipy.signal as signal
import torch
import torch.nn.functional as F
# from faiss.swigfaiss_avx2 import IndexIVFFlat # cause crash on windows' faiss-cpu installed from pip
from fairseq.models.hubert import HubertModel
from transformers import HubertModel as TrHubertModel
from transformers import Wav2Vec2FeatureExtractor

from .models import SynthesizerTrnMs256NSFSid


class VocalConvertPipeline(object):
    def __init__(self, tgt_sr: int, device: Union[str, torch.device], is_half: bool):
        self.x_pad = 3 if is_half else 1
        self.x_query = 10 if is_half else 6
        self.x_center = 60 if is_half else 30
        self.x_max = 65 if is_half else 32

        self.sr = 16000  # hubert input sample rate
        self.window = 160  # hubert input window
        self.t_pad = self.sr * self.x_pad  # padding time for each utterance
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query  # query time before and after query point
        self.t_center = self.sr * self.x_query  # query cut point position
        self.t_max = self.sr * self.x_max  # max time for no query
        self.device = device
        self.is_half = is_half

    def get_f0(
        self,
        x: np.ndarray,
        p_len: int,
        f0_up_key: int,
        f0_method: str,
        inp_f0: np.ndarray = None,
    ):
        time_step = self.window / self.sr * 1000
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        if f0_method == "pm":
            f0 = (
                parselmouth.Sound(x, self.sr)
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
                fs=self.sr,
                f0_ceil=f0_max,
                f0_floor=f0_min,
                frame_period=10,
            )
            f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.sr)
            f0 = signal.medfilt(f0, 3)
        f0 *= pow(2, f0_up_key / 12)
        tf0 = self.sr // self.window  # f0 points per second
        if inp_f0 is not None:
            delta_t = np.round(
                (inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1
            ).astype("int16")
            replace_f0 = np.interp(
                list(range(delta_t)), inp_f0[:, 0] * 100, inp_f0[:, 1]
            )
            shape = f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)].shape[0]
            f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)] = replace_f0[
                :shape
            ]

        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int)
        return f0_coarse, f0bak  # 1-0

    def _convert(
        self,
        model: Union[HubertModel, Tuple[Wav2Vec2FeatureExtractor, TrHubertModel]],
        net_g: SynthesizerTrnMs256NSFSid,
        sid: int,
        audio: np.ndarray,
        pitch: np.ndarray,
        pitchf: np.ndarray,
        index: faiss.IndexIVFFlat,
        big_npy: np.ndarray,
        index_rate: float,
    ):
        feats = torch.from_numpy(audio)
        if self.is_half:
            feats = feats.half()
        else:
            feats = feats.float()
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)

        is_feats_dim_768 = net_g.emb_channels == 768
        
        if isinstance(model, tuple):
            feats = model[0](feats.squeeze(0).squeeze(0).to(self.device), return_tensors="pt", sampling_rate=16000)
            if self.is_half:
                feats = feats.input_values.to(self.device).half()
            else:
                feats = feats.input_values.to(self.device)
            with torch.no_grad():
                if is_feats_dim_768:
                    feats = model[1](feats).last_hidden_state
                else:
                    feats = model[1](feats).extract_features
        else:
            inputs = (
                {
                    "source": feats.to(self.device),
                    "padding_mask": padding_mask,
                    "output_layer": 9,  # layer 9
                }
                if not is_feats_dim_768
                else {
                    "source": feats.to(self.device),
                    "padding_mask": padding_mask,
                    # no pass "output_layer"
                }
            )

            with torch.no_grad():
                logits = model.extract_features(**inputs)
                if is_feats_dim_768:
                    feats = logits[0]
                else:
                    feats = model.final_proj(logits[0])

        if (
            isinstance(index, type(None)) == False
            and isinstance(big_npy, type(None)) == False
            and index_rate != 0
        ):
            npy = feats[0].cpu().numpy()
            if self.is_half:
                npy = npy.astype("float32")
            D, I = index.search(npy, 1)
            npy = big_npy[I.squeeze()]
            if self.is_half:
                npy = npy.astype("float16")
            feats = (
                torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
                + (1 - index_rate) * feats
            )

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

        p_len = audio.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch != None and pitchf != None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]
        p_len = torch.tensor([p_len], device=self.device).long()
        with torch.no_grad():
            if pitch != None and pitchf != None:
                audio1 = (
                    (net_g.infer(feats, p_len, pitch, pitchf, sid)[0][0, 0] * 32768)
                    .data.cpu()
                    .float()
                    .numpy()
                    .astype(np.int16)
                )
            else:
                audio1 = (
                    (net_g.infer(feats, p_len, sid)[0][0, 0] * 32768)
                    .data.cpu()
                    .float()
                    .numpy()
                    .astype(np.int16)
                )
        del feats, p_len, padding_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio1

    def __call__(
        self,
        model: Union[HubertModel, Tuple[Wav2Vec2FeatureExtractor, TrHubertModel]],
        net_g: SynthesizerTrnMs256NSFSid,
        sid: int,
        audio: np.ndarray,
        transpose: int,
        f0_method: str,
        file_index: str,
        file_big_npy: str,
        index_rate: float,
        if_f0: bool,
        f0_file: str = None,
    ):
        if (
            file_big_npy != ""
            and file_index != ""
            and os.path.exists(file_big_npy)
            and os.path.exists(file_index)
            and index_rate != 0
        ):
            try:
                index = faiss.read_index(file_index)
                big_npy = np.load(file_big_npy)
                print(f"Loaded {file_index} and {file_big_npy}")
            except:
                traceback.print_exc()
                index = big_npy = None
        else:
            index = big_npy = None

        bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)
        audio = signal.filtfilt(bh, ah, audio)

        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += audio_pad[i : i - self.window]
            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t
                    - self.t_query
                    + np.where(
                        np.abs(audio_sum[t - self.t_query : t + self.t_query])
                        == np.abs(audio_sum[t - self.t_query : t + self.t_query]).min()
                    )[0][0]
                )

        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        inp_f0 = None
        if hasattr(f0_file, "name"):
            try:
                with open(f0_file.name, "r") as f:
                    lines = f.read().strip("\n").split("\n")
                inp_f0 = []
                for line in lines:
                    inp_f0.append([float(i) for i in line.split(",")])
                inp_f0 = np.array(inp_f0, dtype="float32")
            except:
                traceback.print_exc()
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        pitch, pitchf = None, None
        if if_f0 == 1:
            pitch, pitchf = self.get_f0(audio_pad, p_len, transpose, f0_method, inp_f0)
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()

        audio_opt = []

        s = 0
        t = None

        for t in opt_ts:
            t = t // self.window * self.window
            if if_f0 == 1:
                audio_opt.append(
                    self._convert(
                        model,
                        net_g,
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        pitch[:, s // self.window : (t + self.t_pad2) // self.window],
                        pitchf[:, s // self.window : (t + self.t_pad2) // self.window],
                        index,
                        big_npy,
                        index_rate,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            else:
                audio_opt.append(
                    self._convert(
                        model,
                        net_g,
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        None,
                        None,
                        index,
                        big_npy,
                        index_rate,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            s = t
        if if_f0 == 1:
            audio_opt.append(
                self._convert(
                    model,
                    net_g,
                    sid,
                    audio_pad[t:],
                    pitch[:, t // self.window :] if t is not None else pitch,
                    pitchf[:, t // self.window :] if t is not None else pitchf,
                    index,
                    big_npy,
                    index_rate,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        else:
            audio_opt.append(
                self._convert(
                    model,
                    net_g,
                    sid,
                    audio_pad[t:],
                    None,
                    None,
                    index,
                    big_npy,
                    index_rate,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        audio_opt = np.concatenate(audio_opt)
        del pitch, pitchf, sid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio_opt
