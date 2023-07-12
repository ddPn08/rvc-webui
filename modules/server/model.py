import os
import re
from typing import *

import faiss
import numpy as np
import pyworld
import scipy.signal as signal
import torch
import torch.nn.functional as F
import torchaudio
import torchcrepe
from fairseq import checkpoint_utils
from fairseq.models.hubert.hubert import HubertModel
from pydub import AudioSegment
from torch import Tensor

from lib.rvc.models import (SynthesizerTrnMs256NSFSid,
                            SynthesizerTrnMs256NSFSidNono)
from lib.rvc.pipeline import VocalConvertPipeline
from modules.cmd_opts import opts
from modules.models import (EMBEDDINGS_LIST, MODELS_DIR, get_embedder,
                            get_vc_model, update_state_dict)
from modules.shared import ROOT_DIR, device, is_half

MODELS_DIR = opts.models_dir or os.path.join(ROOT_DIR, "models")
vc_model: Optional["VoiceServerModel"] = None
embedder_model: Optional[HubertModel] = None
loaded_embedder_model = ""


class VoiceServerModel:
    def __init__(self, rvc_model_file: str, faiss_index_file: str) -> None:
        # setting vram
        global device, is_half
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == "cuda":
            vram = torch.cuda.get_device_properties(device).total_memory / 1024**3
        else:
            vram = None
        if vram is not None and vram <= 4:
            self.x_pad = 1
            self.x_query = 5
            self.x_center = 30
            self.x_max = 32
        elif vram is not None and vram <= 5:
            self.x_pad = 1
            self.x_query = 6
            self.x_center = 38
            self.x_max = 41
        else:
            self.x_pad = 3
            self.x_query = 10
            self.x_center = 60
            self.x_max = 65

        # load_model
        state_dict = torch.load(rvc_model_file, map_location="cpu")
        update_state_dict(state_dict)
        self.state_dict = state_dict
        self.tgt_sr = state_dict["params"]["sr"]
        self.f0 = state_dict.get("f0", 1)
        state_dict["params"]["spk_embed_dim"] = state_dict["weight"][
            "emb_g.weight"
        ].shape[0]
        if not "emb_channels" in state_dict["params"]:
            if state_dict.get("version", "v1") == "v1":
                state_dict["params"]["emb_channels"] = 256  # for backward compat.
                state_dict["embedder_output_layer"] = 9
            else:
                state_dict["params"]["emb_channels"] = 768  # for backward compat.
                state_dict["embedder_output_layer"] = 12
        if self.f0 == 1:
            self.net_g = SynthesizerTrnMs256NSFSid(
                **state_dict["params"], is_half=is_half
            )
        else:
            self.net_g = SynthesizerTrnMs256NSFSidNono(**state_dict["params"])
        del self.net_g.enc_q
        self.net_g.load_state_dict(state_dict["weight"], strict=False)
        self.net_g.eval().to(device)
        if is_half:
            self.net_g = self.net_g.half()
        else:
            self.net_g = self.net_g.float()

        emb_name = state_dict.get("embedder_name", "contentvec")
        if emb_name == "hubert_base":
            emb_name = "contentvec"
        emb_file = os.path.join(MODELS_DIR, "embeddings", EMBEDDINGS_LIST[emb_name][0])
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            [emb_file],
            suffix="",
        )
        embedder_model = models[0]
        embedder_model = embedder_model.to(device)

        if is_half:
            embedder_model = embedder_model.half()
        else:
            embedder_model = embedder_model.float()
        embedder_model.eval()
        self.embedder_model = embedder_model

        self.embedder_output_layer = state_dict["embedder_output_layer"]

        self.index = None
        if faiss_index_file != "" and os.path.exists(faiss_index_file):
            self.index = faiss.read_index(faiss_index_file)
            self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)

        self.n_spk = state_dict["params"]["spk_embed_dim"]

        self.sr = 16000  # hubert input sample rate
        self.window = 160  # hubert input window
        self.t_pad = self.sr * self.x_pad  # padding time for each utterance
        self.t_pad_tgt = self.tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query  # query time before and after query point
        self.t_center = self.sr * self.x_center  # query cut point position
        self.t_max = self.sr * self.x_max  # max time for no query
        self.device = device
        self.is_half = is_half

    def __call__(
        self,
        audio: np.ndarray,
        sr: int,
        sid: int,
        transpose: int,
        f0_method: str,
        index_rate: float,
    ):
        # bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)
        # audio = signal.filtfilt(bh, ah, audio)
        if sr != self.sr:
            audio = torchaudio.functional.resample(torch.from_numpy(audio), sr, self.sr, rolloff=0.99).detach().cpu().numpy()
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect" if audio.shape[0] > self.window // 2 else "constant")

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
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect" if audio.shape[0] > self.t_pad else "constant")
        p_len = audio_pad.shape[0] // self.window

        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        pitch, pitchf = None, None
        if self.f0 == 1:
            pitch, pitchf = get_f0(audio_pad, self.sr, p_len, transpose, f0_method)
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            if self.device.type == "mps":
                pitchf = pitchf.astype(np.float32)
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()

        audio_opt = []

        s = 0
        t = None

        for t in opt_ts:
            t = t // self.window * self.window
            if self.f0 == 1:
                audio_opt.append(
                    self._convert(
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        pitch[:, s // self.window : (t + self.t_pad2) // self.window],
                        pitchf[:, s // self.window : (t + self.t_pad2) // self.window],
                        index_rate,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            else:
                audio_opt.append(
                    self._convert(
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        None,
                        None,
                        index_rate,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            s = t
        if self.f0 == 1:
            audio_opt.append(
                self._convert(
                    sid,
                    audio_pad[t:],
                    pitch[:, t // self.window :] if t is not None else pitch,
                    pitchf[:, t // self.window :] if t is not None else pitchf,
                    index_rate,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        else:
            audio_opt.append(
                self._convert(
                    sid,
                    audio_pad[t:],
                    None,
                    None,
                    index_rate,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        audio_opt = np.concatenate(audio_opt)
        del pitch, pitchf, sid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio_opt


    def _convert(
        self,
        sid: int,
        audio: np.ndarray,
        pitch: Optional[np.ndarray],
        pitchf: Optional[np.ndarray],
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

        half_support = (
            self.device.type == "cuda"
            and torch.cuda.get_device_capability(self.device)[0] >= 5.3
        )
        is_feats_dim_768 = self.net_g.emb_channels == 768

        if isinstance(self.embedder_model, tuple):
            feats = self.embedder_model[0](
                feats.squeeze(0).squeeze(0).to(self.device),
                return_tensors="pt",
                sampling_rate=16000,
            )
            if self.is_half:
                feats = feats.input_values.to(self.device).half()
            else:
                feats = feats.input_values.to(self.device)
            with torch.no_grad():
                if is_feats_dim_768:
                    feats = self.embedder_model[1](feats).last_hidden_state
                else:
                    feats = self.embedder_model[1](feats).extract_features
        else:
            inputs = {
                "source": feats.half().to(self.device)
                if half_support
                else feats.to(self.device),
                "padding_mask": padding_mask.to(self.device),
                "output_layer": self.embedder_output_layer,
            }

            if not half_support:
                self.embedder_model = self.embedder_model.float()
                inputs["source"] = inputs["source"].float()

            with torch.no_grad():
                logits = self.embedder_model.extract_features(**inputs)
                if is_feats_dim_768:
                    feats = logits[0]
                else:
                    feats = self.embedder_model.final_proj(logits[0])

        if (
            isinstance(self.index, type(None)) == False
            and isinstance(self.big_npy, type(None)) == False
            and index_rate != 0
        ):
            npy = feats[0].cpu().numpy()
            if self.is_half:
                npy = npy.astype("float32")

            _, ix = self.index.search(npy, k=1)
            npy = self.big_npy[ix[:, 0]]

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
                    (self.net_g.infer(feats, p_len, pitch, pitchf, sid)[0][0, 0] * 32768)
                    .data.cpu()
                    .float()
                    .numpy()
                    .astype(np.int16)
                )
            else:
                audio1 = (
                    (self.net_g.infer(feats, p_len, sid)[0][0, 0] * 32768)
                    .data.cpu()
                    .float()
                    .numpy()
                    .astype(np.int16)
                )
        del feats, p_len, padding_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio1


# F0 computation
def get_f0_crepe_computation(
        x,
        sr,
        f0_min,
        f0_max,
        p_len,
        model="full", # Either use crepe-tiny "tiny" or crepe "full". Default is full
):
    hop_length = sr // 100
    x = x.astype(np.float32) # fixes the F.conv2D exception. We needed to convert double to float.
    x /= np.quantile(np.abs(x), 0.999)
    torch_device = self.get_optimal_torch_device()
    audio = torch.from_numpy(x).to(torch_device, copy=True)
    audio = torch.unsqueeze(audio, dim=0)
    if audio.ndim == 2 and audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True).detach()
    audio = audio.detach()
    print("Initiating prediction with a crepe_hop_length of: " + str(hop_length))
    pitch: Tensor = torchcrepe.predict(
        audio,
        sr,
        sr // 100,
        f0_min,
        f0_max,
        model,
        batch_size=hop_length * 2,
        device=torch_device,
        pad=True
    )
    p_len = p_len or x.shape[0] // hop_length
    # Resize the pitch for final f0
    source = np.array(pitch.squeeze(0).cpu().float().numpy())
    source[source < 0.001] = np.nan
    target = np.interp(
        np.arange(0, len(source) * p_len, len(source)) / p_len,
        np.arange(0, len(source)),
        source
    )
    f0 = np.nan_to_num(target)
    return f0 # Resized f0

def get_f0_official_crepe_computation(
        x,
        sr,
        f0_min,
        f0_max,
        model="full",
):
    # Pick a batch size that doesn't cause memory errors on your gpu
    batch_size = 512
    # Compute pitch using first gpu
    audio = torch.tensor(np.copy(x))[None].float()
    f0, pd = torchcrepe.predict(
        audio,
        sr,
        sr // 100,
        f0_min,
        f0_max,
        model,
        batch_size=batch_size,
        device=device,
        return_periodicity=True,
    )
    pd = torchcrepe.filter.median(pd, 3)
    f0 = torchcrepe.filter.mean(f0, 3)
    f0[pd < 0.1] = 0
    f0 = f0[0].cpu().numpy()
    return f0

def get_f0(
    x: np.ndarray,
    sr: int,
    p_len: int,
    f0_up_key: int,
    f0_method: str,
):
    f0_min = 50
    f0_max = 1100
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)

    if f0_method == "harvest":
        f0, t = pyworld.harvest(
            x.astype(np.double),
            fs=sr,
            f0_ceil=f0_max,
            f0_floor=f0_min,
            frame_period=10,
        )
        f0 = pyworld.stonemask(x.astype(np.double), f0, t, sr)
        f0 = signal.medfilt(f0, 3)
    elif f0_method == "dio":
        f0, t = pyworld.dio(
            x.astype(np.double),
            fs=sr,
            f0_ceil=f0_max,
            f0_floor=f0_min,
            frame_period=10,
        )
        f0 = pyworld.stonemask(x.astype(np.double), f0, t, sr)
        f0 = signal.medfilt(f0, 3)
    elif f0_method == "mangio-crepe":
        f0 = get_f0_crepe_computation(x, sr, f0_min, f0_max, p_len, "full")
    elif f0_method == "crepe":
        f0 = get_f0_official_crepe_computation(x, sr, f0_min, f0_max, "full")

    f0 *= pow(2, f0_up_key / 12)
    f0bak = f0.copy()
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
        f0_mel_max - f0_mel_min
    ) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    f0_coarse = np.rint(f0_mel).astype(np.int32)
    return f0_coarse, f0bak  # 1-0