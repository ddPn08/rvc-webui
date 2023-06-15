import os
import traceback
from typing import *

import faiss
import numpy as np
import pyworld
import scipy.signal as signal
import torch
import torch.nn.functional as F
import torchcrepe
from torch import Tensor
# from faiss.swigfaiss_avx2 import IndexIVFFlat # cause crash on windows' faiss-cpu installed from pip
from fairseq.models.hubert import HubertModel

from .models import SynthesizerTrnMs256NSFSid


class VocalConvertPipeline(object):
    def __init__(self, tgt_sr: int, device: Union[str, torch.device], is_half: bool):
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

        self.sr = 16000  # hubert input sample rate
        self.window = 160  # hubert input window
        self.t_pad = self.sr * self.x_pad  # padding time for each utterance
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query  # query time before and after query point
        self.t_center = self.sr * self.x_center  # query cut point position
        self.t_max = self.sr * self.x_max  # max time for no query
        self.device = device
        self.is_half = is_half

    def get_optimal_torch_device(self, index: int = 0) -> torch.device:
        # Get cuda device
        if torch.cuda.is_available():
            return torch.device(f"cuda:{index % torch.cuda.device_count()}") # Very fast
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        # Insert an else here to grab "xla" devices if available. TO DO later. Requires the torch_xla.core.xla_model library
        # Else wise return the "cpu" as a torch device, 
        return torch.device("cpu")

    def get_f0_crepe_computation(
            self, 
            x, 
            f0_min,
            f0_max,
            p_len,
            hop_length=64, # 512 before. Hop length changes the speed that the voice jumps to a different dramatic pitch. Lower hop lengths means more pitch accuracy but longer inference time.
            model="full", # Either use crepe-tiny "tiny" or crepe "full". Default is full
    ):
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
            self.sr,
            hop_length,
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
            self,
            x,
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
            self.sr,
            self.window,
            f0_min,
            f0_max,
            model,
            batch_size=batch_size,
            device=self.device,
            return_periodicity=True,
        )
        pd = torchcrepe.filter.median(pd, 3)
        f0 = torchcrepe.filter.mean(f0, 3)
        f0[pd < 0.1] = 0
        f0 = f0[0].cpu().numpy()
        return f0

    def get_f0(
        self,
        x: np.ndarray,
        p_len: int,
        f0_up_key: int,
        f0_method: str,
        inp_f0: np.ndarray = None,
    ):
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)

        if f0_method == "harvest":
            f0, t = pyworld.harvest(
                x.astype(np.double),
                fs=self.sr,
                f0_ceil=f0_max,
                f0_floor=f0_min,
                frame_period=10,
            )
            f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.sr)
            f0 = signal.medfilt(f0, 3)
        elif f0_method == "dio":
            f0, t = pyworld.dio(
                x.astype(np.double),
                fs=self.sr,
                f0_ceil=f0_max,
                f0_floor=f0_min,
                frame_period=10,
            )
            f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.sr)
            f0 = signal.medfilt(f0, 3)
        elif f0_method == "mangio-crepe":
            f0 = self.get_f0_crepe_computation(x, f0_min, f0_max, p_len, 160, "full")
        elif f0_method == "crepe":
            f0 = self.get_f0_official_crepe_computation(x, f0_min, f0_max, "full")

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
        model: HubertModel,
        embedding_output_layer: int,
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

        half_support = (
            self.device.type == "cuda"
            and torch.cuda.get_device_capability(self.device)[0] >= 5.3
        )
        is_feats_dim_768 = net_g.emb_channels == 768

        if isinstance(model, tuple):
            feats = model[0](
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
                    feats = model[1](feats).last_hidden_state
                else:
                    feats = model[1](feats).extract_features
        else:
            inputs = {
                "source": feats.half().to(self.device)
                if half_support
                else feats.to(self.device),
                "padding_mask": padding_mask.to(self.device),
                "output_layer": embedding_output_layer,
            }

            if not half_support:
                model = model.float()
                inputs["source"] = inputs["source"].float()

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

            score, ix = index.search(npy, k=8)
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)

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
        model: HubertModel,
        embedding_output_layer: int,
        net_g: SynthesizerTrnMs256NSFSid,
        sid: int,
        audio: np.ndarray,
        transpose: int,
        f0_method: str,
        file_index: str,
        index_rate: float,
        if_f0: bool,
        f0_file: str = None,
    ):
        if file_index != "" and os.path.exists(file_index) and index_rate != 0:
            try:
                index = faiss.read_index(file_index)
                # big_npy = np.load(file_big_npy)
                big_npy = index.reconstruct_n(0, index.ntotal)
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
            if self.device.type == "mps":
                pitchf = pitchf.astype(np.float32)
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
                        embedding_output_layer,
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
                        embedding_output_layer,
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
                    embedding_output_layer,
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
                    embedding_output_layer,
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
