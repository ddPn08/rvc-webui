import os
from typing import *

import torch
from fairseq import checkpoint_utils

from .cmd_opts import opts
from .inference.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono
from .inference.pipeline import VC
from .shared import ROOT_DIR, device, is_half
from .utils import load_audio


class VC_MODEL:
    def __init__(self, model_name: str, weight: Dict[str, Any]) -> None:
        self.model_name = model_name
        self.weight = weight
        self.tgt_sr = weight["config"][-1]
        f0 = weight.get("f0", 1)
        weight["config"][-3] = weight["weight"]["emb_g.weight"].shape[0]

        if f0 == 1:
            self.net_g = SynthesizerTrnMs256NSFsid(*weight["config"], is_half=is_half)
        else:
            self.net_g = SynthesizerTrnMs256NSFsid_nono(*weight["config"])

        del self.net_g.enc_q

        self.net_g.load_state_dict(weight["weight"], strict=False)
        self.net_g.eval().to(device)

        if is_half:
            self.net_g = self.net_g.half()
        else:
            self.net_g = self.net_g.float()

        self.vc = VC(self.tgt_sr, device, is_half)
        self.n_spk = weight["config"][-3]

    def single(
        self,
        sid,
        input_audio,
        f0_up_key,
        f0_file,
        f0_method,
        file_index,
        file_big_npy,
        index_rate,
    ):
        if input_audio is None:
            return "You need to upload an audio", None
        f0_up_key = int(f0_up_key)
        audio = load_audio(input_audio, 16000)
        times = [0, 0, 0]
        if hubert_model == None:
            load_hubert()
        f0 = self.weight.get("f0", 1)
        audio_opt = self.vc.pipeline(
            hubert_model,
            self.net_g,
            sid,
            audio,
            times,
            f0_up_key,
            f0_method,
            file_index,
            file_big_npy,
            index_rate,
            f0,
            f0_file=f0_file,
        )
        return audio_opt


MODELS_DIR = opts.models_dir or os.path.join(ROOT_DIR, "models")
vc_model: Optional[VC_MODEL] = None
hubert_model = None


def get_models():
    dir = os.path.join(ROOT_DIR, "models", "weights")
    os.makedirs(dir, exist_ok=True)
    return [file for file in os.listdir(dir) if file.endswith(".pth")]


def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [os.path.join(MODELS_DIR, "hubert_base.pt")],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(device)
    if is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()


def load_model(model_name: str):
    global vc_model
    model_path = os.path.join(MODELS_DIR, "weights", model_name)
    weight = torch.load(model_path, map_location="cpu")
    vc_model = VC_MODEL(model_name, weight)
