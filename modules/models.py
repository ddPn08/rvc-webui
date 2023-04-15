import asyncio
import os
from typing import *

import torch
from fairseq import checkpoint_utils

from .cmd_opts import opts
from .inference.models import SynthesizerTrnMs256NSFSid, SynthesizerTrnMs256NSFSidNono
from .inference.pipeline import VC
from .shared import ROOT_DIR, device, is_half
from .utils import donwload_file, load_audio


def update_state_dict(state_dict):
    if "params" in state_dict:
        return
    keys = [
        "spec_channels",
        "segment_size",
        "inter_channels",
        "hidden_channels",
        "filter_channels",
        "n_heads",
        "n_layers",
        "kernel_size",
        "p_dropout",
        "resblock",
        "resblock_kernel_sizes",
        "resblock_dilation_sizes",
        "upsample_rates",
        "upsample_initial_channel",
        "upsample_kernel_sizes",
        "spk_embed_dim",
        "gin_channels",
        "sr",
    ]
    for i, key in enumerate(keys):
        state_dict["params"][key] = state_dict["config"][i]


class VC_MODEL:
    def __init__(self, model_name: str, state_dict: Dict[str, Any]) -> None:
        update_state_dict(state_dict)
        self.model_name = model_name
        self.weight = state_dict
        self.tgt_sr = state_dict["params"]["sr"]
        f0 = state_dict.get("f0", 1)
        state_dict["params"]["spk_embed_dim"] = state_dict["weight"][
            "emb_g.weight"
        ].shape[0]

        if f0 == 1:
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

        self.vc = VC(self.tgt_sr, device, is_half)
        self.n_spk = state_dict["params"]["spk_embed_dim"]

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
        audio_opt = self.vc(
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


def download_models():
    loop = asyncio.new_event_loop()
    tasks = []
    for template in [
        "D{}k",
        "G{}k",
        "f0D{}k",
        "f0G{}k",
    ]:
        for sr in ["32", "40", "48"]:
            url = f"https://huggingface.co/ddPn08/rvc_pretrained/resolve/main/{template.format(sr)}.pth"
            out = os.path.join(MODELS_DIR, "pretrained", f"{template.format(sr)}.pth")
            if os.path.exists(out):
                continue
            tasks.append(loop.run_in_executor(None, donwload_file, url, out))

    if len(tasks) > 0:
        loop.run_until_complete(asyncio.gather(*tasks))

    url = "https://huggingface.co/ddPn08/rvc_pretrained/resolve/main/hubert_base.pt"
    out = os.path.join(MODELS_DIR, "hubert_base.pt")
    if not os.path.exists(out):
        donwload_file(url, out)


def get_models():
    dir = os.path.join(ROOT_DIR, "models", "checkpoints")
    os.makedirs(dir, exist_ok=True)
    return [
        file
        for file in os.listdir(dir)
        if any([x for x in [".ckpt", ".pth"] if file.endswith(x)])
    ]


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
    model_path = os.path.join(MODELS_DIR, "checkpoints", model_name)
    weight = torch.load(model_path, map_location="cpu")
    vc_model = VC_MODEL(model_name, weight)
