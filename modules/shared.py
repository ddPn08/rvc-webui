import os
import sys

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")


def has_mps() -> bool:
    if sys.platform != "darwin":
        return False
    else:
        if not getattr(torch, "has_mps", False):
            return False
        try:
            torch.zeros(1).to(torch.device("mps"))
            return True
        except Exception:
            return False


is_half = True

device = "cuda:0"

if not torch.cuda.is_available():
    if has_mps():
        print("Using MPS")
        device = "mps"
    else:
        print("Using CPU")
        device = "cpu"
        is_half = False

if device not in ["cpu", "mps"]:
    gpu_name = torch.cuda.get_device_name(int(device.split(":")[-1]))
    if "16" in gpu_name or "MX" in gpu_name:
        is_half = False

device = torch.device(device)
