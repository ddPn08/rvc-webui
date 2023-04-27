import os
import sys

import torch

from modules.cmd_opts import opts

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")


def has_mps():
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


is_half = opts.precision == "fp16"
half_support = (
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 5.3
)

if not half_support:
    print("WARNING: FP16 is not supported on this GPU")
    is_half = False

device = "cuda:0"

if not torch.cuda.is_available():
    if has_mps():
        print("Using MPS")
        device = "mps"
    else:
        print("Using CPU")
        device = "cpu"

device = torch.device(device)
