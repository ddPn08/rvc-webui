import os

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")

is_half = False

device = "cuda:0"

if device != "cpu":
    gpu_name = torch.cuda.get_device_name(int(device.split(":")[-1]))
    if "16" in gpu_name or "MX" in gpu_name:
        is_half = False

device = torch.device(device)
