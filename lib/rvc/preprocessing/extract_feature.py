import os
import traceback
from concurrent.futures import ProcessPoolExecutor
from typing import *

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from fairseq import checkpoint_utils
from tqdm import tqdm


def load_embedder(embedder_path: str, device):
    models, cfg, _ = checkpoint_utils.load_model_ensemble_and_task(
        [embedder_path],
        suffix="",
    )
    embedder_model = models[0]
    embedder_model = embedder_model.to(device)
    if device != "cpu":
        embedder_model = embedder_model.half()
    else:
        embedder_model = embedder_model.float()
    embedder_model.eval()

    return embedder_model, cfg


# wave must be 16k, hop_size=320
def readwave(wav_path, normalize=False):
    wav, sr = sf.read(wav_path)
    assert sr == 16000
    feats = torch.from_numpy(wav).float()
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    feats = feats.view(1, -1)
    return feats


def processor(
    todo: List[str],
    device: torch.device,
    embedder_path: str,
    is_feats_dim_768: bool,
    wav_dir: str,
    out_dir: str,
    process_id: int,
):
    if not os.path.exists(embedder_path):
        return f"Embedder not found: {embedder_path}"
    model, cfg = load_embedder(embedder_path, device)

    for file in tqdm(todo, position=1 + process_id):
        try:
            if file.endswith(".wav"):
                wav_filepath = os.path.join(wav_dir, file)
                out_filepath = os.path.join(out_dir, file.replace("wav", "npy"))

                if os.path.exists(out_filepath):
                    continue

                os.makedirs(os.path.dirname(out_filepath), exist_ok=True)

                feats = readwave(wav_filepath, normalize=cfg.task.normalize)
                padding_mask = torch.BoolTensor(feats.shape).fill_(False)
                inputs = (
                    {
                        "source": feats.half().to(device)
                        if device != "cpu"
                        else feats.to(device),
                        "padding_mask": padding_mask.to(device),
                        "output_layer": 9,  # layer 9
                    }
                    if not is_feats_dim_768
                    else {
                        "source": feats.half().to(device)
                        if device != "cpu"
                        else feats.to(device),
                        "padding_mask": padding_mask.to(device),
                        # no pass "output_layer"
                    }
                )
                with torch.no_grad():
                    logits = model.extract_features(**inputs)
                    if is_feats_dim_768:
                        feats = logits[0]
                    else:
                        feats = model.final_proj(logits[0])

                feats = feats.squeeze(0).float().cpu().numpy()
                if np.isnan(feats).sum() == 0:
                    np.save(out_filepath, feats, allow_pickle=False)
                else:
                    print(f"{file} contains nan")
        except Exception as e:
            print(f"Error: {e} {file}")
            traceback.print_exc()


def run(
    training_dir: str,
    embedder_path: str,
    is_feats_dim_768: bool,
    gpu_ids: List[int],
    device: torch.device = None,
):
    wav_dir = os.path.join(training_dir, "1_16k_wavs")
    out_dir = os.path.join(training_dir, "3_feature256")

    num_gpus = len(gpu_ids)

    for gpu_id in gpu_ids:
        if num_gpus < gpu_id + 1:
            print(f"GPU {gpu_id} is not available")
            return

    if os.path.exists(out_dir):
        return

    os.makedirs(out_dir, exist_ok=True)

    todo = [
        os.path.join(dir, f)
        for dir in sorted(list(os.listdir(wav_dir)))
        if os.path.isdir(os.path.join(wav_dir, dir))
        for f in sorted(list(os.listdir(os.path.join(wav_dir, dir))))
    ]

    if device is not None:
        processor(
            todo,
            device,
            embedder_path,
            is_feats_dim_768,
            wav_dir,
            out_dir,
            process_id=0,
        )
    else:
        with ProcessPoolExecutor() as executor:
            for i, id in enumerate(gpu_ids):
                executor.submit(
                    processor,
                    todo[i::num_gpus],
                    f"cuda:{id}",
                    embedder_path,
                    is_feats_dim_768,
                    wav_dir,
                    out_dir,
                    process_id=i,
                )
