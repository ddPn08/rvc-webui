from collections import OrderedDict
from typing import *

import torch
import tqdm


def merge(
    path_a: str,
    path_b: str,
    path_c: str,
    alpha: float,
    weights: Dict[str, float],
    method: str,
):
    def extract(ckpt: Dict[str, Any]):
        a = ckpt["model"]
        opt = OrderedDict()
        opt["weight"] = {}
        for key in a.keys():
            if "enc_q" in key:
                continue
            opt["weight"][key] = a[key]
        return opt

    def load_weight(path: str):
        print(f"Loading {path}...")
        state_dict = torch.load(path, map_location="cpu")
        if "model" in state_dict:
            weight = extract(state_dict)
        else:
            weight = state_dict["weight"]
        return weight, state_dict

    def get_alpha(key: str):
        try:
            filtered = sorted(
                [x for x in weights.keys() if key.startswith(x)], key=len, reverse=True
            )
            if len(filtered) < 1:
                return alpha
            return weights[filtered[0]]
        except:
            return alpha

    weight_a, state_dict = load_weight(path_a)
    weight_b, _ = load_weight(path_b)
    if path_c is not None:
        weight_c, _ = load_weight(path_c)

    if sorted(list(weight_a.keys())) != sorted(list(weight_b.keys())):
        raise RuntimeError("Failed to merge models.")

    merged = OrderedDict()
    merged["weight"] = {}

    def merge_weight(a, b, c, alpha):
        if method == "weight_sum":
            return (1 - alpha) * a + alpha * b
        elif method == "add_diff":
            return a + (b - c) * alpha

    for key in tqdm.tqdm(weight_a.keys()):
        a = get_alpha(key)
        if path_c is not None:
            merged["weight"][key] = merge_weight(
                weight_a[key], weight_b[key], weight_c[key], a
            )
        else:
            merged["weight"][key] = merge_weight(weight_a[key], weight_b[key], None, a)
    merged["config"] = state_dict["config"]
    merged["params"] = state_dict["params"] if "params" in state_dict else None
    merged["sr"] = state_dict["sr"]
    merged["f0"] = state_dict["f0"]
    merged["info"] = state_dict["info"]
    return merged
