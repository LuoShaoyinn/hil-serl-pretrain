# load_params.py
#   Load pretrained ResNet-10 parameters into the encoder.

import os
import pickle as pkl
from typing import Mapping, Any

from flax.core import freeze, unfreeze
from jax import tree_util

def load_resnet10_params(params: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return params with the encoder subtree replaced by pretrained weights."""
    file_name = "resnet10_params.pkl"
    if not os.path.exists(file_name):
        raise FileNotFoundError(
            f"[load_params] Missing {file_name}. Download the pretrained weights first."
        )

    with open(file_name, "rb") as f:
        pretrained_params = pkl.load(f)

    count = sum(x.size for x in tree_util.tree_leaves(pretrained_params))
    print(f"[load_params] Loaded {count/1e6:.2f}M parameters from ResNet-10 (ImageNet-1K).")

    params_mutable = unfreeze(params)
    for key, value in pretrained_params.items():
        if key in params_mutable["encoder"]:
            params_mutable["encoder"][key] = value
            print(f"[load_params] Replaced encoder weight '{key}'.")
            
    return freeze(params_mutable)


def save_resnet10_params(params: Mapping[str, Any], filename):
    params_mutable = unfreeze(params)
    encoder_params = params_mutable.get("encoder")
    if encoder_params is None:
        raise KeyError("[save_params] Missing 'encoder' subtree in params.")

    count = sum(x.size for x in tree_util.tree_leaves(encoder_params))
    with open(filename, "wb") as f:
        pkl.dump(encoder_params, f)
    print(f"[save_params] Saved {count/1e6:.2f}M parameters as ResNet-10.")
