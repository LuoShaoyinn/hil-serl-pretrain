# load_params.py
#   Load pretrained ResNet-10 parameters into the encoder.

import os, pickle
from typing import Mapping, Any

import jax, optax
import jax.numpy as jnp
from flax.core import freeze, unfreeze
from flax.training import train_state, checkpoints

from train import ResNetAutoEncoder

def load_resnet10_params(params: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return params with the encoder subtree replaced by pretrained weights."""
    file_name = "resnet10_params.pkl"
    if not os.path.exists(file_name):
        raise FileNotFoundError(
            f"[load_params] Missing {file_name}. Download the pretrained weights first."
        )

    with open(file_name, "rb") as f:
        pretrained_params = pickle.load(f)

    print(f"[load_params] Loaded parameters from ResNet-10 (ImageNet-1K).")

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

    with open(filename, "wb") as f:
        pickle.dump(encoder_params, f)
    print(f"[save_params] Saved parameters as ResNet-10.")


def create_train_state(
    rng: jax.Array,
    learning_rate: float,
    decay_steps: int = 10000,
    decay_rate: float = 0.95,
) -> train_state.TrainState:
    model = ResNetAutoEncoder()
    dummy = jnp.zeros((1, 128, 128, 3), dtype=jnp.float32)
    params = model.init(rng, dummy, train=True)["params"]
    try:
        params = load_resnet10_params(params)
    except FileNotFoundError as exc:
        print(exc)
    schedule = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True,
    )
    tx = optax.adam(schedule)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state


def save_checkpoint(parallel_state, epoch: int):
    host_state = jax.device_get(jax.tree.map(lambda x: x[0], parallel_state))
    ckpt_dir = os.path.abspath("checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoints.save_checkpoint(
        ckpt_dir=ckpt_dir,
        target=host_state.params,
        step=epoch,
        overwrite=True,
        keep_every_n_steps=1
    )
    print(f"Checkpoint saved for epoch {epoch}.")


def load_checkpoint(
    model: ResNetAutoEncoder,
    ckpt_dir: str,
    rng: jax.Array,
) -> dict:
    ckpt_dir = os.path.abspath(ckpt_dir)
    dummy = jnp.zeros((1, 128, 128, 3), dtype=jnp.float32)
    params = model.init(rng, dummy, train=False)["params"]
    latest = checkpoints.latest_checkpoint(ckpt_dir)
    if latest is None:
        return None
    print(f"Restoring parameters from {latest}")
    params = checkpoints.restore_checkpoint(ckpt_dir, target=params)
    return params
