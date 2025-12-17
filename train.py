# train.py
#   Auto-encoder training loop that wires the ResNet encoder with the canvas decoder.

import os
import time
import functools
from typing import Iterator, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state, checkpoints
from flax.metrics.tensorboard import SummaryWriter

import load_data
import resnet_encoder
import self_decoder
import load_params

MASK_KEEP_PROB = 0.25


class ResNetAutoEncoder(nn.Module):
    """Wrap ResNet encoder and decoder into a single auto-encoder module."""
    def setup(self):
        self.encoder = resnet_encoder.ResNetEncoder()
        self.decoder = self_decoder.SelfDecoder()

    def __call__(self, inputs: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        features = self.encoder(inputs, train=train)
        return self.decoder(features)


def create_train_state(
    rng: jax.Array, learning_rate: float
) -> train_state.TrainState:
    model = ResNetAutoEncoder()
    dummy = jnp.zeros((1, 128, 128, 3), dtype=jnp.float32)
    params = model.init(rng, dummy, train=True)["params"]
    try:
        params = load_params.load_resnet10_params(params)
    except FileNotFoundError as exc:
        print(exc)
    tx = optax.adam(learning_rate)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state


def save(parallel_state, epoch: int):
    host_state = jax.device_get(jax.tree_map(lambda x: x[0], parallel_state))
    ckpt_dir = os.path.abspath("checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoints.save_checkpoint(
        ckpt_dir=ckpt_dir,
        target=host_state.params,
        step=epoch,
        overwrite=True,
    )
    print(f"Checkpoint saved for epoch {epoch}.")


def apply_random_mask(batch: jnp.ndarray, rng: jax.Array):
    mask_shape = batch.shape[:-1] + (1,)
    keep_mask = jax.random.bernoulli(rng, p=MASK_KEEP_PROB, shape=mask_shape)
    keep_mask = keep_mask.astype(batch.dtype)
    masked_inputs = batch * keep_mask
    # Reconstruction loss uses the masked-out regions (1 - keep_mask)
    recon_mask = 1.0 - keep_mask
    return masked_inputs, recon_mask


def masked_mse_loss(params, apply_fn, batch, rng):
    masked_inputs, recon_mask = apply_random_mask(batch, rng)
    recon = apply_fn({"params": params}, masked_inputs, train=True)
    diff = (recon - batch) * recon_mask
    denom = jnp.maximum(jnp.sum(recon_mask), 1.0)
    return jnp.sum(diff * diff) / denom


@functools.partial(jax.pmap, axis_name="device")
def train_step(state: train_state.TrainState, batch: jnp.ndarray, rng: jnp.ndarray):
    def loss_fn(params):
        return masked_mse_loss(params, state.apply_fn, batch, rng)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    loss = jax.lax.pmean(loss, axis_name="device")
    grads = jax.lax.pmean(grads, axis_name="device")
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss


def iterate_minibatches(
    data: np.ndarray, batch_size: int, rng: np.random.Generator
) -> Iterator[np.ndarray]:
    idx = rng.permutation(len(data))
    for start in range(0, len(data) - batch_size + 1, batch_size):
        excerpt = idx[start : start + batch_size]
        yield data[excerpt]


def train(
    data_filename: str = "data.pkl",
    global_batch_size: int = 512,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    log_dir: Optional[str] = None,
):

    num_devices = jax.local_device_count()
    if global_batch_size % num_devices != 0:
        raise ValueError(f"Global batch size {global_batch_size} must be divisible by device count {num_devices}.")
    per_device_batch = global_batch_size // num_devices

    # Load dataset
    dataset = np.asarray(load_data.load_data(data_filename, global_batch_size))

    # Initialize model and training state
    init_rng, train_rng = jax.random.split(jax.random.PRNGKey(0))
    state = create_train_state(init_rng, learning_rate)
    parallel_state = jax.device_put_replicated(state, jax.devices())

    np_rng = np.random.default_rng(0)
    if log_dir is None:
        dataset_name = os.path.splitext(os.path.basename(data_filename))[0]
        log_dir = os.path.join(
            "runs", f"{dataset_name}_bs{global_batch_size}_lr{learning_rate}_{int(time.time())}"
        )
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"Training for {num_epochs} epochs on {num_devices} device(s).")
    print(f"[tensorboard] Logging to {os.path.abspath(log_dir)}")
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        steps = 0

        print(f"Epoch {epoch:3d}/{num_epochs}", end="", flush=True)
        for batch in iterate_minibatches(dataset, global_batch_size, np_rng):
            if len(batch) != global_batch_size:
                # Skip incomplete batches to keep per-device shapes consistent.
                print(
                    f"\n[train] Skipping partial batch of size {len(batch)}; expected {global_batch_size}.",
                    flush=True,
                )
                continue

            batch = batch.astype(np.float32) / 255.0
            sharded = batch.reshape(
                (num_devices, per_device_batch, 128, 128, 3)
            )

            train_rng, step_rng = jax.random.split(train_rng)
            step_rngs = jax.random.split(step_rng, num_devices)

            parallel_state, loss = train_step(parallel_state, sharded, step_rngs)
            epoch_loss += float(jax.device_get(loss).mean())
            steps += 1
            
            if(steps % 2) == 0:
                print(".", end="", flush=True)

        avg_loss = epoch_loss / max(steps, 1)
        print(f" - MSE Loss: {avg_loss:.6f}")
        writer.scalar("train/avg_loss", avg_loss, epoch)

        if epoch % 10 == 0:
            save(parallel_state, epoch)

    writer.flush()
    writer.close()

if __name__ == "__main__":
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    train(   
        data_filename = "data.pkl",
        global_batch_size = 512, 
        num_epochs = 5, 
        learning_rate = 1e-4
    )
