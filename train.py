# train.py
#   Auto-encoder training loop that wires the ResNet encoder with the canvas decoder.

import os
import time
import functools
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training import train_state
from flax.metrics.tensorboard import SummaryWriter

import data_utils
import encoder
import decoder
import params_utils

MASK_RATE = 0.6  # portion of pixels to mask out


class ResNetAutoEncoder(nn.Module):
    """Wrap ResNet encoder and decoder into a single auto-encoder module."""
    def setup(self):
        self.encoder = encoder.ResNetEncoder()
        self.decoder = decoder.SelfDecoder()

    def __call__(self, inputs: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        features = self.encoder(inputs, train=train)
        return self.decoder(features)


@jax.jit
def apply_random_mask(batch: jnp.ndarray, rng: jax.Array):
    mask_shape = batch.shape[:-1] + (1,)
    keep_prob = 1.0 - MASK_RATE
    keep_mask = jax.random.bernoulli(rng, p=keep_prob, shape=mask_shape)
    keep_mask = keep_mask.astype(jnp.float32)
    masked_inputs = batch * keep_mask
    return masked_inputs


@functools.partial(jax.jit, static_argnums=(1,))
def masked_mse_loss(params, apply_fn, batch, rng):
    batch = batch.astype(jnp.float32)
    masked_inputs = apply_random_mask(batch, rng)
    target = batch / 255.0  # Decoder outputs [0, 1]
    recon = apply_fn({"params": params}, masked_inputs, train=True)
    diff = recon - target
    return jnp.mean(diff * diff)


@functools.partial(jax.pmap, axis_name="device")
def train_step(state: train_state.TrainState, batch: jnp.ndarray, rng: jnp.ndarray):
    def loss_fn(params):
        return masked_mse_loss(params, state.apply_fn, batch, rng)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    loss = jax.lax.pmean(loss, axis_name="device")
    grads = jax.lax.pmean(grads, axis_name="device")
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

@functools.partial(jax.pmap, axis_name="device")
def eval_step(state: train_state.TrainState, batch: jnp.ndarray, rng: jnp.ndarray):
    loss = masked_mse_loss(state.params, state.apply_fn, batch, rng)
    return jax.lax.pmean(loss, axis_name="device")



def train(
    train_batch_size: int = 512,
    valid_batch_size: int = 128,
    num_epochs: int = 400,
    learning_rate: float = 1e-4,
    lr_decay_rate: float = 0.95,
    lr_decay_steps: Optional[int] = None,
    log_dir: Optional[str] = None,
):
    num_devices = jax.local_device_count()
    if train_batch_size % num_devices != 0:
        raise ValueError(
            f"Global batch size {train_batch_size} must be divisible by device count {num_devices}."
        )
    per_device_batch = train_batch_size // num_devices

    # Load dataset and construct jax-dataloader pipeline
    train_loader = data_utils.load_data("dataset/data.pkl",  train_batch_size, augment=True)
    valid_loader = data_utils.load_data("dataset/valid.pkl", valid_batch_size, augment=False)
    print(f"Train dataset batches: {len(train_loader)}, Valid dataset batches: {len(valid_loader)}")
    
    # Initialize model and training state
    init_rng, train_rng = jax.random.split(jax.random.PRNGKey(0))
    decay_steps = lr_decay_steps or max(1, len(train_loader))
    state = params_utils.create_train_state(
        init_rng,
        learning_rate,
        decay_steps=decay_steps,
        decay_rate=lr_decay_rate,
    )
    parallel_state = jax.device_put_replicated(state, jax.devices())

    if log_dir is None:
        log_dir = os.path.join("runs", f"{int(time.time())}")
                               
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"Training for {num_epochs} epochs on {num_devices} device(s).")
    print(f"Logging to {os.path.abspath(log_dir)}")

    eval_epoch = np.clip(num_epochs // 10, 1, 10)

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch:3d}/{num_epochs}", end="", flush=True)
        epoch_loss = 0.0
        steps = 0

        for batch in train_loader:
            batch = jnp.asarray(batch, dtype=jnp.float32)
            assert batch.shape[0] == train_batch_size
            sharded = batch.reshape((num_devices, per_device_batch) + batch.shape[1:])

            train_rng, step_rng = jax.random.split(train_rng)
            step_rngs = jax.random.split(step_rng, num_devices)

            parallel_state, loss = train_step(parallel_state, sharded, step_rngs)
            step_loss = float(np.mean(jax.device_get(loss)))
            epoch_loss += step_loss
            steps += 1

            if (steps % 5) == 0:
                print(".", end="", flush=True)

        avg_loss = epoch_loss / max(steps, 1)
        print(f" - MSE Loss: {avg_loss:.6f}")
        writer.scalar("train/loss", avg_loss, epoch)

        if epoch % eval_epoch == 0 or epoch == num_epochs:
            eval_loss_avg = 0.0
            for valid_batch in valid_loader:
                valid_batch = jnp.asarray(valid_batch, dtype=jnp.float32)
                assert valid_batch.shape[0] == valid_batch_size
                valid_shared = valid_batch.reshape((num_devices, valid_batch.shape[0] // num_devices) + valid_batch.shape[1:])
                train_rng, step_rng = jax.random.split(train_rng)
                step_rngs = jax.random.split(step_rng, num_devices)
                eval_loss = eval_step(parallel_state, valid_shared, step_rngs)
                eval_loss_avg += float(np.mean(jax.device_get(eval_loss)))
            eval_loss_avg /= len(valid_loader)
            print(f"==== Eval MSE Loss: {eval_loss_avg:.6f} ====")
            writer.scalar("eval/loss", eval_loss_avg, epoch)
            params_utils.save_checkpoint(parallel_state, epoch)

    writer.flush()
    writer.close()

if __name__ == "__main__":
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    train(   
        train_batch_size = 512,
        valid_batch_size = 128,
        num_epochs = 400, 
        learning_rate = 1e-4
    )
