# eval_autoencoder.py
# Evaluate a trained auto-encoder checkpoint without modifying the training code.

import argparse
import os
from typing import Optional, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints
import matplotlib.pyplot as plt

import load_data, load_params
from train import ResNetAutoEncoder



def _load_checkpoint_params(
    model: ResNetAutoEncoder,
    ckpt_dir: str,
    rng: jax.Array,
) -> dict:
    ckpt_dir = os.path.abspath(ckpt_dir)
    dummy = jnp.zeros((1, 128, 128, 3), dtype=jnp.float32)
    params = model.init(rng, dummy, train=False)["params"]
    latest = checkpoints.latest_checkpoint(ckpt_dir)
    if latest is None:
        raise FileNotFoundError(
            f"No checkpoints found in '{os.path.abspath(ckpt_dir)}'."
        )
    print(f"[eval] Restoring parameters from {latest}")
    params = checkpoints.restore_checkpoint(ckpt_dir, target=params)
    return params


def _display_reconstructions(pairs: List[Tuple[np.ndarray, np.ndarray]]) -> None:
    rows = len(pairs)
    if rows == 0:
        return

    fig, axes = plt.subplots(rows, 2, figsize=(6, 3 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, (original, reconstructed) in enumerate(pairs):
        original = np.clip(original, 0.0, 1.0)
        reconstructed = np.clip(reconstructed, 0.0, 1.0)

        axes[idx, 0].imshow(original)
        axes[idx, 0].axis("off")
        axes[idx, 0].set_title("Original")

        axes[idx, 1].imshow(reconstructed)
        axes[idx, 1].axis("off")
        axes[idx, 1].set_title("Reconstruction")

    fig.tight_layout()
    plt.show()


def evaluate(
    checkpoint_dir: str = "checkpoints",
    data_filename: str = "data.pkl",
    batch_size: int = 64,
    max_batches: Optional[int] = None,
    show_samples: int = 0,
) -> float:
    """Return average reconstruction MSE for the saved model."""
    model = ResNetAutoEncoder()
    params = _load_checkpoint_params(model, checkpoint_dir, jax.random.PRNGKey(0))

    raw_data = np.asarray(load_data.load_data(data_filename, batch_size))
    dataset = load_data.normalize_data(raw_data)
    num_samples = len(dataset)
    total_loss, total_items = 0.0, 0
    viz_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    effective_samples = num_samples
    if max_batches is not None:
        effective_samples = min(effective_samples, max_batches * batch_size)
    sample_count = min(show_samples, effective_samples)
    sample_indices = set()
    if sample_count > 0:
        rng = np.random.default_rng()
        sample_indices = set(
            map(int, rng.choice(effective_samples, size=sample_count, replace=False))
        )

    for start in range(0, num_samples, batch_size):
        if max_batches is not None and (start // batch_size) >= max_batches:
            break

        batch = dataset[start : start + batch_size]
        if len(batch) == 0:
            continue

        preds = model.apply({"params": params}, jnp.array(batch), train=False)
        mse = jnp.mean((preds - batch) ** 2)
        preds_np = np.array(preds)

        total_loss += float(mse) * len(batch)
        total_items += len(batch)
        if sample_indices:
            for idx_in_batch in range(len(batch)):
                global_idx = start + idx_in_batch
                if global_idx in sample_indices:
                    viz_pairs.append(
                        (np.array(batch[idx_in_batch]), np.array(preds_np[idx_in_batch]))
                    )
                    sample_indices.remove(global_idx)
                    if not sample_indices:
                        break

    final_loss = total_loss / max(total_items, 1)
    print(f"[eval] Average reconstruction MSE on {total_items} samples: {final_loss:.6f}")
    if viz_pairs:
        _display_reconstructions(viz_pairs)

    load_params.save_resnet10_params(params, "resnet10_params_new.pkl")
    
    return final_loss


if __name__ == "__main__":
    evaluate(
        checkpoint_dir="checkpoints",
        data_filename="data.pkl",
        batch_size=64,
        max_batches=128,
        show_samples=5,
    )
