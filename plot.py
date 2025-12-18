import argparse
import pickle
from typing import Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import params_utils
from train import ResNetAutoEncoder


def _load_dataset(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return np.asarray(data, dtype=np.uint8)


def _sample_images(data: np.ndarray, count: int) -> Tuple[np.ndarray, np.ndarray]:
    count = min(count, len(data))
    rng = np.random.default_rng()
    indices = rng.choice(len(data), size=count, replace=False)
    originals = data[indices]
    inputs = originals.astype(np.float32)  # keep 0-255 scale for encoder
    return originals, inputs


def _plot_pairs(originals: np.ndarray, reconstructions: np.ndarray) -> None:
    rows = len(originals)
    fig, axes = plt.subplots(rows, 2, figsize=(6, 3 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx in range(rows):
        axes[idx, 0].imshow(originals[idx])
        axes[idx, 0].axis("off")
        axes[idx, 0].set_title("Original")

        axes[idx, 1].imshow(np.clip(reconstructions[idx], 0.0, 1.0))
        axes[idx, 1].axis("off")
        axes[idx, 1].set_title("Decoded")

    fig.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize decoded samples.")
    parser.add_argument(
        "--dataset",
        default="dataset/valid.pkl",
        help="Path to dataset pickle file.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="checkpoints",
        help="Directory containing model checkpoints.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of random samples to visualize.",
    )
    args = parser.parse_args()

    data = _load_dataset(args.dataset)
    originals, inputs = _sample_images(data, args.num_samples)

    model = ResNetAutoEncoder()
    params = params_utils.load_checkpoint(model, args.checkpoint_dir, jax.random.PRNGKey(0))

    decoded = model.apply({"params": params}, jnp.array(inputs), train=False)
    decoded_np = np.array(decoded)

    _plot_pairs(originals, decoded_np)

    params_utils.save_resnet10_params(params, "retrained_resnet10_params.pkl")


if __name__ == "__main__":
    main()
