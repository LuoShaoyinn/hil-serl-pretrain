# load_data.py
#   Data loading utilities from data/

import pickle
import numpy as np
import jax_dataloader as jdl


class AugmentedArrayDataset(jdl.Dataset):
    """ArrayDataset wrapper that performs simple on-the-fly augmentations."""

    def __init__(self, array: np.ndarray, augment: bool):
        self.array = array
        self.augment = augment
        self._rng = np.random.default_rng()

    def __len__(self):
        return len(self.array)

    def __getitem__(self, idx):
        sample = np.array(self.array[idx], copy=True)
        if self.augment:
            sample = self._augment_sample(sample)
        return sample

    def _augment_sample(self, sample: np.ndarray) -> np.ndarray:
        if sample.ndim >= 3:
            if self._rng.random() < 0.5:
                sample = np.flip(sample, axis=1)  # horizontal flip
            if self._rng.random() < 0.5:
                sample = np.flip(sample, axis=0)  # vertical flip
        if self._rng.random() < 0.5:
            sample = self._random_brightness(sample, max_delta=20)
        return sample

    def _random_brightness(self, sample: np.ndarray, max_delta: int) -> np.ndarray:
        dtype = sample.dtype
        if np.issubdtype(dtype, np.integer):
            delta = int(self._rng.integers(-max_delta, max_delta + 1))
            if delta == 0:
                return sample
            adjusted = np.clip(sample.astype(np.int16) + delta, 0, 255)
            return adjusted.astype(dtype)
        scale = max_delta / 255.0
        delta = float(self._rng.uniform(-scale, scale))
        adjusted = np.clip(sample + delta, 0.0, 1.0)
        return adjusted.astype(dtype)


def load_data(filename: str, batch_size: int, augment: bool = False):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    data = np.asarray(data)
    dataset = AugmentedArrayDataset(data, augment=augment)

    return jdl.DataLoader(
        dataset,
        backend="jax",     # recommended for pure JAX workflows
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=jdl.Generator(),
    )
