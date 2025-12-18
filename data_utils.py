# load_data.py
#   Data loading utilities from data/

import pickle
import numpy as np
import jax_dataloader as jdl

def load_data(filename: str, batch_size: int):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    data = np.asarray(data)
    dataset = jdl.ArrayDataset(data)

    return jdl.DataLoader(
        dataset,
        backend="jax",     # recommended for pure JAX workflows
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=jdl.Generator(),
    )

