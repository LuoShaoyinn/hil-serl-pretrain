#! python3

import pickle
import jax.numpy as jnp


KEYS = ['side_classifier', 'side_policy', 'wrist_1', 'wrist_2']

all_data = dict()
for key in KEYS:
    all_data[key] = list()

try:
    for ids in range(1, 100000):
        filename = f"buffer/transitions_{ids}000.pkl"
        print(f"[+] Load from {filename}")
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        all_obs = [obs["observations"] for obs in data]
        for key in KEYS:
            obs_key = jnp.array([obs_key[key] for obs_key in all_obs])
            obs_key = jnp.squeeze(obs_key, axis=1)
            all_data[key].append(obs_key)
except FileNotFoundError:
    print(f"[+] All files extracted")
except Exception as e:
    print(f"[!] Unwanted exception: {e}")
finally:
    for key in KEYS:
        data_key = jnp.concatenate(all_data[key], axis=0)
        filename = f"data/{key}.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(data_key, file)
        print(f"[+] Saved {len(data_key)} samples in key {key}, shape: {data_key.shape}")
