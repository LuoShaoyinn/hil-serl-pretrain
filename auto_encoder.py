#! python3

import os
# PREVENT JAX FROM HOGGING ALL VRAM
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from functools import partial

# 1. Model Architecture (Inputs expected to be 0.0 to 1.0)
class Autoencoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Encoder
        x = nn.Conv(features=8, kernel_size=(3, 3), strides=(2, 2))(x)  # 64, 64
        x = nn.relu(x)
        x = nn.Conv(features=16, kernel_size=(3, 3), strides=(2, 2))(x) # 32, 32
        x = nn.relu(x)
        x = nn.Conv(features=16, kernel_size=(3, 3), strides=(2, 2))(x) # 16, 16
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2))(x) # 8, 8
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(8, 8))(x) # 1, 1
        x = nn.relu(x)
        # Decoder
        x = nn.ConvTranspose(features=32, kernel_size=(8, 8), strides=(8, 8))(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=8, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=3, kernel_size=(3, 3), strides=(2, 2))(x)
        return nn.sigmoid(x) # Output 0.0 to 1.0

# 2. Loss and Step Functions
def mse_loss(params, apply_fn, batch):
    reconstructed = apply_fn({'params': params}, batch)
    return jnp.mean((batch - reconstructed) ** 2)

@partial(jax.pmap, axis_name='batch')
def train_step(state, batch):
    # Normalize on-the-fly inside the GPU to save CPU/RAM memory
    batch = batch.astype(jnp.float32) / 255.0
    
    grad_fn = jax.value_and_grad(mse_loss)
    loss, grads = grad_fn(state.params, state.apply_fn, batch)
    grads = jax.lax.pmean(grads, axis_name='batch')
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

# 3. Main Execution
def main():
    num_devices = jax.local_device_count()
    global_batch_size = 128  # Reduced to avoid OOM
    local_batch_size = global_batch_size // num_devices
    
    # Load Data (Keep as uint8 0-255 to save RAM)
    print("Loading data...")
    with open('data/side_policy.pkl', 'rb') as f:
        data = pickle.load(f) 
    
    if data.dtype != np.uint8:
        data = data.astype(np.uint8)

    # Split Data
    train_data, _ = train_test_split(data, test_size=0.2, random_state=42)
    # Ensure divisible by batch size
    train_data = train_data[:(len(train_data) // global_batch_size) * global_batch_size]
    
    # Initialize Model
    rng = jax.random.PRNGKey(0)
    model = Autoencoder()
    params = model.init(rng, jnp.ones((1, 128, 128, 3)))['params']
    state = train_state.TrainState.create(
        apply_fn=model.apply, 
        params=params, 
        tx=optax.adam(1e-4) # Lower learning rate often helps stability
    )
    
    # Replicate state to all GPUs
    parallel_state = jax.device_put_replicated(state, jax.devices())

    print(f"Training on {num_devices} GPUs with Global Batch Size: {global_batch_size}")
    for epoch in range(30):
        epoch_loss = 0
        
        # Shuffle indices each epoch
        perms = np.random.permutation(len(train_data))
        
        for i in range(0, len(train_data), global_batch_size):
            idx = perms[i : i + global_batch_size]
            batch = train_data[idx]
            
            # Reshape for pmap: (num_devices, local_batch, 128, 128, 3)
            sharded_batch = batch.reshape((num_devices, local_batch_size, 128, 128, 3))
            
            parallel_state, loss = train_step(parallel_state, sharded_batch)
            epoch_loss += jnp.mean(loss)
            
        print(f"Epoch {epoch} | Mean MSE Loss: {epoch_loss / (len(train_data)//global_batch_size):.6f}")

if __name__ == "__main__":
    main()

