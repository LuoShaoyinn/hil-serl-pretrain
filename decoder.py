# self_decoder.py
#    Decodes the ResNet encoder output (4x4x512) back to RGB observations.

import math

import jax.numpy as jnp
from flax import linen as nn


class SelfDecoder(nn.Module):
    """Simple 3-layer MLP decoder that outputs 128x128x3 images."""

    hidden_dims: tuple[int, ...] = (4096, 2048)
    output_shape: tuple[int, int, int] = (128, 128, 3)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        assert x.shape[-3:] == (4, 4, 512), f"Unexpected input shape: {x.shape}"

        batch = x.shape[0]
        x = jnp.reshape(x, (batch, -1))

        for features in self.hidden_dims:
            x = nn.Dense(features)(x)
            x = nn.relu(x)

        output_dim = math.prod(self.output_shape)
        x = nn.Dense(output_dim)(x)
        x = nn.sigmoid(x)
        return jnp.reshape(x, (batch,) + self.output_shape)
