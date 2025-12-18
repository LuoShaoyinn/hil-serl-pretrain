# self_decoder.py
#    Decodes the ResNet encoder output (4x4x512) back to RGB observations.

import jax
import jax.numpy as jnp
from flax import linen as nn


class SelfDecoder(nn.Module):
    """Nearest-neighbor upsampling decoder to 128x128x3 outputs."""

    def _upsample(self, x: jnp.ndarray) -> jnp.ndarray:
        b, h, w, c = x.shape
        return jax.image.resize(
            x, shape=(b, h * 2, w * 2, c), method="nearest"
        )

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        assert x.shape[-3:] == (4, 4, 512), f"Unexpected input shape: {x.shape}"

        x = nn.Conv(features=512, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)

        for features in (256, 128, 64, 32, 16):
            x = self._upsample(x)
            x = nn.Conv(features=features, kernel_size=(3, 3), padding="SAME")(x)
            x = nn.relu(x)

        x = nn.Conv(features=3, kernel_size=(3, 3), padding="SAME")(x)
        return nn.sigmoid(x)  # Output scaled to [0, 1]
