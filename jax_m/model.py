import jax.numpy as jnp
from flax import linen as nn


class FraudNet(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)

        x = nn.Dense(1)(x)
        return x
