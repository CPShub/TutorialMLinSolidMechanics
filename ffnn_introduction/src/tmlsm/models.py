"""Model implementations."""

from typing import Callable

import equinox as eqx
import jax
from jax.nn.initializers import he_normal
from jaxtyping import PRNGKeyArray
import klax


class Model(eqx.Module):
    """A custom trainable `equinox.Module`."""

    layers: tuple[Callable, ...]
    activations: tuple[Callable, ...]

    def __init__(self, *, key: PRNGKeyArray):
        self.layers = (
            klax.nn.Linear("scalar", 16, weight_init=he_normal(), key=key),
            klax.nn.Linear(16, 16, weight_init=he_normal(), key=key),
            klax.nn.Linear(16, 16, weight_init=he_normal(), key=key),
            klax.nn.Linear(16, "scalar", weight_init=he_normal(), key=key),
        )

        self.activations = (
            jax.nn.softplus,
            jax.nn.softplus,
            lambda x: x,
        )

    def __call__(self, x):
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))
        return x


def build(*, key: PRNGKeyArray):
    """Build and return a model instance."""
    return Model(key=key)
