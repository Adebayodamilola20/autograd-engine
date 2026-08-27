"""Phase 7/8 -- a small neural-network library on top of the autodiff engine.

Everything here is built from ``Value`` objects. No layer computes a gradient
itself; they build expressions, and ``backward()`` does the rest. That is the
whole point of the layering: if the engine is right, these are right -- and the
test suite gradient-checks a complete MLP to prove it.

    >>> from nabla.nn import MLP
    >>> model = MLP(2, [8, 8], 1, activation='tanh', seed=0)
    >>> out = model([0.5, -1.0])
"""

from .activations import (
    ACTIVATIONS,
    LeakyReLU,
    Linear,
    ReLU,
    Sigmoid,
    Softplus,
    Tanh,
    get_activation,
    leaky_relu,
    linear,
    relu,
    sigmoid,
    softplus,
    tanh,
)
from .init import (
    for_activation,
    get_initializer,
    he_normal,
    he_uniform,
    lecun_normal,
    xavier_normal,
    xavier_uniform,
)
from .layer import Layer
from .mlp import MLP
from .module import Module, as_values
from .neuron import Neuron
from .parameter import Parameter

__all__ = [
    # structure
    "Module",
    "Parameter",
    "Neuron",
    "Layer",
    "MLP",
    "as_values",
    # activations (functional)
    "linear",
    "tanh",
    "relu",
    "sigmoid",
    "leaky_relu",
    "softplus",
    "get_activation",
    "ACTIVATIONS",
    # activations (modules)
    "Linear",
    "Tanh",
    "ReLU",
    "Sigmoid",
    "LeakyReLU",
    "Softplus",
    # initialisation
    "xavier_uniform",
    "xavier_normal",
    "he_uniform",
    "he_normal",
    "lecun_normal",
    "for_activation",
    "get_initializer",
]
