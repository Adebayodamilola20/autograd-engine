r"""Weight initialisation -- and why it decides whether a network trains at all.

The problem
-----------
Set every weight to zero and every neuron in a layer computes the same thing,
receives the same gradient, and updates identically. The layer collapses to a
single unit and never recovers. This is the **symmetry problem**, and it is why
weights must be random.

But *how* random matters enormously. Consider a signal passing through a layer:

.. math:: y_j = \sum_{i=1}^{n_{\text{in}}} w_{ji} x_i + b_j

If the :math:`w_{ji}` are independent with variance :math:`\sigma^2` and the
inputs are independent with variance :math:`\text{Var}(x)`, then

.. math:: \text{Var}(y) = n_{\text{in}}\,\sigma^{2}\,\text{Var}(x)

So each layer multiplies the signal's variance by
:math:`n_{\text{in}}\sigma^{2}`. Stack :math:`L` layers and the factor becomes
:math:`(n_{\text{in}}\sigma^{2})^{L}` -- exponential in depth. Too large and
activations explode (and saturate tanh/sigmoid, where the gradient dies); too
small and they collapse toward zero (and so does the signal). The gradient
suffers the mirror-image fate on the way back.

The fix is to choose :math:`\sigma` so the factor is 1.

Xavier / Glorot
---------------
Forward stability wants :math:`\sigma^{2} = 1/n_{\text{in}}`; backward
stability wants :math:`\sigma^{2} = 1/n_{\text{out}}`. You cannot have both
unless the layer is square, so Glorot & Bengio (2010) split the difference:

.. math:: \sigma^{2} = \frac{2}{n_{\text{in}} + n_{\text{out}}}

For a uniform distribution on :math:`[-a, a]`, :math:`\text{Var} = a^{2}/3`, so
matching variance gives :math:`a = \sqrt{6/(n_{\text{in}}+n_{\text{out}})}`.

Use with **tanh** and **sigmoid**, whose derivation assumes an activation that
is roughly linear and symmetric near zero.

He / Kaiming
------------
ReLU breaks that assumption: it zeroes half its inputs, halving the variance of
what passes through. He et al. (2015) compensate with a factor of 2:

.. math:: \sigma^{2} = \frac{2}{n_{\text{in}}}

Use with **ReLU** and its variants. Using Xavier with a deep ReLU network makes
activations decay by :math:`(1/2)^{L/2}` -- the network trains, but slowly and
worse. ``for_activation()`` picks the right one automatically, and Phase 22
measures the difference.

Determinism
-----------
Every function takes an explicit ``rng``. There are no calls to the global
``random`` module anywhere in this library, so a seed fully determines a run
(decision D9).
"""

from __future__ import annotations

import math
import random
from typing import Callable

__all__ = [
    "zeros",
    "constant",
    "uniform",
    "normal",
    "xavier_uniform",
    "xavier_normal",
    "he_uniform",
    "he_normal",
    "lecun_normal",
    "for_activation",
    "get_initializer",
]

Initializer = Callable[[int, int, random.Random], float]


def zeros(fan_in: int, fan_out: int, rng: random.Random) -> float:
    """All zeros. Correct for **biases**, catastrophic for weights.

    A bias does not suffer the symmetry problem -- each neuron's bias is
    attached to a different weight vector, so they diverge as soon as training
    starts. Zero is the natural neutral choice.
    """
    return 0.0


def constant(value: float) -> Initializer:
    """Fixed value. For tests and for deliberately broken baselines."""
    return lambda fan_in, fan_out, rng: value


def uniform(low: float = -1.0, high: float = 1.0) -> Initializer:
    """Uniform on ``[low, high]``, ignoring layer size.

    The naive choice. Included so Phase 22 can show what goes wrong with it in
    a deep network.
    """
    return lambda fan_in, fan_out, rng: rng.uniform(low, high)


def normal(mean: float = 0.0, std: float = 1.0) -> Initializer:
    return lambda fan_in, fan_out, rng: rng.gauss(mean, std)


def xavier_uniform(fan_in: int, fan_out: int, rng: random.Random) -> float:
    r"""Glorot uniform: :math:`U(-a, a)` with :math:`a=\sqrt{6/(n_{in}+n_{out})}`.

    For **tanh** and **sigmoid**.
    """
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit)


def xavier_normal(fan_in: int, fan_out: int, rng: random.Random) -> float:
    r"""Glorot normal: :math:`\mathcal{N}(0, 2/(n_{in}+n_{out}))`."""
    return rng.gauss(0.0, math.sqrt(2.0 / (fan_in + fan_out)))


def he_uniform(fan_in: int, fan_out: int, rng: random.Random) -> float:
    r"""He uniform: :math:`U(-a,a)` with :math:`a=\sqrt{6/n_{in}}`.

    For **ReLU**. The factor of 2 over Xavier compensates for ReLU discarding
    half the signal.
    """
    limit = math.sqrt(6.0 / fan_in)
    return rng.uniform(-limit, limit)


def he_normal(fan_in: int, fan_out: int, rng: random.Random) -> float:
    r"""He normal: :math:`\mathcal{N}(0, 2/n_{in})`. For **ReLU**."""
    return rng.gauss(0.0, math.sqrt(2.0 / fan_in))


def lecun_normal(fan_in: int, fan_out: int, rng: random.Random) -> float:
    r"""LeCun normal: :math:`\mathcal{N}(0, 1/n_{in})`.

    Preserves forward variance exactly, ignoring the backward direction. The
    right choice for SELU, and a reasonable default for linear layers.
    """
    return rng.gauss(0.0, math.sqrt(1.0 / fan_in))


_BY_NAME: dict[str, Initializer] = {
    "zeros": zeros,
    "xavier_uniform": xavier_uniform,
    "xavier": xavier_uniform,
    "glorot": xavier_uniform,
    "xavier_normal": xavier_normal,
    "he_uniform": he_uniform,
    "he": he_normal,
    "kaiming": he_normal,
    "he_normal": he_normal,
    "lecun": lecun_normal,
    "lecun_normal": lecun_normal,
}


def for_activation(activation: str) -> Initializer:
    """Pick the initialiser whose derivation matches the activation.

    ============================  ==================
    activation                    initialiser
    ============================  ==================
    relu, leaky_relu, softplus    He normal
    tanh, sigmoid                 Xavier uniform
    linear (and anything else)    LeCun normal
    ============================  ==================
    """
    name = activation.lower()
    if name in ("relu", "leaky_relu", "softplus", "elu"):
        return he_normal
    if name in ("tanh", "sigmoid"):
        return xavier_uniform
    return lecun_normal


def get_initializer(spec: str | Initializer, activation: str = "tanh") -> Initializer:
    """Resolve an initialiser from a name, a callable, or ``'auto'``."""
    if callable(spec):
        return spec
    key = spec.lower()
    if key == "auto":
        return for_activation(activation)
    if key not in _BY_NAME:
        raise ValueError(
            f"unknown initializer {spec!r}; choose from "
            f"{sorted(set(_BY_NAME)) + ['auto']}"
        )
    return _BY_NAME[key]
