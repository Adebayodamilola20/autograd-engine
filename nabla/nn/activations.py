r"""Phase 8 -- activation functions, and why a network needs them.

The central fact
----------------
Without a nonlinearity, depth is an illusion. A layer computes
:math:`y = Wx + b`. Stack two:

.. math::
    y = W_2(W_1x + b_1) + b_2 = \underbrace{(W_2W_1)}_{W'}x + \underbrace{(W_2b_1 + b_2)}_{b'}

which is *a single linear layer*. Stack a hundred and it is still a single
linear layer. Composing linear maps gives a linear map -- that is what linear
means. A 100-layer linear network has exactly the representational power of
logistic regression, and no amount of training will make it learn XOR.

Insert any nonlinear :math:`\sigma` between the layers and the composition
:math:`W_2\sigma(W_1x+b_1)+b_2` is no longer expressible as one affine map. The
universal approximation theorem then applies: one hidden layer with enough
units and a non-polynomial activation can approximate any continuous function
on a compact set to arbitrary accuracy.

``examples/train_xor.py`` demonstrates the consequence directly -- the same
network with and without an activation, one solving XOR and one stuck at 50%.

Choosing one
------------
The forward shape matters less than the **derivative**, because the derivative
is what multiplies the gradient on its way back through the layer.

======================  ==============  ==============  ========================
activation              range           max f'(x)       consequence
======================  ==============  ==============  ========================
sigmoid                 (0, 1)          0.25            attenuates >=4x per layer
tanh                    (-1, 1)         1.0             ok shallow, saturates
relu                    [0, inf)        1.0 (exact)     gradient survives depth
leaky relu              (-inf, inf)     1.0             ReLU, minus the dying units
softplus                (0, inf)        1.0             smooth ReLU, costlier
======================  ==============  ==============  ========================

That table is the entire history of why ReLU replaced tanh in deep networks.
Phase 22 measures it.

Implementation note
-------------------
``tanh``, ``sigmoid`` and ``relu`` are engine primitives with hand-derived
backward rules. ``leaky_relu`` and ``softplus`` are **composed** from existing
primitives and get correct gradients for free -- a small but genuine
demonstration that once the primitives are right, the chain rule does the rest.
"""

from __future__ import annotations

from typing import Callable

from ..core.value import Value
from .module import Module

__all__ = [
    "linear",
    "tanh",
    "relu",
    "sigmoid",
    "leaky_relu",
    "softplus",
    "Linear",
    "Tanh",
    "ReLU",
    "Sigmoid",
    "LeakyReLU",
    "Softplus",
    "get_activation",
    "ACTIVATIONS",
]


# ======================================================================
# functional forms
# ======================================================================


def linear(x: Value) -> Value:
    r"""Identity, :math:`f(x)=x`, :math:`f'(x)=1`.

    Not an activation so much as the *absence* of one. Used on the output layer
    of a regression model, and on the logits of a classifier -- where softmax
    lives inside the loss (Phase 9) rather than in the network.
    """
    return x


def tanh(x: Value) -> Value:
    r"""Hyperbolic tangent. Range :math:`(-1,1)`, :math:`f' = 1 - f^{2}`.

    Zero-centred, which is its main advantage over sigmoid: if activations are
    always positive, every weight feeding the next layer receives a gradient of
    the same sign, and updates zig-zag instead of moving diagonally.

    Its weakness is saturation. Past :math:`|x|\approx 3` the derivative is
    under 0.01, so a saturated unit is nearly disconnected from the loss --
    the vanishing-gradient problem.
    """
    return x.tanh()


def relu(x: Value) -> Value:
    r"""Rectified linear unit, :math:`\max(0,x)`.

    A gate: gradient passes untouched or not at all. Because the "on"
    derivative is exactly 1, gradients survive arbitrarily deep stacks -- which
    is most of why deep networks became trainable. It is also cheap (a
    comparison, no transcendental) and induces sparse activations.

    The cost is **dying ReLU**: a unit whose pre-activation is negative for
    every input in the dataset gets zero gradient forever and cannot recover.
    Usually caused by too high a learning rate driving the bias very negative.
    """
    return x.relu()


def sigmoid(x: Value) -> Value:
    r"""Logistic sigmoid, :math:`1/(1+e^{-x})`. Range :math:`(0,1)`.

    Reads as a probability, which is why it survives as the *output* activation
    for binary classification. As a hidden activation it is dominated by tanh:
    its peak derivative is 0.25, so it attenuates gradient by at least 4x per
    layer even when not saturated, and its output is never negative.
    """
    return x.sigmoid()


def leaky_relu(x: Value, slope: float = 0.01) -> Value:
    r"""Leaky ReLU: :math:`x` if :math:`x>0`, else :math:`\alpha x`.

    Fixes dying ReLU by leaving a small gradient :math:`\alpha` on the negative
    side, so a unit that has drifted negative can still be pushed back.

    Composed, not primitive
    -----------------------
    .. math:: \text{leaky}(x) = \text{relu}(x) - \alpha\,\text{relu}(-x)

    For :math:`x>0` the second term is 0, giving :math:`x`. For :math:`x<0` the
    first is 0 and :math:`\text{relu}(-x) = -x`, giving :math:`\alpha x`. ✓

    We wrote no backward rule for this function. The engine composes ``relu``,
    ``neg`` and ``sub``, and the chain rule produces the correct derivative
    automatically -- verified by finite differences in the test suite. That is
    what "the primitives are enough" means in practice.
    """
    return x.relu() - slope * (-x).relu()


def softplus(x: Value) -> Value:
    r"""Softplus, :math:`\ln(1+e^{x})` -- a smooth ReLU.

    Its derivative is exactly the sigmoid:
    :math:`\frac{d}{dx}\ln(1+e^{x}) = \frac{e^{x}}{1+e^{x}} = \sigma(x)`,
    a pleasing connection between the two families.

    Numerical stability
    -------------------
    The naive form overflows: at :math:`x=800`, :math:`e^{x}` is ``inf``. Use
    the identity :math:`\ln(1+e^{x}) = x + \ln(1+e^{-x})` for :math:`x>0`, so
    the exponent is always :math:`\le 0`:

    .. math::
        \text{softplus}(x) = \begin{cases}
            x + \ln(1+e^{-x}) & x > 0\\
            \ln(1+e^{x})      & x \le 0
        \end{cases}

    Branching on ``x.data`` is legitimate here and worth noticing: this is a
    *define-by-run* engine, so the graph records whichever branch executed. The
    two branches are algebraically identical, so the derivative is the same
    either way -- we are choosing a numerically better route to the same
    number, not changing the function.
    """
    if x.data > 0.0:
        return x + (1.0 + (-x).exp()).log()
    return (1.0 + x.exp()).log()


# ======================================================================
# Module wrappers
# ======================================================================


class _Activation(Module):
    """Base for the callable-object form of an activation.

    Layers hold a plain function internally; these exist so an activation can
    be an element of a model tree, printed by ``summary()`` and swapped at
    configuration time. They hold no parameters.
    """

    fn: Callable[[Value], Value]

    def forward(self, x):  # type: ignore[override]
        if isinstance(x, (list, tuple)):
            return [type(self).fn(v) for v in x]
        return type(self).fn(x)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Linear(_Activation):
    fn = staticmethod(linear)


class Tanh(_Activation):
    fn = staticmethod(tanh)


class ReLU(_Activation):
    fn = staticmethod(relu)


class Sigmoid(_Activation):
    fn = staticmethod(sigmoid)


class Softplus(_Activation):
    fn = staticmethod(softplus)


class LeakyReLU(Module):
    """Leaky ReLU with a configurable negative slope."""

    def __init__(self, slope: float = 0.01) -> None:
        super().__init__()
        self.slope = slope

    def forward(self, x):  # type: ignore[override]
        if isinstance(x, (list, tuple)):
            return [leaky_relu(v, self.slope) for v in x]
        return leaky_relu(x, self.slope)

    def __repr__(self) -> str:
        return f"LeakyReLU(slope={self.slope})"


# ======================================================================
# registry
# ======================================================================

ACTIVATIONS: dict[str, Callable[[Value], Value]] = {
    "linear": linear,
    "identity": linear,
    "none": linear,
    "tanh": tanh,
    "relu": relu,
    "sigmoid": sigmoid,
    "leaky_relu": leaky_relu,
    "softplus": softplus,
}


def get_activation(spec: str | Callable[[Value], Value]) -> Callable[[Value], Value]:
    """Resolve an activation from a name or a callable.

    Lets a model be configured from a string -- which is what makes experiment
    sweeps (Phase 22) and the CLI (Phase 23) possible without editing code.
    """
    if callable(spec):
        return spec
    key = spec.lower()
    if key not in ACTIVATIONS:
        raise ValueError(
            f"unknown activation {spec!r}; choose from {sorted(ACTIVATIONS)}"
        )
    return ACTIVATIONS[key]
