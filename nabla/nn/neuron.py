r"""``Neuron`` -- the smallest unit that learns.

What a neuron computes
----------------------
.. math::
    z = \sum_{i=1}^{n} w_i x_i + b, \qquad a = \sigma(z)

Two steps, and they do different jobs:

* the **affine part** :math:`\sum w_ix_i + b` is a weighted vote. Each weight
  says how much its input matters and in which direction; the bias shifts the
  threshold at which the neuron responds at all.
* the **activation** :math:`\sigma` bends the result. Without it, stacking
  neurons gains nothing (see ``activations.py``).

Geometrically the affine part defines a hyperplane, and :math:`z` is
(proportional to) the signed distance from the input to it. The activation then
maps that distance to a response.

Gradients, and what they mean
-----------------------------
The engine derives these, but they are worth reading once:

.. math::
    \frac{\partial a}{\partial w_i} = \sigma'(z)\,x_i, \qquad
    \frac{\partial a}{\partial b} = \sigma'(z), \qquad
    \frac{\partial a}{\partial x_i} = \sigma'(z)\,w_i

Three things fall out immediately:

1. :math:`\partial a/\partial w_i \propto x_i`. **A weight whose input is zero
   receives no gradient.** MNIST border pixels are 0 for nearly every image, so
   their weights barely move -- visible directly in the weight images of
   Phase 13.
2. :math:`\partial a/\partial w_i \propto \sigma'(z)`. **A saturated neuron
   learns nothing**, whatever its inputs.
3. The bias gradient is :math:`\sigma'(z)` alone, unscaled by any input --
   which is why biases are safe to initialise at zero.
"""

from __future__ import annotations

import random
from typing import Callable, Sequence

from ..core.value import Value
from .activations import get_activation
from .init import get_initializer
from .module import Module, as_values
from .parameter import Parameter

__all__ = ["Neuron"]


class Neuron(Module):
    """A single unit: weighted sum of inputs, plus bias, through an activation.

    Parameters
    ----------
    n_inputs
        Length of the input vector.
    activation
        Name (``'tanh'``, ``'relu'``, ``'sigmoid'``, ``'leaky_relu'``,
        ``'softplus'``, ``'linear'``) or a callable ``Value -> Value``.
    initializer
        Name, callable, or ``'auto'`` to match the activation (He for ReLU,
        Xavier for tanh/sigmoid). See ``init.py``.
    rng
        Explicit ``random.Random``. Required for reproducibility; one is
        created if omitted, but then the result is not reproducible.
    label
        Prefix for parameter labels, which show up in graph visualisations.

    Examples
    --------
    >>> import random
    >>> n = Neuron(3, activation='tanh', rng=random.Random(0))
    >>> out = n([0.5, -1.0, 2.0])
    >>> out.backward()
    >>> len(n.parameters())          # 3 weights + 1 bias
    4
    """

    def __init__(
        self,
        n_inputs: int,
        activation: str | Callable[[Value], Value] = "tanh",
        *,
        initializer: str | Callable = "auto",
        rng: random.Random | None = None,
        label: str = "",
    ) -> None:
        super().__init__()
        if n_inputs < 1:
            raise ValueError(f"n_inputs must be >= 1, got {n_inputs}")

        rng = rng or random.Random()
        act_name = activation if isinstance(activation, str) else "linear"
        init = get_initializer(initializer, act_name)

        self.n_inputs = n_inputs
        self.activation_name = act_name
        self.activation = get_activation(activation)

        # fan_out is 1 for a single neuron; the Layer that owns it passes the
        # real fan_out when it matters (Xavier needs both).
        self.weights = [
            Parameter(init(n_inputs, 1, rng), label=f"{label}w{i}")
            for i in range(n_inputs)
        ]
        # Bias at zero: no symmetry problem, and no reason to prefer a shift.
        self.bias = Parameter(0.0, label=f"{label}b")

    def forward(self, xs: Sequence[Value | float]) -> Value:
        """Compute ``activation(Σ wᵢxᵢ + b)``.

        Implementation note
        -------------------
        The sum is accumulated in a left-leaning chain, so the graph for one
        neuron is ``n_inputs`` deep. For MNIST's 784 inputs that is a 786-node
        chain **per neuron, per sample** -- 3,138 nodes in total, all Python
        objects. That number is the honest reason a scalar engine cannot train
        a full-size network in reasonable time, and it is what the ``Tensor``
        engine (Phase 18) collapses into a single dot-product node.
        """
        xs = as_values(xs)
        if len(xs) != self.n_inputs:
            raise ValueError(
                f"expected {self.n_inputs} inputs, got {len(xs)}"
            )

        z = self.bias
        for w, x in zip(self.weights, xs):
            z = z + w * x
        return self.activation(z)

    def __repr__(self) -> str:
        return f"Neuron({self.n_inputs} -> 1, {self.activation_name})"
