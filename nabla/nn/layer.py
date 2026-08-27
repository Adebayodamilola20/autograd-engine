r"""``Layer`` -- a row of neurons reading the same input.

.. math::
    \mathbf{a} = \sigma(W\mathbf{x} + \mathbf{b}), \qquad
    W \in \mathbb{R}^{n_{\text{out}} \times n_{\text{in}}}

Every neuron in a layer sees the **same** input vector and produces one output,
so a layer of :math:`n_{\text{out}}` neurons turns an :math:`n_{\text{in}}`-vector
into an :math:`n_{\text{out}}`-vector. The neurons are independent of each other
in the forward pass -- there are no connections *within* a layer -- which is
exactly the structure that makes the whole thing a matrix multiply.

That last point is worth dwelling on, because it is the bridge to Phase 18.
Writing the layer neuron-by-neuron makes the mathematics obvious and the
implementation slow. Writing it as :math:`W\mathbf{x}` makes it one BLAS call.
They compute the same numbers; only one of them trains MNIST before lunch.

Parameter count: :math:`n_{\text{in}} \times n_{\text{out}}` weights plus
:math:`n_{\text{out}}` biases.
"""

from __future__ import annotations

import random
from typing import Callable, Sequence

from ..core.value import Value
from .init import get_initializer
from .module import Module, as_values
from .neuron import Neuron
from .parameter import Parameter

__all__ = ["Layer"]


class Layer(Module):
    """A fully-connected layer: ``n_in`` inputs -> ``n_out`` outputs.

    Examples
    --------
    >>> import random
    >>> layer = Layer(3, 4, activation='relu', rng=random.Random(0))
    >>> outs = layer([1.0, 2.0, 3.0])
    >>> len(outs), len(layer.parameters())      # 4 outputs; 3*4 + 4 params
    (4, 16)
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        activation: str | Callable[[Value], Value] = "tanh",
        *,
        initializer: str | Callable = "auto",
        rng: random.Random | None = None,
        label: str = "",
    ) -> None:
        super().__init__()
        if n_in < 1 or n_out < 1:
            raise ValueError(f"layer sizes must be >= 1, got {n_in} -> {n_out}")

        rng = rng or random.Random()
        act_name = activation if isinstance(activation, str) else "linear"
        init = get_initializer(initializer, act_name)

        self.n_in = n_in
        self.n_out = n_out
        self.activation_name = act_name

        # Build the neurons here rather than delegating, so the initialiser
        # sees the true fan_out. Xavier depends on both fans, and a Neuron on
        # its own can only assume fan_out=1 -- which would make every layer
        # look like a 1-unit layer and scale the weights wrongly.
        neurons = []
        for j in range(n_out):
            neuron = Neuron(
                n_in, activation, initializer=init, rng=rng, label=f"{label}n{j}_"
            )
            for i, w in enumerate(neuron.weights):
                w.data = init(n_in, n_out, rng)
                w.label = f"{label}w{j},{i}"
            neuron.bias.label = f"{label}b{j}"
            neurons.append(neuron)
        self.neurons = neurons

    def forward(self, xs: Sequence[Value | float]) -> list[Value]:
        """Apply every neuron to the same input vector."""
        xs = as_values(xs)
        if len(xs) != self.n_in:
            raise ValueError(f"expected {self.n_in} inputs, got {len(xs)}")
        return [neuron(xs) for neuron in self.neurons]

    def weight_matrix(self) -> list[list[float]]:
        """Weights as a plain ``n_out x n_in`` nested list.

        Used by the visualisation and checkpoint code; keeps them independent
        of how the layer stores its parameters internally.
        """
        return [[w.data for w in n.weights] for n in self.neurons]

    def bias_vector(self) -> list[float]:
        return [n.bias.data for n in self.neurons]

    def __repr__(self) -> str:
        return (
            f"Layer({self.n_in} -> {self.n_out}, {self.activation_name}, "
            f"{self.num_parameters():,} params)"
        )
