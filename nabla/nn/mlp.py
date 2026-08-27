r"""``MLP`` -- a stack of layers, configurable rather than hard-coded.

A multi-layer perceptron is a chain of affine maps separated by nonlinearities:

.. math::
    \mathbf{h}_1 = \sigma(W_1\mathbf{x} + \mathbf{b}_1), \quad
    \mathbf{h}_2 = \sigma(W_2\mathbf{h}_1 + \mathbf{b}_2), \quad \ldots, \quad
    \mathbf{y} = W_L\mathbf{h}_{L-1} + \mathbf{b}_L

Two design points that are easy to get wrong:

**The output layer usually has no activation.** For classification the network
emits raw **logits** and the softmax lives inside the loss (Phase 9), for
reasons of numerical stability that are spelled out there. For regression the
output must be free to take any real value. So ``output_activation`` defaults
to ``'linear'``, and squashing the output is the exception, not the rule.

**Depth is not free.** Each extra layer multiplies the gradient by another
:math:`\sigma'` on the way back. With tanh (:math:`\sigma' \le 1`) a deep stack
attenuates gradient exponentially; with ReLU (:math:`\sigma' = 1` when active)
it does not. This is the single most important practical consequence of the
chain rule, and Phase 22 measures it by training the same network at depths
1, 2, 4 and 8.
"""

from __future__ import annotations

import random
from typing import Callable, Sequence

from ..core.value import Value
from .layer import Layer
from .module import Module, as_values

__all__ = ["MLP"]


class MLP(Module):
    """A configurable multi-layer perceptron.

    Parameters
    ----------
    n_in
        Input dimension (784 for flattened MNIST).
    hidden
        Hidden layer sizes, e.g. ``[128, 64]``.
    n_out
        Output dimension (10 for MNIST digits).
    activation
        Applied after every hidden layer.
    output_activation
        Applied to the final layer. Defaults to ``'linear'`` -- see above.
    initializer
        ``'auto'`` matches the activation (He for ReLU, Xavier for tanh).
    seed
        Convenience for reproducibility; creates ``random.Random(seed)``.
    rng
        An explicit generator, if you are threading one through already.

    Examples
    --------
    >>> model = MLP(784, [128, 64], 10, activation='relu', seed=0)
    >>> model.num_parameters()
    109386
    >>> len(model([0.0] * 784))
    10
    """

    def __init__(
        self,
        n_in: int,
        hidden: Sequence[int],
        n_out: int,
        *,
        activation: str | Callable[[Value], Value] = "tanh",
        output_activation: str | Callable[[Value], Value] = "linear",
        initializer: str | Callable = "auto",
        seed: int | None = None,
        rng: random.Random | None = None,
    ) -> None:
        super().__init__()
        if rng is None:
            rng = random.Random(seed)

        sizes = [n_in, *hidden, n_out]
        self.sizes = sizes
        self.n_in = n_in
        self.n_out = n_out
        self.activation_name = activation if isinstance(activation, str) else "custom"
        self.output_activation_name = (
            output_activation if isinstance(output_activation, str) else "custom"
        )

        layers = []
        for i in range(len(sizes) - 1):
            is_last = i == len(sizes) - 2
            layers.append(
                Layer(
                    sizes[i],
                    sizes[i + 1],
                    output_activation if is_last else activation,
                    initializer=initializer,
                    rng=rng,
                    label=f"L{i}.",
                )
            )
        self.layers = layers

    def forward(self, xs: Sequence[Value | float]) -> list[Value]:
        """Run the input through every layer in turn.

        Returns a list of ``Value`` -- the logits for a classifier, or the
        predictions for a regressor. A one-output model still returns a
        one-element list, so callers never have to special-case it.
        """
        out = as_values(xs)
        for layer in self.layers:
            out = layer(out)
        return out

    def predict(self, xs: Sequence[float]) -> list[float]:
        """Forward pass returning plain floats.

        Still builds the graph -- a scalar engine has no equivalent of
        ``torch.no_grad()``, because the graph *is* the computation. The
        wasted bookkeeping is one of the costs measured in Phase 17.
        """
        return [v.data for v in self.forward(xs)]

    def summary(self) -> str:
        """Architecture and parameter count, layer by layer."""
        head = " → ".join(str(s) for s in self.sizes)
        lines = [
            f"MLP  {head}",
            f"  hidden activation: {self.activation_name}"
            f"    output activation: {self.output_activation_name}",
            "",
            f"  {'#':<3} {'layer':<16} {'activation':<12} {'weights':>10} "
            f"{'biases':>8} {'params':>10}",
            "  " + "-" * 64,
        ]
        total = 0
        for i, layer in enumerate(self.layers):
            weights = layer.n_in * layer.n_out
            biases = layer.n_out
            total += weights + biases
            lines.append(
                f"  {i:<3} {f'{layer.n_in} → {layer.n_out}':<16} "
                f"{layer.activation_name:<12} {weights:>10,} {biases:>8,} "
                f"{weights + biases:>10,}"
            )
        lines.append("  " + "-" * 64)
        lines.append(f"  {'total':<33} {'':>10} {'':>8} {total:>10,}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        head = "→".join(str(s) for s in self.sizes)
        return f"MLP({head}, {self.num_parameters():,} params)"
