r"""Phase 18 -- the same network, built on ``Tensor`` instead of ``Value``.

Read this file next to ``neuron.py`` / ``layer.py`` / ``mlp.py``. The
mathematics is identical; only the granularity changed.

===========================  ==============================  ==========================
                             scalar (``nn/layer.py``)        tensor (this file)
===========================  ==============================  ==========================
a layer's forward pass       ``n_out`` Python loops over     one ``x @ W + b``
                             ``n_in`` multiply-adds
graph nodes for 784→128      ~200,000 per sample             2 per batch
who does the arithmetic      CPython, one float at a time    BLAS, cache-blocked SIMD
bias handling                one ``Parameter`` per neuron    one row, broadcast
===========================  ==============================  ==========================

The weights are stored as a single ``(n_in, n_out)`` matrix rather than
``n_out`` separate weight vectors. That is the whole trick: it lets the layer
be **one** node in the graph, so ``backward()`` visits 2 nodes per layer instead
of 200,000. The gradient it computes is bit-for-bit the same thing the scalar
engine computes -- ``tests/test_tensor.py`` asserts exactly that, by building
both and comparing.

Note that ``Module``, ``Optimizer``, ``Trainer`` and the checkpoint code are
reused **unchanged**. Only the leaf type differs, which is a good sign the
abstractions were drawn in the right places.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from ..core.tensor import Tensor
from .module import Module
from .parameter import Learnable

__all__ = ["TensorParameter", "Linear", "TensorMLP"]


class TensorParameter(Tensor, Learnable):
    """A learnable ``Tensor``.

    Same role as ``Parameter`` for the scalar engine: the ``Learnable`` marker
    is what makes ``Module.parameters()`` pick it up, so ``Module`` needs no
    knowledge of which engine is in use.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        name = f"{self.label}=" if self.label else ""
        return f"TensorParameter({name}shape={self.shape})"


def _init_matrix(
    fan_in: int, fan_out: int, scheme: str, rng: np.random.Generator
) -> np.ndarray:
    r"""Initialise a weight matrix. Same schemes as ``nn/init.py``, vectorised.

    * ``he``     -- :math:`\mathcal{N}(0, 2/n_{in})`, for ReLU
    * ``xavier`` -- :math:`U(\pm\sqrt{6/(n_{in}+n_{out})})`, for tanh/sigmoid
    * ``lecun``  -- :math:`\mathcal{N}(0, 1/n_{in})`, for linear
    """
    if scheme == "he":
        return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))
    if scheme == "xavier":
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-limit, limit, size=(fan_in, fan_out))
    if scheme == "lecun":
        return rng.normal(0.0, np.sqrt(1.0 / fan_in), size=(fan_in, fan_out))
    raise ValueError(f"unknown init scheme {scheme!r}")


def _scheme_for(activation: str) -> str:
    if activation in ("relu", "leaky_relu"):
        return "he"
    if activation in ("tanh", "sigmoid"):
        return "xavier"
    return "lecun"


class Linear(Module):
    r"""A fully-connected layer as a single matrix multiply.

    .. math:: Y = \sigma(XW + \mathbf{b})

    with :math:`X` of shape ``(batch, n_in)``, :math:`W` of shape
    ``(n_in, n_out)`` and :math:`\mathbf{b}` of shape ``(n_out,)``.

    The bias is broadcast across the batch, which means its gradient is the
    **sum over the batch axis** -- handled automatically by ``_unbroadcast``
    in ``tensor.py``, and the single most important thing to get right when
    writing a tensor engine.

    Examples
    --------
    >>> import numpy as np
    >>> layer = Linear(4, 3, activation='relu', rng=np.random.default_rng(0))
    >>> out = layer(Tensor(np.zeros((8, 4))))
    >>> out.shape
    (8, 3)
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        *,
        activation: str = "linear",
        rng: np.random.Generator | None = None,
        label: str = "",
    ) -> None:
        super().__init__()
        rng = rng or np.random.default_rng()
        self.n_in = n_in
        self.n_out = n_out
        self.activation_name = activation

        self.W = TensorParameter(
            _init_matrix(n_in, n_out, _scheme_for(activation), rng), label=f"{label}W"
        )
        # Zero bias: no symmetry problem, since each bias sits behind a
        # different (random) weight column.
        self.b = TensorParameter(np.zeros(n_out), label=f"{label}b")

    def forward(self, x: Tensor) -> Tensor:
        """``(batch, n_in) -> (batch, n_out)``, activation applied."""
        z = x @ self.W + self.b
        if self.activation_name in ("linear", "none", "identity"):
            return z
        if self.activation_name == "relu":
            return z.relu()
        if self.activation_name == "tanh":
            return z.tanh()
        if self.activation_name == "sigmoid":
            return z.sigmoid()
        if self.activation_name == "leaky_relu":
            # Composed, exactly as in the scalar engine: relu(x) - a·relu(-x)
            return z.relu() - 0.01 * (-z).relu()
        raise ValueError(f"unknown activation {self.activation_name!r}")

    def __repr__(self) -> str:
        return (
            f"Linear({self.n_in} → {self.n_out}, {self.activation_name}, "
            f"{self.num_parameters():,} params)"
        )


class TensorMLP(Module):
    """A multi-layer perceptron on the tensor engine.

    Drop-in equivalent of ``nn.MLP`` -- same architecture, same initialisation
    schemes, same parameter count -- but processes a whole batch in one graph.

    Parameters
    ----------
    n_in, hidden, n_out
        Architecture, e.g. ``TensorMLP(784, [128, 64], 10)``.
    activation
        Hidden-layer activation.
    output_activation
        Defaults to ``'linear'``: classifiers emit logits and the softmax lives
        in the loss, for the stability reasons in ``losses/cross_entropy.py``.
    seed
        Reproducibility.

    Examples
    --------
    >>> model = TensorMLP(784, [128, 64], 10, activation='relu', seed=0)
    >>> model.num_parameters()
    109386
    """

    def __init__(
        self,
        n_in: int,
        hidden: Sequence[int],
        n_out: int,
        *,
        activation: str = "relu",
        output_activation: str = "linear",
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        rng = rng or np.random.default_rng(seed)

        sizes = [n_in, *hidden, n_out]
        self.sizes = sizes
        self.n_in = n_in
        self.n_out = n_out
        self.activation_name = activation
        self.output_activation_name = output_activation

        layers = []
        for i in range(len(sizes) - 1):
            is_last = i == len(sizes) - 2
            layers.append(
                Linear(
                    sizes[i],
                    sizes[i + 1],
                    activation=output_activation if is_last else activation,
                    rng=rng,
                    label=f"L{i}.",
                )
            )
        self.layers = layers

    def forward(self, x: Tensor | np.ndarray | Sequence[Sequence[float]]) -> Tensor:
        """``(batch, n_in) -> (batch, n_out)`` logits.

        Accepts a ``Tensor``, an ndarray, or a nested list, so the same
        ``Trainer`` can feed it the batches it feeds the scalar model.

        ``Tensor.__init__`` already calls ``np.asarray(..., dtype=float64)``, so
        doing it here too was a redundant conversion on the hottest path in the
        library -- and when ``x`` arrives as a nested list, that conversion is
        not cheap. Prefer ``DataLoader(..., as_arrays=True)``, which skips the
        list round trip entirely (Phase 19).
        """
        if not isinstance(x, Tensor):
            x = Tensor(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        for layer in self.layers:
            x = layer(x)
        return x

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Class indices for a batch of inputs."""
        return np.asarray(self.forward(x).data).argmax(axis=-1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Softmax probabilities for a batch of inputs."""
        return np.asarray(self.forward(x).softmax(axis=-1).data)

    def summary(self) -> str:
        head = " → ".join(str(s) for s in self.sizes)
        lines = [
            f"TensorMLP  {head}",
            f"  hidden activation: {self.activation_name}"
            f"    output activation: {self.output_activation_name}",
            "",
            f"  {'#':<3} {'layer':<16} {'activation':<12} {'W shape':>14} "
            f"{'b shape':>10} {'params':>10}",
            "  " + "-" * 72,
        ]
        total = 0
        for i, layer in enumerate(self.layers):
            count = layer.num_parameters()
            total += count
            lines.append(
                f"  {i:<3} {f'{layer.n_in} → {layer.n_out}':<16} "
                f"{layer.activation_name:<12} {str(layer.W.shape):>14} "
                f"{str(layer.b.shape):>10} {count:>10,}"
            )
        lines.append("  " + "-" * 72)
        lines.append(f"  {'total':<48} {total:>23,}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        head = "→".join(str(s) for s in self.sizes)
        return f"TensorMLP({head}, {self.num_parameters():,} params)"
