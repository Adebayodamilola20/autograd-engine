r"""Phase 18 -- ``Tensor``: the same autodiff idea, one dimension up.

The scalar engine is correct and unusably slow. This file explains why, and
fixes it, without changing a single idea.

What stays exactly the same
---------------------------
Everything conceptual:

* a node holds ``data``, ``grad``, ``_prev``, ``_op`` and a ``_backward`` closure,
* operations build a DAG as a side effect of ordinary Python,
* ``backward()`` topologically sorts, seeds the output, sweeps in reverse,
* every rule **accumulates** with ``+=``.

``topological_sort`` from ``graph.py`` is reused *unmodified* -- it only ever
required a ``_prev`` attribute. That reuse is the proof that scalar and tensor
autodiff are the same algorithm.

What changes
------------
``data`` is now an ``np.ndarray`` instead of a float, and three genuinely new
problems appear:

1. **Broadcasting.** ``(32, 128) + (128,)`` is legal in the forward direction
   because NumPy stretches the bias across the batch. The gradient must
   therefore be **summed back down** over the stretched axes -- and getting
   this wrong is the single most common bug in a hand-written tensor engine.
   See ``_unbroadcast``.
2. **Matrix multiplication.** Not a scalar rule at all. :math:`C = AB` gives
   :math:`\bar A = \bar C B^{\top}` and :math:`\bar B = A^{\top}\bar C`.
3. **Reductions.** ``sum`` and ``mean`` collapse axes forward, so their
   backward rules must *expand* a gradient back out. ``max`` routes gradient
   only to the elements that won.

Why this is ~1000x faster
-------------------------
Not because the arithmetic changed -- it is the same multiply-accumulates.
Because of what surrounds it:

============================  ==================  =====================
                              scalar engine       tensor engine
============================  ==================  =====================
graph nodes per forward pass  ~220,000 per sample ~15 for the whole batch
Python-level operations       ~220,000            ~15
where the arithmetic happens  CPython bytecode    BLAS (compiled, SIMD)
memory per value              ~56 B Python object 8 B in a packed array
============================  ==================  =====================

A ``Value`` operation is dominated by interpreter overhead -- allocate an
object, build a closure, chase pointers -- with the actual multiply being a
rounding error in the cost. A ``Tensor`` operation pays that overhead **once**
and then hands a million multiply-accumulates to a hand-tuned BLAS kernel that
uses SIMD registers and a cache-blocked loop order.

*That* is why PyTorch is built around tensors. Not because tensors are
mathematically necessary -- our scalar engine computes identical gradients --
but because the graph must be small enough that Python is not the bottleneck.
``benchmarks/`` measures the gap.
"""

from __future__ import annotations

from itertools import count
from typing import Any

import numpy as np

from .graph import topological_sort

__all__ = ["Tensor"]

ArrayLike = Any


def _noop() -> None:
    """Backward function for leaves."""


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    r"""Reduce ``grad`` back to ``shape``, undoing NumPy broadcasting.

    **The most important eight lines in this file.**

    Broadcasting in the forward pass is an implicit *copy*: adding a ``(128,)``
    bias to a ``(32, 128)`` activation uses each bias element 32 times. Section
    10 of the theory doc says a value used :math:`k` times accumulates
    :math:`k` gradient contributions -- so the bias gradient must be the **sum**
    over the batch axis, not a single row of it.

    Broadcasting is therefore a *reduction* in reverse, exactly as a reduction
    is a broadcast in reverse. The two operations are adjoint to each other,
    which is a genuinely elegant fact and not a coincidence: it falls out of the
    fact that the transpose of a copy matrix is a summing matrix.

    Two cases to undo, in order:

    1. NumPy prepended axes (``(128,)`` became ``(1, 128)``): sum them away.
    2. An axis of size 1 was stretched (``(1, 128)`` -> ``(32, 128)``): sum
       along it, keeping the dimension.

    Getting this wrong is subtle. Skipping it produces a shape error --
    noisy, easy to find. Summing the *wrong* axis produces correct shapes and
    wrong numbers, which is why ``tests/test_tensor.py`` gradient-checks every
    broadcasting pattern.
    """
    # 1. extra leading axes NumPy added
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # 2. axes that were size 1 and got stretched
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """An n-dimensional array that records how it was computed.

    Parameters
    ----------
    data
        Anything ``np.asarray`` accepts. Stored as ``float64``.
    label
        Display name for debugging and visualisation.

    Attributes
    ----------
    data : np.ndarray
        Forward values.
    grad : np.ndarray
        Same shape as ``data``; ``∂L/∂self`` after ``backward()``.

    Examples
    --------
    >>> x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> w = Tensor([[0.5], [-0.5]])
    >>> y = (x @ w).sum()
    >>> y.backward()
    >>> w.grad.ravel().tolist()      # column sums of x
    [4.0, 6.0]
    """

    __slots__ = ("data", "grad", "label", "_prev", "_op", "_backward", "_id")

    _counter = count()

    # Hand dispatch back to us when a raw ndarray is on the left.
    #
    # ``arr + t`` and ``arr @ t`` do *not* start at ``Tensor.__radd__``. Python
    # offers the left operand first, and ``ndarray`` accepts almost anything:
    # it calls ``np.asarray(t)``, gets a 0-d object array, and either builds
    # nonsense elementwise or raises "operand does not have enough dimensions".
    # Either way our graph is never built and the gradient is silently lost.
    #
    # Setting ``__array_ufunc__ = None`` makes every numpy ufunc return
    # ``NotImplemented`` for operands involving a ``Tensor``, so Python falls
    # through to our reflected ``__radd__``/``__rmatmul__``/... as it should.
    # The cost is that ``np.exp(tensor)`` now raises ``TypeError`` instead of
    # quietly bypassing autodiff -- which is the outcome we want. Use
    # ``tensor.exp()``, which records a node.
    __array_ufunc__ = None

    def __init__(
        self,
        data: ArrayLike,
        _children: tuple["Tensor", ...] = (),
        _op: str = "",
        label: str = "",
    ) -> None:
        self.data: np.ndarray = np.asarray(data, dtype=np.float64)
        # np.zeros(shape) rather than np.zeros_like(data): identical result,
        # since data is float64 and C-contiguous by construction one line up,
        # but ~1.6x faster because it skips inspecting the source array's
        # dtype, order and subclass. Called once per node, so it showed up in
        # the Phase 19 profile at ~8,300 calls per epoch.
        self.grad: np.ndarray = np.zeros(self.data.shape)
        self._prev: tuple["Tensor", ...] = tuple(_children)
        self._op: str = _op
        self.label: str = label
        self._backward = _noop
        self._id: int = next(Tensor._counter)

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def zeros(*shape: int, label: str = "") -> "Tensor":
        return Tensor(np.zeros(shape), label=label)

    @staticmethod
    def ones(*shape: int, label: str = "") -> "Tensor":
        return Tensor(np.ones(shape), label=label)

    @staticmethod
    def randn(
        *shape: int, std: float = 1.0, rng: np.random.Generator | None = None,
        label: str = "",
    ) -> "Tensor":
        rng = rng or np.random.default_rng()
        return Tensor(rng.normal(0.0, std, size=shape), label=label)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def size(self) -> int:
        return int(self.data.size)

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    @property
    def is_leaf(self) -> bool:
        return not self._prev

    def item(self) -> float:
        return float(self.data.reshape(()))

    def reset_grad(self) -> None:
        """Zero this tensor's gradient, preserving shape.

        Allocates a fresh array rather than filling in place: a previous
        backward pass may have handed this ``.grad`` out to something that kept
        a reference, and zeroing it underneath them would be a spooky
        action-at-a-distance bug. Allocation is cheap; that class of bug is not.
        """
        self.grad = np.zeros(self.data.shape)

    def __repr__(self) -> str:
        name = f"{self.label}=" if self.label else ""
        op = f", op={self._op!r}" if self._op else ""
        return f"Tensor({name}shape={self.shape}{op})"

    @staticmethod
    def _coerce(other: Any) -> "Tensor":
        if isinstance(other, Tensor):
            return other
        if isinstance(other, (int, float, np.ndarray, list, tuple)):
            return Tensor(other)
        raise TypeError(f"cannot combine Tensor with {type(other).__name__}")

    # ==================================================================
    # ELEMENTWISE BINARY OPS (broadcasting-aware)
    # ==================================================================

    def __add__(self, other: Any) -> "Tensor":
        r"""Elementwise addition with broadcasting.

        Local derivatives are 1 as in the scalar case -- but each parent's
        gradient must be *unbroadcast* back to that parent's own shape. This is
        precisely how a bias vector shared across a batch collects the sum of
        the batch's gradients.
        """
        other = self._coerce(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad = self.grad + _unbroadcast(out.grad, self.shape)
            other.grad = other.grad + _unbroadcast(out.grad, other.shape)

        out._backward = _backward
        return out

    def __mul__(self, other: Any) -> "Tensor":
        r"""Elementwise (Hadamard) product. Not matrix multiplication -- see ``@``.

        Same swap-the-inputs rule as the scalar case, applied elementwise, then
        unbroadcast.
        """
        other = self._coerce(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad = self.grad + _unbroadcast(other.data * out.grad, self.shape)
            other.grad = other.grad + _unbroadcast(self.data * out.grad, other.shape)

        out._backward = _backward
        return out

    def __pow__(self, exponent: float) -> "Tensor":
        r"""Elementwise power with a constant exponent: :math:`n a^{n-1}`."""
        if isinstance(exponent, Tensor):
            raise TypeError(
                "Tensor ** Tensor is not implemented; use (b * a.log()).exp()"
            )
        n = float(exponent)
        # A negative exponent is a division, and `0 ** -1` is `inf`, not an
        # error, in NumPy. Since `a / b` is built as `a * b ** -1.0`, letting
        # that through means division by zero silently produces `inf` and then
        # `nan` gradients, while the scalar engine raises ZeroDivisionError for
        # the identical expression. `log` already guards its own domain here;
        # this closes the matching hole.
        if n < 0.0 and self.data.size and np.any(self.data == 0.0):
            raise ZeroDivisionError(
                f"zero to the power {n:g}: division by zero in "
                f"{int(np.count_nonzero(self.data == 0.0))} of "
                f"{self.data.size} elements"
            )
        out = Tensor(self.data**n, (self,), f"**{n:g}")

        def _backward() -> None:
            self.grad = self.grad + n * self.data ** (n - 1.0) * out.grad

        out._backward = _backward
        return out

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __sub__(self, other: Any) -> "Tensor":
        return self + (-self._coerce(other))

    def __truediv__(self, other: Any) -> "Tensor":
        return self * (self._coerce(other) ** -1.0)

    def __radd__(self, other: Any) -> "Tensor":
        return self + other

    def __rmul__(self, other: Any) -> "Tensor":
        return self * other

    def __rsub__(self, other: Any) -> "Tensor":
        return self._coerce(other) + (-self)

    def __rtruediv__(self, other: Any) -> "Tensor":
        return self._coerce(other) * (self**-1.0)

    # ==================================================================
    # MATRIX MULTIPLICATION
    # ==================================================================

    def __matmul__(self, other: Any) -> "Tensor":
        r"""Matrix product :math:`C = AB`, with the rule that runs deep learning.

        Derivation
        ----------
        Write it in index form: :math:`C_{ij} = \sum_k A_{ik}B_{kj}`. Then

        .. math::
            \frac{\partial C_{ij}}{\partial A_{mn}}
            = \delta_{im}B_{nj}

        and the chain rule sums over all outputs :math:`C_{ij}` that
        :math:`A_{mn}` influenced:

        .. math::
            \bar{A}_{mn}
            = \sum_{ij}\bar{C}_{ij}\,\delta_{im}B_{nj}
            = \sum_{j}\bar{C}_{mj}B_{nj}
            = \left(\bar{C}B^{\top}\right)_{mn}

        and symmetrically

        .. math::
            \bar{B} = A^{\top}\bar{C}

        So:

        .. math:: \boxed{\;\bar A = \bar C B^{\top}, \qquad \bar B = A^{\top}\bar C\;}

        Two sanity checks that make this memorable:

        * **Shapes must work out.** :math:`A` is :math:`(m,k)`, :math:`B` is
          :math:`(k,n)`, :math:`\bar C` is :math:`(m,n)`. Then
          :math:`\bar C B^\top` is :math:`(m,n)(n,k) = (m,k)` ✓ and
          :math:`A^\top \bar C` is :math:`(k,m)(m,n) = (k,n)` ✓. There is only
          one way to arrange the transposes that type-checks, which is a
          genuinely useful mnemonic.
        * **It reduces to the scalar rule.** For :math:`1\times1` matrices this
          is :math:`\bar a = \bar c\, b` and :math:`\bar b = a\,\bar c` -- the
          multiplication rule from ``value.py``, unchanged.

        The backward pass of a matmul is *two more matmuls*, which is why a
        training step costs roughly 3x a forward pass, and why GEMM performance
        determines nearly everything about training speed.
        """
        other = self._coerce(other)
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward() -> None:
            grad_self = out.grad @ np.swapaxes(other.data, -1, -2)
            grad_other = np.swapaxes(self.data, -1, -2) @ out.grad
            # Batched matmul can broadcast leading dimensions too.
            self.grad = self.grad + _unbroadcast(grad_self, self.shape)
            other.grad = other.grad + _unbroadcast(grad_other, other.shape)

        out._backward = _backward
        return out

    def __rmatmul__(self, other: Any) -> "Tensor":
        return self._coerce(other) @ self

    # ==================================================================
    # REDUCTIONS
    # ==================================================================

    def sum(
        self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> "Tensor":
        r"""Sum over ``axis`` (all elements if ``None``).

        Backward rule
        -------------
        :math:`\partial(\sum_i x_i)/\partial x_j = 1` for every :math:`j`, so
        the incoming gradient is **broadcast back** to the input shape,
        unchanged.

        Note the duality with ``_unbroadcast``: summing forward means
        broadcasting backward, and broadcasting forward means summing backward.
        Reduction and broadcast are adjoint operations.
        """
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum"
        )
        in_shape = self.shape

        def _backward() -> None:
            grad = out.grad
            if axis is not None and not keepdims:
                # Put the collapsed axes back so broadcasting lines up.
                axes = (axis,) if isinstance(axis, int) else axis
                for ax in sorted(a % len(in_shape) for a in axes):
                    grad = np.expand_dims(grad, ax)
            self.grad = self.grad + np.broadcast_to(grad, in_shape).copy()

        out._backward = _backward
        return out

    def mean(
        self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> "Tensor":
        r"""Arithmetic mean.

        Since :math:`\text{mean} = \frac{1}{N}\sum`, the backward rule is the
        sum rule scaled by :math:`1/N`. Composed from ``sum`` and a division so
        no new derivative is needed -- the chain rule supplies it.
        """
        if axis is None:
            n = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else axis
            n = int(np.prod([self.shape[a] for a in axes]))
        return self.sum(axis=axis, keepdims=keepdims) / float(n)

    def max(
        self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> "Tensor":
        r"""Maximum over ``axis``.

        Backward rule
        -------------
        ``max`` is a **router**: the output equals exactly one input, so all the
        gradient goes to that input and none to the others.

        .. math::
            \frac{\partial \max_i x_i}{\partial x_j} =
            \begin{cases} 1 & x_j \text{ is the max} \\ 0 & \text{otherwise}\end{cases}

        Ties are genuinely ambiguous (the function is not differentiable there),
        and we split the gradient evenly among the tied elements -- which keeps
        the total correct and is what a subgradient permits.
        """
        result = self.data.max(axis=axis, keepdims=keepdims)
        out = Tensor(result, (self,), "max")
        in_shape = self.shape

        def _backward() -> None:
            grad = out.grad
            expanded = result
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else axis
                for ax in sorted(a % len(in_shape) for a in axes):
                    grad = np.expand_dims(grad, ax)
                    expanded = np.expand_dims(expanded, ax)
            mask = (self.data == expanded).astype(np.float64)
            # Share the gradient among tied maxima so the total is preserved.
            mask /= mask.sum(
                axis=axis if axis is not None else None, keepdims=True
            )
            self.grad = self.grad + mask * grad

        out._backward = _backward
        return out

    # ==================================================================
    # SHAPE OPERATIONS
    # ==================================================================

    def reshape(self, *shape: int) -> "Tensor":
        r"""Reshape without moving data.

        Backward rule: reshape the gradient back. A pure re-indexing, so no
        arithmetic and no scaling -- every output element *is* an input
        element.
        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(self.data.reshape(shape), (self,), "reshape")
        in_shape = self.shape

        def _backward() -> None:
            self.grad = self.grad + out.grad.reshape(in_shape)

        out._backward = _backward
        return out

    def transpose(self, *axes: int) -> "Tensor":
        r"""Permute axes; with no arguments, reverse them.

        Backward rule: apply the **inverse permutation** to the gradient. For
        the 2-D case that is just another transpose, since transposition is its
        own inverse.
        """
        order = axes if axes else tuple(reversed(range(self.ndim)))
        out = Tensor(np.transpose(self.data, order), (self,), "T")
        inverse = np.argsort(order)

        def _backward() -> None:
            self.grad = self.grad + np.transpose(out.grad, inverse)

        out._backward = _backward
        return out

    def __getitem__(self, index: Any) -> "Tensor":
        r"""Indexing and slicing.

        Backward rule: **scatter-add** the gradient into a zero array at the
        same positions. ``np.add.at`` rather than ``[] +=`` because fancy
        indexing with repeated indices must accumulate, not overwrite -- the
        same accumulation rule as everywhere else in the engine, in NumPy's
        clothing.
        """
        out = Tensor(self.data[index], (self,), "index")
        in_shape = self.shape

        def _backward() -> None:
            buffer = np.zeros(in_shape)
            np.add.at(buffer, index, out.grad)
            self.grad = self.grad + buffer

        out._backward = _backward
        return out

    # ==================================================================
    # ELEMENTWISE FUNCTIONS
    # ==================================================================

    def exp(self) -> "Tensor":
        r""":math:`e^{x}`, elementwise. Derivative is the output.

        Numerics
        --------
        Overflows above ~709.78 in double precision. We raise rather than
        return ``inf``, for the same reason ``Value.exp`` does: an ``inf``
        here does not stay here. It reaches the backward pass, where
        ``inf * 0.0`` is ``nan``, and from there it spreads to every parameter
        upstream. The run continues, the loss prints as ``nan``, and the cause
        is many operations behind the symptom.

        ``log_softmax`` subtracts the row maximum before exponentiating, so
        every real workload in this repository stays inside the safe range by
        construction. Hitting this error means a genuinely unstable
        formulation, not an unlucky input.
        """
        # `np.max` of an empty array raises, and an empty exp is trivially
        # in range, so the size check comes first.
        if self.data.size and np.max(self.data) > 709.78:
            offender = float(np.max(self.data))
            raise OverflowError(
                f"exp({offender:g}) overflows float64 (limit ~709.78). "
                "Use a numerically stable formulation -- see nabla/losses/."
            )
        out = Tensor(np.exp(self.data), (self,), "exp")

        def _backward() -> None:
            self.grad = self.grad + out.data * out.grad

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        r""":math:`\ln x`, elementwise. Derivative :math:`1/x`."""
        if np.any(self.data <= 0.0):
            raise ValueError("log() requires all elements to be positive")
        out = Tensor(np.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad = self.grad + out.grad / self.data

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        r""":math:`\tanh x`. Derivative :math:`1 - \tanh^{2}x`."""
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")

        def _backward() -> None:
            self.grad = self.grad + (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        r""":math:`\sigma(x)`. Derivative :math:`\sigma(1-\sigma)`.

        Computed branchlessly but stably: ``np.where`` on the sign, evaluating
        both branches with clipped exponents so neither overflows. (Evaluating
        both is cheap and keeps the operation vectorised -- a real example of
        the tensor engine trading a little redundant arithmetic for the ability
        to stay in a single NumPy call.)
        """
        x = self.data
        positive = x >= 0
        s = np.empty_like(x)
        exp_neg = np.exp(-np.abs(x))
        s[positive] = 1.0 / (1.0 + exp_neg[positive])
        s[~positive] = exp_neg[~positive] / (1.0 + exp_neg[~positive])

        out = Tensor(s, (self,), "sigmoid")

        def _backward() -> None:
            self.grad = self.grad + s * (1.0 - s) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        r""":math:`\max(0,x)`. Derivative is 1 where the input was positive."""
        out = Tensor(np.maximum(self.data, 0.0), (self,), "relu")

        def _backward() -> None:
            self.grad = self.grad + (self.data > 0.0) * out.grad

        out._backward = _backward
        return out

    # ==================================================================
    # SOFTMAX FAMILY
    # ==================================================================

    def log_softmax(self, axis: int = -1) -> "Tensor":
        r"""Numerically stable log-softmax along ``axis``.

        Same log-sum-exp trick as the scalar version, vectorised: subtract the
        per-row maximum (a constant, since softmax is shift-invariant, so it
        contributes no gradient), then

        .. math:: \ln p_k = z_k - m - \ln\sum_j e^{z_j - m}

        Built entirely from ``sub``, ``exp``, ``sum`` and ``log``, so the
        gradient comes from the chain rule rather than a hand-written rule.
        """
        shifted = self - Tensor(self.data.max(axis=axis, keepdims=True))
        return shifted - shifted.exp().sum(axis=axis, keepdims=True).log()

    def softmax(self, axis: int = -1) -> "Tensor":
        r"""Softmax probabilities along ``axis``."""
        return self.log_softmax(axis=axis).exp()

    # ==================================================================
    # BACKPROPAGATION
    # ==================================================================

    def backward(self, gradient: np.ndarray | None = None) -> None:
        r"""Reverse-mode autodiff, identical in structure to ``Value.backward``.

        Parameters
        ----------
        gradient
            The seed :math:`\bar{\text{self}}`. Defaults to ones, which is only
            meaningful when ``self`` is a scalar.

        Why non-scalar outputs need a seed
        ----------------------------------
        For a scalar output, :math:`\partial L/\partial L = 1` is the obvious
        seed. For an output with :math:`m` elements, "the derivative of self
        with respect to self" is an :math:`m \times m` identity *Jacobian*, and
        reverse mode computes one row of it per sweep -- so you must say which
        combination you want.

        Passing a vector :math:`v` computes the **vector-Jacobian product**
        :math:`v^{\top}J` in a single sweep. That is exactly what
        ``torch.Tensor.backward(gradient=...)`` does, and why PyTorch raises
        "grad can be implicitly created only for scalar outputs" without it.
        Now the error message makes sense.

        Accumulation semantics
        ----------------------
        Identical to ``Value.backward`` -- leaf gradients accumulate across
        calls, intermediates are cleared first. See that docstring for why the
        two cases have to differ.
        """
        if gradient is None:
            if self.data.size != 1:
                raise RuntimeError(
                    f"backward() on a non-scalar Tensor of shape {self.shape} "
                    "needs an explicit `gradient` -- the seed is a Jacobian "
                    "row, not a number. Reduce with .sum() or .mean() first, "
                    "or pass gradient=..."
                )
            gradient = np.ones_like(self.data)

        order = topological_sort(self)

        # Clear intermediates only; leaves are running totals. See
        # ``Value.backward`` for the worked example of what goes wrong without
        # this (a second sweep re-reads pass-one residue as a real gradient).
        for node in order:
            if node._prev:
                node.reset_grad()

        self.grad = self.grad + np.asarray(gradient, dtype=np.float64).reshape(
            self.shape
        )
        for node in reversed(order):
            node._backward()

    def zero_grad(self) -> None:
        """Reset gradients on every node reachable from this one."""
        for node in topological_sort(self):
            node.reset_grad()
