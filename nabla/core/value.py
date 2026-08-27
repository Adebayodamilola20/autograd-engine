"""``Value`` -- a scalar node in a reverse-mode automatic differentiation graph.

This is the foundation of the entire project. Roughly 400 lines here support
every neural network we will build: if ``Value`` is correct, the MLP is correct.

The idea in one paragraph
-------------------------
A ``Value`` wraps a single number. When you combine two ``Value`` objects with
an arithmetic operator, the result is a *new* ``Value`` that remembers (a) the
number it computed, (b) which nodes it came from, and (c) a closure that knows
how to convert "the gradient that arrived here" into "the gradient each parent
deserves". Doing that for every operation builds a graph as a side effect of
ordinary Python. Calling ``.backward()`` walks that graph in reverse and fills
in ``.grad`` on every node.

The one equation
----------------
Every backward rule in this file is an instance of the chain rule, written in
adjoint form. With ``L`` the final output and ``v̄ ≡ ∂L/∂v``::

    parent.grad  +=  (∂out/∂parent)  *  out.grad
    └── v̄ of the ──┘  └─ the LOCAL ─┘   └─ what arrived
        parent          derivative,       from downstream
                        known by this
                        operation alone

Read it as: *gradient arriving from downstream, times the local derivative,
accumulated into the parent.* The ``+=`` is not a stylistic choice -- it is the
multivariable chain rule, and using ``=`` silently halves the gradient of any
value that is used more than once. See ``docs/01-*.md`` section 10.

Design decisions (see docs/00-architecture.md)
----------------------------------------------
* ``__slots__``  -- one MNIST forward pass allocates ~10^5-10^6 nodes, so
  removing the per-instance ``__dict__`` is a real memory and speed win.
* closures       -- each derivative sits directly beside the forward code that
  motivates it, and ``backward()`` never has to grow to accommodate a new op.
* accumulate     -- ``.backward()`` adds into ``.grad`` and never clears it,
  matching PyTorch exactly. Use ``zero_grad()`` between steps.
"""

from __future__ import annotations

import math
from itertools import count
from typing import Iterable, Union

from .graph import topological_sort

__all__ = ["Value"]

Number = Union[int, float]
ValueLike = Union["Value", Number]


def _noop() -> None:
    """Default backward function, used by leaves.

    A leaf has no parents, so it has nothing to propagate to. Sharing one
    module-level function (rather than allocating a fresh ``lambda`` per node)
    saves an object per leaf -- which matters when leaves number in the
    hundreds of thousands.
    """


class Value:
    """A scalar with a gradient and a memory of how it was computed.

    Parameters
    ----------
    data : float
        The number this node holds.
    label : str, optional
        A human-readable name, used by visualisation and debugging output.

    Attributes
    ----------
    data : float
        The value computed on the forward pass.
    grad : float
        The adjoint ``∂L/∂self`` for whichever output ``L`` you last called
        ``.backward()`` on. It is ``0.0`` until then -- a gradient is only
        meaningful relative to some output, and before ``.backward()`` no
        output has been named.
    label : str
        Display name.

    Examples
    --------
    >>> a = Value(2.0, label='a')
    >>> b = Value(3.0, label='b')
    >>> d = a * b + a          # d = ab + a
    >>> d.data
    8.0
    >>> d.backward()
    >>> a.grad                 # ∂d/∂a = b + 1 = 4
    4.0
    >>> b.grad                 # ∂d/∂b = a = 2
    2.0
    """

    __slots__ = ("data", "grad", "label", "_prev", "_op", "_backward", "_id")

    # Monotonic ids give every node a stable, readable name for visualisation
    # and debugging. ``id()`` would work but is neither small nor ordered.
    _counter = count()

    def __init__(
        self,
        data: Number,
        _children: tuple["Value", ...] = (),
        _op: str = "",
        label: str = "",
    ) -> None:
        self.data: float = float(data)
        self.grad: float = 0.0
        self._prev: tuple["Value", ...] = tuple(_children)
        self._op: str = _op
        self.label: str = label
        self._backward = _noop
        self._id: int = next(Value._counter)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce(other: ValueLike) -> "Value":
        """Wrap a plain number in a constant ``Value`` so ops are uniform.

        This is what lets ``x * 2`` work: the ``2`` becomes a leaf node with no
        parents. It *does* receive a gradient during the backward pass; we
        simply never look at it, because nobody optimises a literal.
        """
        if isinstance(other, Value):
            return other
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return Value(other)
        raise TypeError(
            f"cannot combine Value with {type(other).__name__}; "
            "expected Value, int or float"
        )

    @property
    def is_leaf(self) -> bool:
        """True if this node has no parents (an input or a parameter)."""
        return not self._prev

    def __repr__(self) -> str:
        name = f"{self.label}=" if self.label else ""
        op = f", op={self._op!r}" if self._op else ""
        return f"Value({name}data={self.data:.6g}, grad={self.grad:.6g}{op})"

    # ==================================================================
    # ARITHMETIC OPERATIONS
    #
    # For each one: the forward formula, the derivative derived from first
    # principles, then the implementation. Nothing below is asserted without
    # being derived, and every rule is verified against finite differences in
    # tests/test_gradients.py.
    # ==================================================================

    def __add__(self, other: ValueLike) -> "Value":
        r"""Addition.  ``out = self + other``

        Derivation
        ----------
        With :math:`f(a, b) = a + b`, hold :math:`b` fixed and differentiate:

        .. math::
            \frac{\partial f}{\partial a}
              = \lim_{h\to 0}\frac{(a+h+b)-(a+b)}{h}
              = \lim_{h\to 0}\frac{h}{h} = 1

        and by symmetry :math:`\partial f/\partial b = 1`.

        Interpretation
        --------------
        Addition is a **gradient router**: it passes the incoming gradient to
        both parents completely unchanged. Nudge either input by ``h`` and the
        sum moves by exactly ``h``. This is why bias terms are so well behaved
        -- gradient reaches them undiminished, never scaled by a weight.

        Tiny example
        ------------
        ``a=2, b=3, out=5``. Seed ``out.grad=1`` -> ``a.grad=1, b.grad=1``.
        """
        other = self._coerce(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += 1.0 * out.grad  # ∂out/∂self  = 1
            other.grad += 1.0 * out.grad  # ∂out/∂other = 1

        out._backward = _backward
        return out

    def __mul__(self, other: ValueLike) -> "Value":
        r"""Multiplication.  ``out = self * other``

        Derivation
        ----------
        With :math:`f(a,b) = ab`, holding :math:`b` fixed:

        .. math::
            \frac{\partial f}{\partial a}
              = \lim_{h\to 0}\frac{(a+h)b - ab}{h}
              = \lim_{h\to 0}\frac{hb}{h} = b

        and symmetrically :math:`\partial f/\partial b = a`.

        Interpretation
        --------------
        Multiplication is a **gradient scaler that swaps its inputs**: each
        parent's gradient is scaled by the *other* parent's value. This single
        fact explains two of deep learning's most famous pathologies. Gradient
        flowing back through a weight is multiplied by that weight, so a chain
        of weights all > 1 makes gradients **explode**, and a chain all < 1
        makes them **vanish**. It is also why input scaling matters: a feature
        with magnitude 255 hands its weight a gradient 255x larger than a
        feature scaled to 1.

        Tiny example
        ------------
        ``a=2, b=3, out=6``. Seed ``out.grad=1`` -> ``a.grad=3, b.grad=2``.
        Check: raising ``a`` to 2.001 gives ``out=6.003``, a rise of 0.003 = 3h.
        """
        other = self._coerce(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad  # ∂out/∂self  = other
            other.grad += self.data * out.grad  # ∂out/∂other = self

        out._backward = _backward
        return out

    def __pow__(self, other: ValueLike) -> "Value":
        r"""Exponentiation.  ``out = self ** other``

        Two cases, because the general one needs a logarithm.

        Case 1 -- constant exponent :math:`n` (the common case)

        .. math::
            \frac{d}{da}a^{n} = n\,a^{n-1}

        the ordinary power rule.

        Case 2 -- both base and exponent are ``Value``. Write
        :math:`a^{b} = e^{b\ln a}` (valid for :math:`a>0`) and differentiate:

        .. math::
            \frac{\partial}{\partial a}a^{b} = b\,a^{b-1},
            \qquad
            \frac{\partial}{\partial b}a^{b} = a^{b}\ln a

        The second follows because :math:`\frac{d}{db}e^{b\ln a} =
        e^{b\ln a}\cdot\ln a`. Note it reuses the forward output.

        Domain
        ------
        A negative base with a fractional exponent is complex-valued (Python
        would silently hand back a ``complex``), so we reject it. A ``Value``
        exponent additionally requires ``base > 0`` for :math:`\ln a` to exist.

        Tiny example
        ------------
        ``a=3, out=a**2=9``. Seed ``out.grad=1`` -> ``a.grad = 2*3 = 6``.
        Check: ``3.001**2 = 9.006001``, a rise of ~0.006 = 6h.
        """
        if isinstance(other, Value):
            if self.data <= 0.0:
                raise ValueError(
                    f"Value**Value requires a positive base (got {self.data}); "
                    "the derivative involves log(base)"
                )
            out = Value(self.data**other.data, (self, other), "**")

            def _backward_vv() -> None:
                # ∂out/∂base     = b * a**(b-1)
                self.grad += other.data * self.data ** (other.data - 1.0) * out.grad
                # ∂out/∂exponent = a**b * ln(a) = out * ln(a)
                other.grad += out.data * math.log(self.data) * out.grad

            out._backward = _backward_vv
            return out

        if not isinstance(other, (int, float)) or isinstance(other, bool):
            raise TypeError("exponent must be a Value, int or float")

        exponent = float(other)
        if self.data < 0.0 and not exponent.is_integer():
            raise ValueError(
                f"({self.data}) ** {exponent} is complex; "
                "a negative base requires an integer exponent"
            )

        out = Value(self.data**exponent, (self,), f"**{exponent:g}")

        def _backward() -> None:
            # ∂out/∂self = n * self**(n-1)
            self.grad += exponent * self.data ** (exponent - 1.0) * out.grad

        out._backward = _backward
        return out

    def __neg__(self) -> "Value":
        r"""Negation.  ``out = -self``

        .. math:: \frac{d}{da}(-a) = -1

        Implemented as ``self * -1`` rather than as a primitive: the
        multiplication rule already gives exactly the right derivative, and one
        fewer primitive is one fewer place to be wrong. The graph gains one
        constant node, which is a fair price.
        """
        return self * -1.0

    def __sub__(self, other: ValueLike) -> "Value":
        r"""Subtraction.  ``out = self - other``

        .. math::
            \frac{\partial}{\partial a}(a-b) = 1,
            \qquad
            \frac{\partial}{\partial b}(a-b) = -1

        Composed as ``self + (-other)``. The sign flip on the second argument
        falls out of the negation rule automatically -- a good demonstration
        that composed operations get correct derivatives for free.
        """
        return self + (-self._coerce(other))

    def __truediv__(self, other: ValueLike) -> "Value":
        r"""Division.  ``out = self / other``

        Derivation
        ----------
        Write :math:`a/b = a\cdot b^{-1}`. Then

        .. math::
            \frac{\partial}{\partial a}\frac{a}{b} = \frac{1}{b},
            \qquad
            \frac{\partial}{\partial b}\frac{a}{b}
              = a\cdot(-1)b^{-2} = -\frac{a}{b^{2}}

        which is the quotient rule, obtained here from the product and power
        rules we already have.

        Note the ``-a/b²`` term blows up as ``b -> 0``. That is not a flaw in
        the engine -- the true derivative really is unbounded there. It is one
        reason numerically-stable formulations (Phase 9) avoid dividing by
        quantities that can approach zero.
        """
        return self * (self._coerce(other) ** -1.0)

    # --- reflected operators: make ``2 + x``, ``2 * x``, ``2 - x`` work ---

    def __radd__(self, other: ValueLike) -> "Value":
        return self + other

    def __rmul__(self, other: ValueLike) -> "Value":
        return self * other

    def __rsub__(self, other: ValueLike) -> "Value":
        return self._coerce(other) + (-self)

    def __rtruediv__(self, other: ValueLike) -> "Value":
        return self._coerce(other) * (self**-1.0)

    def __rpow__(self, other: Number) -> "Value":
        r"""Constant base raised to a ``Value``.  ``out = c ** self``

        .. math::
            \frac{d}{dx}c^{x} = c^{x}\ln c = \text{out}\cdot\ln c

        Requires ``c > 0``. Reuses the forward output, like ``exp``.
        """
        base = float(other)
        if base <= 0.0:
            raise ValueError(
                f"{base} ** Value requires a positive base; "
                "the derivative involves log(base)"
            )
        out = Value(base**self.data, (self,), f"{base:g}**")
        log_base = math.log(base)

        def _backward() -> None:
            self.grad += out.data * log_base * out.grad

        out._backward = _backward
        return out

    # ==================================================================
    # TRANSCENDENTAL FUNCTIONS
    # ==================================================================

    def exp(self) -> "Value":
        r"""Natural exponential.  ``out = e**self``

        Derivation
        ----------
        :math:`e^{x}` is *defined* as the function that is its own derivative:

        .. math:: \frac{d}{dx}e^{x} = e^{x} = \text{out}

        Interpretation
        --------------
        The backward rule reuses the **forward output** rather than recomputing
        anything. This is the first example of a pattern that recurs in
        ``tanh``, ``sigmoid`` and softmax, and it is a concrete reason training
        needs more memory than inference: the framework must keep every
        activation alive until the backward pass consumes it. ``torch.no_grad()``
        exists precisely to skip that bookkeeping.

        Numerics
        --------
        ``exp`` overflows for inputs above ~709.78 in double precision. We
        raise a clear error rather than returning ``inf`` and poisoning the
        whole graph with ``nan``. Phase 9 shows how the log-sum-exp trick keeps
        real workloads inside the safe range.

        Tiny example
        ------------
        ``x=0, out=1``. Seed ``out.grad=1`` -> ``x.grad = 1``.
        """
        try:
            value = math.exp(self.data)
        except OverflowError as err:  # pragma: no cover - defensive
            raise OverflowError(
                f"exp({self.data}) overflows float64 (limit ~709.78). "
                "Use a numerically stable formulation -- see nabla/losses/."
            ) from err

        out = Value(value, (self,), "exp")

        def _backward() -> None:
            self.grad += out.data * out.grad  # ∂out/∂self = e**self = out

        out._backward = _backward
        return out

    def log(self) -> "Value":
        r"""Natural logarithm.  ``out = ln(self)``

        Derivation
        ----------
        :math:`\ln` is the inverse of :math:`\exp`, so from
        :math:`x = e^{y}` and the inverse-function rule
        :math:`dy/dx = 1/(dx/dy)`:

        .. math:: \frac{d}{dx}\ln x = \frac{1}{e^{y}} = \frac{1}{x}

        Interpretation
        --------------
        The :math:`1/x` derivative grows without bound as :math:`x\to 0^{+}`.
        In cross-entropy this is a feature, not a bug: a model that assigns
        probability ~0 to the correct class receives an enormous corrective
        gradient. It is also why naive ``log(softmax(x))`` is dangerous -- if
        the probability underflows to exactly 0, you get ``-inf``. Phase 9
        computes log-softmax directly and never forms the probability first.

        Domain
        ------
        Requires ``self.data > 0``.

        Tiny example
        ------------
        ``x=2, out=0.693``. Seed ``out.grad=1`` -> ``x.grad = 0.5``.
        """
        if self.data <= 0.0:
            raise ValueError(
                f"log() requires a positive argument, got {self.data}"
            )
        out = Value(math.log(self.data), (self,), "log")
        recip = 1.0 / self.data  # captured before the closure runs

        def _backward() -> None:
            self.grad += recip * out.grad  # ∂out/∂self = 1/self

        out._backward = _backward
        return out

    # ==================================================================
    # ACTIVATION FUNCTIONS
    #
    # These are the nonlinearities that make a neural network more than a
    # linear map. Phase 8 covers their behaviour; the derivatives are here
    # because they are elementary operations of the engine.
    # ==================================================================

    def tanh(self) -> "Value":
        r"""Hyperbolic tangent.  ``out = tanh(self)``

        Forward
        -------
        .. math::
            \tanh x = \frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}
                    = \frac{e^{2x}-1}{e^{2x}+1}

        squashing :math:`\mathbb{R}` into :math:`(-1, 1)`, with
        :math:`\tanh 0 = 0`.

        Derivation
        ----------
        Apply the quotient rule to :math:`\sinh/\cosh`, using
        :math:`\sinh' = \cosh` and :math:`\cosh' = \sinh`:

        .. math::
            \frac{d}{dx}\tanh x
              = \frac{\cosh^{2}x - \sinh^{2}x}{\cosh^{2}x}
              = 1 - \tanh^{2}x
              = 1 - \text{out}^{2}

        (using the identity :math:`\cosh^{2}-\sinh^{2}=1`).

        Interpretation
        --------------
        The derivative is at most 1 (at :math:`x=0`) and decays towards 0 as
        :math:`|x|` grows. That decay is the **vanishing gradient problem**: a
        saturated ``tanh`` unit multiplies the incoming gradient by nearly
        zero, and stacking ten such layers can shrink a gradient by
        :math:`10^{-10}`. It is the main reason ReLU displaced ``tanh`` in deep
        networks -- see Phase 8 and the Phase 22 experiments.

        We compute the forward with ``math.tanh``, which is implemented
        stably (the naive :math:`e^{x}` formula overflows for large ``x``).

        Tiny example
        ------------
        ``x=0, out=0``. Seed ``out.grad=1`` -> ``x.grad = 1 - 0² = 1``.
        ``x=2, out=0.9640`` -> ``x.grad = 1 - 0.9640² = 0.0707``. Nearly flat.
        """
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1.0 - t * t) * out.grad  # ∂out/∂self = 1 - tanh²

        out._backward = _backward
        return out

    def sigmoid(self) -> "Value":
        r"""Logistic sigmoid.  ``out = 1 / (1 + e^{-self})``

        Forward
        -------
        .. math:: \sigma(x) = \frac{1}{1+e^{-x}}

        squashing :math:`\mathbb{R}` into :math:`(0,1)`, with
        :math:`\sigma(0)=1/2` -- which is why it reads as a probability.

        Derivation
        ----------
        Write :math:`\sigma = (1+e^{-x})^{-1}` and use the chain rule:

        .. math::
            \sigma'(x) = -(1+e^{-x})^{-2}\cdot(-e^{-x})
                       = \frac{e^{-x}}{(1+e^{-x})^{2}}

        Now the classic trick -- split the fraction:

        .. math::
            = \frac{1}{1+e^{-x}}\cdot\frac{e^{-x}}{1+e^{-x}}
            = \sigma(x)\bigl(1-\sigma(x)\bigr)

        because :math:`\frac{e^{-x}}{1+e^{-x}} = \frac{(1+e^{-x})-1}{1+e^{-x}}
        = 1-\sigma(x)`. So the derivative is expressible purely in terms of the
        output: :math:`\text{out}(1-\text{out})`.

        Interpretation
        --------------
        The maximum derivative is :math:`\sigma'(0)=0.25` -- so sigmoid
        attenuates gradient by at least 4x *even at its best*, and far more
        once saturated. Stacked sigmoids vanish gradients even faster than
        ``tanh``, whose peak derivative is 1. Sigmoid also isn't zero-centred
        (its output is always positive), which biases the gradients of all
        downstream weights in the same direction and slows convergence. Both
        facts are why it survives mainly as an *output* activation for binary
        classification, not as a hidden activation.

        Numerics
        --------
        The naive formula computes ``exp(-x)``, which overflows for very
        negative ``x``. We branch on the sign so the exponent is always
        :math:`\le 0`:

        * :math:`x \ge 0`:  :math:`\sigma = 1/(1+e^{-x})`
        * :math:`x < 0`:    :math:`\sigma = e^{x}/(1+e^{x})`

        Both are algebraically identical (multiply the second by
        :math:`e^{-x}/e^{-x}`) but only this pairing never overflows.

        Tiny example
        ------------
        ``x=0, out=0.5``. Seed ``out.grad=1`` -> ``x.grad = 0.5*0.5 = 0.25``.
        """
        x = self.data
        if x >= 0.0:
            s = 1.0 / (1.0 + math.exp(-x))
        else:
            e = math.exp(x)
            s = e / (1.0 + e)

        out = Value(s, (self,), "sigmoid")

        def _backward() -> None:
            self.grad += s * (1.0 - s) * out.grad  # ∂out/∂self = σ(1-σ)

        out._backward = _backward
        return out

    def relu(self) -> "Value":
        r"""Rectified linear unit.  ``out = max(0, self)``

        Forward
        -------
        .. math::
            \text{ReLU}(x) = \begin{cases} x & x > 0 \\ 0 & x \le 0 \end{cases}

        Derivation
        ----------
        Piecewise-linear, so differentiate each piece:

        .. math::
            \frac{d}{dx}\text{ReLU}(x)
              = \begin{cases} 1 & x > 0 \\ 0 & x < 0 \end{cases}

        Interpretation
        --------------
        ReLU is a **gate**: it either passes the gradient through untouched
        (x > 0) or blocks it entirely (x < 0). Because the "on" derivative is
        exactly 1 -- not 0.25, not something that decays -- gradients survive
        arbitrarily deep stacks of ReLUs. That single property is most of why
        deep networks became trainable.

        The cost is the **dying ReLU** problem: a unit whose pre-activation is
        negative for every input receives zero gradient forever and can never
        recover. Leaky ReLU exists to patch this.

        The kink at zero
        ----------------
        At exactly :math:`x=0` there is no derivative: the left slope is 0, the
        right slope is 1, and no single tangent line exists. We *choose* 0.
        That choice is a legitimate **subgradient** (any value in
        :math:`[0,1]` is), and it is what PyTorch chooses too. It is harmless
        in practice because landing on exactly ``0.0`` in floating point is
        vanishingly unlikely, and gradient descent tolerates disagreement on a
        measure-zero set.

        Note this is the one operation whose backward rule inspects the
        *output* to recover a *condition* (``out.data > 0`` is true exactly
        when ``self.data > 0``), rather than to reuse a value.

        Tiny example
        ------------
        ``x=-2, out=0``. Seed ``out.grad=1`` -> ``x.grad = 0``. Blocked.
        ``x= 3, out=3``. Seed ``out.grad=1`` -> ``x.grad = 1``. Passed through.
        """
        out = Value(self.data if self.data > 0.0 else 0.0, (self,), "relu")

        def _backward() -> None:
            self.grad += (1.0 if out.data > 0.0 else 0.0) * out.grad

        out._backward = _backward
        return out

    # ==================================================================
    # BACKPROPAGATION
    # ==================================================================

    def backward(self) -> None:
        r"""Run reverse-mode autodiff from this node.

        Fills in ``.grad = ∂self/∂node`` on every node reachable from ``self``.

        The algorithm, in five lines of real work
        ----------------------------------------
        1. **Topologically sort** the graph reachable from ``self`` (parents
           before children).
        2. **Seed** ``self.grad = 1``, because ``∂self/∂self = 1`` -- the output
           is perfectly sensitive to itself.
        3. **Reverse** the ordering, so every node comes after all of its
           consumers.
        4. **Walk** it, calling each node's ``_backward()``.
        5. Each ``_backward()`` **accumulates** ``local_derivative * self.grad``
           into its parents.

        Steps 1 and 3 guarantee a node's gradient is *complete* before it is
        used; step 5 guarantees it *collected everything*. Both are required,
        and together they are correct on any DAG. See ``docs/01-*.md`` §10-11.

        Accumulation semantics (important)
        ----------------------------------
        Leaf gradients **accumulate**, exactly like ``torch.Tensor.backward()``.
        Two consequences:

        * Calling ``backward()`` twice sums two gradients -- which is the
          mechanism behind gradient accumulation over micro-batches.
        * A training loop **must** call ``zero_grad()`` each step, or gradients
          from every previous step keep contributing. This is the classic
          PyTorch bug, and we reproduce it on purpose rather than hide it.

        Why intermediates are cleared but leaves are not
        -----------------------------------------------
        ``.grad`` is quietly doing *two* jobs: it stores the answer
        :math:`\partial L/\partial\text{node}`, and during the sweep it is also
        the **carrier** that ferries the gradient from a node down to its
        parents. For leaves those jobs agree. For intermediates they conflict.

        Consider ``u = x*x; L = u + 0`` at ``x = 3``, and call ``backward()``
        twice without zeroing. The first sweep leaves ``u.grad = 1`` and
        ``x.grad = 6``. The second sweep seeds ``L.grad = 1``, and the ``+``
        rule accumulates into ``u``, giving ``u.grad = 2`` -- but that ``2`` is
        pass-one residue, not a real derivative. The ``*`` rule then reads it
        as the incoming gradient and pushes ``2·(2·3) = 12`` into ``x``, for a
        total of ``18`` instead of the correct ``12``.

        The gradient *flowing along an edge* belongs to one sweep and must not
        outlive it; only the gradient *landing on a leaf* is a running total.
        So we reset every non-leaf reachable node before sweeping. PyTorch
        reaches the same place by a different route: it never populates
        ``.grad`` on non-leaf tensors at all, holding edge gradients in a
        temporary buffer that dies with the pass. That is why ``.grad`` is
        ``None`` on intermediates there unless you ask for ``retain_grad()``.
        We keep intermediates visible -- they are the whole point of a teaching
        engine -- and pay for it with this one explicit reset.

        Cost
        ----
        O(V + E): one pass to sort, one to propagate. That is the *cheap
        gradient principle* -- the gradient of a function with n parameters
        costs a constant multiple of evaluating the function, independent of n.

        Note there is no "is this a scalar?" check to make: every ``Value``
        holds exactly one number, so ``∂self/∂self = 1`` is always well
        defined. The tensor engine (Phase 18) has to check, because
        ``∂self/∂self`` for a non-scalar output is a Jacobian, not a number.
        """
        order = topological_sort(self)

        # Intermediates carry *this* sweep's edge gradients and nothing else,
        # so clear them. Leaves are the running totals, so leave them alone.
        for node in order:
            if node._prev:
                node.reset_grad()

        # ∂self/∂self = 1. Every other gradient is derived from this seed.
        # `+=` rather than `=`: if `self` is a leaf it was deliberately not
        # cleared above, and clobbering it would break accumulation. If `self`
        # is an intermediate it was just zeroed, so `+=` is exactly `=`.
        self.grad += 1.0

        for node in reversed(order):
            node._backward()

    def reset_grad(self) -> None:
        """Zero this node's own gradient.

        The one-node counterpart to ``zero_grad()``. Exists so ``Value`` and
        ``Tensor`` present the same interface to ``Module`` and the optimisers
        -- a ``Tensor`` must reset to a zero *array* of the right shape, which
        a bare ``= 0.0`` would not do.
        """
        self.grad = 0.0

    def zero_grad(self) -> None:
        """Reset ``.grad`` to 0 on every node reachable from this one.

        Call between training steps, or before a second ``backward()`` whose
        result should not include the first. Note this walks the *graph*, so it
        clears intermediates too; the optimiser's ``zero_grad()`` (Phase 10)
        only needs to clear the parameters, which is cheaper.
        """
        for node in topological_sort(self):
            node.grad = 0.0

    # ==================================================================
    # CONVENIENCES
    # ==================================================================

    def __lt__(self, other: ValueLike) -> bool:
        """Compare by ``data``. **Not differentiable** -- returns a plain bool.

        Provided so ``max(logits, key=...)`` and argmax-style code reads
        naturally. Deliberately does *not* define ``__eq__``, which would break
        hashing and stop ``Value`` objects being usable in sets and dicts (the
        graph traversal relies on identity, not value equality).
        """
        return self.data < (other.data if isinstance(other, Value) else other)

    def __gt__(self, other: ValueLike) -> bool:
        return self.data > (other.data if isinstance(other, Value) else other)

    def __le__(self, other: ValueLike) -> bool:
        return self.data <= (other.data if isinstance(other, Value) else other)

    def __ge__(self, other: ValueLike) -> bool:
        return self.data >= (other.data if isinstance(other, Value) else other)

    def __float__(self) -> float:
        return self.data

    def item(self) -> float:
        """Return the underlying Python float (mirrors ``torch.Tensor.item``)."""
        return self.data

    @staticmethod
    def sum(values: Iterable["Value"]) -> "Value":
        """Sum an iterable of ``Value`` objects.

        Builds a left-leaning chain of ``+`` nodes, so the graph is ``n`` deep
        for ``n`` terms. Fine for our sizes; the ``Tensor`` engine (Phase 18)
        replaces this with a single reduction node, which is one of the reasons
        it is dramatically faster.
        """
        it = iter(values)
        try:
            total = next(it)
        except StopIteration:
            return Value(0.0)
        for v in it:
            total = total + v
        return total
