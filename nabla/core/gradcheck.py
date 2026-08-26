"""Phase 5 -- verifying gradients with finite differences.

Why bother, when the engine already computes exact gradients?
-------------------------------------------------------------
Because "exact" is only true if every backward rule was *derived correctly*. A
sign error in one derivative produces gradients that are exact solutions to the
wrong problem, and the forward pass will not complain. Unit tests against
hand-computed values help, but they are written by the same person who wrote the
derivative -- if the derivation was wrong, the test is wrong in the same way.

Finite differences break that circularity. They compute the derivative from its
*definition as a limit*, using nothing but the forward pass:

.. math:: f'(x) \\approx \\frac{f(x+h) - f(x-h)}{2h}

They share **no code and no reasoning** with the analytic backward rules. When
the two agree to ten digits across dozens of operations and random inputs, the
derivations are right. This is the single most valuable testing tool in the
project, and it is why every phase from here on can be trusted.

Why the central difference, and not the forward difference
----------------------------------------------------------
Taylor-expand around :math:`x`:

.. math::
    f(x+h) &= f(x) + hf'(x) + \\tfrac{h^{2}}{2}f''(x) + \\tfrac{h^{3}}{6}f'''(x) + \\cdots \\\\
    f(x-h) &= f(x) - hf'(x) + \\tfrac{h^{2}}{2}f''(x) - \\tfrac{h^{3}}{6}f'''(x) + \\cdots

Subtracting cancels the :math:`f(x)` and :math:`f''` terms exactly:

.. math::
    \\frac{f(x+h)-f(x-h)}{2h} = f'(x) + \\frac{h^{2}}{6}f'''(x) + O(h^{4})

So the error is :math:`O(h^{2})`. The one-sided difference
:math:`(f(x+h)-f(x))/h` keeps the :math:`f''` term and is only :math:`O(h)` --
for :math:`h=10^{-5}` that is the difference between ~10 correct digits and ~5.
Two extra function evaluations for five extra digits is an excellent trade.

Choosing h
----------
Two errors pull in opposite directions:

* **truncation** :math:`\\sim h^{2}` -- wants :math:`h` small,
* **round-off** :math:`\\sim \\varepsilon/h` -- wants :math:`h` large, because
  :math:`f(x+h)` and :math:`f(x-h)` are nearly equal and subtracting them
  destroys significant digits (catastrophic cancellation).

Minimising :math:`h^{2} + \\varepsilon/h` gives
:math:`h \\approx \\varepsilon^{1/3} \\approx 6\\times10^{-6}`, so the default is
``eps=1e-5``. ``epsilon_sweep()`` in this module measures the resulting U-curve
directly -- run ``examples/gradient_check.py`` to see it.

This is exactly the trade-off reverse-mode autodiff does not have, and it is the
reason we test with finite differences but never train with them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .value import Value

__all__ = [
    "GradCheckResult",
    "analytic_gradient",
    "numerical_gradient",
    "relative_error",
    "check_gradients",
    "check_parameter_gradients",
    "epsilon_sweep",
]

# Below this magnitude, a relative comparison is meaningless -- comparing 1e-18
# with 3e-18 gives a "relative error" of 0.67 for two numbers that are both
# indistinguishable from zero. Fall back to an absolute difference instead.
_SCALE_FLOOR = 1e-8

ScalarFn = Callable[..., Value]


# ======================================================================
# The two gradients
# ======================================================================


def analytic_gradient(fn: ScalarFn, inputs: Sequence[float]) -> list[float]:
    """Gradient of ``fn`` at ``inputs``, via our reverse-mode engine.

    Parameters
    ----------
    fn
        Takes one ``Value`` per input and returns a single ``Value``.
    inputs
        The point at which to differentiate.

    Cost: **one** forward pass plus one backward pass, regardless of how many
    inputs there are. Compare with ``numerical_gradient`` below.
    """
    values = [Value(x) for x in inputs]
    out = fn(*values)
    if not isinstance(out, Value):
        raise TypeError(
            f"fn must return a Value, got {type(out).__name__}. "
            "Gradient checking is defined for scalar-valued functions."
        )
    out.backward()
    return [v.grad for v in values]


def numerical_gradient(
    fn: ScalarFn,
    inputs: Sequence[float],
    *,
    eps: float = 1e-5,
    order: int = 2,
) -> list[float]:
    """Gradient of ``fn`` at ``inputs``, via finite differences.

    Uses **only** the forward pass -- it never touches ``.grad``, ``.backward()``
    or any backward rule. That independence is the entire point.

    Parameters
    ----------
    eps
        Step size :math:`h`. Default ``1e-5``, near the theoretical optimum
        :math:`\\varepsilon^{1/3}` for order 2.
    order
        ``2`` for the 3-point central difference (error :math:`O(h^{2})`,
        2 evaluations per input) or ``4`` for the 5-point stencil

        .. math::
            f'(x) \\approx \\frac{f(x-2h) - 8f(x-h) + 8f(x+h) - f(x+2h)}{12h}

        (error :math:`O(h^{4})`, 4 evaluations per input). Order 4 buys another
        two or three digits and is used when a check is borderline.

    Cost
    ----
    ``order * len(inputs)`` forward passes. For an MLP with 109,386 parameters
    that is over 200,000 forward passes for **one** gradient -- which is why
    this is a testing tool and never a training tool.
    """
    if order not in (2, 4):
        raise ValueError(f"order must be 2 or 4, got {order}")

    def evaluate(point: Sequence[float]) -> float:
        return fn(*[Value(x) for x in point]).data

    grads: list[float] = []
    base = list(inputs)

    for i in range(len(base)):
        original = base[i]

        if order == 2:
            base[i] = original + eps
            f_plus = evaluate(base)
            base[i] = original - eps
            f_minus = evaluate(base)
            grad = (f_plus - f_minus) / (2.0 * eps)
        else:
            base[i] = original + 2 * eps
            f_pp = evaluate(base)
            base[i] = original + eps
            f_p = evaluate(base)
            base[i] = original - eps
            f_m = evaluate(base)
            base[i] = original - 2 * eps
            f_mm = evaluate(base)
            grad = (f_mm - 8.0 * f_m + 8.0 * f_p - f_pp) / (12.0 * eps)

        base[i] = original  # restore before moving on
        grads.append(grad)

    return grads


def relative_error(analytic: float, numerical: float) -> float:
    """Scale-invariant disagreement between two gradient estimates.

    .. math::
        \\text{err} = \\frac{|a - n|}{\\max(|a|, |n|)}

    Relative rather than absolute, because a gradient of magnitude
    :math:`10^{6}` and one of :math:`10^{-6}` deserve the same standard. When
    both values are below ``_SCALE_FLOOR`` the ratio is noise, so the absolute
    difference is reported instead.

    Rules of thumb for interpreting the result (with ``eps=1e-5``, order 2):

    ==================  ==========================================
    ``< 1e-7``          the derivative is right
    ``1e-7 .. 1e-5``    fine for a deep or badly-scaled expression
    ``1e-5 .. 1e-2``    suspicious -- check the point is smooth
    ``> 1e-2``          a real bug
    ==================  ==========================================
    """
    diff = abs(analytic - numerical)
    scale = max(abs(analytic), abs(numerical))
    if scale < _SCALE_FLOOR:
        return diff
    return diff / scale


# ======================================================================
# The comparison harness
# ======================================================================


@dataclass
class GradCheckResult:
    """Outcome of comparing analytic gradients against finite differences."""

    passed: bool
    analytic: list[float]
    numerical: list[float]
    errors: list[float]
    names: list[str]
    tol: float
    eps: float
    order: int
    label: str = ""
    failures: list[int] = field(default_factory=list)

    @property
    def max_error(self) -> float:
        return max(self.errors) if self.errors else 0.0

    def __bool__(self) -> bool:
        return self.passed

    def __str__(self) -> str:
        head = self.label or "gradient check"
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"{head}: {status}   (max rel err {self.max_error:.3e}, "
            f"tol {self.tol:.0e}, h={self.eps:.0e}, order {self.order})",
            f"  {'input':<12} {'analytic':>16} {'numerical':>16} {'rel err':>11}",
            "  " + "-" * 58,
        ]
        for i, name in enumerate(self.names):
            mark = "  <-- FAIL" if i in self.failures else ""
            lines.append(
                f"  {name:<12} {self.analytic[i]:>16.10f} "
                f"{self.numerical[i]:>16.10f} {self.errors[i]:>11.3e}{mark}"
            )
        return "\n".join(lines)


def check_gradients(
    fn: ScalarFn,
    inputs: Sequence[float],
    *,
    eps: float = 1e-5,
    tol: float = 1e-6,
    order: int = 2,
    names: Sequence[str] | None = None,
    label: str = "",
    raise_on_failure: bool = False,
) -> GradCheckResult:
    """Compare our engine's gradients with finite differences.

    Examples
    --------
    >>> result = check_gradients(lambda x, y: (x * y).tanh(), [2.0, 3.0])
    >>> bool(result)
    True

    >>> print(check_gradients(lambda x: x ** 3, [2.0], names=['x']))  # doctest: +SKIP
    gradient check: PASS   (max rel err 1.9e-11, tol 1e-06, h=1e-05, order 2)
      input            analytic        numerical     rel err
      ----------------------------------------------------------
      x            12.0000000000    12.0000000002   1.855e-11

    Caveats
    -------
    Finite differences assume ``fn`` is smooth at the evaluation point. They
    are **expected to fail** at a kink -- ``relu`` at exactly 0, ``abs`` at 0 --
    because the two-sided difference straddles the discontinuity in the
    derivative and reports the average of the two slopes. That is a property of
    the function, not a bug in the engine. See
    ``tests/test_gradients.py::TestKnownLimitations``.

    Parameters
    ----------
    raise_on_failure
        Raise ``AssertionError`` with the full table instead of returning a
        failed result. Convenient inside library code that must not proceed on
        a bad gradient.
    """
    analytic = analytic_gradient(fn, inputs)
    numerical = numerical_gradient(fn, inputs, eps=eps, order=order)
    errors = [relative_error(a, n) for a, n in zip(analytic, numerical)]
    failures = [i for i, e in enumerate(errors) if not (e <= tol)]

    result = GradCheckResult(
        passed=not failures,
        analytic=analytic,
        numerical=numerical,
        errors=errors,
        names=list(names) if names else [f"x[{i}]" for i in range(len(inputs))],
        tol=tol,
        eps=eps,
        order=order,
        label=label,
        failures=failures,
    )

    if raise_on_failure and not result.passed:
        raise AssertionError(str(result))
    return result


def check_parameter_gradients(
    loss_fn: Callable[[], Value],
    params: Sequence[Value],
    *,
    eps: float = 1e-5,
    tol: float = 1e-5,
    order: int = 2,
    names: Sequence[str] | None = None,
    label: str = "",
    raise_on_failure: bool = False,
) -> GradCheckResult:
    """Gradient-check an existing set of ``Value`` parameters in place.

    ``check_gradients`` builds fresh inputs from floats, which is ideal for
    testing a single operation. A neural network is different: the parameters
    already exist as ``Value`` objects held by the model, and the loss closes
    over them. This version perturbs ``param.data`` directly and re-runs
    ``loss_fn()``, which rebuilds the graph from the perturbed values.

    This is what lets us verify gradients through an *entire MLP* -- every
    layer, activation and loss composed together -- rather than one operation
    at a time. Phase 7 uses it as its acceptance test.

    Parameters
    ----------
    loss_fn
        Zero-argument callable returning a scalar ``Value``. Must recompute the
        forward pass on each call (i.e. read ``param.data`` fresh).
    params
        The parameters to check. Their ``.data`` is restored exactly after
        each perturbation.

    Notes
    -----
    ``tol`` defaults to ``1e-5`` here, looser than for single operations: a
    deep composition accumulates round-off, and its third derivative (which
    sets the truncation error) can be large.
    """
    for p in params:
        p.grad = 0.0
    loss = loss_fn()
    loss.backward()
    analytic = [p.grad for p in params]

    numerical: list[float] = []
    for p in params:
        original = p.data
        try:
            if order == 2:
                p.data = original + eps
                f_plus = loss_fn().data
                p.data = original - eps
                f_minus = loss_fn().data
                numerical.append((f_plus - f_minus) / (2.0 * eps))
            else:
                p.data = original + 2 * eps
                f_pp = loss_fn().data
                p.data = original + eps
                f_p = loss_fn().data
                p.data = original - eps
                f_m = loss_fn().data
                p.data = original - 2 * eps
                f_mm = loss_fn().data
                numerical.append(
                    (f_mm - 8.0 * f_m + 8.0 * f_p - f_pp) / (12.0 * eps)
                )
        finally:
            p.data = original  # restore even if loss_fn raises

    errors = [relative_error(a, n) for a, n in zip(analytic, numerical)]
    failures = [i for i, e in enumerate(errors) if not (e <= tol)]

    result = GradCheckResult(
        passed=not failures,
        analytic=analytic,
        numerical=numerical,
        errors=errors,
        names=list(names) if names else [f"p[{i}]" for i in range(len(params))],
        tol=tol,
        eps=eps,
        order=order,
        label=label,
        failures=failures,
    )

    if raise_on_failure and not result.passed:
        raise AssertionError(str(result))
    return result


# ======================================================================
# Diagnostics
# ======================================================================


def epsilon_sweep(
    fn: ScalarFn,
    inputs: Sequence[float],
    *,
    index: int = 0,
    order: int = 2,
    exponents: Sequence[int] = range(-1, -15, -1),
) -> list[tuple[float, float, float]]:
    """Measure the accuracy-vs-step-size U-curve.

    Returns ``(h, numerical_estimate, relative_error)`` for each
    :math:`h = 10^{e}`, comparing against the analytic gradient of input
    ``index``.

    This makes the central claim of section 3 of the theory doc *measurable*
    rather than asserted: error falls as :math:`h^{2}` while truncation
    dominates, bottoms out around :math:`h \\approx 10^{-5}`--:math:`10^{-6}`,
    then climbs again as catastrophic cancellation takes over. There is no
    value of :math:`h` that recovers full double precision.

    Used by ``examples/gradient_check.py`` and quoted in the docs.
    """
    truth = analytic_gradient(fn, inputs)[index]
    rows: list[tuple[float, float, float]] = []
    for e in exponents:
        h = 10.0**e
        estimate = numerical_gradient(fn, inputs, eps=h, order=order)[index]
        rows.append((h, estimate, relative_error(truth, estimate)))
    return rows
