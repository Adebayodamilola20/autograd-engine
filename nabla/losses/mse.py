r"""Phase 9 -- squared-error losses, and where they are the wrong tool.

Mean squared error
------------------
.. math:: \mathcal{L} = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^{2}

with the beautifully simple gradient

.. math:: \frac{\partial \mathcal{L}}{\partial \hat{y}_i} = \frac{2}{N}(\hat{y}_i - y_i)

**The gradient is proportional to the error.** Predict 0.1 too high and you get
a small push down; predict 100 too high and you get a large one. That
proportionality is what makes gradient descent on MSE well behaved, and it is a
good first loss to reason about because there is nothing hidden in it.

Why squared, and not absolute
-----------------------------
Three reasons, all real:

1. **Differentiable everywhere.** :math:`|x|` has no derivative at 0.
2. **It punishes outliers harder.** Being wrong by 10 costs 100, not 10. This
   is a feature when large errors are genuinely worse, and a bug when the data
   has outliers you would rather ignore -- which is when you switch to MAE or
   Huber.
3. **It is the maximum-likelihood estimator under Gaussian noise.** If
   :math:`y = f(x) + \varepsilon` with :math:`\varepsilon \sim
   \mathcal{N}(0,\sigma^{2})`, maximising the likelihood is *exactly*
   minimising the sum of squares. MSE is not an arbitrary choice; it encodes an
   assumption about your noise.

Why MSE is wrong for classification
-----------------------------------
This matters enough to state carefully, because it is a mistake people make.

Suppose you attach a sigmoid to the output and train with MSE on a 0/1 target.
The chain rule gives

.. math::
    \frac{\partial\mathcal{L}}{\partial z}
    = 2(\sigma(z) - y)\cdot\sigma'(z)

Look at what happens when the model is **confidently wrong**: :math:`y=1` but
:math:`z=-10`, so :math:`\sigma(z)\approx 0.000045`. The error term
:math:`(\sigma(z)-y) \approx -1` is as large as it can be -- but
:math:`\sigma'(z) \approx 0.000045`, so the product is about
:math:`-9\times10^{-5}`. **The gradient nearly vanishes precisely when the
model is most wrong.** Learning stalls.

Cross-entropy fixes this exactly: its gradient is :math:`\sigma(z) - y`, with
no :math:`\sigma'` factor, because the :math:`1/p` from the logarithm cancels
it (see ``cross_entropy.py``). Same wrong prediction, gradient of magnitude
~1, error corrected immediately.

So: **MSE for regression, cross-entropy for classification.** Not convention --
a property of the gradients. ``experiments/`` measures the difference on real
training runs.
"""

from __future__ import annotations

from typing import Sequence

from ..core.value import Value

__all__ = ["mse_loss", "mse", "sse_loss", "mae_loss", "huber_loss"]


def _as_value(x: Value | float) -> Value:
    return x if isinstance(x, Value) else Value(x)


def mse_loss(
    predictions: Sequence[Value] | Value,
    targets: Sequence[float] | float,
) -> Value:
    r"""Mean squared error, :math:`\frac{1}{N}\sum(\hat{y}-y)^{2}`.

    Accepts either a single prediction/target pair or matching sequences.

    Averaging rather than summing keeps the gradient scale independent of how
    many outputs there are, so a learning rate tuned on a 1-output model still
    works on a 10-output one.

    Examples
    --------
    >>> p = [Value(2.0), Value(3.0)]
    >>> loss = mse_loss(p, [1.0, 5.0])
    >>> loss.data                      # (1 + 4) / 2
    2.5
    >>> loss.backward()
    >>> p[0].grad                      # 2(2-1)/2
    1.0
    """
    if isinstance(predictions, Value):
        predictions = [predictions]
        targets = [targets]  # type: ignore[list-item]

    predictions = list(predictions)
    targets = list(targets)  # type: ignore[arg-type]
    if len(predictions) != len(targets):
        raise ValueError(
            f"got {len(predictions)} predictions for {len(targets)} targets"
        )
    if not predictions:
        raise ValueError("empty prediction list")

    total = Value(0.0)
    for pred, target in zip(predictions, targets):
        total = total + (_as_value(pred) - target) ** 2
    return total / len(predictions)


#: Short alias, because ``mse(preds, targets)`` reads well at a call site.
mse = mse_loss


def sse_loss(
    predictions: Sequence[Value], targets: Sequence[float]
) -> Value:
    r"""Sum of squared errors -- MSE without the :math:`1/N`.

    Included because the difference is instructive: the gradients are
    identical up to the constant factor :math:`N`, so SSE with learning rate
    :math:`\eta/N` takes exactly the same steps as MSE with :math:`\eta`. A
    reminder that a loss and a learning rate are only meaningful together.
    """
    total = Value(0.0)
    for pred, target in zip(predictions, targets):
        total = total + (_as_value(pred) - target) ** 2
    return total


def mae_loss(predictions: Sequence[Value], targets: Sequence[float]) -> Value:
    r"""Mean absolute error, :math:`\frac{1}{N}\sum|\hat{y}-y|`.

    Robust to outliers -- the gradient is :math:`\pm 1/N` regardless of how
    large the error is, so one wild data point cannot dominate the update.
    The flip side is that the gradient does not shrink as you approach the
    optimum, so training tends to oscillate near the minimum unless the
    learning rate is decayed.

    Built from ``relu``, since :math:`|x| = \text{relu}(x) + \text{relu}(-x)`,
    which means we write no backward rule for it. Not differentiable at
    :math:`x=0`, where the composition yields the subgradient 0.
    """
    total = Value(0.0)
    for pred, target in zip(predictions, targets):
        diff = _as_value(pred) - target
        total = total + diff.relu() + (-diff).relu()
    return total / len(list(predictions))


def huber_loss(
    predictions: Sequence[Value], targets: Sequence[float], *, delta: float = 1.0
) -> Value:
    r"""Huber loss -- quadratic near zero, linear in the tails.

    .. math::
        \mathcal{L}_\delta(e) = \begin{cases}
            \tfrac{1}{2}e^{2} & |e| \le \delta\\
            \delta\left(|e| - \tfrac{1}{2}\delta\right) & \text{otherwise}
        \end{cases}

    The best of both: MSE's shrinking gradient near the optimum (so it settles)
    and MAE's bounded gradient far from it (so outliers cannot dominate). The
    two pieces are constructed to agree in both value and slope at
    :math:`|e|=\delta`, which is what makes it differentiable there.

    Branching on ``e.data`` is safe for the same reason as in ``softplus``:
    define-by-run records the branch that executed, and the two pieces meet
    smoothly, so the derivative is continuous across the boundary.
    """
    total = Value(0.0)
    n = 0
    for pred, target in zip(predictions, targets):
        e = _as_value(pred) - target
        if abs(e.data) <= delta:
            total = total + 0.5 * e**2
        else:
            abs_e = e if e.data > 0 else -e
            total = total + delta * (abs_e - 0.5 * delta)
        n += 1
    return total / n
