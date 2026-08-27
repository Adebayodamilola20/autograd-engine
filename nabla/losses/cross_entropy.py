r"""Phase 9 -- softmax and cross-entropy, derived and made numerically stable.

This file is the mathematical centre of the classification pipeline, so it is
worth reading end to end. Four ideas, in order.

1. Logits
---------
The network's last layer emits :math:`K` unconstrained real numbers
:math:`z_1,\ldots,z_K`, one per class. They are called **logits**. They are not
probabilities: they can be negative, they need not sum to anything. They are
*scores* -- higher means "more evidence for this class".

Why leave them unconstrained? Because forcing the network to output valid
probabilities directly (with a sigmoid per class, say) makes the optimisation
harder and the gradients worse. Better to let the network say whatever it
likes and convert afterwards.

2. Softmax
----------
Softmax turns :math:`K` scores into :math:`K` probabilities:

.. math::
    p_k = \text{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}

Every :math:`p_k > 0` (exponentials are positive) and :math:`\sum_k p_k = 1`
(the denominator is exactly that sum). It is called *soft*max because it is a
smooth relaxation of "take the max": a large :math:`z_k` dominates the sum and
drives :math:`p_k \to 1`, but smoothly, so it has a derivative -- and a hard
``argmax`` does not, which would make training impossible.

Two properties worth internalising:

* **Shift invariance.** :math:`\text{softmax}(\mathbf{z} + c) =
  \text{softmax}(\mathbf{z})` for any constant :math:`c`, because
  :math:`e^{z_k+c} = e^{c}e^{z_k}` and the :math:`e^{c}` cancels top and
  bottom. Only *differences* between logits matter. We exploit this for
  stability below.
* **Scale sensitivity.** Multiplying all logits by 10 makes the distribution
  far peakier. This is where the "temperature" knob in language models comes
  from.

3. Cross-entropy
----------------
Given a predicted distribution :math:`\mathbf{p}` and a true distribution
:math:`\mathbf{q}`, cross-entropy measures the cost of using :math:`\mathbf{p}`
to encode outcomes drawn from :math:`\mathbf{q}`:

.. math:: H(\mathbf{q},\mathbf{p}) = -\sum_k q_k \ln p_k

For classification the truth is one-hot -- the label is class :math:`y` with
certainty -- so every term vanishes except one:

.. math:: \mathcal{L} = -\ln p_y

That is the entire loss: **the negative log probability the model assigned to
the correct class.**

* :math:`p_y = 1` gives :math:`\mathcal{L}=0`. Perfect, no loss.
* :math:`p_y = 0.5` gives :math:`\mathcal{L}=0.693`.
* :math:`p_y \to 0` gives :math:`\mathcal{L} \to \infty`. Confident and wrong
  is punished without bound.

That last property is why cross-entropy beats squared error for
classification. MSE on probabilities caps the penalty for being confidently
wrong, and -- worse -- its gradient *vanishes* exactly there, because it gets
multiplied by :math:`\sigma'` which is near zero when saturated. Cross-entropy
composed with softmax cancels that factor entirely. See below.

4. The gradient, and why it is beautiful
----------------------------------------
Differentiate :math:`\mathcal{L} = -\ln p_y` with respect to the logits.

First, softmax's own Jacobian. For :math:`k = i`:

.. math::
    \frac{\partial p_i}{\partial z_i}
    = \frac{e^{z_i}S - e^{z_i}e^{z_i}}{S^{2}} = p_i(1-p_i),
    \qquad S = \sum_j e^{z_j}

and for :math:`k \ne i`:

.. math::
    \frac{\partial p_k}{\partial z_i} = \frac{0 - e^{z_k}e^{z_i}}{S^{2}} = -p_kp_i

Now the chain rule, with :math:`\mathcal{L} = -\ln p_y` so
:math:`\partial\mathcal{L}/\partial p_y = -1/p_y`:

.. math::
    \frac{\partial \mathcal{L}}{\partial z_i}
    = -\frac{1}{p_y}\frac{\partial p_y}{\partial z_i}
    = \begin{cases}
        -\frac{1}{p_y}\,p_y(1-p_y) = p_y - 1 & i = y\\[4pt]
        -\frac{1}{p_y}\,(-p_yp_i) = p_i       & i \ne y
      \end{cases}

Both cases collapse into one line:

.. math:: \boxed{\;\frac{\partial \mathcal{L}}{\partial z_i} = p_i - y_i\;}

where :math:`y_i` is the one-hot target. **The gradient is just
prediction minus truth.** No :math:`\sigma'` factor anywhere -- the
:math:`1/p_y` from the log cancels the :math:`p_y` from the softmax exactly.
That cancellation is why softmax and cross-entropy are always paired: apart,
each has a saturating gradient; together, they do not. A confidently wrong
prediction produces a gradient of magnitude ~1 and gets fixed fast.

Our engine derives this automatically by composing ``exp``, ``log``, ``+`` and
``/``. ``tests/test_losses.py`` confirms the analytic result matches, which is
a satisfying check on both the maths and the engine.

5. Numerical stability -- why the naive formula must not be used
----------------------------------------------------------------
:math:`e^{z}` overflows above :math:`z \approx 709`, and logits routinely reach
the hundreds mid-training. Worse, at the other end :math:`e^{-800}` underflows
to exactly ``0.0``, and then :math:`\ln 0 = -\infty` poisons the whole graph
with ``nan`` on the first backward pass.

Both are fixed by the **log-sum-exp trick**. Using shift invariance with
:math:`m = \max_j z_j`:

.. math::
    \ln \sum_j e^{z_j}
    = \ln\left(e^{m}\sum_j e^{z_j - m}\right)
    = m + \ln \sum_j e^{z_j - m}

Every exponent :math:`z_j - m` is now :math:`\le 0`, so :math:`e^{z_j-m} \in
(0, 1]` -- no overflow possible. And the sum contains the term :math:`j = \arg\max`
which equals exactly 1, so the sum is :math:`\ge 1` and its log is :math:`\ge 0` --
no underflow to zero either.

Then log-softmax is computed **directly**, never via ``log(softmax(z))``:

.. math:: \ln p_k = z_k - m - \ln\sum_j e^{z_j-m}

so a probability of :math:`10^{-300}` yields a perfectly ordinary log-probability
of about :math:`-690` instead of ``-inf``.

This is precisely why PyTorch offers ``log_softmax`` and
``cross_entropy(logits, target)`` and warns you against applying softmax
yourself first. Now you know what it is protecting you from.
"""

from __future__ import annotations

from typing import Sequence

from ..core.value import Value

__all__ = [
    "log_sum_exp",
    "softmax",
    "log_softmax",
    "cross_entropy",
    "softmax_cross_entropy",
    "nll_loss",
    "binary_cross_entropy",
]


def log_sum_exp(logits: Sequence[Value]) -> Value:
    r"""Stable :math:`\ln\sum_j e^{z_j}`.

    Subtracts the maximum before exponentiating, so every exponent is
    :math:`\le 0`. Mathematically identical to the naive form; numerically it
    is the difference between working and returning ``inf``.

    The shift ``m`` is taken as a **constant**, read off ``.data`` rather than
    kept in the graph. That is legitimate: softmax is exactly shift-invariant,
    so :math:`\partial\mathcal{L}/\partial m = 0` and the term contributes no
    gradient. Treating it as a constant is not an approximation -- it just
    keeps the graph smaller. (``tests/test_losses.py`` verifies the gradients
    against finite differences, which would catch it if this reasoning were
    wrong.)
    """
    if not logits:
        raise ValueError("log_sum_exp requires at least one logit")

    m = max(v.data for v in logits)
    total = Value(0.0)
    for v in logits:
        total = total + (v - m).exp()
    return total.log() + m


def log_softmax(logits: Sequence[Value]) -> list[Value]:
    r"""Stable :math:`\ln p_k = z_k - \ln\sum_j e^{z_j}`.

    Computed directly, never as ``log(softmax(z))``. The intermediate
    probability can underflow to exactly zero, and ``log(0)`` is ``-inf``;
    the log-probability itself stays a perfectly ordinary finite number.
    """
    lse = log_sum_exp(logits)
    return [z - lse for z in logits]


def softmax(logits: Sequence[Value]) -> list[Value]:
    r"""Probabilities :math:`p_k = e^{z_k}/\sum_j e^{z_j}`.

    Computed as ``exp(log_softmax(z))`` so the stability work is not repeated.
    Use this when you actually want probabilities to display; use
    ``log_softmax`` inside a loss.
    """
    return [lp.exp() for lp in log_softmax(logits)]


def nll_loss(log_probs: Sequence[Value], target: int) -> Value:
    r"""Negative log-likelihood: :math:`-\ln p_y`, from log-probabilities.

    Split out from ``cross_entropy`` so the two halves are visible separately.
    ``cross_entropy(logits, y) == nll_loss(log_softmax(logits), y)``, which is
    exactly the relationship PyTorch's ``NLLLoss`` and ``CrossEntropyLoss``
    have.
    """
    if not 0 <= target < len(log_probs):
        raise ValueError(
            f"target {target} out of range for {len(log_probs)} classes"
        )
    return -log_probs[target]


def cross_entropy(logits: Sequence[Value], target: int) -> Value:
    r"""Softmax cross-entropy for one sample, from **logits**.

    Parameters
    ----------
    logits
        Raw network outputs, one per class. Do **not** apply softmax first.
    target
        Index of the correct class.

    Returns
    -------
    Value
        :math:`-\ln p_{\text{target}}`, whose gradient with respect to the
        logits is :math:`p_i - y_i`.

    Examples
    --------
    >>> logits = [Value(2.0), Value(1.0), Value(0.1)]
    >>> loss = cross_entropy(logits, 0)
    >>> loss.backward()
    >>> round(loss.data, 4)          # -ln(0.6590)
    0.4170
    >>> round(logits[0].grad, 4)     # p0 - 1
    -0.341
    """
    return nll_loss(log_softmax(logits), target)


def softmax_cross_entropy(
    batch_logits: Sequence[Sequence[Value]],
    targets: Sequence[int],
) -> Value:
    r"""Mean cross-entropy over a batch.

    Averaging (rather than summing) keeps the gradient magnitude independent of
    batch size, so a learning rate tuned at batch 8 still behaves sensibly at
    batch 64. With a sum, doubling the batch would double every gradient and
    effectively double the learning rate -- a real and confusing bug.
    """
    if len(batch_logits) != len(targets):
        raise ValueError(
            f"got {len(batch_logits)} predictions for {len(targets)} targets"
        )
    if not batch_logits:
        raise ValueError("empty batch")

    total = Value(0.0)
    for logits, target in zip(batch_logits, targets):
        total = total + cross_entropy(logits, target)
    return total / len(batch_logits)


def binary_cross_entropy(
    prediction: Value, target: float, *, from_logits: bool = True
) -> Value:
    r"""Binary cross-entropy for a single output.

    .. math:: \mathcal{L} = -\left[y\ln p + (1-y)\ln(1-p)\right]

    With ``from_logits=True`` (the default and the right choice), ``prediction``
    is a raw logit and the stable identity is used:

    .. math::
        \mathcal{L} = \max(z,0) - zy + \ln\left(1 + e^{-|z|}\right)

    Derivation: for :math:`z \ge 0`, :math:`-\ln\sigma(z) = \ln(1+e^{-z})` and
    :math:`-\ln(1-\sigma(z)) = z + \ln(1+e^{-z})`; the :math:`\max(z,0)` form
    covers both signs with the exponent always :math:`\le 0`. The naive route
    -- sigmoid, then log -- produces ``log(0)`` as soon as the sigmoid
    saturates, which for a confident model happens routinely.

    Both branch decisions use the same ``z.data > 0`` test, deliberately. The
    expression has a kink at exactly :math:`z=0` where :math:`\max(z,0)` is not
    differentiable; taking the ``<= 0`` branch there makes the gradient come
    out as :math:`\sigma(0) - y = 0.5 - y`, agreeing with the closed form.
    Splitting the tests differently would silently give :math:`-0.5 - y` at
    that one point.

    With ``from_logits=False``, ``prediction`` must already be a probability in
    :math:`(0,1)`; provided for teaching comparison, and it *will* produce
    infinities at the extremes, which is the point.
    """
    if not from_logits:
        eps = 1e-12
        p = prediction
        if not (0.0 <= p.data <= 1.0):
            raise ValueError(
                f"from_logits=False expects a probability, got {p.data}"
            )
        return -(target * (p + eps).log() + (1.0 - target) * (1.0 - p + eps).log())

    z = prediction
    positive = z.data > 0.0
    abs_z = z if positive else -z          # |z|, as a differentiable expression
    max_term = z if positive else Value(0.0)  # max(z, 0)
    return max_term - z * target + (1.0 + (-abs_z).exp()).log()
