r"""Phase 18 -- losses for the tensor engine.

Same mathematics as ``cross_entropy.py`` and ``mse.py``, computed for a whole
batch at once. See those files for the derivations; this one is about how the
same formulas look when the granularity changes.

The one interesting difference: **one-hot as a matrix**
--------------------------------------------------------
The scalar version selects the target's log-probability by indexing:
``log_probs[target]``. The vectorised version instead multiplies by a one-hot
matrix and sums:

.. math:: \mathcal{L} = -\frac{1}{N}\sum_{i}\sum_{k} Y_{ik}\,\ln p_{ik}

where :math:`Y` is a constant :math:`(N, K)` one-hot array. Those are the same
number, but the second is a single vectorised expression -- and, crucially,
it is built **only from operations the engine already has** (``mul``, ``sum``,
``div``, ``log_softmax``). We write no new backward rule, and the chain rule
still produces the exact :math:`(p - y)/N` gradient. ``examples/`` checks that
against the closed form.

PyTorch instead ships a *fused* kernel for this, which avoids materialising the
one-hot matrix and computes :math:`p - y` in one pass. That is a performance
optimisation, not a mathematical one -- and knowing it is fused is why
``F.cross_entropy(logits, targets)` takes integer targets rather than one-hot.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..core.tensor import Tensor

__all__ = [
    "tensor_cross_entropy",
    "tensor_mse",
    "one_hot_array",
    "tensor_accuracy",
]


def one_hot_array(targets: Sequence[int], n_classes: int) -> np.ndarray:
    """Constant one-hot matrix of shape ``(len(targets), n_classes)``.

    A *constant*, not a parameter: it carries no gradient because the labels
    are given, not learned.

    Labels are range-checked, which NumPy will not do for you in one
    direction. ``out[rows, targets] = 1.0`` raises for a label of ``n_classes``
    or above, but a **negative** label is a valid NumPy index that counts from
    the end, so ``-1`` silently one-hots the last class. That matters more
    than it sounds: ``-1`` is the usual sentinel for "unlabelled" or "ignore",
    so the natural way to mark a row as having no label quietly trains it
    against the final class instead.
    """
    targets = np.asarray(targets, dtype=int)
    if targets.size:
        low, high = int(targets.min()), int(targets.max())
        if low < 0 or high >= n_classes:
            offender = low if low < 0 else high
            raise ValueError(
                f"class label {offender} is outside [0, {n_classes - 1}]"
                + (
                    "; negative labels are not an 'ignore' sentinel here, they "
                    "would silently one-hot a class counting from the end"
                    if low < 0
                    else ""
                )
            )
    out = np.zeros((len(targets), n_classes), dtype=np.float64)
    out[np.arange(len(targets)), targets] = 1.0
    return out


def tensor_cross_entropy(logits: Tensor, targets: Sequence[int]) -> Tensor:
    r"""Mean softmax cross-entropy over a batch, from logits.

    Parameters
    ----------
    logits
        ``(batch, n_classes)``. Raw network outputs -- do **not** softmax first.
    targets
        Integer class indices, length ``batch``.

    Returns
    -------
    Tensor
        A scalar, whose gradient with respect to ``logits`` is
        :math:`(p - y)/N`.

    Examples
    --------
    >>> logits = Tensor([[2.0, 1.0, 0.1]])
    >>> loss = tensor_cross_entropy(logits, [0])
    >>> round(loss.item(), 4)
    0.417
    """
    if logits.ndim != 2:
        raise ValueError(
            f"expected logits of shape (batch, n_classes), got {logits.shape}"
        )
    batch, n_classes = logits.shape
    if len(targets) != batch:
        raise ValueError(f"{batch} rows of logits but {len(targets)} targets")

    log_probs = logits.log_softmax(axis=-1)
    y = Tensor(one_hot_array(targets, n_classes), label="onehot")
    # -(1/N) Σᵢ Σₖ Yᵢₖ ln pᵢₖ  -- the mean over the batch keeps the gradient
    # magnitude independent of batch size.
    return -(log_probs * y).sum() / float(batch)


def tensor_mse(predictions: Tensor, targets: np.ndarray | Tensor) -> Tensor:
    r"""Mean squared error over every element.

    .. math:: \frac{1}{N}\sum(\hat y - y)^{2}
    """
    if not isinstance(targets, Tensor):
        targets = Tensor(np.asarray(targets, dtype=np.float64))
    diff = predictions - targets
    return (diff * diff).mean()


def tensor_accuracy(logits: Tensor, targets: Sequence[int]) -> float:
    """Fraction correct, as a plain float. Not differentiable, and not meant to be.

    Range-checked for the same reason as ``one_hot_array``, and to keep the
    two agreeing. Without it a label of ``-1`` reports a near-zero loss (the
    one-hot wrapped around to the last class, which the model predicted) and
    simultaneously zero accuracy (``argmax`` returns ``2``, which does not
    equal ``-1``). Two metrics contradicting each other on the same batch is a
    much harder thing to debug than one clear error.
    """
    targets = np.asarray(targets, dtype=int)
    predicted = np.asarray(logits.data).argmax(axis=-1)
    if predicted.shape != targets.shape:
        raise ValueError(
            f"{predicted.shape[0]} rows of logits but {targets.shape[0]} targets"
        )
    if targets.size:
        n_classes = logits.shape[-1]
        low, high = int(targets.min()), int(targets.max())
        if low < 0 or high >= n_classes:
            raise ValueError(
                f"class label {low if low < 0 else high} is outside "
                f"[0, {n_classes - 1}]"
            )
    return float(np.mean(predicted == targets))
