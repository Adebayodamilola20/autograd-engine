"""Phase 9 -- loss functions, with the mathematics in the open.

A loss turns "how wrong is the model?" into a single ``Value``, which is the
root of the graph that ``backward()`` walks. Choosing one is choosing what
gradients the model will receive, so the choice is not cosmetic -- see the
module docstrings for why MSE is wrong for classification and why softmax and
cross-entropy must be computed together.

    >>> from nabla.losses import mse_loss, cross_entropy
"""

from .cross_entropy import (
    binary_cross_entropy,
    cross_entropy,
    log_softmax,
    log_sum_exp,
    nll_loss,
    softmax,
    softmax_cross_entropy,
)
from .mse import huber_loss, mae_loss, mse, mse_loss, sse_loss

__all__ = [
    "mse_loss",
    "mse",
    "sse_loss",
    "mae_loss",
    "huber_loss",
    "cross_entropy",
    "softmax_cross_entropy",
    "softmax",
    "log_softmax",
    "log_sum_exp",
    "nll_loss",
    "binary_cross_entropy",
]
