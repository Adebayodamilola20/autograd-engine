"""Phase 10 -- gradient-based optimisers, written from scratch.

Backpropagation says which way is uphill. These decide what to do about it.

    >>> from nabla.optim import SGD, Adam
"""

from .adam import Adam, AdamW
from .optimizer import Optimizer
from .sgd import SGD

__all__ = ["Optimizer", "SGD", "Adam", "AdamW"]
