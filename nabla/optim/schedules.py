r"""Learning-rate schedules for ``Trainer.fit(lr_schedule=...)``.

The Trainer calls ``schedule(epoch, current_lr)`` at the start of every epoch
(epochs count from 1) and uses the returned value. Each factory here returns
such a function, computed from the epoch alone so that resuming from a
checkpoint at epoch ``k`` gives exactly the rate a straight run would have had.

    >>> sched = cosine(0.1, total_epochs=10)
    >>> round(sched(1, 0.1), 4), round(sched(10, 0.1), 4)
    (0.1, 0.0)

Why schedule at all: a large rate early covers ground quickly, a small rate
late lets the optimiser settle into a minimum instead of bouncing around it.
``warmup`` addresses the opposite end -- Adam's second-moment estimate is
noisy for the first few steps, and starting small avoids an early blow-up.
"""

from __future__ import annotations

import math
from typing import Callable

__all__ = ["Schedule", "constant", "step_decay", "exponential", "cosine", "warmup"]

Schedule = Callable[[int, float], float]


def _check_epoch(epoch: int) -> None:
    if epoch < 1:
        raise ValueError(f"epochs count from 1, got {epoch}")


def constant(lr: float) -> Schedule:
    """The same rate every epoch (useful as the ``after`` of a warmup)."""
    def schedule(epoch: int, current: float) -> float:
        _check_epoch(epoch)
        return lr
    return schedule


def step_decay(lr: float, every: int, gamma: float = 0.1) -> Schedule:
    """Multiply the rate by ``gamma`` every ``every`` epochs: lr, lr, ..., lr*gamma, ..."""
    if every < 1:
        raise ValueError(f"every must be >= 1, got {every}")
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")

    def schedule(epoch: int, current: float) -> float:
        _check_epoch(epoch)
        return lr * gamma ** ((epoch - 1) // every)
    return schedule


def exponential(lr: float, gamma: float) -> Schedule:
    """``lr * gamma ** (epoch - 1)``: a smooth decay by the same factor each epoch."""
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")

    def schedule(epoch: int, current: float) -> float:
        _check_epoch(epoch)
        return lr * gamma ** (epoch - 1)
    return schedule


def cosine(lr: float, total_epochs: int, min_lr: float = 0.0) -> Schedule:
    r"""Half a cosine from ``lr`` at epoch 1 down to ``min_lr`` at ``total_epochs``.

    .. math:: \eta_e = \eta_{min} + \tfrac12(\eta - \eta_{min})(1 + \cos(\pi \tfrac{e-1}{T-1}))

    Epochs past ``total_epochs`` stay at ``min_lr``.
    """
    if total_epochs < 1:
        raise ValueError(f"total_epochs must be >= 1, got {total_epochs}")
    if min_lr > lr:
        raise ValueError(f"min_lr ({min_lr}) must not exceed lr ({lr})")

    def schedule(epoch: int, current: float) -> float:
        _check_epoch(epoch)
        if total_epochs == 1:
            return lr
        progress = min(1.0, (epoch - 1) / (total_epochs - 1))
        return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    return schedule


def warmup(lr: float, epochs: int, after: Schedule | None = None) -> Schedule:
    """Ramp linearly up to ``lr`` over ``epochs`` epochs, then hand over to ``after``.

    ``after`` sees epochs renumbered from 1 when the warmup ends, so
    ``warmup(0.1, 3, cosine(0.1, 17))`` is a 20-epoch run: 3 warm, 17 cosine.
    """
    if epochs < 1:
        raise ValueError(f"warmup epochs must be >= 1, got {epochs}")
    following = after or constant(lr)

    def schedule(epoch: int, current: float) -> float:
        _check_epoch(epoch)
        if epoch <= epochs:
            return lr * epoch / epochs
        return following(epoch - epochs, current)
    return schedule
