"""Phase 11 -- metrics and running statistics.

Loss is what we optimise; accuracy is what we care about. They are not the same
thing and they can move in opposite directions -- a model can grow more
confident about samples it already gets right (loss falls) while flipping a few
borderline ones the wrong way (accuracy falls). Tracking both is the point.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

__all__ = ["RunningAverage", "Timer", "accuracy", "argmax", "confusion_counts", "EpochMetrics"]


class RunningAverage:
    """Streaming mean, weighted by batch size.

    Weighting matters: a final partial batch of 7 samples must not count as
    much as a full batch of 32, or the epoch average is quietly wrong. Computed
    incrementally so a long epoch never holds every value in memory.
    """

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, weight: int = 1) -> None:
        self.total += float(value) * weight
        self.count += weight

    @property
    def value(self) -> float:
        return self.total / self.count if self.count else 0.0

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def __float__(self) -> float:
        return self.value

    def __repr__(self) -> str:
        return f"RunningAverage({self.value:.6f}, n={self.count})"


class Timer:
    """Wall-clock timing, as a context manager or by hand.

    Uses ``perf_counter``, which is monotonic and has the highest resolution
    the platform offers -- unlike ``time.time()``, which can jump if the system
    clock is adjusted mid-run.

    >>> with Timer() as t:
    ...     pass
    >>> t.elapsed >= 0
    True
    """

    def __init__(self) -> None:
        self.start = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self.start

    def lap(self) -> float:
        now = time.perf_counter()
        self.elapsed = now - self.start
        return self.elapsed


def argmax(values: Sequence[float]) -> int:
    """Index of the largest element.

    The prediction rule for a classifier. Note softmax is **monotonic**, so
    ``argmax(logits) == argmax(softmax(logits))`` -- you never need to compute
    probabilities just to make a prediction, only to report confidence.
    """
    best = 0
    for i in range(1, len(values)):
        if values[i] > values[best]:
            best = i
    return best


def accuracy(predictions: Sequence[int], targets: Sequence[int]) -> float:
    """Fraction of exactly-correct predictions, in ``[0, 1]``.

    A blunt instrument: it treats a 51% guess and a 99% certainty identically,
    and it is misleading on imbalanced data (99% accuracy is trivial when 99%
    of samples share a label). Cross-entropy sees what accuracy cannot, which
    is why we train on one and report both.
    """
    if not targets:
        return 0.0
    correct = sum(1 for p, t in zip(predictions, targets) if int(p) == int(t))
    return correct / len(targets)


def confusion_counts(
    predictions: Sequence[int], targets: Sequence[int], n_classes: int
) -> list[list[int]]:
    """``matrix[true][predicted]`` counts. Diagonal = correct."""
    matrix = [[0] * n_classes for _ in range(n_classes)]
    for p, t in zip(predictions, targets):
        matrix[int(t)][int(p)] += 1
    return matrix


@dataclass
class EpochMetrics:
    """Everything measured during one epoch."""

    epoch: int
    train_loss: float = 0.0
    train_acc: float | None = None
    val_loss: float | None = None
    val_acc: float | None = None
    seconds: float = 0.0
    samples: int = 0
    grad_norm: float | None = None
    learning_rate: float | None = None
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def samples_per_second(self) -> float:
        return self.samples / self.seconds if self.seconds > 0 else 0.0

    def format_line(self, total_epochs: int | None = None) -> str:
        """One compact line per epoch, aligned so columns are scannable."""
        head = (
            f"epoch {self.epoch:>3}/{total_epochs}"
            if total_epochs
            else f"epoch {self.epoch:>3}"
        )
        parts = [head, f"loss {self.train_loss:.4f}"]
        if self.train_acc is not None:
            parts.append(f"acc {self.train_acc * 100:5.2f}%")
        if self.val_loss is not None:
            parts.append(f"val_loss {self.val_loss:.4f}")
        if self.val_acc is not None:
            parts.append(f"val_acc {self.val_acc * 100:5.2f}%")
        if self.grad_norm is not None:
            parts.append(f"|g| {self.grad_norm:.3f}")
        parts.append(f"{self.seconds:.1f}s")
        if self.samples_per_second:
            parts.append(f"{self.samples_per_second:.0f} samp/s")
        return "  ".join(parts)

    def to_dict(self) -> dict[str, float]:
        out = {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "seconds": self.seconds,
            "samples": self.samples,
        }
        for key in ("train_acc", "val_loss", "val_acc", "grad_norm", "learning_rate"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        out.update(self.extra)
        return out
