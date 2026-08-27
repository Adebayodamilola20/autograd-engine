"""Phase 11 -- the training system: loop, metrics, checkpoints.

    >>> from nabla.training import Trainer
"""

from .checkpoint import load_checkpoint, load_npz, save_checkpoint, save_npz
from .metrics import EpochMetrics, RunningAverage, Timer, accuracy, argmax, confusion_counts
from .trainer import Trainer, TrainingHistory

__all__ = [
    "Trainer",
    "TrainingHistory",
    "EpochMetrics",
    "RunningAverage",
    "Timer",
    "accuracy",
    "argmax",
    "confusion_counts",
    "save_checkpoint",
    "load_checkpoint",
    "save_npz",
    "load_npz",
]
