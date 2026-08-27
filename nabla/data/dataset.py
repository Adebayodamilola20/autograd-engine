"""Datasets, batching and shuffling.

Deliberately small. The engine does not need a data framework -- it needs
something that hands out reproducible batches and gets out of the way.

Why batching exists at all
--------------------------
Three separate reasons, often conflated:

1. **Gradient quality.** One sample gives a very noisy estimate of the true
   gradient; the whole dataset gives an exact one but only one update per pass.
   A batch of 32 averages 32 noisy estimates, cutting the standard error by
   :math:`\\sqrt{32} \\approx 5.7` while still allowing many updates per epoch.
2. **Hardware.** On a vectorised backend, 32 samples cost barely more than 1
   because the matrix multiply is the same shape with a bigger inner dimension.
   This reason does **not** apply to our scalar engine -- a batch of 32 costs
   exactly 32x -- which is itself a useful thing to notice.
3. **Regularisation.** The residual noise helps escape sharp minima. Very large
   batches are known to generalise slightly worse for this reason.

Reproducibility
---------------
Shuffling draws from an explicitly-seeded ``numpy`` generator (decision D9), so
a seed plus a config fully determines a run -- including the order samples were
seen in, which matters more than people expect when comparing experiments.
"""

from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np

__all__ = ["Dataset", "DataLoader", "train_val_split", "one_hot"]


class Dataset:
    """Features and labels held as NumPy arrays.

    NumPy is used here purely as a container and for indexing -- no gradient
    ever touches it. Samples are converted to plain Python floats on the way
    out, because that is what the scalar engine consumes.

    Parameters
    ----------
    x
        Shape ``(n_samples, n_features)``.
    y
        Shape ``(n_samples,)`` -- class indices for classification, or targets
        of shape ``(n_samples, n_outputs)`` for regression.
    name
        Used in log lines and plot titles.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, *, name: str = "dataset") -> None:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        if len(x) != len(y):
            raise ValueError(f"x has {len(x)} rows but y has {len(y)}")
        if x.ndim != 2:
            raise ValueError(f"x must be 2-D (n_samples, n_features), got {x.shape}")

        self.x = x
        self.y = y
        self.name = name

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[list[float], int | list[float]]:
        target = self.y[index]
        return (
            self.x[index].tolist(),
            target.tolist() if isinstance(target, np.ndarray) else target.item(),
        )

    @property
    def n_features(self) -> int:
        return int(self.x.shape[1])

    @property
    def n_classes(self) -> int:
        """Number of distinct integer labels. Meaningless for regression."""
        return int(np.max(self.y)) + 1 if self.y.ndim == 1 else int(self.y.shape[1])

    def subset(self, n: int, *, seed: int | None = None) -> "Dataset":
        """A random subset of ``n`` samples.

        The scalar engine is slow enough that MNIST experiments run on subsets;
        this makes that explicit and reproducible rather than ad hoc slicing.
        """
        if n >= len(self):
            return self
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self), size=n, replace=False)
        return Dataset(self.x[idx], self.y[idx], name=f"{self.name}[{n}]")

    def class_counts(self) -> dict[int, int]:
        labels, counts = np.unique(self.y, return_counts=True)
        return {int(k): int(v) for k, v in zip(labels, counts)}

    def __repr__(self) -> str:
        return (
            f"Dataset({self.name}: {len(self):,} samples, "
            f"{self.n_features} features)"
        )


class DataLoader:
    """Yields shuffled mini-batches from a ``Dataset``.

    Each iteration produces ``(batch_x, batch_y)`` where ``batch_x`` is a list
    of feature lists and ``batch_y`` a list of targets -- the plain-Python form
    the scalar engine consumes directly.

    Parameters
    ----------
    dataset
        The data to iterate.
    batch_size
        Samples per batch. ``len(dataset)`` gives full-batch gradient descent.
    shuffle
        Reshuffle every epoch. Always on for training: a fixed order lets the
        model exploit correlations between adjacent samples, and if the data is
        sorted by class it can be catastrophic.
    drop_last
        Discard a final partial batch. Off by default -- with a slow engine
        every sample counts.
    seed
        Seeds the shuffle generator.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.dataset = dataset
        self.batch_size = min(batch_size, len(dataset)) if len(dataset) else batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[tuple[list[list[float]], list]]:
        n = len(self.dataset)
        order = self.rng.permutation(n) if self.shuffle else np.arange(n)

        for start in range(0, n, self.batch_size):
            idx = order[start : start + self.batch_size]
            if self.drop_last and len(idx) < self.batch_size:
                break
            xs = self.dataset.x[idx].tolist()
            ys = self.dataset.y[idx]
            yield xs, ys.tolist()

    def __repr__(self) -> str:
        return (
            f"DataLoader({self.dataset.name}, batch_size={self.batch_size}, "
            f"{len(self)} batches)"
        )


def train_val_split(
    dataset: Dataset,
    *,
    val_fraction: float = 0.1,
    seed: int | None = None,
    stratify: bool = True,
) -> tuple[Dataset, Dataset]:
    """Split into training and validation sets.

    Why a validation set exists
    ---------------------------
    Training loss measures how well the model fits data it has already seen,
    which it can always improve by memorising. Validation loss measures
    performance on data it has never been updated on, which is the only number
    that estimates generalisation. When training loss falls while validation
    loss rises, the model has started memorising -- and you can only see that
    if you held data back.

    Parameters
    ----------
    stratify
        Preserve class proportions in both splits. Matters when a split is
        small: a random 10% of a 500-sample set can easily miss a class
        entirely, which makes the validation accuracy meaningless.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    rng = np.random.default_rng(seed)
    n = len(dataset)

    if stratify and dataset.y.ndim == 1:
        train_idx: list[int] = []
        val_idx: list[int] = []
        for label in np.unique(dataset.y):
            members = np.flatnonzero(dataset.y == label)
            rng.shuffle(members)
            cut = max(1, int(round(len(members) * val_fraction)))
            val_idx.extend(members[:cut].tolist())
            train_idx.extend(members[cut:].tolist())
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
    else:
        order = rng.permutation(n)
        cut = int(round(n * val_fraction))
        val_idx = order[:cut].tolist()
        train_idx = order[cut:].tolist()

    return (
        Dataset(dataset.x[train_idx], dataset.y[train_idx], name=f"{dataset.name}/train"),
        Dataset(dataset.x[val_idx], dataset.y[val_idx], name=f"{dataset.name}/val"),
    )


def one_hot(labels: Sequence[int], n_classes: int) -> np.ndarray:
    """Convert class indices to one-hot rows.

    Not needed by ``cross_entropy`` -- it takes the index directly, which is
    both faster and clearer. Provided for MSE-based comparisons and for the
    experiments that deliberately use the wrong loss to show what happens.
    """
    out = np.zeros((len(labels), n_classes), dtype=np.float64)
    out[np.arange(len(labels)), np.asarray(labels, dtype=int)] = 1.0
    return out
