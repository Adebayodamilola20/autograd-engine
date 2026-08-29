"""Train nabla on your own data -- nothing here is MNIST-specific.

Run:  python examples/train_your_own_data.py
      python examples/train_your_own_data.py --csv path/to/your.csv

Every other example in this repository trains on a dataset the repository
itself supplies. That is good for reproducibility and bad for answering the
question people actually arrive with: *can I point this at my data?*

Yes. The engine never knew what a digit was. ``TensorMLP`` takes a matrix of
numbers and a vector of labels; where they came from is not its concern.

What you need
-------------
Two arrays:

    X   shape (n_rows, n_features)   float -- your columns
    y   shape (n_rows,)              int   -- the class of each row, 0..k-1

That is the whole contract. With no ``--csv`` this script generates a
synthetic 8-feature, 2-class problem so it runs anywhere; with ``--csv`` it
reads a real file, treating the last column as the label.

Three things that decide whether this works
-------------------------------------------
**Standardise your features.** MNIST pixels are already on a common [0, 1]
scale. Real tabular columns are not -- an "income" column in the tens of
thousands next to an "age" column under 100 means the first dominates the
first layer's gradients purely by magnitude, and the second is effectively
ignored. We subtract the mean and divide by the standard deviation, both
computed on the **training split only**: using the full dataset's statistics
leaks information about rows the model must not have seen. This single step
matters more than the architecture.

**Stratify the split.** ``train_val_split(..., stratify=True)`` keeps the class
proportions equal in both halves. On a balanced problem it changes little; on
an imbalanced one (fraud, defects, rare diagnoses) an unstratified split can
put almost none of the minority class in validation, and the score it reports
is then measuring nothing.

**Watch the gap, not the training accuracy.** Training accuracy always
improves -- a big enough network memorises the training set exactly. The
number that means something is validation accuracy, and the moment it stops
tracking training accuracy is the moment the model started memorising rather
than learning. ``early_stopping_patience`` below acts on precisely that.

Scale
-----
The tensor engine is comfortable to roughly one to two million parameters on a
CPU. Past that it still runs, but a step costs enough that training stops
being interactive -- see docs/18-performance.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from nabla.data import DataLoader, Dataset, train_val_split  # noqa: E402
from nabla.losses.tensor_losses import tensor_cross_entropy  # noqa: E402
from nabla.nn.tensor_mlp import TensorMLP  # noqa: E402
from nabla.optim import Adam  # noqa: E402
from nabla.training import Trainer  # noqa: E402


def rule(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m\n" + "─" * 74)


# ----------------------------------------------------------------------
# getting data in
# ----------------------------------------------------------------------


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read a CSV whose **last column is the label**.

    Deliberately minimal -- ``np.genfromtxt`` and no pandas dependency. If your
    file has strings, dates or missing values, clean it before this point;
    encoding categorical columns sensibly is a data question, not an autodiff
    one, and pretending otherwise would hide the interesting part.
    """
    raw = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    names = list(raw.dtype.names)
    table = np.array([[float(row[c]) for c in names] for row in raw])

    x, y = table[:, :-1], table[:, -1].astype(int)
    if np.isnan(x).any():
        raise ValueError(
            f"{path} contains missing values -- fill or drop them first, "
            "since a NaN feature makes every gradient downstream NaN"
        )
    return x, y, names[:-1]


def synthetic(n_rows: int = 2000, n_features: int = 8) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """A learnable-but-not-trivial stand-in, so the script runs with no input.

    A random linear rule with noise added: separable enough that a working
    engine should reach the low nineties, noisy enough that 100% would mean
    something had gone wrong.
    """
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n_rows, n_features))
    weights = rng.normal(size=n_features)
    y = ((x @ weights + 0.4 * rng.normal(size=n_rows)) > 0).astype(int)
    return x, y, [f"feature_{i}" for i in range(n_features)]


def standardise(train_x: np.ndarray, val_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero mean, unit variance -- using the **training** statistics for both.

    The ``1e-8`` guards a constant column, which has zero standard deviation
    and would otherwise divide by zero. Such a column carries no information
    anyway; this keeps it harmless instead of poisoning the whole matrix
    with NaN.
    """
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0) + 1e-8
    return (train_x - mean) / std, (val_x - mean) / std


# ----------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--csv", type=Path, help="CSV file; last column is the label")
    p.add_argument("--hidden", type=int, nargs="+", default=[32, 16])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rule("1. The data")
    if args.csv:
        x, y, columns = load_csv(args.csv)
        print(f"  read {args.csv}")
    else:
        x, y, columns = synthetic()
        print("  no --csv given, using a synthetic 8-feature, 2-class problem")

    n_classes = int(y.max()) + 1
    counts = np.bincount(y, minlength=n_classes)
    print(f"  {len(x):,} rows x {x.shape[1]} features -> {n_classes} classes")
    print(f"  columns: {', '.join(columns[:6])}{' ...' if len(columns) > 6 else ''}")
    print(f"  class balance: {dict(enumerate(counts.tolist()))}")

    if counts.min() < 2:
        raise SystemExit(
            "every class needs at least 2 rows to appear in both splits; "
            f"class {int(counts.argmin())} has {int(counts.min())}"
        )

    rule("2. Split, then standardise -- in that order")
    train, val = train_val_split(
        Dataset(x, y, name="your data"),
        val_fraction=args.val_fraction,
        stratify=True,
        seed=args.seed,
    )
    train.x, val.x = standardise(train.x, val.x)
    print(f"  train {len(train):,}   validation {len(val):,}   (stratified)")
    print("  standardised on training statistics only -- val never leaks in")

    rule("3. The model")
    model = TensorMLP(
        n_in=x.shape[1],
        hidden=args.hidden,
        n_out=n_classes,
        activation="relu",
        seed=args.seed,
    )
    print(" ", model.summary().replace("\n", "\n  "))

    rule("4. Training")
    # A majority-class guesser is the bar any real model has to clear. Quoting
    # accuracy without it is how a 95%-accurate model on a 95%-imbalanced
    # problem gets mistaken for a good one.
    baseline = counts.max() / counts.sum()
    print(f"  always-guess-the-majority baseline: {baseline:.2%}\n")

    trainer = Trainer(
        model,
        Adam(model.parameters(), lr=args.lr),
        tensor_cross_entropy,
        metric="accuracy",
    )
    history = trainer.fit(
        DataLoader(train, batch_size=args.batch_size, shuffle=True, seed=args.seed),
        DataLoader(val, batch_size=256),
        epochs=args.epochs,
        early_stopping_patience=8,
    )

    rule("Result")
    # `best` returns a 0-based index into the history; +1 to match the epoch
    # numbers printed above, which count from 1.
    index, best = history.best("val_acc")
    final_train = history["train_acc"][-1]
    print(f"    best validation accuracy   {best:.2%}  (epoch {index + 1})")
    print(f"    final training accuracy    {final_train:.2%}")
    print(f"    majority-class baseline    {baseline:.2%}")
    print(f"    parameters                 {model.num_parameters():,}")

    gap = final_train - best
    print()
    if best <= baseline + 0.01:
        print("    The model has not beaten guessing. Either the features do not")
        print("    predict the label, or training needs longer -- check whether the")
        print("    loss fell at all before reaching for a bigger network.")
    elif gap > 0.10:
        print(f"    Training accuracy is {gap:.1%} above validation: the network is")
        print("    memorising. Fewer parameters or more data, not more epochs.")
    else:
        print("    Training and validation are tracking each other, and the model")
        print("    beats the baseline. That is what learning looks like.")

    print("\n    To use your own file:")
    print("      python examples/train_your_own_data.py --csv your.csv")
    print("    Last column is the label; every other column is a feature.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
