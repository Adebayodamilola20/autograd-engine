"""Phase 22 -- shared machinery for running controlled experiments.

An experiment is only worth running if it can change your mind, which means
everything except the variable under test has to be held fixed. This module
exists so each experiment file can express *only* what it varies.

What is held fixed by default
-----------------------------
Same data, same subset, same split, same seed, same architecture, same
optimiser, same batch size, same number of epochs. Every experiment prints the
configuration it actually ran so the controls are visible rather than implied.

Seeds and honesty
-----------------
A single seed can make a bad configuration look good. Every result here is
averaged over ``--seeds`` runs (default 3) and reported with its spread, so a
difference smaller than the seed-to-seed variation is visibly *not* a
difference. This is the single easiest way to stop fooling yourself, and it
costs nothing but time.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "experiments"

from nabla.data import DataLoader, train_val_split  # noqa: E402
from nabla.data.mnist import load_mnist  # noqa: E402
from nabla.losses.tensor_losses import tensor_cross_entropy  # noqa: E402
from nabla.nn.tensor_mlp import TensorMLP  # noqa: E402
from nabla.optim import SGD, Adam  # noqa: E402
from nabla.training import Trainer  # noqa: E402

# Defaults every experiment inherits unless it deliberately overrides one.
DEFAULTS = dict(
    hidden=[128, 64],
    activation="relu",
    optimizer="adam",
    lr=1e-3,
    momentum=0.9,
    batch_size=64,
    epochs=8,
    n_train=8000,
    n_test=2000,
    grad_clip=None,
)


@dataclass
class RunResult:
    """One training run, reduced to the numbers an experiment compares."""

    label: str
    config: dict[str, Any]
    seed: int
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    grad_mean: list[float] = field(default_factory=list)
    grad_max: list[float] = field(default_factory=list)
    per_layer_grad: list[list[float]] = field(default_factory=list)
    test_acc: float = 0.0
    seconds: float = 0.0
    diverged: bool = False
    note: str = ""

    @property
    def best_val_acc(self) -> float:
        return max(self.val_acc) if self.val_acc else 0.0

    @property
    def final_train_loss(self) -> float:
        return self.train_loss[-1] if self.train_loss else float("nan")


# ======================================================================
# data -- loaded once and reused, so every run sees identical samples
# ======================================================================

_CACHE: dict[tuple, Any] = {}


def get_data(n_train: int, n_test: int, seed: int = 0):
    """Load and split MNIST, memoised.

    Memoisation is not just a speed trick here: it guarantees that every
    configuration in a sweep is scored on *the same* validation and test
    samples. Re-drawing the split per run would add a second, uncontrolled
    variable.
    """
    key = (n_train, n_test, seed)
    if key not in _CACHE:
        train_full, test = load_mnist(n_train=n_train, n_test=n_test, seed=seed)
        train, val = train_val_split(train_full, val_fraction=0.15, seed=seed)
        _CACHE[key] = (train, val, test)
    return _CACHE[key]


# ======================================================================
# a single run
# ======================================================================


def run_once(
    label: str,
    *,
    seed: int = 0,
    track_gradients: bool = True,
    **overrides: Any,
) -> RunResult:
    """Train one configuration and return its curves.

    Divergence (a non-finite loss) is caught and recorded rather than raised.
    "This configuration blows up" is a legitimate experimental result, and
    several experiments here are specifically about producing it on purpose.
    """
    config = {**DEFAULTS, **overrides}
    train, val, test = get_data(config["n_train"], config["n_test"])

    model = TensorMLP(
        784,
        config["hidden"],
        10,
        activation=config["activation"],
        seed=seed,
    )

    if config["optimizer"] == "adam":
        optimizer = Adam(model.parameters(), lr=config["lr"])
    else:
        optimizer = SGD(
            model.parameters(),
            lr=config["lr"],
            momentum=config["momentum"] if config["optimizer"] == "momentum" else 0.0,
        )

    trainer = Trainer(
        model,
        optimizer,
        tensor_cross_entropy,
        metric="accuracy",
        grad_clip=config["grad_clip"],
        verbose=False,
    )

    result = RunResult(label=label, config=config, seed=seed)

    # A callback samples gradient health once per epoch. Cheap, and it is the
    # only way to distinguish "not learning" from "learning slowly".
    def on_epoch_end(metrics: Any) -> None:
        if not track_gradients:
            return
        stats = model.gradient_stats()
        result.grad_mean.append(stats["mean"])
        result.grad_max.append(stats["max"])
        result.per_layer_grad.append(
            [float(np.abs(layer.W.grad).mean()) for layer in model.layers]
        )

    loaders = dict(as_arrays=True)
    train_loader = DataLoader(
        train, batch_size=config["batch_size"], seed=seed, **loaders
    )
    val_loader = DataLoader(val, batch_size=256, shuffle=False, **loaders)
    test_loader = DataLoader(test, batch_size=256, shuffle=False, **loaders)

    started = time.perf_counter()
    try:
        history = trainer.fit(
            train_loader,
            val_loader,
            epochs=config["epochs"],
            callbacks=[on_epoch_end],
        )
        result.train_loss = list(history["train_loss"])
        result.val_loss = list(history.get("val_loss", []))
        result.train_acc = list(history.get("train_acc", []))
        result.val_acc = list(history.get("val_acc", []))
        _, result.test_acc = trainer.evaluate(test_loader)
    except RuntimeError as err:
        # The Trainer raises when the loss stops being finite.
        result.diverged = True
        result.note = str(err)[:160]
        result.train_loss = list(trainer.history.get("train_loss", []))
        result.val_acc = list(trainer.history.get("val_acc", []))

    result.seconds = time.perf_counter() - started
    return result


def run_sweep(
    variants: Sequence[tuple[str, dict[str, Any]]],
    *,
    seeds: Sequence[int] = (0, 1, 2),
    track_gradients: bool = True,
    progress: bool = True,
) -> list[RunResult]:
    """Run every variant under every seed. Returns a flat list of results."""
    results: list[RunResult] = []
    total = len(variants) * len(seeds)
    done = 0
    for label, overrides in variants:
        for seed in seeds:
            done += 1
            if progress:
                print(f"    [{done:>2}/{total}] {label}  seed={seed}", flush=True)
            results.append(
                run_once(label, seed=seed, track_gradients=track_gradients, **overrides)
            )
    return results


# ======================================================================
# aggregation and reporting
# ======================================================================


@dataclass
class Aggregate:
    """One variant's results, averaged across seeds."""

    label: str
    test_acc_mean: float
    test_acc_std: float
    best_val_mean: float
    final_loss_mean: float
    seconds_mean: float
    diverged: int
    runs: int

    def acc_text(self) -> str:
        if self.diverged == self.runs:
            return "diverged"
        return f"{self.test_acc_mean * 100:.2f}% ± {self.test_acc_std * 100:.2f}"


def aggregate(results: Sequence[RunResult]) -> list[Aggregate]:
    """Group by label, preserving first-seen order."""
    order: list[str] = []
    groups: dict[str, list[RunResult]] = {}
    for r in results:
        if r.label not in groups:
            groups[r.label] = []
            order.append(r.label)
        groups[r.label].append(r)

    out = []
    for label in order:
        runs = groups[label]
        accs = [r.test_acc for r in runs]
        out.append(
            Aggregate(
                label=label,
                test_acc_mean=float(np.mean(accs)),
                test_acc_std=float(np.std(accs)),
                best_val_mean=float(np.mean([r.best_val_acc for r in runs])),
                final_loss_mean=float(
                    np.mean([r.final_train_loss for r in runs if np.isfinite(r.final_train_loss)])
                    if any(np.isfinite(r.final_train_loss) for r in runs)
                    else float("nan")
                ),
                seconds_mean=float(np.mean([r.seconds for r in runs])),
                diverged=sum(1 for r in runs if r.diverged),
                runs=len(runs),
            )
        )
    return out


def mean_curve(results: Sequence[RunResult], label: str, key: str) -> np.ndarray:
    """Average one curve across seeds, truncating to the shortest run.

    Runs can differ in length when one diverges early or early stopping fires,
    and averaging ragged lists silently pads with garbage.
    """
    curves = [getattr(r, key) for r in results if r.label == label and getattr(r, key)]
    if not curves:
        return np.array([])
    n = min(len(c) for c in curves)
    return np.mean([c[:n] for c in curves], axis=0)


def rule(title: str, width: int = 78) -> None:
    print(f"\n\033[1m{title}\033[0m\n" + "─" * width)


def results_table(aggregates: Sequence[Aggregate], variable: str = "variant") -> str:
    headers = [variable, "test accuracy", "best val", "final loss", "time", "diverged"]
    rows = []
    for a in aggregates:
        rows.append(
            [
                a.label,
                a.acc_text(),
                f"{a.best_val_mean * 100:.2f}%",
                f"{a.final_loss_mean:.4f}" if np.isfinite(a.final_loss_mean) else "—",
                f"{a.seconds_mean:.1f}s",
                f"{a.diverged}/{a.runs}" if a.diverged else "—",
            ]
        )
    widths = [max(len(str(r[i])) for r in [headers, *rows]) for i in range(len(headers))]

    def line(cells):
        out = [str(cells[0]).ljust(widths[0])]
        out += [str(c).rjust(w) for c, w in zip(cells[1:], widths[1:])]
        return "    " + "  ".join(out)

    sep = "    " + "  ".join("─" * w for w in widths)
    return "\n".join([line(headers), sep, *(line(r) for r in rows)])


def save(name: str, results: Sequence[RunResult], extra: dict | None = None) -> Path:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": name,
        "defaults": DEFAULTS,
        "results": [asdict(r) for r in results],
        "aggregates": [asdict(a) for a in aggregate(results)],
        **(extra or {}),
    }
    path = ARTIFACTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=float))
    return path


def figure(name: str):
    """Open a matplotlib figure, returning ``(plt, path)``; caller saves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    return plt, ARTIFACTS / f"{name}.png"


def add_seed_arguments(parser) -> None:
    parser.add_argument("--seeds", type=int, default=3, help="runs per variant")
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--n-train", type=int, default=DEFAULTS["n_train"])
    parser.add_argument("--no-plot", action="store_true")
