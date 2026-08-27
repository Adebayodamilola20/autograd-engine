r"""Experiment 5 -- batch size, and the three things it changes at once.

Run:
    python experiments/exp_batch_size.py
    python experiments/exp_batch_size.py --seeds 5

Batch size looks like a performance knob. It is really three knobs bolted
together, and they pull in different directions.

**1. Gradient quality.** The true gradient is the average over the whole
training set. A mini-batch of :math:`B` samples estimates it, with standard
error falling as :math:`1/\sqrt{B}`. Note the square root: going from 32 to
128 samples is 4x the work for 2x less noise. Diminishing returns are built
into the mathematics.

**2. Number of updates.** With :math:`N` samples, one epoch performs
:math:`N/B` updates. **Doubling the batch halves the number of updates.** At
fixed epochs, large batches take far fewer steps -- and steps, not samples,
are what move the parameters. This is usually the dominant effect, and the one
people forget.

**3. Speed per sample.** Large batches turn many small matmuls into few large
ones, which is exactly what BLAS is good at. Our own Phase 17 numbers show the
per-sample cost dropping steeply with batch size.

The noise is not only a cost
----------------------------
Small-batch gradient noise acts as a regulariser. It jitters the parameters,
which discourages settling into sharp minima and often *improves*
generalisation. So "less noise" is not automatically "better" — one of the few
places in optimisation where a worse estimate can produce a better model.

The practical consequence
-------------------------
If you raise the batch size, you generally have to raise the learning rate to
compensate for taking fewer steps -- the "linear scaling rule": multiply lr by
the same factor as the batch. This experiment runs both fixed-lr and scaled-lr
arms, because with fixed lr the result is dominated by the update-count effect
and tells you almost nothing about batching itself.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from experiments.common import (  # noqa: E402
    aggregate,
    figure,
    mean_curve,
    results_table,
    rule,
    run_sweep,
    save,
)

SIZES = [8, 32, 64, 128, 512]
BASE_BATCH = 64
BASE_LR = 1e-3


def main() -> int:
    p = argparse.ArgumentParser(description="Batch size sweep.")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--n-train", type=int, default=8000)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    common = dict(epochs=args.epochs, n_train=args.n_train)
    seeds = tuple(range(args.seeds))
    n_train_used = int(args.n_train * 0.85)  # after the 15% validation split

    rule("Experiment 5a: batch size at a fixed learning rate")
    print(
        f"    Adam lr={BASE_LR}, ReLU, {args.epochs} epochs, {args.seeds} seeds.\n"
        "    Updates per epoch shown -- this is the variable that actually\n"
        "    drives the result.\n"
    )
    for b in SIZES:
        print(f"      batch {b:>4}   {n_train_used // b:>5,} updates/epoch")
    print()

    fixed = run_sweep(
        [(f"batch {b}", dict(batch_size=b, lr=BASE_LR, **common)) for b in SIZES],
        seeds=seeds,
    )
    fixed_aggs = aggregate(fixed)
    print()
    print(results_table(fixed_aggs, "batch (fixed lr)"))

    # ------------------------------------------------------------------
    rule("Experiment 5b: batch size with the linear scaling rule")
    print(
        f"    lr scaled proportionally: lr = {BASE_LR} × (batch / {BASE_BATCH}).\n"
        "    This holds the *distance travelled per epoch* roughly constant,\n"
        "    isolating the effect of gradient quality from update count.\n"
    )
    scaled = run_sweep(
        [
            (
                f"batch {b} (lr×{b / BASE_BATCH:g})",
                dict(batch_size=b, lr=BASE_LR * b / BASE_BATCH, **common),
            )
            for b in SIZES
        ],
        seeds=seeds,
    )
    scaled_aggs = aggregate(scaled)
    print()
    print(results_table(scaled_aggs, "batch (scaled lr)"))

    # ------------------------------------------------------------------
    rule("What happened")

    small_fixed, large_fixed = fixed_aggs[0], fixed_aggs[-1]
    large_scaled = scaled_aggs[-1]

    print(
        f"    Fixed lr:  batch 8 → {small_fixed.acc_text()},  "
        f"batch 512 → {large_fixed.acc_text()}\n"
        f"    Scaled lr: batch 512 → {large_scaled.acc_text()}\n"
    )
    print(
        f"    At batch 512 the model gets {n_train_used // 512} updates per epoch\n"
        f"    against {n_train_used // 8} at batch 8 -- a {(n_train_used // 8) // max(n_train_used // 512, 1)}x difference in how\n"
        "    many times the parameters move. At a fixed learning rate, that\n"
        "    difference is most of what the first table measures. It is not\n"
        "    really telling you that big batches are bad; it is telling you\n"
        "    that few steps are bad.\n\n"
        "    The second table corrects for it. Scaling the learning rate with\n"
        "    the batch recovers much of the gap, which is the evidence for the\n"
        "    linear scaling rule being a real effect rather than folklore."
    )

    print(
        "\n    Wall-clock time is the other half of the story. Compare the time\n"
        "    column: large batches are dramatically cheaper *per sample*\n"
        "    because they hand BLAS bigger matrices. The practical choice is\n"
        "    the largest batch whose accuracy you can recover by tuning the\n"
        "    learning rate -- not the batch with the best accuracy at some\n"
        "    arbitrary fixed lr."
    )

    save("batch_size", fixed + scaled)

    if not args.no_plot:
        _plot(fixed, fixed_aggs, scaled, scaled_aggs)
        print("\n    artifacts/experiments/batch_size.png")
    print("    artifacts/experiments/batch_size.json")
    print()
    return 0


def _plot(fixed, fixed_aggs, scaled, scaled_aggs) -> None:
    plt, path = figure("batch_size")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    colors = plt.cm.cividis(np.linspace(0, 0.9, len(SIZES)))

    ax = axes[0]
    for b, color in zip(SIZES, colors):
        curve = mean_curve(fixed, f"batch {b}", "val_acc")
        if len(curve):
            ax.plot(range(1, len(curve) + 1), curve * 100, label=f"batch {b}",
                    color=color, marker="o", markersize=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy (%)")
    ax.set_title(f"Fixed lr={BASE_LR}: fewer updates, less progress")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for b, color in zip(SIZES, colors):
        label = f"batch {b} (lr×{b / BASE_BATCH:g})"
        curve = mean_curve(scaled, label, "val_acc")
        if len(curve):
            ax.plot(range(1, len(curve) + 1), curve * 100, label=f"batch {b}",
                    color=color, marker="o", markersize=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy (%)")
    ax.set_title("lr scaled with batch: the gap largely closes")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.errorbar(SIZES, [a.test_acc_mean * 100 for a in fixed_aggs],
                yerr=[a.test_acc_std * 100 for a in fixed_aggs],
                marker="o", capsize=4, label="fixed lr", color="#4c72b0")
    ax.errorbar(SIZES, [a.test_acc_mean * 100 for a in scaled_aggs],
                yerr=[a.test_acc_std * 100 for a in scaled_aggs],
                marker="s", capsize=4, label="scaled lr", color="#dd8452")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch size (log)")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Scaling the lr recovers most of the loss")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("Batch size changes gradient noise, update count, and speed at once",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
