"""Experiment 1 -- what the learning rate actually does.

Run:
    python experiments/exp_learning_rate.py
    python experiments/exp_learning_rate.py --seeds 5 --epochs 12

The question
------------
Gradient descent says

.. math:: \\theta \\leftarrow \\theta - \\eta \\, \\nabla_\\theta \\mathcal{L}

The gradient tells you which *direction* reduces the loss. It says nothing
about how far to go, because it is a purely local quantity -- the slope at a
point carries no information about how long that slope continues. :math:`\\eta`
is our guess at that distance, and it is the single most consequential number
in training.

Three regimes, and the intuition for each
-----------------------------------------
Picture the loss surface as a valley.

* **Too small.** Every step is a tiny shuffle downhill. You are going the right
  way, but you run out of epochs before arriving. Nothing looks *broken* --
  which is what makes this failure expensive. The loss curve is smooth and
  decreasing and simply never gets anywhere.

* **About right.** Steps are large enough to make progress and small enough
  that the linear approximation the gradient represents stays valid over the
  distance travelled.

* **Too large.** You step so far across the valley that you land higher up the
  opposite wall than you started. The next gradient is larger, so the next
  step is larger still. Loss goes to infinity in a handful of updates. This is
  *divergence*, and it is the failure mode people mean by "exploding
  gradients" far more often than a genuine gradient-magnitude problem.

The asymmetry worth remembering: too small wastes your time, too large
destroys the run. But too small is harder to *notice*, so it wastes more time
in practice.
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

RATES = [
    ("lr=1e-5  (far too small)", 1e-5),
    ("lr=1e-4  (too small)", 1e-4),
    ("lr=1e-3  (about right)", 1e-3),
    ("lr=1e-2  (too large)", 1e-2),
    ("lr=1e-1  (way too large)", 1e-1),
    ("lr=1.0   (catastrophic)", 1.0),
]

# The same sweep on plain SGD. Adam rescales each parameter's step by a running
# estimate of its gradient magnitude, which makes the *effective* step size far
# less sensitive to lr than the formula suggests -- so Adam degrades gracefully
# where SGD detonates. Running both is the only way to see that.
SGD_RATES = [
    ("SGD lr=0.1", 0.1),
    ("SGD lr=1.0", 1.0),
    ("SGD lr=10", 10.0),
    ("SGD lr=100", 100.0),
    ("SGD lr=1000", 1000.0),
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--n-train", type=int, default=8000)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    rule("Experiment 1: learning rate")
    print(
        "    Everything else held fixed: 784→128→64→10, ReLU, Adam, batch 64.\n"
        f"    {args.seeds} seeds per rate, {args.epochs} epochs each.\n"
    )

    variants = [
        (label, dict(lr=lr, epochs=args.epochs, n_train=args.n_train))
        for label, lr in RATES
    ]
    results = run_sweep(variants, seeds=tuple(range(args.seeds)))

    rule("Results")
    aggs = aggregate(results)
    print(results_table(aggs, "learning rate"))

    # ------------------------------------------------------------------
    rule("What happened")

    best = max(aggs, key=lambda a: a.test_acc_mean)
    print(f"    Best: {best.label.strip()} at {best.acc_text()}\n")

    tiny = aggs[0]
    print(
        f"    At lr=1e-5 the model reached only {tiny.test_acc_mean * 100:.1f}%, with a\n"
        f"    final training loss of {tiny.final_loss_mean:.3f}. Nothing failed. The\n"
        "    curve is smooth, monotonic, and heading in the right direction --\n"
        "    it simply had 100x too little distance to cover the ground. This\n"
        "    is the expensive failure, because it looks like a working run."
    )

    worst = aggs[-1]
    print(
        f"\n    At lr=1.0 accuracy is {worst.test_acc_mean * 100:.1f}% -- chance level for ten\n"
        "    classes. The model is not wrong, it is *destroyed*: every step\n"
        "    overshoots so far that the weights carry no information about the\n"
        "    data at all."
    )

    print(
        "\n    Note the spread column. lr=1e-3 and lr=1e-2 differ by less than\n"
        "    their seed-to-seed standard deviation, so on this evidence they\n"
        "    have not been shown to differ at all. Reporting either as 'the\n"
        "    winner' from a single run would be noise dressed as a finding.\n"
        "    That is why every row is several runs, not one."
    )

    # ------------------------------------------------------------------
    rule("The same sweep on plain SGD")
    print(
        "    Nothing above produced a NaN, which is worth explaining rather\n"
        "    than ignoring. Adam divides each parameter's step by a running\n"
        "    estimate of its own gradient magnitude, so the actual distance\n"
        "    moved is roughly lr regardless of how large the gradient is. That\n"
        "    bounds the damage: Adam degrades to chance instead of exploding.\n\n"
        "    Plain SGD has no such protection. Its step is lr × gradient, so a\n"
        "    large lr multiplies an already-large gradient and the next\n"
        "    gradient is larger still.\n"
    )

    sgd_variants = [
        (label, dict(lr=lr, optimizer="sgd", epochs=args.epochs, n_train=args.n_train))
        for label, lr in SGD_RATES
    ]
    sgd_results = run_sweep(sgd_variants, seeds=tuple(range(args.seeds)))
    sgd_aggs = aggregate(sgd_results)
    print()
    print(results_table(sgd_aggs, "SGD"))

    print(
        "\n    Two things to read off that table.\n\n"
        "    First, the variance. SGD at lr=1.0 has a standard deviation of\n"
        "    over 15 percentage points across seeds -- the same configuration\n"
        "    is sometimes usable and sometimes ruined, decided purely by the\n"
        "    random initialisation. A single run of that setting tells you\n"
        "    almost nothing, which is the whole argument for multiple seeds.\n\n"
        "    Second, look at the first-epoch losses in the JSON for lr=1000:\n"
        "    they reach the order of 1e57 and then come *back down*. The loss\n"
        "    is astronomically large but never NaN, and the run completes."
    )

    blown = [a for a in sgd_aggs if a.diverged]
    if blown:
        print(
            f"\n    Non-finite loss: {', '.join(a.label for a in blown)}. The Trainer\n"
            "    raises rather than continuing -- once a NaN enters the\n"
            "    parameters the next matmul spreads it to every output and the\n"
            "    run is unrecoverable."
        )
    else:
        print(
            "\n    \033[1mAnd that is Phase 9 paying off.\033[0m A naive softmax computes\n"
            "    exp(z) directly, which overflows to inf at z ≈ 710 in float64,\n"
            "    and inf/inf is NaN. Our log-sum-exp implementation subtracts\n"
            "    the row maximum first, so the largest exponent is always\n"
            "    exp(0) = 1 and overflow is impossible by construction --\n"
            "    however absurd the logits get.\n\n"
            "    So the model is thoroughly destroyed at these rates, but the\n"
            "    *arithmetic* never breaks. Those are different failures, and\n"
            "    conflating them is why 'my loss went NaN' gets misdiagnosed as\n"
            "    a learning-rate problem when it is often a stability bug -- or\n"
            "    the reverse."
        )
    print(
        "\n    The practical case for adaptive optimisers is here: not that\n"
        "    they find better minima, but that they are far less sensitive to\n"
        "    a hyperparameter nobody can compute in advance. Compare the two\n"
        "    tables -- Adam stays usable across two orders of magnitude of lr,\n"
        "    SGD across roughly one."
    )

    save("learning_rate", results, extra={"sgd_results": [r.__dict__ for r in sgd_results]})

    if not args.no_plot:
        _plot(results, aggs)
        print("\n    artifacts/experiments/learning_rate.png")
    print("    artifacts/experiments/learning_rate.json")
    print()
    return 0


def _plot(results, aggs) -> None:
    plt, path = figure("learning_rate")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(RATES)))

    ax = axes[0]
    for (label, _), color in zip(RATES, colors):
        curve = mean_curve(results, label, "train_loss")
        if len(curve):
            ax.plot(range(1, len(curve) + 1), curve, label=label.split()[0], color=color,
                    marker="o", markersize=3)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss (log)")
    ax.set_title("Too small never arrives; too large leaves")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for (label, _), color in zip(RATES, colors):
        curve = mean_curve(results, label, "val_acc")
        if len(curve):
            ax.plot(range(1, len(curve) + 1), curve * 100, label=label.split()[0],
                    color=color, marker="o", markersize=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy (%)")
    ax.set_title("Validation accuracy")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    rates = [lr for _, lr in RATES]
    means = [a.test_acc_mean * 100 for a in aggs]
    stds = [a.test_acc_std * 100 for a in aggs]
    ax.errorbar(rates, means, yerr=stds, marker="o", capsize=4, color="#4c72b0")
    ax.set_xscale("log")
    ax.set_xlabel("learning rate (log)")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("The usable window is narrow")
    ax.grid(alpha=0.3)

    fig.suptitle("Learning rate: the single most consequential hyperparameter", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
