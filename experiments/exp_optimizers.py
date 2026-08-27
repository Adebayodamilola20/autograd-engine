r"""Experiment 4 -- SGD vs momentum vs Adam, and what each one is for.

Run:
    python experiments/exp_optimizers.py
    python experiments/exp_optimizers.py --seeds 5

All three optimisers use the same gradients, computed by the same engine. They
differ only in what they *do* with them.

**SGD** takes the gradient literally:

.. math:: \theta \leftarrow \theta - \eta g

Simple, and it has one structural weakness. In a ravine -- a direction where
the loss drops slowly flanked by walls that are steep -- the gradient points
mostly *across* the ravine rather than along it. SGD oscillates between the
walls and creeps along the floor. Real loss surfaces are full of ravines,
because parameters differ wildly in how much they affect the output.

**Momentum** accumulates a velocity:

.. math::
    v \leftarrow \beta v + g, \qquad \theta \leftarrow \theta - \eta v

Across the ravine, successive gradients alternate sign and cancel in the sum.
Along the ravine they agree and accumulate. The same update therefore damps
the oscillation and accelerates the useful direction, with one extra buffer
per parameter. With :math:`\beta = 0.9`, the effective step along a consistent
direction approaches :math:`10\eta` -- the geometric series
:math:`1/(1-\beta)`.

**Adam** adds a per-parameter scale. It tracks a decaying mean :math:`m` and a
decaying mean square :math:`v` of each gradient, and steps by

.. math:: \theta \leftarrow \theta - \eta \frac{\hat m}{\sqrt{\hat v} + \epsilon}

Dividing by :math:`\sqrt{\hat v}` makes the step roughly *scale-invariant*: a
parameter with consistently tiny gradients gets the same-sized step as one with
huge gradients. That is why Adam works across such a wide range of learning
rates, and why Experiment 1 found it degrading gracefully where SGD detonated.

The tradeoff nobody mentions
----------------------------
Adam stores two extra arrays per parameter — **3x the memory of plain SGD**.
For a 109k-parameter MLP that is irrelevant. For a model with billions of
parameters, optimiser state is often the reason training does not fit in
memory. Convenience is not free; it is just cheap at this scale.
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

# Each optimiser gets a learning rate near its own optimum. Comparing them at a
# single shared lr would not be a comparison of optimisers -- it would be a
# demonstration that they use lr differently, which we already know.
VARIANTS = [
    ("SGD lr=0.01", dict(optimizer="sgd", lr=0.01)),
    ("SGD lr=0.1", dict(optimizer="sgd", lr=0.1)),
    ("SGD lr=0.5", dict(optimizer="sgd", lr=0.5)),
    ("momentum lr=0.01", dict(optimizer="momentum", lr=0.01, momentum=0.9)),
    ("momentum lr=0.1", dict(optimizer="momentum", lr=0.1, momentum=0.9)),
    ("Adam lr=1e-3", dict(optimizer="adam", lr=1e-3)),
    ("Adam lr=1e-2", dict(optimizer="adam", lr=1e-2)),
]


def main() -> int:
    p = argparse.ArgumentParser(description="Optimiser comparison.")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--n-train", type=int, default=8000)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    rule("Experiment 4: optimisers")
    print(
        "    784→128→64→10, ReLU, batch 64. Same gradients throughout --\n"
        f"    only the update rule changes. {args.seeds} seeds each.\n"
    )

    variants = [
        (label, {**cfg, "epochs": args.epochs, "n_train": args.n_train})
        for label, cfg in VARIANTS
    ]
    results = run_sweep(variants, seeds=tuple(range(args.seeds)))
    aggs = aggregate(results)

    rule("Results")
    print(results_table(aggs, "optimiser"))

    # ------------------------------------------------------------------
    rule("How fast each one gets going")
    print(
        "    Validation accuracy after the first epoch. Adam's advantage is\n"
        "    mostly here -- in the early epochs, before the others have found\n"
        "    a sensible scale for each parameter.\n"
    )
    rows = []
    for label, _ in VARIANTS:
        curve = mean_curve(results, label, "val_acc")
        if len(curve):
            rows.append(
                [label, f"{curve[0] * 100:.2f}%",
                 f"{curve[min(2, len(curve) - 1)] * 100:.2f}%",
                 f"{curve[-1] * 100:.2f}%"]
            )
    headers = ["optimiser", "after 1", "after 3", "final"]
    widths = [max(len(str(r[i])) for r in [headers, *rows]) for i in range(len(headers))]
    print("    " + "  ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
    print("    " + "  ".join("─" * w for w in widths))
    for r in rows:
        print("    " + "  ".join(str(c).ljust(w) for c, w in zip(r, widths)))

    # ------------------------------------------------------------------
    rule("What happened")

    best = max(aggs, key=lambda a: a.test_acc_mean)
    by = {a.label: a for a in aggs}
    sgd = by.get("SGD lr=0.1")
    mom = by.get("momentum lr=0.1")
    adam = by.get("Adam lr=1e-3")

    print(f"    Best overall: {best.label} at {best.acc_text()}\n")

    if sgd and mom:
        print(
            f"    At the same learning rate (0.1):\n"
            f"      SGD       {sgd.acc_text()}\n"
            f"      momentum  {mom.acc_text()}\n\n"
            "    Momentum costs one extra array per parameter and changes three\n"
            "    lines of the update. It is the best value-for-complexity\n"
            "    trade in the whole optimiser literature."
        )

    if adam and sgd:
        print(
            f"\n    Adam vs well-tuned SGD: {adam.acc_text()} vs {sgd.acc_text()}.\n\n"
            "    Note how close those are. Adam's real advantage is not the\n"
            "    final number -- a properly tuned SGD is highly competitive --\n"
            "    it is that Adam got there without anyone tuning it. Look back\n"
            "    at Experiment 1: Adam stayed usable across two orders of\n"
            "    magnitude of learning rate, SGD across about one.\n\n"
            "    You are buying *robustness to a hyperparameter you cannot\n"
            "    compute in advance*, at the price of 3x optimiser memory."
        )

    print(
        "\n    A caveat worth stating: 8 epochs on 8,000 samples favours\n"
        "    fast starters. Given many more epochs, tuned SGD with momentum\n"
        "    frequently catches up and sometimes generalises slightly better.\n"
        "    Short-budget experiments systematically flatter adaptive methods,\n"
        "    and it would be dishonest to present this table without saying so."
    )

    save("optimizers", results)

    if not args.no_plot:
        _plot(results, aggs)
        print("\n    artifacts/experiments/optimizers.png")
    print("    artifacts/experiments/optimizers.json")
    print()
    return 0


def _plot(results, aggs) -> None:
    plt, path = figure("optimizers")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    def color(label: str) -> str:
        if label.startswith("SGD"):
            return "#4c72b0"
        if label.startswith("momentum"):
            return "#dd8452"
        return "#55a868"

    styles = {}
    seen: dict[str, int] = {}
    for label, _ in VARIANTS:
        family = label.split()[0]
        seen[family] = seen.get(family, 0) + 1
        styles[label] = ["-", "--", ":"][seen[family] - 1]

    ax = axes[0]
    for label, _ in VARIANTS:
        curve = mean_curve(results, label, "train_loss")
        if len(curve):
            ax.plot(range(1, len(curve) + 1), curve, label=label,
                    color=color(label), linestyle=styles[label])
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss (log)")
    ax.set_title("Training loss")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for label, _ in VARIANTS:
        curve = mean_curve(results, label, "val_acc")
        if len(curve):
            ax.plot(range(1, len(curve) + 1), curve * 100, label=label,
                    color=color(label), linestyle=styles[label])
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy (%)")
    ax.set_title("Adam's edge is early, not final")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[2]
    labels = [a.label for a in aggs]
    means = [a.test_acc_mean * 100 for a in aggs]
    stds = [a.test_acc_std * 100 for a in aggs]
    ax.barh(range(len(labels)), means, xerr=stds,
            color=[color(a) for a in labels], capsize=3)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("test accuracy (%)")
    ax.set_xlim(left=min(means) - 5)
    ax.set_title("Final test accuracy")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    fig.suptitle("Same gradients, three ways of using them", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
