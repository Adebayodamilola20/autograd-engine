r"""Experiment 3 -- how depth and width change what a network can learn.

Run:
    python experiments/exp_architecture.py
    python experiments/exp_architecture.py --seeds 5 --epochs 12

Two knobs, two different jobs
-----------------------------
**Width** (units per layer) sets how many features a layer can represent at
once. A layer of :math:`n` units computes :math:`n` different linear functions
of its input and passes each through the activation. More width means more
patterns detectable *in parallel*.

**Depth** (number of layers) sets how many times features can be *composed*.
Layer 1 sees pixels; layer 2 sees combinations of layer-1 features; layer 3
sees combinations of those. Depth builds hierarchy, and hierarchy is why a
network can express things a wide-but-shallow model cannot express compactly.

The universal approximation theorem says one hidden layer, made wide enough,
can approximate any continuous function arbitrarily well. That is true and
much less useful than it sounds: "wide enough" can mean exponentially many
units, whereas depth often achieves the same function with a polynomial
number. Depth is about *efficiency* of representation, not possibility.

What to expect, and the honest caveat
-------------------------------------
Both curves should rise and then flatten or fall. Two different causes:

* Adding capacity past what the data supports stops helping. 8,000 MNIST
  samples do not contain enough information to fit an arbitrarily large model.
* Depth eventually *hurts* for optimisation reasons -- more layers means a
  longer chain for the gradient to survive, which is what Experiment 2's
  gradient table was measuring.

Distinguishing "not enough capacity" from "capacity I failed to train" needs
the gradient statistics, not just the accuracy. Both are recorded here.
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
from nabla.nn.tensor_mlp import TensorMLP  # noqa: E402

DEPTHS = [
    ("0 hidden (linear model)", []),
    ("1 hidden", [128]),
    ("2 hidden", [128, 64]),
    ("3 hidden", [128, 96, 64]),
    ("4 hidden", [128, 96, 64, 48]),
    ("6 hidden", [128, 112, 96, 80, 64, 48]),
]

WIDTHS = [
    ("width 4", [4, 4]),
    ("width 16", [16, 16]),
    ("width 32", [32, 32]),
    ("width 64", [64, 64]),
    ("width 128", [128, 128]),
    ("width 256", [256, 256]),
]


def param_count(hidden: list[int]) -> int:
    return TensorMLP(784, hidden, 10, seed=0).num_parameters()


def main() -> int:
    p = argparse.ArgumentParser(description="Depth and width sweeps.")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--n-train", type=int, default=8000)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    common = dict(epochs=args.epochs, n_train=args.n_train)
    seeds = tuple(range(args.seeds))

    # ══════════════════════════════════════════════════════════════════
    rule("Experiment 3a: depth")
    print(
        "    Width held near-constant, layers added. ReLU, Adam, batch 64.\n"
        f"    {args.seeds} seeds each.\n"
    )
    depth_results = run_sweep(
        [(label, dict(hidden=h, **common)) for label, h in DEPTHS], seeds=seeds
    )
    depth_aggs = aggregate(depth_results)

    print()
    print(results_table(depth_aggs, "depth"))
    print("\n    parameter counts:")
    for label, hidden in DEPTHS:
        print(f"      {label:<26} {param_count(hidden):>9,}")

    best_depth = max(depth_aggs, key=lambda a: a.test_acc_mean)
    linear_depth = depth_aggs[0]
    print(
        f"\n    Best depth: {best_depth.label} at {best_depth.acc_text()}\n"
        f"    No hidden layer at all: {linear_depth.acc_text()}\n\n"
        "    The first hidden layer buys the largest single improvement --\n"
        "    it is the one that introduces a nonlinearity where there was\n"
        "    none. Every layer after it is refining a hierarchy that already\n"
        "    exists, and the returns diminish quickly at this scale.\n\n"
        "    Watch the deepest rows especially. If accuracy falls there while\n"
        "    parameter count rises, the model has *more* capacity and *worse*\n"
        "    results -- which can only be an optimisation failure, not a\n"
        "    representational one. The gradient columns below say which."
    )

    # ══════════════════════════════════════════════════════════════════
    rule("Experiment 3b: width")
    print(
        "    Two hidden layers throughout, units per layer varied.\n"
    )
    width_results = run_sweep(
        [(label, dict(hidden=h, **common)) for label, h in WIDTHS], seeds=seeds
    )
    width_aggs = aggregate(width_results)

    print()
    print(results_table(width_aggs, "width"))
    print("\n    parameter counts:")
    for label, hidden in WIDTHS:
        print(f"      {label:<26} {param_count(hidden):>9,}")

    narrow, wide = width_aggs[0], width_aggs[-1]
    print(
        f"\n    {narrow.label}: {narrow.acc_text()}   ({param_count(WIDTHS[0][1]):,} params)\n"
        f"    {wide.label}: {wide.acc_text()}   ({param_count(WIDTHS[-1][1]):,} params)\n\n"
        "    Four units per layer is a genuine bottleneck: everything the\n"
        "    network knows about a digit has to pass through 4 numbers. Ten\n"
        "    classes cannot be cleanly separated by that, and no amount of\n"
        "    training fixes it -- this is a capacity failure, the one case\n"
        "    where 'the model is too small' is literally true.\n\n"
        "    Past ~64 units the curve flattens hard. A 60x increase in\n"
        "    parameters from width 32 to width 256 buys very little, because\n"
        "    the limit stopped being the model and became the data."
    )

    # ══════════════════════════════════════════════════════════════════
    rule("Gradient survival with depth")
    print(
        "    Mean |∂L/∂W| at the first layer, final epoch. The deeper the\n"
        "    network, the further the gradient has to travel to get there.\n"
    )
    for label, _ in DEPTHS:
        runs = [r for r in depth_results if r.label == label and r.per_layer_grad]
        if not runs:
            continue
        first = np.mean([r.per_layer_grad[-1][0] for r in runs])
        last = np.mean([r.per_layer_grad[-1][-1] for r in runs])
        n_layers = len(runs[0].per_layer_grad[-1])
        print(
            f"      {label:<26} layers={n_layers}  first={first:.2e}  "
            f"last={last:.2e}  ratio={last / first if first else float('inf'):>8,.1f}x"
        )
    print(
        "\n    ReLU keeps this ratio manageable -- its derivative is exactly 1\n"
        "    on the active path, so depth costs the gradient nothing in\n"
        "    principle. The growth you do see comes from the weight matrices\n"
        "    themselves, not the activation. Swap in sigmoid (Experiment 2)\n"
        "    and the same depths become untrainable."
    )

    save("architecture", depth_results + width_results)

    if not args.no_plot:
        _plot(depth_results, depth_aggs, width_results, width_aggs)
        print("\n    artifacts/experiments/architecture.png")
    print("    artifacts/experiments/architecture.json")
    print()
    return 0


def _plot(depth_results, depth_aggs, width_results, width_aggs) -> None:
    plt, path = figure("architecture")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0][0]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(DEPTHS)))
    for (label, _), color in zip(DEPTHS, colors):
        curve = mean_curve(depth_results, label, "val_acc")
        if len(curve):
            ax.plot(range(1, len(curve) + 1), curve * 100, label=label, color=color,
                    marker="o", markersize=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy (%)")
    ax.set_title("Depth: validation accuracy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    counts = [param_count(h) for _, h in DEPTHS]
    means = [a.test_acc_mean * 100 for a in depth_aggs]
    stds = [a.test_acc_std * 100 for a in depth_aggs]
    ax.errorbar(range(len(DEPTHS)), means, yerr=stds, marker="o", capsize=4,
                color="#4c72b0")
    ax.set_xticks(range(len(DEPTHS)))
    ax.set_xticklabels([str(len(h)) for _, h in DEPTHS])
    ax.set_xlabel("hidden layers")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Depth: the first layer matters most")
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    colors = plt.cm.plasma(np.linspace(0, 0.85, len(WIDTHS)))
    for (label, _), color in zip(WIDTHS, colors):
        curve = mean_curve(width_results, label, "val_acc")
        if len(curve):
            ax.plot(range(1, len(curve) + 1), curve * 100, label=label, color=color,
                    marker="o", markersize=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy (%)")
    ax.set_title("Width: validation accuracy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    counts = [param_count(h) for _, h in WIDTHS]
    means = [a.test_acc_mean * 100 for a in width_aggs]
    stds = [a.test_acc_std * 100 for a in width_aggs]
    ax.errorbar(counts, means, yerr=stds, marker="o", capsize=4, color="#dd8452")
    ax.set_xscale("log")
    ax.set_xlabel("parameters (log)")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Width: sharp returns, then a plateau")
    ax.grid(alpha=0.3)

    fig.suptitle("Depth composes features; width detects them in parallel", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
