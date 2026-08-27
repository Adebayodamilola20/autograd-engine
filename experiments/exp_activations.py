r"""Experiment 2 -- activations, and why a nonlinearity is not optional.

Run:
    python experiments/exp_activations.py
    python experiments/exp_activations.py --seeds 5

The claim to test
-----------------
A network without a nonlinearity between its layers is **exactly equivalent to
a single linear layer**, no matter how many layers it has. Not "roughly", not
"almost" -- exactly.

The proof is two lines. A linear layer is :math:`f(x) = xW + b`. Compose two:

.. math::
    f_2(f_1(x)) = (xW_1 + b_1)W_2 + b_2
                = x\underbrace{(W_1W_2)}_{W'} + \underbrace{(b_1W_2 + b_2)}_{b'}

which is a single linear layer with :math:`W' = W_1W_2` and
:math:`b' = b_1W_2 + b_2`. Stack a hundred and the argument repeats a hundred
times. Depth buys **nothing** without a nonlinearity in between.

So a 784→128→64→10 network with linear activations has 109,386 parameters but
only 7,850 *effective* degrees of freedom -- the size of a single 784→10 map.
The extra parameters are not merely useless, they are a redundant
parameterisation of the same small function class.

MNIST is nearly linearly separable, so this failure is quiet: the linear model
still reaches ~92%. The gap to a real network is the part depth was supposed to
provide, and it only exists because of the activation function.

The three activations
---------------------
* **sigmoid** :math:`\sigma(z) = 1/(1+e^{-z})`, into (0, 1).
  :math:`\sigma' = \sigma(1-\sigma)`, which **peaks at 0.25**. Every layer
  multiplies the backward signal by at most a quarter, so gradient magnitude
  falls by :math:`4^{-\text{depth}}` in the worst case. That is the vanishing
  gradient problem, and it is a property of the derivative, not a bug.

* **tanh** into (-1, 1), with :math:`\tanh' = 1 - \tanh^2`, peaking at **1.0**.
  Four times better than sigmoid at the peak, and zero-centred, which keeps
  activations from drifting to one side. Strictly better than sigmoid for
  hidden layers; there is no reason to prefer sigmoid inside a network.

* **ReLU** :math:`\max(0, z)`, derivative exactly **1** for :math:`z > 0` and
  **0** otherwise. The gradient does not shrink with depth on the active path
  at all -- 1 multiplied by itself any number of times is 1. That single fact
  is most of why deep networks became trainable. The cost is that units with
  :math:`z < 0` pass *no* gradient, and one that never activates never
  recovers: a "dead" unit.
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

ACTIVATIONS = ["relu", "tanh", "sigmoid", "linear"]


def main() -> int:
    p = argparse.ArgumentParser(description="Activation function comparison.")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--n-train", type=int, default=8000)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    rule("Experiment 2: activation functions")
    print(
        "    784→128→64→10, Adam lr=1e-3, batch 64. Only the hidden-layer\n"
        f"    activation changes. {args.seeds} seeds each.\n"
    )

    variants = [
        (name, dict(activation=name, epochs=args.epochs, n_train=args.n_train))
        for name in ACTIVATIONS
    ]
    results = run_sweep(variants, seeds=tuple(range(args.seeds)))
    aggs = aggregate(results)

    rule("Results")
    print(results_table(aggs, "activation"))

    # ------------------------------------------------------------------
    rule("Gradient magnitude reaching the first layer")
    print(
        "    Mean |∂L/∂W| per layer at the final epoch. Layer 0 is the input\n"
        "    layer -- the furthest from the loss, and the last to receive any\n"
        "    signal. This is where vanishing gradients show up first.\n"
    )

    rows = []
    for name in ACTIVATIONS:
        runs = [r for r in results if r.label == name and r.per_layer_grad]
        if not runs:
            continue
        final = np.mean([r.per_layer_grad[-1] for r in runs], axis=0)
        ratio = final[-1] / final[0] if final[0] > 0 else float("inf")
        rows.append([name, *[f"{g:.2e}" for g in final], f"{ratio:,.1f}x"])

    headers = ["activation", "layer 0", "layer 1", "layer 2", "last/first"]
    widths = [max(len(str(r[i])) for r in [headers, *rows]) for i in range(len(headers))]
    print("    " + "  ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
    print("    " + "  ".join("─" * w for w in widths))
    for r in rows:
        print("    " + "  ".join(str(c).ljust(w) for c, w in zip(r, widths)))

    # ------------------------------------------------------------------
    rule("What happened")

    by_name = {a.label: a for a in aggs}
    linear = by_name.get("linear")
    relu = by_name.get("relu")
    sigmoid = by_name.get("sigmoid")

    if linear and relu:
        gap = (relu.test_acc_mean - linear.test_acc_mean) * 100
        print(
            f"    linear activations: {linear.acc_text()}\n"
            f"    ReLU:               {relu.acc_text()}\n\n"
            f"    A {gap:.1f} point gap, from adding a function with no parameters\n"
            "    at all. The linear network has all 109,386 weights and the\n"
            "    representational power of 7,850 of them, because the algebra\n"
            "    above collapses it to a single 784→10 map.\n\n"
            "    Note it still reaches a respectable score. MNIST is close to\n"
            "    linearly separable, so this failure does not announce itself\n"
            "    -- it just quietly caps you below what the architecture on\n"
            "    paper should reach. On a genuinely nonlinear problem (XOR, in\n"
            "    examples/train_xor.py) the same model cannot get above 50%."
        )

    if sigmoid and rows:
        sig_row = next((r for r in rows if r[0] == "sigmoid"), None)
        relu_row = next((r for r in rows if r[0] == "relu"), None)
        if sig_row and relu_row:
            print(
                f"\n    Gradient decay from last layer to first:\n"
                f"      sigmoid  {sig_row[-1]}\n"
                f"      ReLU     {relu_row[-1]}\n\n"
                "    sigmoid' peaks at 0.25, so each layer can attenuate the\n"
                "    backward signal fourfold before anything else happens.\n"
                "    ReLU' is exactly 1 on the active path, so depth costs the\n"
                "    gradient nothing. At three layers this is a nuisance; at\n"
                "    thirty it is the difference between training and not."
            )

    save("activations", results)

    if not args.no_plot:
        _plot(results, aggs, rows)
        print("\n    artifacts/experiments/activations.png")
    print("    artifacts/experiments/activations.json")
    print()
    return 0


def _plot(results, aggs, grad_rows) -> None:
    plt, path = figure("activations")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    colors = {"relu": "#4c72b0", "tanh": "#dd8452", "sigmoid": "#55a868",
              "linear": "#c44e52"}

    ax = axes[0]
    for name in ACTIVATIONS:
        curve = mean_curve(results, name, "train_loss")
        if len(curve):
            ax.plot(range(1, len(curve) + 1), curve, label=name, color=colors[name],
                    marker="o", markersize=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    ax.set_title("Training loss")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    for name in ACTIVATIONS:
        curve = mean_curve(results, name, "val_acc")
        if len(curve):
            ax.plot(range(1, len(curve) + 1), curve * 100, label=name,
                    color=colors[name], marker="o", markersize=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy (%)")
    ax.set_title("linear plateaus early -- depth is doing nothing")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2]
    for name in ACTIVATIONS:
        runs = [r for r in results if r.label == name and r.per_layer_grad]
        if not runs:
            continue
        final = np.mean([r.per_layer_grad[-1] for r in runs], axis=0)
        ax.plot(range(len(final)), final, marker="o", label=name, color=colors[name])
    ax.set_yscale("log")
    ax.set_xlabel("layer (0 = input side, furthest from the loss)")
    ax.set_ylabel("mean |gradient| (log)")
    ax.set_title("How much signal reaches each layer")
    ax.set_xticks(range(3))
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("Activation functions: the nonlinearity is the whole point", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
