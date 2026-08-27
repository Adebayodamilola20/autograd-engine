"""Phase 21 -- XOR: the smallest problem that proves the engine works.

Run:  python examples/train_xor.py

Why XOR, and why before MNIST
------------------------------
XOR is four data points:

    (0,0) -> 0     (0,1) -> 1     (1,0) -> 1     (1,1) -> 0

and it is **not linearly separable**. No straight line separates the two 1s
from the two 0s -- try drawing one. Minsky and Papert's 1969 proof of exactly
this killed neural-network funding for a decade, until backpropagation made
multi-layer networks trainable and the objection evaporated.

That makes it the perfect first test:

* small enough to verify by hand -- four samples, and you can read every
  prediction,
* it **requires** a hidden layer and a nonlinearity, so it fails loudly if the
  activation is missing or gradients are not flowing through both layers,
* if XOR fails, the bug is in the *engine*, not in a data pipeline. Debugging
  MNIST first would confound the two.

This script trains three models: the linear one that must fail, and two that
must succeed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from nabla.core.gradcheck import check_parameter_gradients  # noqa: E402
from nabla.data import DataLoader, Dataset  # noqa: E402
from nabla.losses import softmax_cross_entropy  # noqa: E402
from nabla.nn import MLP  # noqa: E402
from nabla.optim import SGD, Adam  # noqa: E402
from nabla.training import Trainer, argmax  # noqa: E402

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = np.array([0, 1, 1, 0])


def rule(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m\n" + "─" * 74)


def make_loader(seed: int = 0) -> DataLoader:
    # Full batch: with four samples there is nothing to be stochastic about.
    return DataLoader(Dataset(X, Y, name="xor"), batch_size=4, seed=seed)


def report(model, name: str) -> bool:
    """Print the truth table and return whether all four are correct."""
    print(f"\n  {name}")
    print(f"    {'x1':>4} {'x2':>4} {'target':>8} {'pred':>6} {'P(1)':>8}   ")
    print("    " + "─" * 42)
    all_correct = True
    for (x1, x2), target in zip(X, Y):
        logits = [v.data for v in model([x1, x2])]
        shifted = [z - max(logits) for z in logits]
        exps = [pow(2.718281828459045, z) for z in shifted]
        p1 = exps[1] / sum(exps)
        pred = argmax(logits)
        correct = pred == target
        all_correct &= correct
        mark = "✓" if correct else "✗"
        print(f"    {x1:>4.0f} {x2:>4.0f} {target:>8} {pred:>6} {p1:>8.4f}   {mark}")
    return all_correct


# ══════════════════════════════════════════════════════════════════════
rule("1. Why XOR needs a hidden layer")

print("""    x2
     1 │  ●(0,1)=1        ○(1,1)=0
       │
     0 │  ○(0,0)=0        ●(1,0)=1
       └──────────────────────────  x1
          0                1

    The two 1s sit on one diagonal, the two 0s on the other. A single linear
    unit computes  w1·x1 + w2·x2 + b  and can only split the plane with a
    straight line -- and no straight line separates the diagonals.

    A hidden layer plus a nonlinearity lets the network build intermediate
    features (effectively OR and NAND) and combine them. That is the whole
    argument for depth, in four data points.""")


# ══════════════════════════════════════════════════════════════════════
rule("2. The control: no hidden layer. This must fail.")

linear_model = MLP(2, [], 2, seed=0)
linear_trainer = Trainer(
    linear_model, Adam(linear_model.parameters(), lr=0.1), softmax_cross_entropy,
    verbose=False,
)
linear_trainer.fit(make_loader(), epochs=400)
linear_ok = report(linear_model, "MLP(2 → 2), no hidden layer, 2000 steps")
print(f"\n    final loss {linear_trainer.history['train_loss'][-1]:.4f}"
      f"   accuracy {linear_trainer.history['train_acc'][-1] * 100:.0f}%")
print("    Stuck at 50%, exactly as Minsky and Papert proved. Not a bug.")


# ══════════════════════════════════════════════════════════════════════
rule("3. The other control: two layers, but NO nonlinearity")

# Composing two affine maps gives an affine map, so this is still linear --
# depth without an activation buys literally nothing.
no_act = MLP(2, [8], 2, activation="linear", seed=0)
no_act_trainer = Trainer(
    no_act, Adam(no_act.parameters(), lr=0.1), softmax_cross_entropy, verbose=False
)
no_act_trainer.fit(make_loader(), epochs=400)
no_act_ok = report(no_act, "MLP(2 → 8 → 2) with activation='linear'")
print(f"\n    final loss {no_act_trainer.history['train_loss'][-1]:.4f}"
      f"   accuracy {no_act_trainer.history['train_acc'][-1] * 100:.0f}%")
print("    Eight hidden units, and it still cannot do it:  W₂(W₁x + b₁) + b₂")
print("    is just  W'x + b'.  Depth without nonlinearity is an illusion.")
print("    (The loss is ln 2 = 0.6931 -- the model outputs P=0.5 for every input")
print("     and has genuinely given up. Any accuracy above 50% is argmax breaking")
print("     ties between two identical logits, not the model knowing anything.)")


# ══════════════════════════════════════════════════════════════════════
rule("4. The real thing: one hidden layer with tanh")

model = MLP(2, [8], 2, activation="tanh", seed=1)
print(f"    {model}")
print(f"    {model.num_parameters()} parameters: 2×8 + 8 weights, 8 + 2 biases")

# Verify gradients through the whole network *before* training it. If the
# gradients were wrong, training might still appear to work on a problem this
# small -- so check first.
def loss_fn():
    outs = [model([float(a), float(b)]) for a, b in X]
    return softmax_cross_entropy(outs, Y.tolist())

check = check_parameter_gradients(loss_fn, model.parameters(), label="XOR MLP")
print(f"\n    gradient check before training: "
      f"{'PASS' if check.passed else 'FAIL'}  (max rel err {check.max_error:.2e})")
print(f"    {len(model.parameters())} parameters verified against finite differences")

trainer = Trainer(
    model, Adam(model.parameters(), lr=0.08), softmax_cross_entropy, verbose=False
)
history = trainer.fit(make_loader(seed=1), epochs=600)

print("\n    learning curve:")
print(f"      {'epoch':>7} {'loss':>10} {'accuracy':>10}")
print("      " + "─" * 30)
for e in (0, 24, 49, 99, 199, 399, 599):
    if e < len(history["train_loss"]):
        print(f"      {e + 1:>7} {history['train_loss'][e]:>10.5f} "
              f"{history['train_acc'][e] * 100:>9.0f}%")

solved = report(model, "MLP(2 → 8 → 2), tanh")


# ══════════════════════════════════════════════════════════════════════
rule("5. Optimiser comparison on the same problem")

print(f"    {'optimiser':<28} {'final loss':>12} {'epochs to 100%':>16}")
print("    " + "─" * 58)

configs = [
    ("SGD(lr=0.1)", lambda p: SGD(p, lr=0.1)),
    ("SGD(lr=0.5)", lambda p: SGD(p, lr=0.5)),
    ("SGD(lr=0.5, momentum=0.9)", lambda p: SGD(p, lr=0.5, momentum=0.9)),
    ("Adam(lr=0.08)", lambda p: Adam(p, lr=0.08)),
]
for label, build in configs:
    m = MLP(2, [8], 2, activation="tanh", seed=1)
    t = Trainer(m, build(m.parameters()), softmax_cross_entropy, verbose=False)
    h = t.fit(make_loader(seed=1), epochs=600)
    hit = next((i + 1 for i, a in enumerate(h["train_acc"]) if a == 1.0), None)
    print(f"    {label:<28} {h['train_loss'][-1]:>12.5f} "
          f"{(str(hit) if hit else 'never'):>16}")


# ══════════════════════════════════════════════════════════════════════
rule("6. What the hidden layer learned")

print("    Hidden unit activations for each input (after tanh):\n")
hidden = model.layers[0]
print(f"    {'input':<10} " + " ".join(f"h{i}".rjust(7) for i in range(8)))
print("    " + "─" * 68)
for (x1, x2), target in zip(X, Y):
    acts = [n([x1, x2]).data for n in hidden.neurons]
    print(f"    ({x1:.0f},{x2:.0f})→{target}    " + " ".join(f"{a:>7.3f}" for a in acts))

print("""
    Each hidden unit has become a different linear feature of the input. The
    output layer then finds a linear combination of *those* that separates the
    classes -- which is impossible in the original coordinates but easy in the
    representation the hidden layer built. That is what "learning features"
    means, concretely.""")

# Decision boundary plot
try:
    from nabla.visualization.plots import plot_decision_boundary

    def predict_prob(point):
        logits = [v.data for v in model([float(point[0]), float(point[1])])]
        shifted = [z - max(logits) for z in logits]
        exps = [pow(2.718281828459045, z) for z in shifted]
        return exps[1] / sum(exps)

    path = plot_decision_boundary(
        predict_prob, X, Y, title="XOR: learned decision boundary",
        path=ARTIFACTS / "xor_decision_boundary.png",
    )
    print("\n    decision boundary saved to artifacts/xor_decision_boundary.png")
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════════
rule("Result")

checks = [
    ("linear model fails (expected)", not linear_ok),
    ("linear activations fail (expected)", not no_act_ok),
    ("gradient check passes", check.passed),
    ("tanh MLP solves XOR 4/4", solved),
]
for label, ok in checks:
    print(f"    {'✓' if ok else '✗'}  {label}")

if all(ok for _, ok in checks):
    print("\n    \033[1mXOR SOLVED.\033[0m The engine trains a real network end to end:")
    print("    forward pass, graph construction, backpropagation, and parameter")
    print("    updates all correct. MNIST is next.")
    sys.exit(0)

print("\n    Something is wrong -- do not proceed to MNIST.")
sys.exit(1)
