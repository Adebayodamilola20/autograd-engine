"""Phase 5 demo -- verifying the engine against finite differences.

Run:  python examples/gradient_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nabla import Value  # noqa: E402
from nabla.core.gradcheck import (  # noqa: E402
    check_gradients,
    check_parameter_gradients,
    epsilon_sweep,
)


def rule(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m\n" + "─" * 78)


# ──────────────────────────────────────────────────────────────────────
rule("1. Every operation, checked against the definition of a derivative")

CHECKS = [
    ("x + y",              lambda x, y: x + y,              [2.0, 3.0]),
    ("x * y",              lambda x, y: x * y,              [2.0, 3.0]),
    ("x - y",              lambda x, y: x - y,              [2.0, 3.0]),
    ("x / y",              lambda x, y: x / y,              [8.0, 4.0]),
    ("-x",                 lambda x: -x,                    [2.5]),
    ("x ** 3",             lambda x: x**3,                  [2.0]),
    ("x ** -2",            lambda x: x**-2,                 [3.0]),
    ("x ** y",             lambda x, y: x**y,               [2.0, 3.0]),
    ("2 ** x",             lambda x: 2**x,                  [3.0]),
    ("exp(x)",             lambda x: x.exp(),               [1.5]),
    ("log(x)",             lambda x: x.log(),               [4.0]),
    ("tanh(x)",            lambda x: x.tanh(),              [0.8]),
    ("sigmoid(x)",         lambda x: x.sigmoid(),           [0.5]),
    ("relu(x)",            lambda x: x.relu(),              [1.7]),
]

print(f"  {'operation':<14} {'analytic':>18} {'numerical':>18} {'rel err':>11}   ok")
print("  " + "─" * 70)
all_passed = True
for label, fn, point in CHECKS:
    r = check_gradients(fn, point, label=label)
    all_passed &= r.passed
    print(
        f"  {label:<14} {r.analytic[0]:>18.12f} {r.numerical[0]:>18.12f} "
        f"{r.errors[0]:>11.3e}   {'✓' if r.passed else '✗'}"
    )
print(f"\n  {'ALL OPERATIONS VERIFIED' if all_passed else 'FAILURES PRESENT'}")
print("  (first input shown; every input of every check was compared)")


# ──────────────────────────────────────────────────────────────────────
rule("2. Composite expressions -- the chain rule composing itself")

COMPOSITE = [
    ("(x·y + b)²",          lambda x, y, b: (x * y + b) ** 2,        [2.0, -3.0, 1.0]),
    ("tanh(x·w + b)",       lambda x, w, b: (x * w + b).tanh(),      [1.5, 2.0, -0.5]),
    ("log(exp(x)+exp(y))",  lambda x, y: (x.exp() + y.exp()).log(),  [1.0, 2.0]),
    ("exp(-x²)",            lambda x: (-(x**2)).exp(),               [0.8]),
    ("x·tanh(x)",           lambda x: x * x.tanh(),                  [1.1]),
    ("x·x·x  (3 paths)",    lambda x: x * x * x,                     [2.0]),
    ("x/x    (cancels)",    lambda x: x / x,                         [4.0]),
]

for label, fn, point in COMPOSITE:
    r = check_gradients(fn, point, label=label)
    grads = "  ".join(f"{g:+.6f}" for g in r.analytic)
    print(f"  {label:<20} grad = [{grads}]   max err {r.max_error:.2e}  {'✓' if r else '✗'}")


# ──────────────────────────────────────────────────────────────────────
rule("3. The step-size trade-off, measured (docs/01 §3)")

print("  f(x) = x³ at x = 2, true derivative = 12\n")
print(f"  {'h':>10} {'central difference':>24} {'relative error':>18}")
print("  " + "─" * 56)

rows = epsilon_sweep(lambda x: x**3, [2.0])
best_h, best_err = min(((h, e) for h, _, e in rows), key=lambda t: t[1])
for h, estimate, err in rows:
    marker = "  ← best" if h == best_h else ""
    print(f"  {h:>10.0e} {estimate:>24.12f} {err:>18.3e}{marker}")

print(f"\n  Error falls like h² (truncation), bottoms out near h≈{best_h:.0e},")
print("  then climbs again as catastrophic cancellation takes over.")
print(f"  Best achievable relative error: {best_err:.1e} -- never the full 1e-16.")
print("  Reverse-mode autodiff has no such trade-off. That is why we test with")
print("  finite differences and train with autodiff.")


# ──────────────────────────────────────────────────────────────────────
rule("4. The checker can fail -- otherwise it proves nothing")


def square_with_wrong_gradient(x: Value) -> Value:
    """x², but the backward rule claims dx = x instead of 2x."""
    out = Value(x.data**2, (x,), "sq")

    def _backward():
        x.grad += 1.0 * x.data * out.grad  # BUG: missing the factor of 2

    out._backward = _backward
    return out


bad = check_gradients(square_with_wrong_gradient, [3.0], names=["x"], label="broken x²")
print("  " + str(bad).replace("\n", "\n  "))
print("\n  A missing factor of 2 -- the single most common autodiff bug -- caught")
print("  immediately, by a method that shares no code with the derivative.")


# ──────────────────────────────────────────────────────────────────────
rule("5. Gradient checking through an entire hand-built network")

# 2 inputs → 2 tanh units → 1 linear output, squared error. Nine parameters.
w = [Value(v) for v in (0.5, -0.3, 0.8, 0.2, -0.6, 0.4, 0.1, -0.9, 0.7)]
x1, x2, target = Value(0.6), Value(-0.4), 0.25


def loss_fn() -> Value:
    h1 = (x1 * w[0] + x2 * w[1] + w[2]).tanh()
    h2 = (x1 * w[3] + x2 * w[4] + w[5]).tanh()
    out = h1 * w[6] + h2 * w[7] + w[8]
    return (out - target) ** 2


net = check_parameter_gradients(
    loss_fn, w, names=[f"w{i}" for i in range(9)], label="2→2→1 network, MSE loss"
)
print("  " + str(net).replace("\n", "\n  "))
print("\n  Nine parameters, gradients flowing through two tanh units and a loss.")
print("  Cost: 1 backward pass (analytic) vs 18 forward passes (numerical).")
print("  For the real MNIST net that ratio becomes 1 vs 218,772.")

print()
