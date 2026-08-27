"""Phase 2 demo -- the engine doing what it was built to do.

Run:  python examples/basic_autograd.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nabla import Value, format_graph, graph_size  # noqa: E402


def rule(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m\n" + "─" * 72)


# ──────────────────────────────────────────────────────────────────────
rule("1. The API from the brief")

a = Value(2.0, label="a")
b = Value(3.0, label="b")

c = a * b
d = c + a

d.backward()

print(f"  a.data = {a.data}      a.grad = {a.grad}   (∂d/∂a = b + 1 = 4)")
print(f"  b.data = {b.data}      b.grad = {b.grad}   (∂d/∂b = a     = 2)")
print(f"  c = a*b   -> {c!r}")
print(f"  d = c+a   -> {d!r}")


# ──────────────────────────────────────────────────────────────────────
rule("2. The running example:  L = (x·w + b)²   with x=2, w=-3, b=1")

x = Value(2.0, label="x")
w = Value(-3.0, label="w")
bias = Value(1.0, label="b")

u = x * w
u.label = "u"
v = u + bias
v.label = "v"
L = v**2
L.label = "L"

print(f"  forward:  u = xw = {u.data:>5.1f}   v = u+b = {v.data:>5.1f}   L = v² = {L.data:>5.1f}")
L.backward()
print(f"  backward: L̄ = {L.grad:>4.1f}   v̄ = 2v = {v.grad:>6.1f}   ū = {u.grad:>6.1f}")
print(f"            x̄ = ū·w = {x.grad:>6.1f}   w̄ = ū·x = {w.grad:>6.1f}   b̄ = {bias.grad:>6.1f}")

print("\n  cross-check against pencil-and-paper calculus:")
print(f"    ∂L/∂x = 2(xw+b)·w = {2 * (2 * -3 + 1) * -3:>6.1f}   engine: {x.grad:>6.1f}  {'✓' if x.grad == 30 else '✗'}")
print(f"    ∂L/∂w = 2(xw+b)·x = {2 * (2 * -3 + 1) * 2:>6.1f}   engine: {w.grad:>6.1f}  {'✓' if w.grad == -20 else '✗'}")
print(f"    ∂L/∂b = 2(xw+b)   = {2 * (2 * -3 + 1):>6.1f}   engine: {bias.grad:>6.1f}  {'✓' if bias.grad == -10 else '✗'}")

print("\n  the graph the engine actually built:\n")
print("    " + format_graph(L).replace("\n", "\n    "))


# ──────────────────────────────────────────────────────────────────────
rule("3. The gradient *means* something: nudge w by ε, watch L move by ε·w̄")

for eps in (1e-1, 1e-3, 1e-6):
    before = (2.0 * -3.0 + 1.0) ** 2
    after = (2.0 * (-3.0 + eps) + 1.0) ** 2
    measured = (after - before) / eps
    print(f"    ε={eps:<8g}  ΔL/ε = {measured:>12.6f}   predicted = {w.grad:>6.1f}")
print("    (converging to -20 as ε shrinks -- this is what Phase 5 automates)")


# ──────────────────────────────────────────────────────────────────────
rule("4. Accumulation: the bug that only shows up when a value is reused")

p = Value(3.0, label="p")
(p * p).backward()
print(f"    L = p·p at p=3   ->   p.grad = {p.grad}   (correct: dL/dp = 2p = 6)")
print("    With '=' instead of '+=' this would read 3.0 -- silently half.")

q = Value(2.0, label="q")
L2 = 2 * q + q**3
L2.backward()
print(f"    L = 2q + q³ at q=2  ->  q.grad = {q.grad}   (correct: 2 + 3q² = 14)")


# ──────────────────────────────────────────────────────────────────────
rule("5. Every operation, and its local derivative, at one point")

point = 0.7
ops = [
    ("x + 2", lambda t: t + 2, 1.0),
    ("x * 3", lambda t: t * 3, 3.0),
    ("x - 1", lambda t: t - 1, 1.0),
    ("-x", lambda t: -t, -1.0),
    ("x / 4", lambda t: t / 4, 0.25),
    ("x ** 3", lambda t: t**3, 3 * point**2),
    ("2 ** x", lambda t: 2**t, 2**point * math.log(2)),
    ("exp(x)", lambda t: t.exp(), math.exp(point)),
    ("log(x)", lambda t: t.log(), 1 / point),
    ("tanh(x)", lambda t: t.tanh(), 1 - math.tanh(point) ** 2),
    ("sigmoid(x)", lambda t: t.sigmoid(), (lambda s: s * (1 - s))(1 / (1 + math.exp(-point)))),
    ("relu(x)", lambda t: t.relu(), 1.0),
]

print(f"  evaluated at x = {point}\n")
print(f"  {'expression':<12} {'forward':>12} {'engine grad':>14} {'by hand':>14}   ok")
print("  " + "─" * 68)
for name, fn, expected in ops:
    t = Value(point)
    out = fn(t)
    out.backward()
    ok = "✓" if abs(t.grad - expected) < 1e-12 else "✗"
    print(f"  {name:<12} {out.data:>12.8f} {t.grad:>14.8f} {expected:>14.8f}   {ok}")


# ──────────────────────────────────────────────────────────────────────
rule("6. Graph scale -- foreshadowing why PyTorch exists")

# One neuron with 784 inputs: the MNIST input layer, one unit.
inputs = [Value(0.1) for _ in range(784)]
weights = [Value(0.01) for _ in range(784)]
neuron = Value.sum(xi * wi for xi, wi in zip(inputs, weights)) + Value(0.0)
activated = neuron.tanh()

stats = graph_size(activated)
print("    ONE neuron with 784 inputs builds:")
print(f"      {stats['nodes']:>7,} nodes")
print(f"      {stats['edges']:>7,} edges")
print(f"      {stats['depth']:>7,} deep  ← the recursive topological sort would")
print("                        die here; ours is iterative (decision D5)")
print("\n    A 784→128→64→10 network has 109,386 parameters.")
print("    Every one of them is a Python object, in a graph rebuilt per sample.")
print("    That is the measurement that motivates Phase 18 (tensors).")

print()
