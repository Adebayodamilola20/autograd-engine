"""Phases 3 & 4 demo -- the graph, the ordering, and the backward pass traced
one node at a time.

Run:  python examples/backprop_step_by_step.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nabla import Value  # noqa: E402
from nabla.core.graph import (  # noqa: E402
    graph_size,
    is_topologically_sorted,
    topological_sort,
)


def rule(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m\n" + "─" * 78)


def name(node: Value) -> str:
    return node.label or f"{node._op or 'const'}({node.data:g})"


def build():
    """L = (x·w + b)²  with x=2, w=-3, b=1."""
    x = Value(2.0, label="x")
    w = Value(-3.0, label="w")
    b = Value(1.0, label="b")
    u = x * w
    u.label = "u"
    v = u + b
    v.label = "v"
    L = v**2
    L.label = "L"
    return L, {"x": x, "w": w, "b": b, "u": u, "v": v, "L": L}


# ──────────────────────────────────────────────────────────────────────
rule("1. The graph that ordinary Python built")

L, nodes = build()

print("""      x=2.0 ──┐
              ├──[ * ]──> u=-6.0 ──┐
      w=-3.0 ─┘                    ├──[ + ]──> v=-5.0 ──[ **2 ]──> L=25.0
                      b=1.0 ───────┘
""")
stats = graph_size(L)
print(f"  {stats['nodes']} nodes, {stats['edges']} edges, "
      f"{stats['leaves']} leaves, depth {stats['depth']}")
print("  Nobody declared this graph. It is a side effect of evaluating")
print("  `(x * w + b) ** 2` -- which is what 'define-by-run' means.")


# ──────────────────────────────────────────────────────────────────────
rule("2. Topological order: every node after all of its parents")

order = topological_sort(L)
print("  forward order  (parents first): " + " → ".join(name(n) for n in order))
print("  backward order (children first): " + " → ".join(name(n) for n in reversed(order)))
print(f"\n  valid topological ordering? {is_topologically_sorted(order)}")
print(f"  is the reverse also valid?  {is_topologically_sorted(list(reversed(order)))}"
      "   ← must be False, or the check is vacuous")


# ──────────────────────────────────────────────────────────────────────
rule("3. The backward pass, one node at a time")

for n in topological_sort(L):
    n.grad = 0.0

print("  seed: ∂L/∂L = 1  (the output is perfectly sensitive to itself)\n")
L.grad = 1.0

print(f"  {'step':<5} {'node':<6} {'op':<8} {'its grad':>10}   pushes to parents")
print("  " + "─" * 68)

for step, node in enumerate(reversed(topological_sort(L)), start=1):
    before = {id(p): p.grad for p in node._prev}
    node._backward()
    pushes = ", ".join(
        f"{name(p)} {before[id(p)]:+g} → {p.grad:+g}" for p in node._prev
    ) or "— (leaf, nothing to push)"
    print(f"  {step:<5} {name(node):<6} {node._op or 'leaf':<8} {node.grad:>10.4g}   {pushes}")

print("\n  final gradients:")
for key in ("x", "w", "b"):
    print(f"    ∂L/∂{key} = {nodes[key].grad:+.1f}")
print("\n  by hand:  ∂L/∂x = 2(xw+b)·w = +30    ∂L/∂w = 2(xw+b)·x = -20"
      "    ∂L/∂b = 2(xw+b) = -10")


# ──────────────────────────────────────────────────────────────────────
rule("4. Why the order matters: the same graph, traversed wrongly")


def backward_in_order(root: Value, sequence: list[Value]) -> None:
    for n in topological_sort(root):
        n.grad = 0.0
    root.grad = 1.0
    for n in sequence:
        n._backward()


# A graph where x reaches L by two separate paths:  L = (x+1)·(x+2)
# dL/dx = (x+2) + (x+1) = 2x + 3 = 9 at x = 3.
x = Value(3.0, label="x")
p = x + 1.0
p.label = "p"
q = x + 2.0
q.label = "q"
out = p * q
out.label = "L"

print("  L = (x+1)·(x+2) at x=3.  Two paths reach x, so dL/dx = 2x+3 = 9.\n")

scenarios = [
    ("reverse topological (correct)", list(reversed(topological_sort(out)))),
    ("forward topological", topological_sort(out)),
    ("p processed before L", [p] + [n for n in reversed(topological_sort(out)) if n is not p]),
]

print(f"  {'traversal order':<32} {'x.grad':>10}   verdict")
print("  " + "─" * 68)
for label, sequence in scenarios:
    backward_in_order(out, sequence)
    ok = abs(x.grad - 9.0) < 1e-12
    note = "correct" if ok else f"WRONG (lost {9.0 - x.grad:+g} of the gradient)"
    print(f"  {label:<32} {x.grad:>10.4f}   {note}")

print("""
  Row 2: walking forwards, every node's gradient is still 0 when it is asked
         to push, so nothing propagates at all.
  Row 3: p pushes to x before L has told p what its gradient is. p contributes
         nothing, and only q's path survives -- 4 instead of 9. Silently wrong,
         with no error and no NaN.

  That is the whole reason `backward()` sorts before it propagates.""")


# ──────────────────────────────────────────────────────────────────────
rule("5. Accumulation: the other half of correctness")

y = Value(3.0, label="y")
z = y * y
z.backward()
print(f"  L = y·y at y=3.  One node, two edges into the same multiply.")
print(f"    edge 1 contributes ∂L/∂y = y = 3")
print(f"    edge 2 contributes ∂L/∂y = y = 3")
print(f"    with '+='  →  y.grad = {y.grad}   ✓ (2y = 6)")
print(f"    with '='   →  y.grad = 3.0   ✗ silently half")
print("""
  Ordering gets the *timing* right: a node is processed only after all of its
  consumers. Accumulation gets the *completeness* right: when it is processed,
  it has collected every contribution. You need both, and together they are
  correct on any DAG.""")

print()
