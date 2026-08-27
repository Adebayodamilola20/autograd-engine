# 5. Backpropagation

> Implemented in [`nabla/core/value.py`](../nabla/core/value.py) (`backward`).
> Tested by [`tests/test_backward.py`](../tests/test_backward.py) (49 tests).
> Narrated live by [`examples/backprop_step_by_step.py`](../examples/backprop_step_by_step.py).

Everything so far has been preparation. This is the algorithm.

---

## 1. The whole thing

```python
def backward(self):
    order = topological_sort(self)

    for node in order:              # clear intermediates (see §5)
        if node._prev:
            node.reset_grad()

    self.grad += 1.0                # ∂self/∂self = 1

    for node in reversed(order):
        node._backward()
```

That is backpropagation. Six lines. Everything else in this repository —
neurons, losses, optimisers, MNIST — is built on top of those six lines, and
PyTorch's `backward()` does the same thing with more engineering.

The steps:

1. **Topologically sort** the graph reachable from the output.
2. **Clear** intermediate gradients, keeping leaf totals.
3. **Seed** `∂L/∂L = 1`, because the output is perfectly sensitive to itself.
4. **Reverse** the order, so each node comes after all its consumers.
5. **Walk**, calling each node's local rule.
6. Each rule **accumulates** `local_derivative × out.grad` into its parents.

Steps 1 and 4 guarantee a gradient is *complete* before it is *used*. Step 6
guarantees it *collected everything*. Both are required; together they are
correct on any DAG.

## 2. Worked by hand: `L = (x·y + b)²`

Take `x = 2`, `y = 3`, `b = 1`.

**Forward:**

```
p = x·y = 6
s = p + b = 7
L = s²   = 49
```

**Backward**, seeding `L.grad = 1`:

| node | local rule | computation | result |
|---|---|---|---|
| `L` | `pow`: ∂L/∂s = 2s | `s.grad += 2(7)·1` | `s.grad = 14` |
| `s` | `add`: passes through | `p.grad += 14`, `b.grad += 14` | both `14` |
| `p` | `mul`: swaps operands | `x.grad += y·14 = 42`<br>`y.grad += x·14 = 28` | `42`, `28` |

**Check analytically.** $L = (xy+b)^2$, so

$$\frac{\partial L}{\partial x} = 2(xy+b)\cdot y = 2(7)(3) = 42 \;✓$$
$$\frac{\partial L}{\partial y} = 2(xy+b)\cdot x = 2(7)(2) = 28 \;✓$$
$$\frac{\partial L}{\partial b} = 2(xy+b) = 14 \;✓$$

The engine agrees. `tests/test_backward.py` runs exactly this case, plus a
dozen more with hand-computed answers.

## 3. Why one sweep is enough — the cheap gradient principle

This is the fact that makes deep learning computationally possible, and it
deserves more attention than it usually gets.

`backward()` is **O(V + E)**: one pass to sort, one to propagate. The cost of
computing the gradient with respect to *all* n parameters is a small constant
multiple of the cost of evaluating the function **once** — independent of n.

Compare the alternative. Finite differences need two function evaluations per
parameter. For our MNIST network:

| method | forward passes for a full gradient |
|---|---|
| finite differences | 2 × 109,386 = **218,772** |
| reverse-mode autodiff | **1** (plus a backward pass ≈ 2× forward) |

Roughly **70,000× cheaper**, and the ratio grows linearly with parameter count.
At GPT scale, with hundreds of billions of parameters, the gap is the
difference between "trains overnight" and "does not finish before the sun
burns out."

The asymmetry comes from the shape of the problem: a loss function has **many
inputs and one output**. Reverse mode computes one row of the Jacobian per
sweep, and for a scalar output there is only one row to compute. Forward mode
computes one *column* per sweep and would need 109,386 of them. That is
[chapter 1 §4-5](01-what-is-automatic-differentiation.md#4-what-reverse-mode-differentiation-is).

## 4. Accumulation semantics

Calling `backward()` twice **sums** two gradients into the leaves:

```python
x = Value(3.0)
u = x * x
L = u + Value(0.0)

L.backward();  x.grad   # 6.0
L.backward();  x.grad   # 12.0  -- accumulated
```

This mirrors PyTorch exactly, and it is not an accident of implementation — it
is the mechanism behind **gradient accumulation over micro-batches**. When a
batch does not fit in memory, you run several backward passes and step once;
the summing is the feature.

The consequence is the classic PyTorch bug: a training loop **must** call
`zero_grad()` each step, or step *n* uses the sum of gradients from steps
1…*n*. We reproduce this on purpose rather than hiding it, because it is a
thing worth understanding rather than being protected from.

## 5. The subtlety: intermediates must be cleared

This one is not obvious, and we got it wrong initially. `.grad` is quietly
doing **two jobs**:

- it **stores** the answer ∂L/∂node, and
- during the sweep it is the **carrier** that ferries gradient from a node down
  to its parents.

For leaves those jobs agree. For intermediates they conflict.

Run the example above without clearing. The first sweep leaves `u.grad = 1`.
The second sweep seeds `L.grad = 1`, and the `+` rule accumulates into `u`,
giving `u.grad = 2` — but that `2` is **pass-one residue, not a derivative**.
The `*` rule then reads it as its incoming gradient and pushes `2·(2·3) = 12`
into `x`, totalling **18** instead of the correct 12.

The resolution:

> Gradient flowing along an **edge** belongs to one sweep and must not outlive
> it. Gradient landing on a **leaf** is a running total. Only leaves survive.

Hence the clear-non-leaves step. PyTorch reaches the same place by a different
route: it never populates `.grad` on non-leaf tensors at all, holding edge
gradients in a temporary buffer that dies with the pass. That is why `.grad` is
`None` on intermediates there unless you ask for `retain_grad()`. We keep
intermediates inspectable — they are the whole point of a teaching engine — and
pay for it with one explicit reset.

**How this was caught:** the original accumulation test used `L = x*x`, which
has *no intermediate between the leaf and the output*, so there was nothing to
contaminate and the test passed against a broken engine. A tensor test that
happened to include a `.sum()` — one extra node — exposed it immediately.

The lesson generalises: **a test that exercises the minimum structure often
cannot see the bug.** `tests/test_backward.py` now pins this at depth 1, depth
6, and the leaf-as-output edge case.

## 6. Edge cases that are tested

| case | why it is dangerous |
|---|---|
| `d = a*b + a` | a variable on multiple paths — the accumulation bug |
| `x + x + x` | three contributions to one leaf |
| 100-deep chain | recursion limits, gradient decay |
| `relu(0.0)` | the non-differentiable kink |
| `x * 0` | gradient is zero but must still *flow* |
| unreached nodes | must keep `grad == 0`, not be skipped incorrectly |
| repeated `backward()` | §4 and §5 above |
| leaf as output | `backward()` on a node with no parents |

---

**Next:** [6. Gradient checking](06-gradient-checking.md) — how we know any of
this is true.
