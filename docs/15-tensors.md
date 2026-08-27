# 15. From scalars to tensors

> Implemented in [`nabla/core/tensor.py`](../nabla/core/tensor.py) and
> [`nabla/nn/tensor_mlp.py`](../nabla/nn/tensor_mlp.py).
> Tested by [`tests/test_tensor.py`](../tests/test_tensor.py) (89 tests).

This is the chapter that connects a micrograd-style toy to how PyTorch actually
works. It is one idea, and the idea is not mathematical.

---

## 1. The problem, in one number

A `784 → 128` layer on the scalar engine performs 100,352 multiplications and
100,352 additions **per sample**. Each builds a `Value`: an object allocation, a
tuple, and a closure.

One MNIST sample:

```
nodes:  328,804
edges:  437,012
depth:    1,000
```

Three hundred thousand Python objects, to classify one 28×28 image. Then
`backward()` walks every one of them, calling a Python closure at each.

The mathematics is correct. The *granularity* is wrong.

## 2. The idea

**Make a node a whole array instead of a single number.**

That is the entire change. Same chain rule, same topological sort, same
`backward()`, same accumulation — the algorithm in
[chapter 5](05-backpropagation.md) is untouched. Only what a node *contains*
changes.

```
scalar engine:  one node = one float,  layer = 200,000 nodes
tensor engine:  one node = an array,   layer = 2 nodes  (a matmul and an add)
```

The same MNIST computation, for a **batch of 64**:

```
nodes:  33
edges:  33
depth:  18
```

33 nodes instead of 328,804 per sample. That is **637,680× fewer objects** for
the same arithmetic and identical gradients.

## 3. What a matmul's backward rule is

The one derivation this chapter really needs. For $C = AB$:

Write it in index form, $C_{ij} = \sum_k A_{ik}B_{kj}$. Then
$\partial C_{ij}/\partial A_{mn} = \delta_{im}B_{nj}$, and the chain rule sums
over every output $C_{ij}$ that $A_{mn}$ influenced:

$$\bar{A}_{mn} = \sum_{ij}\bar{C}_{ij}\,\delta_{im}B_{nj} = \sum_j \bar{C}_{mj}B_{nj} = (\bar{C}B^\top)_{mn}$$

and symmetrically for $B$:

$$\boxed{\;\bar{A} = \bar{C}B^\top, \qquad \bar{B} = A^\top\bar{C}\;}$$

Two checks that make this memorable:

**The shapes force it.** $A$ is $(m,k)$, $B$ is $(k,n)$, $\bar C$ is $(m,n)$.
Then $\bar C B^\top$ is $(m,n)(n,k) = (m,k)$ ✓ and $A^\top\bar C$ is
$(k,m)(m,n) = (k,n)$ ✓. There is only one arrangement of transposes that
type-checks.

**It reduces to the scalar rule.** For $1\times1$ matrices this is
$\bar a = \bar c\,b$ and $\bar b = a\,\bar c$ — the multiplication rule from
[chapter 3](03-operators.md#multiplication), unchanged.

**The backward pass of a matmul is two more matmuls.** That is why a training
step costs roughly 3× a forward pass, and why GEMM performance determines nearly
everything about training speed.

The prediction is checkable, and it checks out. Measured forward+backward ÷
forward: **3.58× for us, 3.34× for PyTorch.** Two independent implementations
land on the same structural ratio, because it is a property of the mathematics
rather than of either codebase.

## 4. Broadcasting, and the rule that is easy to get wrong

The genuinely new thing tensors introduce.

`x @ W + b` adds a `(n_out,)` bias to a `(batch, n_out)` matrix. NumPy
broadcasts the bias across the batch — it is *reused* once per sample.

A value used multiple times accumulates gradient
([chapter 2](02-the-value-class.md#why-grad-uses--and-never-)). Broadcasting is
reuse. Therefore:

> **Broadcasting forward means summing backward.**

The bias gradient is the **sum over the batch axis**. Get this wrong — return a
`(batch, n_out)` gradient for a `(n_out,)` parameter — and shapes stop matching,
or worse, silently broadcast again and train something subtly incorrect.

`_unbroadcast(gradient, target_shape)` handles it generally: sum over any axis
that was broadcast, then reshape. It is the single most important helper in
`tensor.py`, and the most common source of bugs in a hand-written tensor engine.

## 5. Non-scalar outputs need a seed

`Value.backward()` needs no argument: $\partial L/\partial L = 1$ is always
well-defined because a `Value` holds one number.

For a tensor output with $m$ elements, "the derivative of self with respect to
self" is an $m \times m$ **identity Jacobian**, and reverse mode computes one
row per sweep. So you must say which combination you want:

```python
Tensor([[1.0, 2.0]]).backward()
# RuntimeError: backward() on a non-scalar Tensor of shape (1, 2)
# needs an explicit `gradient` -- the seed is a Jacobian row, not a number.
```

Passing a vector $v$ computes the **vector-Jacobian product** $v^\top J$ in one
sweep. That is exactly what `torch.Tensor.backward(gradient=...)` does — and why
PyTorch says *"grad can be implicitly created only for scalar outputs"* without
it. Having built this, that error message is no longer mysterious.

## 6. What did *not* change

The most informative part of this phase.

| reused unchanged | why it worked |
|---|---|
| `topological_sort` | only ever needed `_prev` |
| `Module`, `Parameter` | only needed a "learnable" marker |
| `SGD`, `Adam` | only needed `.data` and `.grad` |
| `Trainer` | only needed `loss.backward()` and `optimizer.step()` |
| checkpointing | only needed `state_dict()` |

`graph.py` was written for `Value` in Phase 3 and runs on `Tensor` without a
single modification. That reuse is a strong signal the abstractions were drawn
in the right places — and it would have been impossible if, say, the optimiser
had assumed gradients were floats.

The only genuinely new code is `tensor.py` itself and `Linear`/`TensorMLP`.

## 7. Verified against the scalar engine

The tests do not merely check that the tensor engine is self-consistent. They
build the **same network both ways** and compare gradients element by element,
and they gradient-check the tensor engine against finite differences
independently.

Both must agree, because they are computing the same mathematics at different
granularity. If they ever diverge, one is wrong.

## 8. The payoff, and the real lesson

| engine | forward + backward, per sample | vs the next |
|---|---|---|
| our scalar `Value` | 1.68 s | — |
| our tensor engine | 6.6 µs | **~254,000× faster** |
| PyTorch | 4.7 µs | 1.4× faster |

Read that twice.

The step from **our scalar engine to our tensor engine** is ~254,000×. The step
from **our tensor engine to PyTorch** — a project with thousands of
contributors, hand-tuned kernels and a C++ core — is about 1.4×.

**Almost the entire performance story is granularity, and we captured nearly all
of it ourselves by changing what a node represents.**

That is the real answer to *"why are frameworks built around tensors?"* Not
because tensors are mathematically necessary — our scalar engine computes
identical gradients. But because **the unit of work has to be big enough that
per-operation interpreter overhead stops mattering.**

There is a second, non-performance argument too. A 33-node graph can be drawn
and understood ([chapter 7](07-visualisation.md#4-scale-is-a-real-constraint)).
A 328,804-node graph cannot be drawn *or* comprehended. The tensor abstraction
buys clarity as well as speed.

---

**Next:** [18. Performance vs PyTorch](18-performance.md) — the full measurement.
