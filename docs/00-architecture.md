# 0. Project Architecture

> **Phase 1 — Architecture.** The design decisions, made before writing the
> implementation, with the reasoning behind each one. Read
> [`01-what-is-automatic-differentiation.md`](./01-what-is-automatic-differentiation.md)
> first for the theory this design serves.

---

## Design goals, in priority order

1. **Correctness** — every derivative is hand-derived, then verified numerically.
   Nothing is trusted because it "looks like micrograd".
2. **Legibility** — the code should read like the maths. A reader who understands
   the chain rule should be able to open `value.py` and recognise it.
3. **Layering** — each layer may only use the layer below it. No shortcuts, no
   reaching around the engine.
4. **Testability** — every layer is independently testable, and gradient correctness
   is checked by a method that shares no code with the thing it checks.
5. **Performance, last** — we measure it (Phase 17/19) and improve it (tensors), but
   never at the cost of the four goals above.

### Non-goals

- Beating PyTorch. We will lose by ~4 orders of magnitude on the scalar engine, and
  understanding *why* is a deliverable, not a failure.
- Covering every operator. We implement what the maths requires and no more.
- Premature abstraction. `Parameter`, `Module`, `Optimizer` appear when a second
  concrete case justifies them, not before.

---

## The layering

Each layer depends **only** on the ones beneath it. This is the load-bearing
architectural constraint of the project — it's what makes the claim "we built the
autodiff ourselves" verifiable rather than aspirational.

```
┌─────────────────────────────────────────────────────────────────┐
│  L6  web/            interactive demo (React + TS + Vite)       │  Phase 14
│      docs/           written explanation                        │  Phase 15
├─────────────────────────────────────────────────────────────────┤
│  L5  examples/  benchmarks/  experiments/                       │  Phases 12,17,22
│      train_xor · train_mnist · vs-pytorch · lr sweeps           │
├─────────────────────────────────────────────────────────────────┤
│  L4  training/       Trainer, metrics, checkpoints, logging     │  Phase 11
│      data/           MNIST loader, batching, splits             │  Phase 12
├─────────────────────────────────────────────────────────────────┤
│  L3  nn/             Module, Parameter, Neuron, Layer, MLP      │  Phase 7-8
│      losses/         MSE, softmax cross-entropy                 │  Phase 9
│      optim/          SGD, Momentum, Adam                        │  Phase 10
├─────────────────────────────────────────────────────────────────┤
│  L2  core/graph.py   topological sort, traversal, statistics    │  Phase 3
│      core/gradcheck  finite differences, comparison harness     │  Phase 5
│      visualization/  SVG / DOT graph rendering, plots           │  Phase 6
├─────────────────────────────────────────────────────────────────┤
│  L1  core/value.py   Value: data, grad, _prev, _op, _backward   │  Phase 2 ◀── the whole
│      core/tensor.py  Tensor: the same idea, vectorised          │  Phase 18     project
└─────────────────────────────────────────────────────────────────┘             rests here

     ⚠️  NumPy is permitted from L2 upward for data handling and inside the
         Tensor kernels. It is NOT permitted to compute a derivative.
         PyTorch appears in exactly one place: benchmarks/, as a baseline.
```

The arrow matters: **L1 is ~400 lines, and everything else is a consequence of it.**
If `Value` is right, the MLP is right. That is why Phases 4 and 5 are so heavily
tested before we build anything on top.

---

## Repository layout

```
autograd-engine/
├── nabla/                        # the library (importable package)
│   ├── __init__.py               # public API: Value, and re-exports
│   ├── core/
│   │   ├── value.py              # ◀ Phase 2  the scalar autodiff node
│   │   ├── graph.py              # ◀ Phase 3  topological sort, graph stats
│   │   ├── gradcheck.py          # ◀ Phase 5  finite-difference verification
│   │   └── tensor.py             # ◀ Phase 18 n-d array autodiff
│   ├── nn/
│   │   ├── module.py             # base class: parameters(), zero_grad(), train/eval
│   │   ├── parameter.py          # a Value that is learnable
│   │   ├── activations.py        # tanh, relu, sigmoid, ... as Modules
│   │   ├── neuron.py             # Σ wᵢxᵢ + b  → activation
│   │   ├── layer.py              # a list of Neurons
│   │   ├── mlp.py                # a list of Layers
│   │   └── init.py               # Xavier / He initialisation
│   ├── losses/
│   │   ├── mse.py
│   │   └── cross_entropy.py      # numerically-stable log-softmax + NLL
│   ├── optim/
│   │   ├── optimizer.py          # base: step(), zero_grad()
│   │   ├── sgd.py                # + momentum, + weight decay
│   │   └── adam.py
│   ├── data/
│   │   ├── mnist.py              # download, cache, normalise, split
│   │   └── dataset.py            # Dataset / DataLoader / batching
│   ├── training/
│   │   ├── trainer.py            # epochs, batches, callbacks, checkpoints
│   │   ├── metrics.py            # loss, accuracy, timing, running averages
│   │   └── checkpoint.py         # save/load weights (JSON + npz)
│   └── visualization/
│       ├── graph_svg.py          # self-contained SVG renderer (no graphviz)
│       ├── graph_dot.py          # optional Graphviz DOT export
│       └── plots.py              # matplotlib: curves, digits, confusion matrix
├── tests/                        # ◀ Phase 16  pytest suite
├── examples/                     # ◀ runnable scripts, smallest → largest
├── benchmarks/                   # ◀ Phase 17  us vs PyTorch
├── experiments/                  # ◀ Phase 22  lr / depth / width / optimiser studies
├── docs/                         # ◀ Phase 15  the written explanation
├── web/                          # ◀ Phase 14  React + TS demo
├── artifacts/                    # generated: plots, checkpoints, SVGs (gitignored)
├── pyproject.toml
├── requirements.txt
└── README.md                     # ◀ Phase 24
```

> **Naming note.** The brief sketched a top-level `engine/` directory. `engine` is a
> generic name that collides easily on `sys.path`, so the package is `nabla/` —
> after ∇, the gradient operator. The internal structure is exactly as specified.
> Imports read as `from nabla import Value`, `from nabla.nn import MLP`.

---

## Core design decisions

### D1 — Scalar `Value` first, `Tensor` second

We build a **scalar** engine before a tensor engine, even though the tensor engine is
what actually trains MNIST at full size.

*Why:* with scalars there is nowhere to hide. Every node holds exactly one number and
one gradient, so `d.grad` can be checked against pencil-and-paper calculus. Broadcasting,
axis reductions and matmul all add real complexity to the *backward rules*
(see Phase 18), and debugging those while also debugging the graph machinery would be
two unknowns in one equation.

*Cost:* a scalar engine is roughly 1000× slower than a vectorised one. That cost is
not a bug to be apologised for — it is **the measurement that motivates Phase 18**, and
it is the honest answer to "why does PyTorch exist?".

### D2 — Dynamic graph, rebuilt every forward pass

The graph is a side effect of running normal Python. There is no `compile()`, no
session, no placeholder.

*Why:* it makes `print` a working debugger, supports arbitrary control flow, and is
what PyTorch does — so the intuition transfers.

*Cost:* Python-object overhead per operation, and no opportunity for graph-level
optimisation (fusion, constant folding). Quantified in Phase 17.

### D3 — Backward rules live in closures on the node that created them

Each operation attaches a small function to its output:

```python
out = Value(a.data * b.data, (a, b), '*')
def _backward():
    a.grad += b.data * out.grad     # ∂out/∂a = b
    b.grad += a.data * out.grad     # ∂out/∂b = a
out._backward = _backward
```

*Why:* the closure captures precisely what the derivative needs (`a`, `b`, `out`) and
nothing else. Adding an operation = writing its forward and its derivative, in one
place, side by side. `backward()` itself never grows.

*Alternative rejected:* a registry of `Op` classes with `forward`/`backward` static
methods (PyTorch's `autograd.Function`). More extensible, more indirection, and it
separates each derivative from its forward pass — the opposite of what a teaching
codebase wants. We revisit this trade-off in the `Tensor` engine, where the extra
structure starts to pay for itself.

### D4 — `+=`, always

Every backward rule accumulates. Never assigns. This is the multivariable chain rule
(§10 of the theory doc) and it is non-negotiable. Tested explicitly for shared inputs,
diamonds, and deep reuse.

### D5 — Iterative topological sort

The textbook DFS is recursive. A scalar MLP produces graphs thousands of nodes deep,
and CPython's recursion limit is ~1000. We use an explicit stack, so depth is bounded
by heap memory. There is a test that builds a 50,000-node chain and differentiates it.

### D6 — PyTorch gradient semantics: `.backward()` accumulates, and does **not** auto-zero

Calling `.backward()` twice sums two gradients into the leaves. We keep this
deliberately, and require `zero_grad()` in the training loop.

*Why:* it is the real semantic, it teaches the real gotcha, and it is what makes
gradient accumulation over micro-batches possible. Hiding it would make our engine
*less* educational than the thing it explains.

### D7 — `__slots__` on `Value`

`Value` defines `__slots__`, so instances carry no `__dict__`.

*Why:* one MNIST forward pass creates ~10⁵–10⁶ nodes. `__slots__` cuts per-node memory
by roughly half and speeds attribute access measurably — both quantified in Phase 19.

*Cost:* you cannot monkey-patch attributes onto a `Value` in a notebook. Acceptable.

### D8 — Verification by an independent method

Gradient checking uses central finite differences, which shares **no code and no
reasoning** with the analytic backward rules. If a derivative is mis-derived, the two
disagree. Every operation, every activation, every loss and the full MLP are checked
this way in `tests/`.

### D9 — Determinism by default

Every source of randomness (initialisation, shuffling, dropout if added) draws from an
explicitly seeded generator threaded through the call site. Reproducing a reported
number must require only the seed.

---

## The public API we are designing toward

Phase 2 delivers only the first block; the rest is the target the architecture is
shaped around.

```python
# ── L1: the engine ────────────────────────────────────────────
from nabla import Value

a = Value(2.0, label='a')
b = Value(3.0, label='b')
d = a * b + a
d.backward()
a.grad, b.grad          # → 4.0, 2.0

# ── L2: verification and visualisation ────────────────────────
from nabla.core.gradcheck import check_gradients
from nabla.visualization import render_svg

check_gradients(lambda x, y: (x*y).tanh(), [2.0, 3.0])   # → passed
render_svg(d, 'artifacts/graph.svg')

# ── L3: the network ───────────────────────────────────────────
from nabla.nn import MLP
from nabla.losses import softmax_cross_entropy
from nabla.optim import Adam

model = MLP(784, [128, 64], 10, activation='relu', seed=0)
opt   = Adam(model.parameters(), lr=1e-3)

# ── L4: training ──────────────────────────────────────────────
from nabla.training import Trainer

trainer = Trainer(model, opt, softmax_cross_entropy)
history = trainer.fit(train_loader, val_loader, epochs=10)
```

Note what is *absent*: no `torch`, no `.backward()` borrowed from anywhere, no
optimiser we did not write. That is the point of the layering diagram.

---

## Testing strategy

Four independent kinds of evidence, because any one of them alone can be fooled:

| Kind | What it catches | Where |
|---|---|---|
| **Hand-computed** derivatives | Wrong local rule | `test_backward.py` |
| **Finite-difference** checks | Wrong local rule, wrong sign, missing term | `test_gradients.py` |
| **Structural** tests (topo order, accumulation, reuse, depth) | Wrong traversal, lost gradients | `test_graph.py` |
| **Behavioural** tests (XOR converges, loss decreases) | Everything integrated wrongly | `test_training.py` |

Plus the edge cases the brief calls out explicitly: zeros, negatives, repeated
variables, branching graphs, deep graphs, and multi-path gradient accumulation.

---

## Build order

Each phase ends with green tests before the next begins. No phase builds on an
unverified one.

| # | Phase | Deliverable | Gate |
|---|---|---|---|
| 1 | Theory + architecture | these two docs | — |
| 2 | `Value` + operations | `core/value.py` | forward + hand-checked backward |
| 3 | Graph | `core/graph.py` | topo-order tests, deep-graph test |
| 4 | Backpropagation | `Value.backward()` | hand-computed multi-path examples |
| 5 | Gradient checking | `core/gradcheck.py` | every op passes finite differences |
| 6 | Visualisation | `visualization/` | renders the running example |
| 7–8 | `nn` + activations | `nn/` | gradcheck through a whole MLP |
| 9 | Losses | `losses/` | stability + gradcheck |
| 10 | Optimisers | `optim/` | converge on a known quadratic |
| 11 | Trainer | `training/` | XOR trains end to end |
| 21 | **XOR** | `examples/train_xor.py` | 4/4 correct |
| 12–13 | MNIST + plots | `examples/train_mnist.py` | target accuracy hit |
| 14–15 | Web demo + docs | `web/`, `docs/` | reproducible from clean clone |
| 16–17 | Full test suite + benchmarks | `tests/`, `benchmarks/` | all green, numbers reported |
| 18–19 | Tensors + performance | `core/tensor.py` | full 784→128→64→10 trained |
| 22–25 | Experiments, README, review | — | another dev can reproduce |

Phase 21 (XOR) is deliberately pulled forward, ahead of MNIST: it is the smallest
problem that *requires* a hidden layer and a nonlinearity, so if XOR fails, the bug is
in the engine — not in the data pipeline.
