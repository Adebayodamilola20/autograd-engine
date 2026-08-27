# 7. Visualising the graph

> Implemented in [`nabla/visualization/`](../nabla/visualization/).
> `graph_svg.py` (no dependencies), `graph_dot.py` (Graphviz), `plots.py`
> (matplotlib).

A computational graph is an abstract object described in prose for six
chapters. Drawing it turns it into something you can point at.

The goal here is **explanatory, not decorative**. A picture that shows only the
shape of the graph is a diagram; one that shows values *and* gradients lets you
follow the chain rule with your finger.

---

## 1. Two renderers, and why both

```python
from nabla.visualization import render_svg, render_dot

L = (x * w + b).tanh()
L.backward()

render_svg(L, "graph.svg")          # zero dependencies
render_dot(L, "graph.dot")          # nicer layout, needs Graphviz
```

**`graph_svg.py` writes SVG by hand** — no libraries at all. That is a
deliberate choice: the primary visualisation of a from-scratch project should
not require an external binary that most people do not have installed. It does
its own layering and node ordering, which is about 350 lines and means the demo
always works.

**`graph_dot.py` emits Graphviz DOT.** Graphviz's layered layout is genuinely
better than what we hand-roll, so when it *is* available the output is nicer.
It is the optional upgrade, not the default.

## 2. What a node shows

Each node displays three things:

```
   ┌──────────────────┐
   │ x                │   label
   │ data   2.0000    │   the forward value
   │ grad  -0.7000    │   ∂L/∂x after backward()
   └──────────────────┘
```

with operation nodes drawn between them:

```
  x ──┐
      ├─▶ ( * ) ──▶ xw ──┐
  w ──┘                  ├─▶ ( + ) ──▶ z ──▶ (tanh) ──▶ L
                     b ──┘
```

Showing `grad` alongside `data` is the point. You can trace a single number
backwards through the picture and watch each local derivative multiply in — the
chain rule stops being a formula and becomes a path through a diagram.

## 3. The canonical example

`examples/basic_autograd.py` builds a single neuron and renders it:

$$L = \tanh(xw + b)$$

with `x = 2, w = -3, b = 1`. Forward: `xw = -6`, `z = -5`, `L = tanh(-5) ≈ -0.9999`.

Backward, seeded with `L.grad = 1`:

- `tanh` rule: `z.grad = (1 - L²)·1 ≈ 0.000181`
- `+` rule: passes it unchanged to `xw` and `b`
- `*` rule: `x.grad = w · 0.000181 ≈ -0.00054`, `w.grad = x · 0.000181 ≈ 0.00036`

Look at how small those numbers are. `tanh(-5)` is deep in saturation, where
the derivative $1 - \tanh^2$ is nearly zero, so almost nothing reaches `x` and
`w`. **That is a saturated unit, visible as small numbers in a picture** — and
it is exactly the vanishing-gradient mechanism from
[chapter 3](03-operators.md#tanh), made concrete on five nodes instead of
asserted about thirty layers.

## 4. Scale is a real constraint

Graph rendering is for **understanding**, not inspection at scale:

| graph | nodes | renderable? |
|---|---|---|
| a single neuron | 6 | yes, and it is beautiful |
| XOR network, one sample | ~60 | yes, busy but readable |
| MNIST scalar, one sample | **328,804** | absolutely not |
| MNIST tensor, one batch | **33** | yes — and this is the point |

The scalar MNIST graph cannot be drawn. Neither can it be understood by looking
at it. This is not a limitation of the renderer; a picture with 328,804 nodes
contains no information a human can extract.

The tensor engine's 33-node graph for the *same computation* is not just faster
to run — it is the only one you can actually look at. That is an argument for
the tensor abstraction that has nothing to do with speed
([chapter 15](15-tensors.md)).

## 5. The other plots

`plots.py` covers the results side, all matplotlib:

| function | shows |
|---|---|
| `plot_training_curves` | train/val loss and accuracy per epoch |
| `plot_digit_grid` | MNIST samples with true label, prediction, confidence |
| `plot_prediction_detail` | one digit with its full probability distribution |
| `plot_confusion_matrix` | which digits get mistaken for which |
| `plot_weight_images` | first-layer weights reshaped to 28×28 |
| `plot_activations` | activation distributions per layer |
| `plot_gradient_histogram` | gradient magnitudes — the health check |
| `plot_decision_boundary` | 2-D problems like XOR |

Two of these earn special mention.

**`plot_weight_images`** reshapes each first-layer weight column back to 28×28
and displays it. Those images are *what each hidden unit is looking for* —
stroke fragments, curves, and regions of "there should be no ink here". It is
the most direct answer available to "what did the network learn?", and it comes
free because the input is an image.

**`plot_gradient_histogram`** is diagnostic rather than presentational. A
histogram piled at zero means vanishing gradients; a long right tail means
exploding. It turns "training isn't working" into a specific, actionable
observation — see [chapter 20](20-experiments.md).

## 6. And the live version

The [web demo](../web/README.md) renders per-layer activations in the browser
for a digit you draw yourself, updating in real time. Watching ~50% of the
first hidden layer go silent, and watching *which* half changes as you redraw,
communicates ReLU sparsity better than any static figure.

---

**Next:** [8. Neurons, layers, and MLPs](08-neural-network.md)
