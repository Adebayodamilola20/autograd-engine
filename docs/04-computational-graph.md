# 4. The computational graph

> Implemented in [`nabla/core/graph.py`](../nabla/core/graph.py).
> Tested by [`tests/test_graph.py`](../tests/test_graph.py).

---

## 1. The graph builds itself

There is no `Graph` class in this project, and that is deliberate. Nobody ever
calls `graph.add_node(...)`. The graph is an *emergent* structure: every
operation records its operands in `_prev`, and the collection of those
references **is** the graph.

```python
x = Value(2.0, label='x')
w = Value(-3.0, label='w')
b = Value(1.0, label='b')

y = x * w + b
```

produces

```
        x ──┐
            ├──▶ (*) ──▶ xw ──┐
        w ──┘                 ├──▶ (+) ──▶ y
                          b ──┘
```

Five nodes: three leaves, one for the product, one for the sum. Running the
forward pass *is* building the graph. They are the same act.

This is what "define-by-run" means, and it is why PyTorch feels like ordinary
Python while TensorFlow 1.x did not. There is no separate compilation step and
no graph object to keep in sync — which means a graph can contain a Python
`if`, a loop, or a recursive call, and it simply records whatever actually
executed.

## 2. It is a DAG, and each property matters

**Directed** — edges point from output to inputs (`_prev`). Backpropagation
walks them in that direction, which is why they are stored that way.

**Acyclic** — you cannot build a cycle, because a node's parents must already
exist when it is constructed. This is guaranteed by construction, not checked.
And it is *necessary*: a cycle would make "the derivative" ill-defined, since a
value would depend on itself.

**Not a tree.** This is the property people forget. A node can have several
consumers:

```python
d = a * b + a       # `a` has two consumers
```

The graph is a DAG, not a tree, and every algorithm that walks it must cope with
arriving at the same node more than once. That is exactly why gradients
accumulate ([chapter 2](02-the-value-class.md#why-grad-uses--and-never-)) and
why we need topological order rather than a simple recursion.

## 3. Why order matters — the failure case

Suppose we backpropagate in the wrong order. Take:

```
u = x * x
L = u + u        # u has two consumers
```

`L.grad = 1`. The `+` rule fires and sends `1` to `u`, twice — but suppose we
process `u`'s own backward rule **after the first contribution and before the
second**. Then `u.grad` is `1` when it should be `2`, and `x` receives half the
gradient it should.

The rule that prevents this:

> **A node's `_backward()` may only run once every one of its consumers has
> already run.**

Because `node.grad` is the carrier — it must be *complete* before it is *used*.

That is precisely the definition of reverse topological order.

## 4. Topological sort

A topological ordering lists every node after all of its parents. Reverse it and
every node comes after all of its consumers — which is the guarantee we need.

```python
def topological_sort(root):
    order, visited = [], set()
    stack = [(root, False)]

    while stack:
        node, expanded = stack.pop()
        if expanded:
            order.append(node)          # all parents already emitted
            continue
        if id(node) in visited:
            continue
        visited.add(id(node))
        stack.append((node, True))      # revisit after parents
        for parent in node._prev:
            stack.append((parent, False))

    return order
```

Three things worth pointing at:

**It is iterative, not recursive.** A recursive DFS is shorter and blows the
Python stack — default limit ~1000 frames — on any genuinely deep graph. Our
MNIST scalar graph has a depth of **1,000** and 328,804 nodes. Recursion would
crash on the real workload while passing every small test, which is the worst
possible failure mode. The explicit stack has no such limit.

**`visited` keys on `id(node)`, not the node.** Nodes are identified by
identity. Two distinct nodes holding `2.0` are different nodes with different
gradients, and `Value` deliberately does not define `__eq__`
([chapter 2](02-the-value-class.md#5-why-__slots__)).

**The `(node, expanded)` flag** is the standard trick for iterative post-order:
push the node twice, once to expand its parents and once to emit it afterwards.

## 5. A worked ordering

For `y = x*w + b`:

```
topological_sort(y)  →  [x, w, xw, b, y]        parents before children
reversed             →  [y, b, xw, w, x]        consumers before producers
```

Walking the reversed list:

1. `y` — the `+` rule: pushes gradient to `xw` and `b`
2. `b` — a leaf, `_noop`
3. `xw` — the `*` rule: **`xw.grad` is now complete**, safe to use
4. `w`, `x` — leaves, `_noop`

`xw` is only processed after `y`, its sole consumer, has contributed. That is
the guarantee, and it holds for arbitrarily complicated graphs.

## 6. Inspecting a graph

`graph.py` provides more than the sort, because "how big is one forward pass?"
turns out to be the question that explains all the performance results:

```python
from nabla import graph_size, format_graph, leaves, build_edges

graph_size(loss)
# {'nodes': 328804, 'edges': 437012, 'leaves': 109387, 'depth': 1000}
```

Those four numbers, for one MNIST sample on the scalar engine, explain
everything in [chapter 18](18-performance.md):

- **328,804 nodes** — Python objects allocated, per sample
- **depth 1,000** — the longest chain the gradient must flow along

The depth number is the more interesting one. **Depth is inherently serial** —
no amount of hardware parallelism shortens a chain of dependencies. Width can be
vectorised away; depth cannot. The tensor engine
([chapter 15](15-tensors.md)) reduces the same computation to **33 nodes and
depth 18**, and that is the entire performance story.

---

**Next:** [5. Backpropagation](05-backpropagation.md)
