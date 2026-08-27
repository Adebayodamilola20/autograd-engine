# 2. Building the `Value` class

> Implemented in [`nabla/core/value.py`](../nabla/core/value.py).
> Tested by [`tests/test_value.py`](../tests/test_value.py).

Everything in this project rests on one class. Get it right and neurons,
layers, losses, optimisers and MNIST all follow as consequences. Get it wrong
and nothing above it can be correct.

So it is worth being slow here.

---

## 1. The intuition

We want to write ordinary arithmetic:

```python
d = a * b + a
```

and afterwards ask *"how much does `d` change if I nudge `a`?"* — without ever
writing down a derivative by hand.

For that to be possible, `a * b` cannot just produce a number. A number has
forgotten where it came from. It has to produce a number **that remembers how
it was made**.

That is the entire idea. A `Value` is a number with a memory of its own
history.

## 2. What each node must store

Five fields, and each one earns its place:

```python
class Value:
    __slots__ = ("data", "grad", "label", "_prev", "_op", "_backward", "_id")
```

| field | what it is | why it must exist |
|---|---|---|
| `data` | the number itself | the forward pass |
| `grad` | ∂output/∂this | where the answer accumulates |
| `_prev` | the nodes this came from | the **edges** of the graph — without them there is nothing to walk backward through |
| `_op` | the operation's name | debugging and visualisation only; the maths does not need it |
| `_backward` | a closure that pushes gradient to `_prev` | the local derivative rule, captured at the moment it is known |
| `label` | optional human name | so a graph diagram is readable |
| `_id` | a monotonic counter | stable ordering in visualisation; identity, not equality, defines a node |

### Why `_backward` is a closure and not a method

This is the design decision that makes the whole thing work, and it is easy to
miss.

Consider `c = a * b`. The derivative of `c` with respect to `a` is `b`. But
*which* `b`? Not "some `b`" — **that specific node**, with the value it had at
the moment the multiplication happened.

A closure captures exactly that:

```python
def __mul__(self, other):
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
        self.grad  += other.data * out.grad     # ∂out/∂self = other
        other.grad += self.data  * out.grad     # ∂out/∂other = self

    out._backward = _backward
    return out
```

`self`, `other` and `out` are all captured by the closure. The rule for *this
multiplication* is stored on *this node*, with references to *these* operands.
Nothing has to be looked up later, and nothing can be confused with another
multiplication elsewhere in the graph.

A method would have had to reconstruct that context. The closure simply keeps
it.

### Why `grad` uses `+=` and never `=`

If a value is used more than once, gradient arrives from more than one place:

```python
d = a * b + a      # `a` is used twice
```

`a` influences `d` through the multiplication **and** through the addition. The
multivariable chain rule says those contributions **sum**:

$$\frac{\partial d}{\partial a} = \underbrace{b}_{\text{via } a*b} + \underbrace{1}_{\text{via } +a} = 4$$

With `=` the second contribution would overwrite the first and we would get
`1`. With `+=` we get `4`, which is correct.

This is the single most common bug in a hand-written autodiff engine, and it is
invisible on any expression where each variable appears exactly once — so it
passes naive tests. `tests/test_backward.py` tests it deliberately.

## 3. A tiny numerical example, by hand

Take `a = 2`, `b = 3`, and `d = a*b + a`.

**Forward.** Three nodes get built:

```
a = 2.0  (leaf)
b = 3.0  (leaf)
c = a*b = 6.0     _prev = (a, b),  _op = '*'
d = c+a = 8.0     _prev = (c, a),  _op = '+'
```

**Backward.** Seed `d.grad = 1` (∂d/∂d = 1), then walk in reverse:

| step | node | rule | effect |
|---|---|---|---|
| 1 | `d` | `+` passes gradient through unchanged | `c.grad += 1` → `1`, `a.grad += 1` → `1` |
| 2 | `c` | `*` sends each operand the *other's* value | `a.grad += 3·1` → `4`, `b.grad += 2·1` → `2` |

Final: `a.grad = 4`, `b.grad = 2`.

Check by hand: `d = ab + a`, so `∂d/∂a = b + 1 = 4` ✓ and `∂d/∂b = a = 2` ✓.

Notice `a.grad` was touched twice, by two different rules, and the `+=` is what
made the answer right.

## 4. The implementation

```python
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data  = float(data)
        self.grad  = 0.0            # no gradient until a backward pass runs
        self._prev = tuple(_children)
        self._op   = _op
        self.label = label
        self._backward = _noop      # leaves have nothing to propagate
        self._id   = next(Value._counter)
```

Two details that look small and are not:

**Leaves get `_noop`, not `None`.** Every node can be called uniformly during
the backward sweep — no `if node._backward is not None` branch in the hot loop,
and no chance of forgetting one.

**`__eq__` is deliberately not defined.** Defining it would make `Value`
unhashable by default and break `set`/`dict` membership, which the graph
traversal depends on. Nodes are identified by **identity**, not by value — two
different nodes that happen to hold `2.0` are different nodes with different
gradients. `__lt__`/`__gt__` *are* defined, for `max(logits, key=...)`, and
they return plain bools because a comparison is not differentiable.

## 5. Why `__slots__`

A `Value` is created once per arithmetic operation. Training MNIST on the
scalar engine builds ~328,000 of them **per sample**.

`__slots__` removes the per-instance `__dict__`, which is roughly a 40–50%
memory saving and a measurable speedup on attribute access. It also makes typos
loud: `v.gard = 1.0` raises `AttributeError` instead of silently creating a new
attribute and leaving the real gradient at zero.

That second benefit matters more than the first. A silently-misspelled gradient
field is a bug that produces plausible-looking wrong numbers.

## 6. The test that proves it

```python
def test_gradient_accumulates_when_a_value_is_reused():
    a = Value(2.0)
    b = Value(3.0)
    d = a * b + a
    d.backward()

    assert d.data == 8.0
    assert a.grad == 4.0     # b + 1 -- both paths counted
    assert b.grad == 2.0     # a
```

Small, but it is simultaneously testing the forward pass, graph construction,
topological ordering, two backward rules, and gradient accumulation across
branching paths. If this passes, the core is sound.

---

**Next:** [3. Implementing the operators](03-operators.md) — every derivative,
derived before it is coded.
