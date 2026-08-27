# 3. Implementing the operators

> Implemented in [`nabla/core/value.py`](../nabla/core/value.py).
> Tested by [`tests/test_operations.py`](../tests/test_operations.py) and
> [`tests/test_gradients.py`](../tests/test_gradients.py) (183 gradient checks).

Every operation needs two things: what it computes, and what its derivative is.
This chapter derives each derivative before implementing it, because a backward
rule you cannot derive is a backward rule you cannot debug.

The pattern is always the same:

```python
def op(self, other):
    out = Value(f(self.data, other.data), (self, other), 'name')

    def _backward():
        self.grad  += (∂out/∂self)  * out.grad
        other.grad += (∂out/∂other) * out.grad

    out._backward = _backward
    return out
```

`out.grad` is ∂L/∂out — what the rest of the network already worked out. The
local derivative ∂out/∂self is the only new information. Their product is the
chain rule.

---

## Addition

$$z = x + y \qquad \frac{\partial z}{\partial x} = 1, \quad \frac{\partial z}{\partial y} = 1$$

```python
self.grad  += out.grad
other.grad += out.grad
```

**Addition is a gradient router.** It passes the incoming gradient to both
parents unchanged. Nudge `x` by ε and `z` moves by exactly ε, regardless of `y`.

This is why bias terms are easy and why residual connections help so much: a
`+` gives the gradient a path that does not attenuate.

## Multiplication

$$z = xy \qquad \frac{\partial z}{\partial x} = y, \quad \frac{\partial z}{\partial y} = x$$

```python
self.grad  += other.data * out.grad
other.grad += self.data  * out.grad
```

**Multiplication is a gradient swapper** — each operand receives the *other's*
value. Intuitively: if `y` is large, a small change in `x` produces a large
change in `xy`, so `x` matters more.

This is also where exploding and vanishing gradients are born. Chain many
multiplications and the gradient is a product of many factors; if they are
mostly > 1 it explodes, mostly < 1 it vanishes.

## Power

$$z = x^n \qquad \frac{\partial z}{\partial x} = n x^{n-1}$$

```python
self.grad += (n * self.data ** (n - 1)) * out.grad
```

Restricted to a constant exponent in the base case, because the general
$x^y$ needs $\partial z/\partial y = x^y \ln x$, which is undefined for
$x \le 0$. Supporting a case that silently produces NaN for half its domain is
worse than not supporting it.

## Negation, subtraction, division — composed, not new

None of these need a new backward rule:

```python
-x       ==  x * -1
x - y    ==  x + (-y)
x / y    ==  x * y**-1
```

Each reuses rules already derived and tested. **Every operation implemented in
terms of existing ones is an operation whose derivative cannot be wrong**, as
long as the ones underneath are right.

This is a real design principle, not laziness. Each new primitive is a new
opportunity for a sign error. `x / y` composed from `mul` and `pow` inherits
their correctness — and gradient checking confirms it gives
$\partial z/\partial y = -x/y^2$ without anyone typing that formula.

## Exponential

$$z = e^x \qquad \frac{\partial z}{\partial x} = e^x = z$$

```python
self.grad += out.data * out.grad     # out.data IS e^x
```

The derivative is the output. No recomputation — we already have it in
`out.data`. `exp` is the only function that is its own derivative, which is
most of why it is everywhere in this subject.

## Logarithm

$$z = \ln x \qquad \frac{\partial z}{\partial x} = \frac{1}{x}$$

```python
self.grad += (1.0 / self.data) * out.grad
```

Note the gradient blows up as `x → 0`. That is not a bug in the rule, it is a
true fact about `ln` — and it is exactly why `cross_entropy` must never be
handed a probability of zero. See [chapter 10](10-losses.md).

## tanh

$$z = \tanh x = \frac{e^{2x}-1}{e^{2x}+1} \qquad \frac{\partial z}{\partial x} = 1 - \tanh^2 x = 1 - z^2$$

```python
self.grad += (1 - out.data ** 2) * out.grad
```

Again the derivative is expressed in terms of the output, so it costs one
multiply.

**The shape of that derivative is the whole story of tanh.** It peaks at
**1.0** when `z = 0`, and falls to 0 as `|z|` grows. A saturated unit —
one with a large input — passes almost no gradient. Nothing downstream can
teach it anything. Output range is (−1, 1) and it is **zero-centred**, which
keeps activations from drifting systematically positive.

## Sigmoid

$$\sigma(x) = \frac{1}{1+e^{-x}} \qquad \sigma'(x) = \sigma(x)\bigl(1-\sigma(x)\bigr)$$

```python
self.grad += out.data * (1 - out.data) * out.grad
```

Derivation, since it is short and worth seeing once. Write $\sigma = (1+e^{-x})^{-1}$:

$$\sigma' = -(1+e^{-x})^{-2}\cdot(-e^{-x}) = \frac{e^{-x}}{(1+e^{-x})^2} = \frac{1}{1+e^{-x}}\cdot\frac{e^{-x}}{1+e^{-x}} = \sigma(1-\sigma)$$

**That derivative peaks at 0.25**, at `x = 0`. This one number explains why deep
sigmoid networks were untrainable: every layer multiplies the backward signal by
at most a quarter, so after `n` layers the gradient is attenuated by up to
$4^{-n}$. Ten layers is a factor of a million.

[Experiment 2](20-experiments.md#2-activations-and-why-nonlinearity-is-mandatory)
measures this: sigmoid attenuates 32.9× from last layer to first, against tanh's
6.8×.

Implemented via a numerically stable branch rather than the naive formula —
`exp(-x)` overflows for large negative `x`.

## ReLU

$$\text{relu}(x) = \max(0, x) \qquad \frac{\partial}{\partial x} = \begin{cases} 1 & x > 0 \\ 0 & x < 0 \end{cases}$$

```python
self.grad += (out.data > 0) * out.grad
```

The derivative is **exactly 1** on the active path. Not 0.25, not "up to 1" —
exactly 1. One multiplied by itself any number of times is one, so **depth costs
the gradient nothing** through ReLU. That single fact is most of why deep
networks became trainable.

The cost: for `x < 0` the gradient is exactly **0**. Not small — zero. A unit
that never activates receives no gradient, so its weights never change, so it
never starts activating. It is dead permanently. In practice a trained network
has ~50% of its hidden units silent on any given input (visible live in the
[web demo](../web/README.md)), and that sparsity is useful rather than wasteful.

### The kink at zero

`relu` is not differentiable at exactly `x = 0`. We define the derivative there
as 0, using `out.data > 0`.

This is a *convention*, and it is worth being honest that it is one. The
justification is practical: `x` being exactly `0.0` in floating point has
measure zero, and both 0 and 1 are valid subgradients. PyTorch makes the same
choice. But it is a choice, not a derivation, and it is the one place in this
file where the maths does not fully determine the code.

---

## How we know these are right

Deriving a rule and typing it correctly are different skills. Every operation
above is verified against finite differences:

$$\frac{\partial f}{\partial x} \approx \frac{f(x+\varepsilon) - f(x-\varepsilon)}{2\varepsilon}$$

`tests/test_gradients.py` contains **183 such checks**, covering every operator,
composition, edge case (zero, negative, repeated variables, branching paths) and
the full networks. If a derivation above is wrong, those tests fail.

That is the subject of [chapter 6](06-gradient-checking.md), and it is not
optional. An unchecked derivative is a guess that happens to be typed in
monospace.

---

**Next:** [4. The computational graph](04-computational-graph.md)
