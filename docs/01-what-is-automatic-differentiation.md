# 1. What Is Automatic Differentiation?

> **Phase 1 — Research.** This document answers the eleven questions from the project
> brief. Every claim is backed by a tiny numerical example you can check by hand.
> Nothing here is magic. By the end you should be able to differentiate a small
> expression on paper the *exact* same way our engine will do it in code.

---

## 0. The one-sentence version

Training a neural network means answering one question, over and over:

> *"If I nudge this parameter a tiny bit, does the loss go up or down, and by how much?"*

That number is a **derivative**. A network has hundreds of thousands of parameters,
so we need hundreds of thousands of those numbers, and we need them fast.
**Automatic differentiation** is the algorithm that computes all of them in
roughly the cost of *one* extra pass over the computation.

That is the whole idea. Everything else in this project is machinery to make it
correct, general, and fast.

---

## 1. What automatic differentiation is

**Automatic differentiation (AD)** is a technique for computing exact derivatives of a
function that is *defined by a program*, by applying the chain rule mechanically to
the elementary operations that program performs.

The key insight is this:

> Any function a computer can evaluate — no matter how complicated — is ultimately a
> composition of a small number of **elementary operations**: `+`, `*`, `exp`, `log`,
> `tanh`, and so on. We know the derivative of each elementary operation exactly.
> Therefore we know the derivative of the whole thing, by the chain rule.

### Intuition

Think of a factory assembly line. Raw materials (inputs) enter on the left. Each
station performs one simple transformation. A finished product (the output) comes out
on the right.

Now someone asks: *"if we increase the amount of raw material #7 by 1 gram, how much
heavier is the final product?"*

You don't need to re-run the factory. You just need to know, at each station, the
**local** sensitivity: "for every extra gram in, how many extra grams out?" Multiply
those local sensitivities along the path from station 7 to the exit, and you have your
answer. That product-of-local-sensitivities *is* the chain rule, and AD is the
bookkeeping that does it for you.

### Why "automatic"

Because you never write down a derivative formula. You write the *forward* computation
in ordinary code:

```python
loss = (x * w + b) ** 2
```

and the engine — having recorded what happened — produces `dloss/dw`, `dloss/db`,
`dloss/dx` on demand. You get derivatives "for free" as a side effect of computing
the value.

### The three properties that matter

| Property | AD | Why it matters |
|---|---|---|
| **Exact** | Yes — accurate to floating-point round-off | No approximation error to debug |
| **Efficient** | Reverse mode: all gradients in ~1 extra pass | Makes deep learning feasible |
| **General** | Works on loops, branches, recursion | Real programs, not just formulas |

---

## 2. How AD differs from **symbolic** differentiation

**Symbolic differentiation** is what you did in calculus class, and what Mathematica
or SymPy do: it manipulates the *algebraic expression* and returns a new *algebraic
expression*.

```
f(x) = x * sin(x)
      → symbolic differentiation →
f'(x) = sin(x) + x*cos(x)      (a formula)
```

AD, by contrast, never produces a formula. It produces a **number**, at the specific
input you asked about.

```
f(x) = x * sin(x),  at x = 2.0
      → automatic differentiation →
f'(2.0) = 0.0770...             (a number)
```

### The problem symbolic differentiation runs into: expression swell

Consider a chain of products:

$$f = u_1 u_2 u_3 \cdots u_n$$

The symbolic derivative by the product rule is

$$f' = u_1' u_2 \cdots u_n + u_1 u_2' \cdots u_n + \cdots + u_1 u_2 \cdots u_n'$$

That's $n$ terms of $n$ factors each — $O(n^2)$ symbols from an $O(n)$ expression. Now
nest that a few levels deep, as a neural network does, and the expression explodes
combinatorially. Symbolic engines fight this with simplification passes, which are
themselves expensive.

AD sidesteps the problem entirely: it never builds the expression, only evaluates
numbers. A 100-layer network costs 100 layers' worth of arithmetic, not
$2^{100}$ symbols.

### The other problem: control flow

Symbolic differentiation needs a closed-form expression. But what is the symbolic
derivative of *this*?

```python
def f(x):
    total = 0.0
    while total < 10.0:      # loop count depends on x
        total = total + x*x
    if x > 0:                # branch depends on x
        return total.log()
    return -total
```

There is no tidy formula. But AD handles it without blinking, because AD doesn't
differentiate the *source code* — it differentiates the **specific sequence of
operations that actually executed** for the input you gave. Loops unroll themselves,
branches resolve themselves. This property is called **define-by-run**, and it is
exactly why PyTorch feels like ordinary Python.

> **Nuance worth knowing:** AD differentiates the branch you *took*. At `x > 0` the
> function may not be differentiable at all (there could be a kink at `x = 0`), and AD
> will silently hand you the derivative of one side. AD is honest about the executed
> trace, not about the mathematical function's global smoothness. `ReLU` at exactly 0
> is the classic case — see §8.

### Summary

| | Symbolic | Automatic |
|---|---|---|
| Output | A formula | A number, at a point |
| Handles loops/branches | No | Yes |
| Expression swell | Yes, severe | No |
| Exactness | Exact | Exact (to round-off) |
| Reusable across inputs | Yes | No — re-run per input |

---

## 3. How AD differs from **numerical** differentiation

**Numerical differentiation** (finite differences) approximates the derivative from
the *definition* of a limit, by actually taking a small step:

$$f'(x) \;\approx\; \frac{f(x+h) - f(x-h)}{2h}$$

This is simple, requires no knowledge of $f$'s internals — and has two serious flaws.

### Flaw 1: it is inexact, and there is no good choice of $h$

You are caught between two errors pulling in opposite directions:

- **Truncation error** — Taylor's theorem gives the central difference an error of
  $O(h^2)$. This shrinks as $h \to 0$. *Wants $h$ small.*
- **Round-off error** — $f(x+h)$ and $f(x-h)$ are nearly equal, so subtracting them
  destroys significant digits (catastrophic cancellation). Dividing by a tiny $h$ then
  amplifies whatever noise is left. Roughly $O(\varepsilon / h)$ where
  $\varepsilon \approx 2.2\times10^{-16}$ is double-precision machine epsilon.
  *Wants $h$ large.*

The total error is minimised around $h \approx \varepsilon^{1/3} \approx 6\times10^{-6}$,
giving you about **10–11 correct digits at best** — never the full 16. Watch it happen,
for $f(x)=x^3$ at $x=2$ (true answer $f'=3x^2=12$):

| $h$ | central difference | absolute error |
|---|---|---|
| $10^{-1}$ | 12.010000000000 | $1.0\times10^{-2}$ |
| $10^{-3}$ | 12.000001000000 | $1.0\times10^{-6}$ |
| $10^{-5}$ | 12.000000000121 | $1.2\times10^{-10}$ |
| $10^{-8}$ | 12.000000116860 | $1.2\times10^{-7}$ ← getting **worse** |
| $10^{-12}$ | 12.000622...    | $6.2\times10^{-4}$ ← much worse |

AD has no such trade-off. It computes $3x^2 = 12$ exactly, because it *knows* the
power rule.

### Flaw 2: it costs one full function evaluation **per parameter**

To get the gradient of a function with $n$ inputs, central differences needs $2n$
evaluations of $f$. For our MNIST network, $n = 109{,}386$. That is **218,772 forward
passes** to get one gradient — for one training step. Reverse-mode AD gets the same
gradient in the cost of about **two** forward passes. That is a ~100,000× difference,
and it is precisely why neural networks are trained with AD and not finite differences.

### So why implement finite differences at all?

Because it is the perfect **independent check**. It shares no code and no reasoning
with our backward rules, so if a hand-derived derivative is wrong, finite differences
will disagree. It's slow and imprecise — but it's *independently* slow and imprecise.
We build this in **Phase 5 (gradient checking)** and use it to validate every single
operation.

### Summary

| | Numerical | Automatic |
|---|---|---|
| Exactness | ~10 digits, tuning-dependent | Exact to round-off |
| Cost for $n$ inputs | $O(n)$ function evals | $O(1)$ extra passes |
| Needs internals of $f$ | No (black box) | Yes (traces the ops) |
| Good for | **Testing** | **Training** |

---

## 4. What reverse-mode differentiation is

AD comes in two flavours. The difference is *the direction you apply the chain rule*.

Take a composition $y = f(g(h(x)))$. Name the intermediates:

$$a = h(x), \qquad b = g(a), \qquad y = f(b)$$

The chain rule says

$$\frac{dy}{dx} = \frac{dy}{db}\cdot\frac{db}{da}\cdot\frac{da}{dx}$$

Multiplication is associative, so we may bracket it two ways — and **that choice is
the entire distinction between forward and reverse mode.**

### Forward mode — bracket from the right (inputs → outputs)

$$\frac{dy}{dx} = \frac{dy}{db}\cdot\left(\frac{db}{da}\cdot\frac{da}{dx}\right)$$

Travel *with* the computation. At each node carry a **tangent**: "how fast does this
value change as $x$ changes?" You seed $\dot{x} = 1$ and push forward:

$$\dot a = \frac{da}{dx}, \quad \dot b = \frac{db}{da}\dot a, \quad \dot y = \frac{dy}{db}\dot b$$

One sweep gives you the derivative of **every output** with respect to **one input**.
To cover $n$ inputs you need $n$ sweeps.

### Reverse mode — bracket from the left (outputs → inputs)

$$\frac{dy}{dx} = \left(\frac{dy}{db}\cdot\frac{db}{da}\right)\cdot\frac{da}{dx}$$

Travel *against* the computation. At each node carry an **adjoint**, conventionally
written $\bar v$:

$$\bar v \;=\; \frac{\partial y}{\partial v} \;=\; \text{"how sensitive is the final output to this intermediate?"}$$

You seed $\bar y = 1$ (the output is perfectly sensitive to itself) and pull backward:

$$\bar b = \bar y \cdot \frac{dy}{db}, \quad \bar a = \bar b \cdot \frac{db}{da}, \quad \bar x = \bar a \cdot \frac{da}{dx}$$

One sweep gives you the derivative of **one output** with respect to **every input**.

**This is what our engine implements, and `Value.grad` is exactly $\bar v$.**

### The two modes side by side

|  | Forward mode | Reverse mode |
|---|---|---|
| Direction | inputs → output | output → inputs |
| Quantity carried | tangent $\dot v = \partial v/\partial x$ | adjoint $\bar v = \partial y/\partial v$ |
| One sweep gives | 1 input → all outputs | all inputs → 1 output |
| Sweeps for $f:\mathbb{R}^n\to\mathbb{R}^m$ | $n$ | $m$ |
| Memory | $O(1)$ extra — no tape needed | $O(\text{ops})$ — must store the graph |
| PyTorch calls it | `jvp` | `.backward()` / `vjp` |

Neither is "better". They are duals, and which one wins depends entirely on the
*shape* of your function — which is the subject of the next section.

---

## 5. Why reverse mode is the right choice for neural networks

Look at the shape of a training step:

$$\underbrace{L}_{\text{1 number}} \;=\; \text{loss}\big(\underbrace{w_1, w_2, \ldots, w_n}_{n = 109{,}386 \text{ parameters}}\big)$$

**Many inputs in, exactly one number out.** That asymmetry is total, and it decides
everything.

- **Forward mode** costs $n$ sweeps → 109,386 forward passes per gradient. Hopeless.
- **Reverse mode** costs $m = 1$ sweep → one backward pass per gradient. Trivial.

Concretely, for our MNIST network:

| Method | Passes per gradient | Relative cost |
|---|---|---|
| Central finite differences | 218,772 | ~100,000× |
| Forward-mode AD | 109,386 | ~50,000× |
| **Reverse-mode AD** | **~2** | **1×** |

A useful rule of thumb, sometimes called the **cheap gradient principle**:

> The cost of computing $\nabla f$ by reverse-mode AD is a small constant multiple
> (typically 2–4×) of the cost of computing $f$ itself — **independent of the number
> of parameters.**

That single fact is why deep learning is possible at all. Scaling a model from 1
million to 1 billion parameters makes the gradient 1000× more expensive, not
1,000,000,000× more expensive. Without it, "just add more parameters" would never
have been a viable research programme.

### What reverse mode costs you: memory

There is no free lunch. To walk backward you must remember what happened on the way
forward. Every intermediate value and every edge of the graph is retained until
`.backward()` runs. This is the **tape** (or **Wengert list**). It is why:

- training uses far more memory than inference (inference needs no tape — hence
  `torch.no_grad()`),
- very deep models use *gradient checkpointing*, which throws away intermediates and
  recomputes them during the backward pass — trading compute for memory,
- our `Value` objects hold references to their parents, which keeps the whole graph
  alive.

**Reverse mode trades memory for time.** For neural networks, that is an
overwhelmingly good trade.

---

## 6. What a computational graph is

A **computational graph** is a **directed acyclic graph (DAG)** that records the
computation as it happens:

- **nodes** = values (inputs, intermediates, the final output),
- **edges** = data dependencies (`u → v` means "$v$ was computed from $u$").

It is *acyclic* by construction: to build a node you must already hold its inputs, so
an edge can never point back in time.

### Worked example — our running example for the rest of this document

$$L = (x \cdot w + b)^2, \qquad x = 2,\; w = -3,\; b = 1$$

Name the intermediates so we can talk about them:

$$u = x\cdot w = -6, \qquad v = u + b = -5, \qquad L = v^2 = 25$$

```
   x=2.0 ──┐
           ├──[ * ]──> u=-6.0 ──┐
   w=-3.0 ─┘                    ├──[ + ]──> v=-5.0 ──[ **2 ]──> L=25.0
                    b=1.0 ──────┘
```

Five nodes, three operations. Read left-to-right you get the **forward pass** (compute
$L$). Read right-to-left you get the **backward pass** (compute the gradients). Same
graph, two directions. That symmetry is the heart of the whole project.

### Why a graph and not a formula

Because the graph is what actually gets *executed*. It records:

- **sharing** — if `x` is used twice, two edges leave it. A formula hides that.
- **the executed trace** — a loop that ran 7 times becomes 7 nodes, no special cases.
- **order** — which we need for correct traversal (§11).

### Static vs. dynamic graphs — a real distinction in the wild

- **Dynamic / define-by-run** (PyTorch, and our engine): the graph is built as a
  side effect of running ordinary Python. Every forward pass builds a fresh graph.
  Debuggable with `print`, works with native control flow, costs Python overhead.
- **Static / define-and-run** (TensorFlow 1.x, JAX under `jit`, ONNX): you define the
  graph once, the framework compiles and optimises it (operator fusion, memory
  planning, kernel selection), then feeds data through. Much faster, much harder to
  debug.

We build a dynamic graph. It is the right choice for a project whose goal is
*understanding*, and it is what PyTorch does.

---

## 7. What a node represents

A node is a single **value**, together with everything needed to differentiate it. In
our engine that is the `Value` class, and it stores five things:

| Field | Meaning | In the example ($v = u + b$) |
|---|---|---|
| `data` | the number computed on the forward pass | `-5.0` |
| `grad` | the adjoint $\bar v = \partial L / \partial v$, filled in on the backward pass | `-10.0` |
| `_prev` | the parent nodes it was computed from | `(u, b)` |
| `_op` | which operation produced it (for debugging/visualisation) | `'+'` |
| `_backward` | a closure that pushes this node's grad to its parents | see §9 |

Two subtleties worth stating plainly:

**1. `data` and `grad` are filled in at different times.** `data` exists the moment the
node is created (forward). `grad` is meaningless — conventionally `0.0` — until
`.backward()` has run. A node's `grad` is only defined *relative to some output*; it
is $\partial L/\partial v$, and until you name the $L$, the question has no answer.

**2. Leaves are the ones you care about.** A node with no parents is a **leaf**: an
input or a learnable parameter. Those are the gradients the optimiser will consume.
Intermediate nodes get gradients too — they're needed as stepping stones — but they're
discarded with the graph.

---

## 8. What local derivatives are

A **local derivative** is the derivative of a *single operation* with respect to *one
of its immediate inputs* — computed with total disregard for everything else in the
graph.

For $v = u + b$, the local derivatives are

$$\frac{\partial v}{\partial u} = 1, \qquad \frac{\partial v}{\partial b} = 1$$

That's it. The `+` node does not know that $u$ came from a multiplication, or that
$v$ feeds a square, or that a loss exists somewhere downstream. **Locality is the
design principle that makes AD tractable**: each operation needs to know only its own
calculus rule, and the graph handles composition. Adding a new operation to the engine
means writing down one derivative — never touching backpropagation itself.

### The local derivatives we will implement

| Operation | Forward | Local derivatives | Note |
|---|---|---|---|
| add | $v = a + b$ | $\partial v/\partial a = 1$, $\partial v/\partial b = 1$ | routes gradient unchanged |
| mul | $v = a \cdot b$ | $\partial v/\partial a = b$, $\partial v/\partial b = a$ | **swaps** the inputs |
| pow (const $n$) | $v = a^n$ | $\partial v/\partial a = n a^{n-1}$ | |
| neg | $v = -a$ | $\partial v/\partial a = -1$ | = mul by $-1$ |
| sub | $v = a - b$ | $1$, $-1$ | = add + neg |
| div | $v = a/b$ | $1/b$, $-a/b^2$ | = mul + pow(−1) |
| exp | $v = e^a$ | $\partial v/\partial a = e^a = v$ | reuses the output |
| log | $v = \ln a$ | $\partial v/\partial a = 1/a$ | needs $a>0$ |
| tanh | $v = \tanh a$ | $\partial v/\partial a = 1 - v^2$ | reuses the output |
| sigmoid | $v = \sigma(a)$ | $\partial v/\partial a = v(1-v)$ | reuses the output |
| relu | $v = \max(0,a)$ | $\partial v/\partial a = \mathbb{1}[a>0]$ | see below |

Two things to notice, because they show up everywhere in real frameworks:

**Some backward rules reuse the forward output.** `exp`, `tanh`, `sigmoid` all express
their derivative in terms of $v$, not $a$. That is why frameworks *cache activations*
during the forward pass — the backward pass needs them. It is a direct, concrete
reason training uses more memory than inference.

**ReLU is not differentiable at 0.** $\max(0,x)$ has a kink there: the left derivative
is 0, the right derivative is 1, and there is no single tangent. We *choose* the
convention $\partial v/\partial a = 0$ at $a = 0$ (this is a **subgradient**, a
legitimate member of the set $[0,1]$ of valid slopes). PyTorch makes the same choice.
It works in practice because hitting exactly `0.0` in floating point is
vanishingly rare, and because gradient descent is robust to a measure-zero set of
disagreements.

---

## 9. How the chain rule is applied repeatedly

The chain rule for a single link is the one from calculus:

$$\frac{\partial L}{\partial u} = \frac{\partial L}{\partial v}\cdot\frac{\partial v}{\partial u}$$

In adjoint notation, with $\bar v \equiv \partial L/\partial v$:

$$\boxed{\;\bar u \;\mathrel{+}=\; \bar v \cdot \frac{\partial v}{\partial u}\;}$$

Read it as a sentence, because this single line **is** backpropagation:

> *The gradient arriving at a node ($\bar v$, from downstream) is multiplied by the
> local derivative ($\partial v / \partial u$, known by the operation) and added into
> the parent's gradient ($\bar u$).*

Every node applies this rule to its own parents, knowing nothing about the rest of the
graph. Do that for every node in the right order and the chain rule composes itself
across the whole network — a hundred layers deep if need be.

### The full backward pass, by hand

Our example: $u = xw$, $v = u+b$, $L = v^2$, with $x=2$, $w=-3$, $b=1$, so
$u=-6$, $v=-5$, $L=25$.

**Seed.** $\bar L = \dfrac{\partial L}{\partial L} = 1$.

**Step 1 — through the square.** Local: $\partial L/\partial v = 2v = -10$.

$$\bar v = \bar L \cdot 2v = 1 \cdot (-10) = -10$$

**Step 2 — through the addition.** Locals are both 1, so `+` simply *copies* the
gradient to both parents:

$$\bar u = \bar v \cdot 1 = -10, \qquad \bar b = \bar v \cdot 1 = -10$$

**Step 3 — through the multiplication.** Locals are the *other* input:

$$\bar x = \bar u \cdot w = (-10)(-3) = +30, \qquad \bar w = \bar u \cdot x = (-10)(2) = -20$$

**Verify against pencil-and-paper calculus.** $L = (xw+b)^2$, so

$$\frac{\partial L}{\partial w} = 2(xw+b)\cdot x = 2(-5)(2) = -20 \;\checkmark$$
$$\frac{\partial L}{\partial x} = 2(xw+b)\cdot w = 2(-5)(-3) = +30 \;\checkmark$$
$$\frac{\partial L}{\partial b} = 2(xw+b)\cdot 1 = -10 \;\checkmark$$

The graph, annotated with `data | grad`:

```
   x  2.0 | +30.0 ──┐
                    ├─[ * ]─> u  -6.0 | -10.0 ─┐
   w -3.0 | -20.0 ──┘                          ├─[ + ]─> v -5.0 | -10.0 ─[ **2 ]─> L 25.0 | 1.0
                              b  1.0 | -10.0 ──┘
```

**Sanity-check the meaning.** $\bar w = -20$ says: increase $w$ by a tiny $\epsilon$
and $L$ falls by about $20\epsilon$. Try $\epsilon = 0.001$, so $w = -2.999$:
$L = (2(-2.999)+1)^2 = (-4.998)^2 = 24.980004$. The change is $-0.019996 \approx -20 \times 0.001$. ✓

That last check — perturb the input, see if the loss moves by the predicted amount —
is exactly what **Phase 5's gradient checker** automates.

---

## 10. Why gradients must be **accumulated**

This is the single most common source of bugs in a hand-written autograd engine, so it
gets its own section.

### The situation

What if a value is used **more than once**? Consider the smallest possible case:

$$L = x \cdot x, \qquad x = 3$$

The graph is *not* a chain. It's a diamond — one node, two edges into the same
multiply:

```
        ┌──────┐
   x=3 ─┤      ├─[ * ]─> L = 9
        └──────┘
     (two edges, one source)
```

We know the answer: $L = x^2$, so $\partial L/\partial x = 2x = 6$.

Now run the backward pass. Seed $\bar L = 1$. The multiply node has two parents —
which both happen to be $x$ — and its local derivatives are "the other input":

- via the **left** edge: $\bar L \cdot x_{\text{right}} = 1 \cdot 3 = 3$
- via the **right** edge: $\bar L \cdot x_{\text{left}} = 1 \cdot 3 = 3$

If we **overwrite**, $x$ ends up with `grad = 3`. **Wrong — off by exactly a factor of 2.**
If we **accumulate**, $x$ ends up with `3 + 3 = 6`. **Correct.**

### The rule, and why it's true

This is the **multivariable chain rule**. If $L$ depends on $x$ through several
intermediate paths $v_1, v_2, \ldots, v_k$, then

$$\frac{\partial L}{\partial x} \;=\; \sum_{i=1}^{k} \frac{\partial L}{\partial v_i}\cdot\frac{\partial v_i}{\partial x}$$

The **sum** is the mathematics. **Accumulation (`+=`) is that sum, computed
incrementally** as each downstream consumer reports in.

The intuition: a small nudge to $x$ propagates down *every* path simultaneously, and
the effects add up at the output. Ignoring a path means ignoring part of the effect.

Hence, in the implementation, the rule is absolute:

```python
self.grad += local_derivative * out.grad     # ✅  always +=
self.grad  = local_derivative * out.grad     # ❌  silently wrong, and only
                                             #     when a node is reused
```

The bug is nasty precisely because it's invisible on simple chains — every test where
each value is used once will pass. It only shows up with branching. Weight sharing,
residual connections, and any reused input hit this path, which is why **Phase 16
tests branching and reuse explicitly**.

### The consequence you'll trip over

Because gradients accumulate, they accumulate *across training steps too*. Call
`.backward()` twice without clearing, and you get the sum of two gradients. This is
exactly why PyTorch makes you write `optimizer.zero_grad()` every iteration — and our
engine reproduces the same semantics deliberately, rather than hiding it. Forgetting
it is a rite of passage.

---

## 11. What topological sorting has to do with backpropagation

### The ordering requirement

A node can only push gradient to its parents once **its own** gradient is final. And
its own gradient is final only after **every** node that consumes it has contributed.

> **Correctness condition:** process a node only after all of its children (consumers)
> have been processed.

Break that rule and you propagate a half-finished gradient. Consider

$$a \to b, \qquad a \to c, \qquad b, c \to L$$

```
        ┌──> b ──┐
   a ───┤        ├──> L
        └──> c ──┘
```

If we process $a$ after $b$ but before $c$, then $a$ pushes to its parents using a
gradient that is missing $c$'s contribution entirely. Silently wrong — and the error
scales with how much of the graph you skipped.

### The fix

A **topological sort** of a DAG is a linear ordering of its nodes such that every edge
$u \to v$ has $u$ appearing before $v$. In our graph, edges point from parents to
children, so topological order means **every node appears after all of its parents**.

Reverse it, and you get: **every node appears after all of its children** — precisely
the correctness condition. So:

```
backward pass = reverse(topological_sort(graph))
```

For our running example, one valid topological order is

$$[\,x,\; w,\; u,\; b,\; v,\; L\,]$$

and reversed:

$$[\,L,\; v,\; b,\; u,\; w,\; x\,]$$

Walk that list, calling each node's `_backward()`, and every gradient is complete
before it is used. (Topological orders aren't unique — $[w, x, u, b, v, L]$ works too.
Any of them is fine; the guarantee is all we need.)

### How we compute it

Depth-first search with post-order output: visit a node, recurse into all its parents
*first*, then append the node. A node is appended only after its entire ancestry, so
the result is topologically ordered by construction. We keep a `visited` set so shared
nodes are emitted exactly once — which matters enormously, since without it a diamond
graph would be traversed exponentially.

One engineering note: the textbook implementation is recursive, and Python's default
recursion limit (~1000) would blow up on a deep graph — and a scalar engine makes
*very* deep graphs. **We implement it iteratively with an explicit stack**, so depth is
bounded by heap memory rather than the C stack. This is a real, tested difference
from the textbook version. See `nabla/core/graph.py`.

### Why this doesn't conflict with accumulation

They solve two different halves of the same problem:

- **Topological order** guarantees a node is processed at the *right time* — after all
  its consumers.
- **Accumulation** guarantees that when it is processed, it has collected *all* the
  contributions those consumers sent.

Order without accumulation: last writer wins, contributions lost.
Accumulation without order: sums computed from incomplete inputs.
You need both, and together they are provably correct on any DAG.

---

## Where this goes next

Everything above is implemented, in order:

| Concept | Lands in |
|---|---|
| Nodes, local derivatives | **Phase 2** — `nabla/core/value.py` |
| The graph, topological sort | **Phase 3** — `nabla/core/graph.py` |
| Reverse sweep, accumulation | **Phase 4** — `Value.backward()` |
| Finite-difference verification | **Phase 5** — `nabla/core/gradcheck.py` |
| Seeing the graph | **Phase 6** — `nabla/visualization/` |
| Scalar → tensor autodiff | **Phase 18** — `nabla/core/tensor.py` |

---

### Quick self-test

If you can answer these without looking, Phase 1 has done its job.

1. Why is reverse mode preferred over forward mode for a loss function with a million parameters?
2. Why must the backward pass use `+=` rather than `=`?
3. What goes wrong if you traverse the graph in an arbitrary order?
4. Why does `tanh`'s backward rule use the *output* rather than the input — and what does that cost you?
5. Why can't symbolic differentiation handle a `while` loop whose trip count depends on the input?
6. Why do we implement finite differences even though we never train with them?
