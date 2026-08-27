# 6. Gradient checking

> Implemented in [`nabla/core/gradcheck.py`](../nabla/core/gradcheck.py).
> Used by [`tests/test_gradients.py`](../tests/test_gradients.py) — 183 checks.
> Runnable: `python examples/gradient_check.py`

**An unchecked derivative is a guess that happens to be typed in monospace.**

This chapter is about the tool that makes every other claim in the project
trustworthy.

---

## 1. Why unit tests are not enough

The obvious objection: we already test gradients against hand-computed values
in `test_backward.py`. Why more?

Because those tests are **circular**. The person who derived
`∂(x·y)/∂x = y` is the same person who wrote the test asserting it. If the
derivation was wrong, the test is wrong in exactly the same way, and it passes.

A sign error in one backward rule produces gradients that are exact solutions to
the *wrong problem*. The forward pass will not complain. Training will still
run, the loss will still go down — just slower, or to a worse place. It is a bug
that looks like a hyperparameter problem, which makes it nearly impossible to
find by observation.

Finite differences break the circle. They compute the derivative from its
**definition as a limit**, using nothing but the forward pass. They share no
code and no reasoning with the analytic rules.

When two independent methods agree to ten significant figures across dozens of
operations and random inputs, the derivations are right.

## 2. Central differences, and why not the obvious one

The definition of a derivative gives the obvious estimator:

$$f'(x) \approx \frac{f(x+h) - f(x)}{h}$$

We use the **central** difference instead:

$$f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}$$

Here is why, and it is worth doing the algebra once. Taylor-expand both terms:

$$f(x+h) = f(x) + hf'(x) + \tfrac{h^2}{2}f''(x) + \tfrac{h^3}{6}f'''(x) + \cdots$$
$$f(x-h) = f(x) - hf'(x) + \tfrac{h^2}{2}f''(x) - \tfrac{h^3}{6}f'''(x) + \cdots$$

Subtracting cancels **both** the $f(x)$ term and the $f''$ term exactly:

$$\frac{f(x+h)-f(x-h)}{2h} = f'(x) + \frac{h^2}{6}f'''(x) + O(h^4)$$

So the error is $O(h^2)$. The one-sided version keeps the $f''$ term and is only
$O(h)$. At $h = 10^{-5}$ that is the difference between roughly **10 correct
digits and 5**.

Two extra function evaluations for five extra digits is an excellent trade.

## 3. Choosing h: a genuine trade-off

Two error sources pull in opposite directions:

- **Truncation error** ~ $h^2$ — wants `h` **small**.
- **Round-off error** ~ $\varepsilon/h$ — wants `h` **large**. As `h` shrinks,
  $f(x+h)$ and $f(x-h)$ become nearly equal, and subtracting two nearly-equal
  floats destroys significant digits. This is **catastrophic cancellation**.

Minimising $h^2 + \varepsilon/h$ gives $h \approx \varepsilon^{1/3} \approx
6\times10^{-6}$ for float64. We default to `eps=1e-5`.

`epsilon_sweep()` measures this U-curve directly rather than asserting it —
`python examples/gradient_check.py` prints it. Error falls as `h` decreases,
bottoms out around $10^{-5}$–$10^{-6}$, then **rises again** as round-off takes
over. Seeing that curve is worth more than reading the paragraph.

**This is exactly the trade-off reverse mode does not have.** Autodiff is exact
to machine precision at any scale. It is why we *test* with finite differences
and never *train* with them.

## 4. Relative error, not absolute

Comparing gradients needs a scale-invariant measure:

$$\text{rel} = \frac{|a - n|}{\max(|a|, |n|, \delta)}$$

An absolute difference of `1e-4` is catastrophic when the gradient is `1e-6` and
irrelevant when it is `1e6`. The $\delta$ floor stops division by zero when both
are legitimately zero.

Thresholds we use:

| relative error | verdict |
|---|---|
| < 1e-7 | exact agreement |
| < 1e-5 | **pass** — the default |
| 1e-5 … 1e-3 | suspicious; usually a genuinely non-smooth point |
| > 1e-3 | **the backward rule is wrong** |

## 5. Using it

```python
from nabla.core.gradcheck import check_gradients

result = check_gradients(lambda x, y: (x * y + 1.0) ** 2, [2.0, 3.0])
print(result)          # PASS  max rel err 3.1e-11
```

And directly on network parameters, which is what actually matters:

```python
from nabla.core.gradcheck import check_parameter_gradients

model = MLP(2, [4], 1)
result = check_parameter_gradients(
    lambda: mse_loss([model(x) for x in xs], ys),
    model.parameters(),
)
assert result.passed
```

That second form is the important one. It checks the gradient of the **real
loss** with respect to the **real parameters**, through the full network —
every layer, every activation, the loss function, and the accumulation across
branching paths, all at once. `examples/train_xor.py` runs it before training,
so a broken engine fails loudly instead of training badly.

## 6. What it caught

Two categories of real bug, both from this project:

**Non-smooth points.** `relu` at exactly `x = 0` fails a gradient check, and it
*should* — the function genuinely has no derivative there. Finite differences
straddle the kink and return `0.5`; we return `0`. The fix was not to the rule
but to the test: check ReLU away from zero, and pin the convention at zero
separately. A gradient checker that never complains is not checking anything.

**Scale.** `exp` at large inputs shows elevated relative error, because
$f(x+h)$ and $f(x-h)$ differ hugely and the finite-difference estimate itself
is imprecise. Here the *checker* is wrong, not the engine — worth understanding
before "fixing" a rule that was correct.

Both cases make the same point: gradient checking tells you the two methods
disagree. Working out **which one is wrong** is still your job.

## 7. Coverage

`tests/test_gradients.py` runs 183 checks:

- every operator, at several inputs including negative and near-zero
- every activation function
- every loss function
- compositions and deeply nested expressions
- variables used on multiple paths (the accumulation case)
- full `Neuron`, `Layer`, and `MLP` parameter gradients
- the tensor engine, against the scalar engine *and* against finite differences

If any derivation in [chapter 3](03-operators.md) were wrong, these fail.

---

**Next:** [7. Visualising the graph](07-visualisation.md)
