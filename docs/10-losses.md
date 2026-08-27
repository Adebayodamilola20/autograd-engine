# 10. Loss functions

> Implemented in [`nabla/losses/`](../nabla/losses/).
> Tested by [`tests/test_losses.py`](../tests/test_losses.py) (63 tests).

The loss is where learning is *defined*. Everything else — the graph, the chain
rule, the optimiser — is machinery for reducing a number that this chapter
chooses.

---

## 1. Mean squared error

$$\mathcal{L} = \frac{1}{N}\sum_i (\hat{y}_i - y_i)^2 \qquad \frac{\partial \mathcal{L}}{\partial \hat{y}_i} = \frac{2(\hat{y}_i - y_i)}{N}$$

Squaring makes errors positive and penalises large errors disproportionately.
The gradient is proportional to the error, which is the property you want: badly
wrong predictions produce large corrections.

MSE is right for **regression**, where outputs are unbounded real numbers.

### Why not MSE for classification

You can use MSE with one-hot targets. It works badly, and the reason is
instructive.

Pair MSE with a sigmoid output. The gradient carries a factor of $\sigma'$,
which is ≈ 0 when the network is confidently wrong. **The more wrong it is, the
less it learns** — exactly backwards. Cross-entropy's gradient, derived below,
has no such factor.

There is also a modelling objection: MSE treats class indices as numbers, so
predicting 8 when the answer is 1 is "worse" than predicting 2. Digit labels are
categories, not magnitudes.

## 2. Logits, softmax, and probabilities

The output layer emits **logits** — unbounded real scores, one per class, with
no constraint to be positive or sum to anything.

**Softmax** turns them into a probability distribution:

$$p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$$

`exp` makes everything positive; dividing by the sum makes it total 1. Softmax is
**shift-invariant** — adding a constant to every logit changes nothing, since the
constant factors out of numerator and denominator. That fact is not a curiosity;
it is what makes §4 possible.

## 3. Cross-entropy

For a true distribution $q$ and a predicted $p$:

$$H(q, p) = -\sum_k q_k \ln p_k$$

With a one-hot target, all terms vanish except the true class:

$$\mathcal{L} = -\ln p_y$$

Read it directly: **the loss is the negative log of the probability assigned to
the correct answer.**

| $p_y$ | loss |
|---|---|
| 1.0 | 0 — perfect |
| 0.5 | 0.69 |
| 0.1 | 2.30 — chance, for 10 classes |
| 0.01 | 4.61 |
| → 0 | → ∞ |

That unbounded tail is the point. Confident *and wrong* is punished without
limit, so the network learns to be uncertain when it should be.

### The derivative that makes it all work

This derivation is worth doing once, because the result is startling.

Combining softmax and cross-entropy, for $\mathcal{L} = -\ln p_y$ where
$p = \text{softmax}(z)$:

$$\boxed{\;\frac{\partial \mathcal{L}}{\partial z_i} = p_i - y_i\;}$$

where $y$ is the one-hot target. That is it. **Predicted minus actual.**

No $\sigma'$ factor, no saturation term, nothing that vanishes when the network
is confidently wrong. If the model says 0.99 for class 3 and the answer is 7,
the gradient on logit 3 is +0.99 — a large, corrective push. This is precisely
the failure mode MSE-plus-sigmoid has, and cross-entropy simply does not.

It is also why `F.cross_entropy` in PyTorch takes **integer targets rather than
one-hot**: the fused kernel computes $p - y$ directly without materialising the
one-hot vector.

We do **not** hand-write this rule. We build the loss out of `exp`, `log`, `sum`
and `div`, and the chain rule produces $p - y$ on its own. `tests/test_losses.py`
asserts that our engine's gradient matches the closed form — which is a genuinely
satisfying test to watch pass, because nobody typed the answer in.

## 4. Numerical stability, which is not optional

The naive implementation:

```python
probs = [exp(z) for z in logits]      # overflows
total = sum(probs)
loss = -log(probs[target] / total)
```

`exp(z)` overflows to `inf` at **z ≈ 710** in float64. Then `inf/inf = NaN`,
NaN propagates through every subsequent operation, and every parameter becomes
NaN on the next update. The run is unrecoverable and the error message points
nowhere useful.

The fix uses softmax's shift-invariance. Subtract the maximum logit:

$$\ln p_k = z_k - m - \ln\sum_j e^{z_j - m}, \qquad m = \max_j z_j$$

Now the largest exponent is always $e^0 = 1$. **Overflow becomes impossible by
construction** — not unlikely, impossible. Underflow of the small terms is
harmless: they were negligible anyway.

Working in **log space** avoids the other trap. `log(softmax(z))` computes a
probability, possibly rounds it to 0, then takes `log(0) = -inf`. `log_softmax`
never forms the probability at all.

### This paid off, measurably

[Experiment 1](20-experiments.md#the-nan-that-never-came) pushed SGD to a
learning rate of 1000. First-epoch loss reached the order of **1e57** — and
never became NaN. The run completed.

The model was thoroughly destroyed at that learning rate, but the *arithmetic*
never broke. Those are different failures, and conflating them is why "my loss
went NaN" gets misdiagnosed as a learning-rate problem when it is often a
stability bug — or the reverse.

## 5. Binary cross-entropy

For two classes:

$$\mathcal{L} = -\bigl[y\ln p + (1-y)\ln(1-p)\bigr]$$

Same idea, and the same stability concern: implemented from logits rather than
from a sigmoid output, so `log(0)` cannot occur.

## 6. What the library provides

```python
from nabla.losses import (
    mse_loss, mae_loss,
    log_sum_exp, log_softmax, softmax,
    nll_loss, cross_entropy, softmax_cross_entropy,
    binary_cross_entropy,
)
```

`softmax_cross_entropy(logits, targets)` is what MNIST uses. It takes **logits,
not probabilities** — the softmax lives inside the loss, which is why the output
layer is linear ([chapter 8](08-neural-network.md#3-the-mlp)). Splitting them
would force the unstable path.

The tensor versions in `losses/tensor_losses.py` compute the same mathematics for
a whole batch. The one interesting difference: the vectorised form multiplies by
a constant one-hot **matrix** and sums, rather than indexing — the same number,
but a single expression built only from operations the engine already has. No new
backward rule.

---

**Next:** [11. Optimisers](11-optimisers.md)
