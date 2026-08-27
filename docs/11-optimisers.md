# 11. Optimisers

> Implemented in [`nabla/optim/`](../nabla/optim/).
> Tested by [`tests/test_optimizers.py`](../tests/test_optimizers.py) (46 tests).
> Measured in [chapter 20 §4](20-experiments.md#4-optimisers).

The engine produces gradients. The optimiser decides what to do with them. All
three below receive **identical gradients** — they differ only in the update
rule.

---

## 1. Gradient descent

$$\theta_{t+1} = \theta_t - \eta\,g_t, \qquad g_t = \nabla_\theta \mathcal{L}$$

```python
for p in self.params:
    p.data -= self.lr * p.grad
```

Two lines. The gradient points in the direction of steepest **increase**, so we
move against it.

Note what the gradient does *not* tell you: how far to go. It is a purely local
quantity — the slope at a point carries no information about how long that slope
continues. The learning rate $\eta$ is a guess at that distance, and it is the
most consequential number in training
([chapter 20 §1](20-experiments.md#1-learning-rate)).

### Stochastic

We compute the gradient on a mini-batch, not the whole dataset. It is a noisy
estimate of the true gradient, with standard error falling as $1/\sqrt{B}$.

Three consequences, and the third surprises people:

1. Updates are far cheaper, so we take many more of them.
2. Noise helps escape saddle points, which vastly outnumber local minima in high
   dimensions.
3. **Noise acts as a regulariser** — jitter discourages settling into sharp
   minima, and often *improves* generalisation. A worse gradient estimate can
   produce a better model. See [chapter 20 §5](20-experiments.md#5-batch-size).

## 2. Momentum

$$v_{t+1} = \beta v_t + g_t, \qquad \theta_{t+1} = \theta_t - \eta\,v_{t+1}$$

```python
self.velocity[i] = self.momentum * self.velocity[i] + p.grad
p.data -= self.lr * self.velocity[i]
```

**The problem it solves.** Picture a *ravine* — a direction where the loss drops
slowly, flanked by steep walls. The gradient points mostly **across** the ravine
rather than along it. Plain SGD bounces between the walls and creeps along the
floor. Real loss surfaces are full of ravines, because parameters differ wildly
in how much they affect the output.

**Why the fix works.** Across the ravine, successive gradients alternate sign and
**cancel in the running sum**. Along the ravine they agree and **accumulate**.
The same update simultaneously damps the oscillation and accelerates the useful
direction.

With $\beta = 0.9$, the effective step along a consistent direction approaches
$10\eta$ — the geometric series $1/(1-\beta)$.

**Measured:** at the same lr=0.1, momentum turned 93.23% into 94.43%. One extra
array per parameter, three changed lines, 1.2 points. It is the best
value-for-complexity trade in the optimiser literature.

## 3. Adam

Track a decaying mean and a decaying mean-square of each gradient:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

$$\theta_{t+1} = \theta_t - \eta\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$$

**The key idea is the division.** Dividing by $\sqrt{\hat v}$ makes the step
roughly **scale-invariant**: a parameter with consistently tiny gradients gets
the same-sized step as one with huge gradients. Each parameter effectively gets
its own learning rate.

### Bias correction, which is not decoration

$m$ and $v$ start at zero. After one step:

$$m_1 = (1-\beta_1)g = 0.1\,g$$

That is **ten times too small** — the estimate is biased toward its zero
initialisation. $v_1$ is biased by a factor of 1000 with $\beta_2 = 0.999$.
Without correction, early steps are badly wrong in a way that can wreck a run
before it starts.

Dividing by $1 - \beta_1^t$ exactly undoes it: at $t=1$ that is $1-0.9 = 0.1$,
recovering $g$. As $t$ grows the correction → 1 and quietly stops mattering.

### $\epsilon$

Prevents division by zero when a gradient has been zero for a long time.
Default `1e-8`. It also caps the maximum step, which is a second, less obvious
job.

## 4. What Adam actually buys you

Our measured result contradicts the folk wisdom, so it is worth stating plainly:

| optimiser | test accuracy | val acc after 1 epoch |
|---|---|---|
| **SGD lr=0.5** | **94.77% ± 0.39** | 78.67% |
| momentum lr=0.1 | 94.43% ± 0.10 | 88.87% |
| **Adam lr=1e-3** | **94.60% ± 0.15** | 87.07% |

**Well-tuned SGD matched Adam.** Overlapping error bars — a tie, and certainly
no Adam advantage in final accuracy.

So what is Adam for? Look at the last column, and at [chapter
20 §1](20-experiments.md#1-learning-rate): **Adam stayed usable across two orders
of magnitude of learning rate; SGD across about one.**

You are not buying a better answer. You are buying **robustness to a
hyperparameter nobody can compute in advance**. On a new problem where you have
no idea what learning rate is right, that is worth a great deal. Once you *have*
tuned, the advantage largely evaporates.

The price: Adam stores two extra arrays per parameter — **3× the memory of plain
SGD**. Irrelevant for 109k parameters. Frequently the binding constraint at
billions, where optimiser state is often why training does not fit in memory.

**One caveat, stated because omitting it would mislead:** 8 epochs on 8,000
samples systematically favours fast starters. Given a longer budget, tuned SGD
with momentum often catches up and sometimes generalises slightly better.

## 5. Shared machinery

`Optimizer` provides what every optimiser needs:

```python
optimizer.zero_grad()             # must be called each step -- see chapter 5
optimizer.step()
optimizer.gradient_norm()         # health check
optimizer.clip_grad_norm(1.0)     # rescale if the norm exceeds a threshold
optimizer.state_dict()            # for checkpointing
```

**Gradient clipping** deserves a note. If $\|g\| > \text{max}$, rescale so it
equals `max`. This preserves the *direction* while capping the *distance* — it
does not distort which way you go, only how far. It is the standard defence
against the occasional huge gradient that would otherwise destroy a run in one
step.

`AdamW` is also provided: it decouples weight decay from the adaptive scaling.
In plain Adam, L2 regularisation added to the loss gets divided by
$\sqrt{\hat v}$ along with everything else, so the effective decay differs per
parameter — which is not what anyone means by weight decay.

## 6. None of this is PyTorch's

Every optimiser here is implemented from the update equations above. No
`torch.optim`. The gradients come from our engine and the updates are applied by
our code, which is the whole point — `p.data -= lr * p.grad` is not magic, and
after this chapter it should not feel like it.

---

**Next:** [12. The training loop](12-training.md)
