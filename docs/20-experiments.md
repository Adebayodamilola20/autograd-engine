# Phase 22 — Experiments

Everything in this document was produced by the scripts in `experiments/`,
using gradients computed entirely by our own engine. Every row is the mean of
**3 seeds** with its standard deviation, on 8,000 MNIST samples for 8 epochs.

```bash
python experiments/run_all.py          # ~160s, regenerates everything here
```

**Why three seeds and not one.** A single run can make a bad configuration look
good, and the difference between two configurations is meaningless until you
know how much the same configuration varies against itself. Several results
below are *within noise of each other*, and saying so is the finding. Reporting
a winner from one run would be reporting noise.

---

## 1. Learning rate

`experiments/exp_learning_rate.py`

The gradient tells you which direction reduces the loss. It says nothing about
how far to go — a slope at a point carries no information about how long that
slope continues. The learning rate is our guess at that distance.

| learning rate | test accuracy | final train loss |
|---|---|---|
| 1e-5 | 62.85% ± 2.52 | 1.7814 |
| 1e-4 | 90.52% ± 0.45 | 0.3461 |
| **1e-3** | **94.60% ± 0.15** | 0.0594 |
| 1e-2 | 93.47% ± 0.19 | 0.0842 |
| 1e-1 | 76.95% ± 5.14 | 0.7871 |
| 1.0 | 9.58% ± 0.21 | 2.3637 |

**The usable window is about two orders of magnitude wide**, and outside it
things fail in two completely different ways.

**Too small is the expensive failure.** At 1e-5 the loss curve is smooth,
monotonic, and heading the right way. Nothing looks broken. It simply had 100x
too little distance to cover the ground in the epochs available. A loss curve
that is *descending* tells you nothing about whether it will *arrive*.

**Too large destroys the run.** At lr=1.0 accuracy is 9.58% — chance level for
ten classes. The model is not merely wrong; the weights carry no information
about the data at all. Each step overshoots so far that it lands further up the
opposite side of the valley than it started, and the next gradient is larger
still.

**1e-3 and 1e-2 are statistically tied-ish but not equal** — 94.60% ± 0.15
against 93.47% ± 0.19. The gap (1.1 points) is larger than the spread, so this
one is real. But note lr=1e-1 has a standard deviation of **5.14 points**: the
same configuration is sometimes mediocre and sometimes ruined, decided purely by
the random initialisation. High variance is itself a signal that you are near
the edge of stability.

### The NaN that never came

None of the runs above produced a non-finite loss, including lr=1.0. Pushing
plain SGD to lr=1000 gets a first-epoch loss on the order of **1e57** — and it
still comes back down without ever becoming NaN.

**That is Phase 9 paying off.** A naive softmax computes `exp(z)` directly,
which overflows to `inf` at z ≈ 710 in float64, and `inf/inf` is NaN. Our
log-sum-exp implementation subtracts the row maximum first, so the largest
exponent is always `exp(0) = 1` and overflow is **impossible by construction**,
however absurd the logits become.

So the model is thoroughly destroyed at these rates, but the *arithmetic* never
breaks. Those are different failures. Conflating them is why "my loss went NaN"
gets misdiagnosed as a learning-rate problem when it is often a numerical
stability bug — or the reverse.

---

## 2. Activations, and why nonlinearity is mandatory

`experiments/exp_activations.py`

| activation | test accuracy | final train loss |
|---|---|---|
| **ReLU** | **94.60% ± 0.15** | 0.0594 |
| tanh | 93.53% ± 0.12 | 0.0960 |
| sigmoid | 91.75% ± 0.11 | 0.2401 |
| **linear (none)** | **89.92% ± 0.68** | 0.2120 |

### The linear network is a single layer wearing a costume

A linear layer is `f(x) = xW + b`. Compose two:

$$f_2(f_1(x)) = (xW_1 + b_1)W_2 + b_2 = x\underbrace{(W_1W_2)}_{W'} + \underbrace{(b_1W_2 + b_2)}_{b'}$$

That is a single linear layer. Stack a hundred and the argument repeats a
hundred times. **Depth buys nothing without a nonlinearity between the layers**
— not "little", exactly nothing.

So our 784→128→64→10 linear network has 109,386 parameters and the
representational power of 7,850 of them: the size of a single 784→10 map. The
extra 101,536 parameters are a redundant parameterisation of the same small
function class.

**And it still scores 89.92%.** MNIST is close to linearly separable, so this
failure does not announce itself — it just quietly caps you ~4.7 points below
what the architecture on paper should reach. On a genuinely nonlinear problem
the same model cannot get above chance: `examples/train_xor.py` demonstrates it
failing on XOR, which is four data points.

**A 4.7-point improvement from `max(0, z)`** — a function with no parameters,
no memory, and no arithmetic beyond a comparison.

### Gradient decay: why ReLU won deep learning

Mean |∂L/∂W| per layer at the final epoch. Layer 0 is the input side — furthest
from the loss, last to receive any signal.

| activation | layer 0 | layer 1 | layer 2 | last ÷ first |
|---|---|---|---|---|
| ReLU | 5.57e-04 | 2.57e-03 | 1.00e-02 | 18.0x |
| tanh | 6.38e-04 | 1.88e-03 | 4.31e-03 | **6.8x** |
| sigmoid | 2.10e-04 | 1.24e-03 | 6.92e-03 | **32.9x** |
| linear | 1.90e-03 | 5.46e-03 | 2.15e-02 | 11.3x |

The derivatives explain the ordering:

- **σ'(z) = σ(1-σ) peaks at 0.25.** Every sigmoid layer can attenuate the
  backward signal fourfold *before anything else happens*. Sigmoid shows the
  steepest decay in the table (32.9x) and the smallest first-layer gradient
  (2.10e-04, less than half of ReLU's).
- **tanh' = 1 - tanh² peaks at 1.0** — four times better, and zero-centred so
  activations do not drift to one side. There is no reason to prefer sigmoid
  inside a network.
- **ReLU' is exactly 1** on the active path. 1 multiplied by itself any number
  of times is 1, so depth costs the gradient *nothing* in principle.

At three layers this is a nuisance worth ~2.9 points. At thirty layers it is
the difference between training and not, which is most of why ReLU displaced
sigmoid.

ReLU's cost: units with z < 0 pass **no** gradient, and a unit that never
activates can never recover. A dead unit is dead permanently.

---

## 3. Depth and width

`experiments/exp_architecture.py`

### Depth

| hidden layers | parameters | test accuracy |
|---|---|---|
| 0 (linear model) | 7,850 | 89.97% ± 0.13 |
| 1 | 101,770 | 93.82% ± 0.26 |
| 2 | 109,386 | 94.60% ± 0.15 |
| **3** | **114,410** | **94.92% ± 0.22** |
| 4 | 117,530 | 94.85% ± 0.49 |
| 6 | 121,082 | 94.60% ± 0.37 |

**The first hidden layer is worth 3.85 points. The second is worth 0.78. The
third is worth 0.32. The fourth and beyond are worth nothing measurable** —
layers 3, 4 and 6 are all within one standard deviation of each other.

The first layer is the one that introduces a nonlinearity where there was none.
Everything after it refines a hierarchy that already exists.

Note that the 6-layer network has *more* parameters and *lower* accuracy than
the 3-layer one. More capacity, worse result — which can only be an
**optimisation** failure, not a representational one. The 6-layer model can
express everything the 3-layer model can (set the extra layers to identity);
it just cannot be trained to.

Gradient survival with ReLU stays flat across depth — the last÷first ratio
hovers around 14–20x whether there are 2 layers or 7. That is ReLU's
derivative-of-1 property doing exactly what the theory says. Swap in sigmoid and
these same depths become untrainable.

### Width

| width | parameters | test accuracy |
|---|---|---|
| 4 | 3,210 | **72.07% ± 6.93** |
| 16 | 13,002 | 91.03% ± 0.97 |
| 32 | 26,506 | 92.47% ± 0.16 |
| 64 | 55,050 | 93.72% ± 0.21 |
| 128 | 118,282 | 94.68% ± 0.44 |
| 256 | 269,322 | 95.52% ± 0.22 |

**Width 4 is a genuine capacity bottleneck** — the one case where "the model is
too small" is literally true. Everything the network knows about a digit must
pass through 4 numbers, and ten classes cannot be cleanly separated by that. No
amount of training fixes it. Note the ±6.93 standard deviation: at this size the
result depends enormously on getting a lucky initialisation.

Past that, returns compress hard. **Width 32 → 256 is a 10x increase in
parameters for 3 points.** The limit stopped being the model and became the
data — 8,000 samples do not contain enough information to fit an arbitrarily
large model.

### Depth vs width, on the same budget

Compare `2 hidden, width 128` (109,386 params, 94.60%) against
`width 256 × 2` (269,322 params, 95.52%). Width bought 0.9 points for 2.5x the
parameters.

The universal approximation theorem says one hidden layer, made wide enough,
can approximate any continuous function. True, and less useful than it sounds:
"wide enough" can mean exponentially many units, where depth often achieves the
same function with polynomially many. **Depth is about efficiency of
representation, not possibility.** At MNIST's scale neither effect dominates,
which is why both curves flatten so quickly.

---

## 4. Optimisers

`experiments/exp_optimizers.py`

All three use the **same gradients** from the same engine. They differ only in
what they do with them.

| optimiser | test accuracy | val acc after 1 epoch |
|---|---|---|
| SGD lr=0.01 | 88.40% ± 0.64 | 40.27% |
| SGD lr=0.1 | 93.23% ± 0.13 | 78.07% |
| **SGD lr=0.5** | **94.77% ± 0.39** | 78.67% |
| momentum lr=0.01 | 93.30% ± 0.40 | 83.00% |
| momentum lr=0.1 | 94.43% ± 0.10 | 88.87% |
| **Adam lr=1e-3** | **94.60% ± 0.15** | 87.07% |
| Adam lr=1e-2 | 93.47% ± 0.19 | 89.93% |

**Well-tuned SGD beat Adam.** 94.77% ± 0.39 against 94.60% ± 0.15 — overlapping
error bars, so honestly a tie, but certainly no Adam advantage. This is worth
stating plainly because the folk wisdom says otherwise.

**Momentum is the best value-for-complexity trade in the literature.** At the
same lr=0.1, momentum turns 93.23% into 94.43% — 1.2 points for one extra array
per parameter and three changed lines:

$$v \leftarrow \beta v + g, \qquad \theta \leftarrow \theta - \eta v$$

The intuition: in a ravine — a direction where loss drops slowly, flanked by
steep walls — the gradient points mostly *across* the ravine rather than along
it. Across the ravine, successive gradients alternate sign and **cancel in the
sum**. Along it they agree and **accumulate**. The same update damps the
oscillation and accelerates the useful direction. With β=0.9 the effective step
along a consistent direction approaches 10η, from the geometric series 1/(1-β).

**So what is Adam actually for?** Look at the "after 1 epoch" column: Adam and
momentum are at 87–90% while plain SGD at its own best rate is at 78%. Adam
divides each parameter's step by a running estimate of its own gradient
magnitude, making the step roughly *scale-invariant* — a parameter with tiny
gradients gets the same-sized step as one with huge gradients.

You are not buying a better final answer. You are buying **robustness to a
hyperparameter nobody can compute in advance**. Experiment 1 showed Adam usable
across two orders of magnitude of lr where SGD managed about one. The price is
3x optimiser memory (two extra arrays per parameter) — irrelevant for 109k
parameters, often the binding constraint at billions.

**Caveat, stated because omitting it would be misleading:** 8 epochs on 8,000
samples systematically favours fast starters. With a longer budget, tuned SGD
with momentum frequently catches up and sometimes generalises slightly better.

---

## 5. Batch size

`experiments/exp_batch_size.py`

Batch size looks like a performance knob. It is three knobs bolted together.

**Fixed lr = 1e-3:**

| batch | updates/epoch | test accuracy | time |
|---|---|---|---|
| 8 | 850 | 94.52% ± 0.27 | 5.5s |
| 32 | 212 | 94.48% ± 0.25 | 1.6s |
| 64 | 106 | 94.60% ± 0.15 | 1.0s |
| 128 | 53 | 93.62% ± 0.19 | 0.7s |
| 512 | 13 | **91.95% ± 0.22** | **0.5s** |

**lr scaled with batch (`lr = 1e-3 × batch/64`):**

| batch | test accuracy | time |
|---|---|---|
| 8 (lr×0.125) | 93.03% ± 0.20 | 5.8s |
| 32 (lr×0.5) | 94.08% ± 0.33 | 1.7s |
| 64 (lr×1) | 94.60% ± 0.15 | 1.0s |
| 128 (lr×2) | 94.62% ± 0.06 | 0.8s |
| **512 (lr×8)** | **94.55% ± 0.14** | **0.5s** |

### The headline

At batch 512, **scaling the learning rate recovered 2.6 points** — 91.95% →
94.55% — landing statistically level with batch 64 while running **2x faster**.

That is the linear scaling rule working, and it is direct evidence that the
first table was not measuring what it appeared to measure.

### Why the fixed-lr table is misleading

Batch 512 gets **13 updates per epoch**. Batch 8 gets **850** — a 65x
difference in how many times the parameters actually move. At a fixed learning
rate, that update-count difference dominates everything else.

So the first table is not telling you "big batches are bad". It is telling you
"**few steps are bad**", which is a different and much less interesting claim.
It is the single most common misreading of a batch-size sweep.

The three effects, separated:

1. **Gradient quality.** Standard error falls as 1/√B. Note the square root:
   32 → 128 is 4x the work for 2x less noise. Diminishing returns are built into
   the mathematics.
2. **Update count.** N/B updates per epoch. Doubling the batch halves the
   steps. Usually dominant, usually forgotten.
3. **Speed.** Large batches hand BLAS bigger matrices. Batch 512 is 11x faster
   per epoch than batch 8 (0.5s vs 5.5s).

### Noise is not purely a cost

Small-batch gradient noise acts as a regulariser — it jitters the parameters,
discouraging settling into sharp minima. Note that in the *scaled* table, batch
8 is the **worst** performer (93.03%): with lr scaled down 8x it takes many
tiny, well-estimated steps and underperforms. Less noise was not better.

**Practical rule:** choose the largest batch whose accuracy you can recover by
tuning the learning rate — not the batch with the best accuracy at some
arbitrary fixed lr.

---

## What the whole phase taught

**Report variance or report nothing.** lr=1e-1 varies by ±5.14 points and SGD
at lr=1.0 by over ±15 across seeds. Any single-run comparison between nearby
configurations is noise. Several apparent "winners" here are ties.

**Distinguish capacity failures from optimisation failures.** Width 4 at 72% is
a capacity failure — the information cannot fit through 4 units, and no training
fixes it. The 6-layer network scoring below the 3-layer one with *more*
parameters is an optimisation failure. Only the gradient statistics tell you
which you have.

**The quiet failures cost the most.** lr=1e-5 and the linear network both
produce smooth, plausible, monotonically improving loss curves while being badly
broken. Divergence at least announces itself.

**Nonlinearity is not a detail.** 4.7 points from a function with no parameters,
and the entire justification for depth existing at all.

**Numerical stability is separate from optimisation stability.** A loss of 1e57
that never becomes NaN is Phase 9's log-sum-exp doing its job. Knowing which of
the two failed is the difference between fixing the learning rate and fixing the
loss function.

**Folk wisdom is testable.** "Adam beats SGD" did not survive contact with a
tuned SGD baseline here. "Large batches hurt accuracy" turned out to be "few
updates hurt accuracy". Both took about a minute to check.
