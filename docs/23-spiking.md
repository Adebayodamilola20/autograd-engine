# 23. Spiking networks, and an A/B test that the new idea lost

This chapter does two things. It explains surrogate-gradient training for
spiking neural networks, which is the one place in this repository where a
backward pass is deliberately *not* the derivative of its forward pass. And it
reports an A/B test against the existing dense network, which the spiking
network lost on both axes it was supposed to win on.

The negative result is the point. It is reported here in full because a
comparison you only publish when it comes out the way you wanted is not a
comparison.

## The problem: a derivative that is zero everywhere

A spiking neuron fires when its membrane potential crosses a threshold. As a
function, that is the Heaviside step:

```
S = Θ(U - θ)
```

Its derivative is zero everywhere it exists, and undefined at the one point
that matters:

```
∂S/∂U = 0   for U ≠ θ,   undefined at U = θ
```

Put that in the chain rule and every gradient upstream is multiplied by zero.
The network does not train badly. It does not train at all. This is the **dead
neuron problem**, and it is why spiking networks were trained for decades by
local biological rules rather than by gradient descent.

`tests/test_spiking.py::TestTheDeadNeuronProblem` states it as a test, because
it is the premise everything else here rests on.

## The fix: forward and backward deliberately disagree

Use the true step function on the forward pass, so the network really is
binary and really is spiking. Substitute a smooth, non-zero derivative on the
backward pass:

```
S = Θ(U - θ)          but          ∂S/∂U := σ'(U - θ)
```

This is the only node in the repository whose backward is not its forward's
derivative, and the deviation is the entire technique rather than an
approximation for speed.

The justification: `Θ` is the limit of a family of smooth sigmoids as their
width goes to zero. The surrogate is the derivative of a member of that family
with non-zero width. So this is the exact gradient of a *nearby* network,
which is enough to descend on. It is the standard method in the field (Neftci,
Mostafa and Zenke, 2019).

Three surrogates are implemented, all normalised to peak at 1 on the threshold
so that changing sharpness does not silently rescale the learning rate:

| name | derivative | character |
|---|---|---|
| `fast_sigmoid` | `1/(1+β\|x\|)²` | heavy tails, recruits silent units |
| `atan` | `1/(1+(πβx/2)²)` | sharper peak, gradient concentrated near threshold |
| `sigmoid` | `4σ(βx)(1-σ(βx))` | the logistic bump, easiest to reason about |

### What can and cannot be verified

The whole repository sells gradient checking, and here it does not apply.
Finite-differencing the real forward function returns zero, because the real
forward function has zero gradient. That is not a failure of the check; it is
the thing being worked around.

What *is* checked is that each surrogate really is the derivative of the smooth
function it claims to stand for, by finite-differencing that function. It
catches an algebra slip in the surrogate, which is the failure mode actually
available. Pretending anything stronger was verified would be dishonest.

## The neuron

Leaky integrate-and-fire, in discrete time:

```
U[t] = λ·U[t-1] + I[t] - θ·S[t-1]
S[t] = Θ(U[t] - θ)
```

Three terms, each doing one job:

- `λ·U[t-1]` is the **leak**. Potential decays toward rest, so inputs that do
  not arrive close together fail to add up. This is what makes the neuron
  sensitive to timing rather than to a running total.
- `I[t]` is the **input current**, an ordinary weighted sum.
- `-θ·S[t-1]` is the **reset**. After firing, drop by one threshold rather than
  to zero: subtraction keeps the remainder above threshold instead of
  discarding it, and preserves a gradient path that reset-to-zero severs.

## BPTT is not an algorithm here

`U[t]` depends on `U[t-1]`, so unrolling `T` timesteps builds a graph that is
deep in time as well as in layers.

Nothing in `spiking.py` implements backpropagation through time. Writing the
recurrence as a Python loop **is** the unrolling, and `backward()` walks the
result because it is one graph like any other. BPTT is usually presented as
its own algorithm with its own derivation; on a define-by-run engine it is a
`for` loop.

## The A/B test

**One starting point, two arms.** Both get the identical memoised
train/validation/test split, identical layer sizes (784 → 128 → 10), an
identical parameter count (101,770 exactly), the same optimiser, the same
learning rate, the same batch size, the same number of epochs, and the same
seed per paired run. They differ in whether hidden activations are real numbers
or binary spikes in time, and in nothing else not forced by that choice.

**Frozen seeds**, three of them, each producing one dense and one spiking run
compared as a pair. **A held-out test set** of 10,000, touched once, after all
training and all model selection.

The pre-registered expectation was that the spiking network would lose on
accuracy and win on energy.

### Result

| arm | test accuracy | spikes/image | simulated energy |
|---|---|---|---|
| dense | **97.69% ± 0.12%** | n/a | **467.5 nJ** |
| spiking | 97.29% ± 0.11% | 1,423 | 489.1 nJ |

Paired accuracy difference: **-0.40% ± 0.22%**, negative on all three seeds
(-0.44%, -0.16%, -0.60%).

Energy ratio: **1.05x**, meaning the spiking network used *more*.

**It lost on accuracy, as expected. It also lost on energy, which was not
expected, and that is the more interesting half.**

### Why it lost on energy

The energy model (`nabla/nn/energy.py`) costs an operation count with the
standard published figures (Horowitz, ISSCC 2014, 45nm): 4.6 pJ for a 32-bit
float multiply-accumulate, 0.9 pJ for an accumulate alone.

The spiking argument is a single correct observation: when the input to a
synapse is a binary spike, `w·x` is either `w` or nothing, so there is no
multiply, and a synapse whose input did not spike does no work at all. Below
roughly 20% activity (`0.9/4.6`) a spiking layer wins.

The itemised budget shows where that argument runs out:

| term | ops/image | energy |
|---|---|---|
| static input layer (MACs) | 100,352 | **461.6 nJ** |
| spike-driven synapses | 14,556 | 13.1 nJ |
| membrane updates | 3,200 | 14.7 nJ |

The spike-driven part is genuinely cheap: 13.1 nJ against the dense network's
467.5 nJ, a 36x saving on that portion. It does not matter, because it is 3% of
the budget.

Two costs swamp it, and both are routinely omitted from published comparisons:

1. **The static input layer.** The image is injected as constant analog
   current, so the first layer is a full dense layer of real multiplies. It is
   computed once rather than `T` times, but with a 784-pixel input and one
   hidden layer of 128, that single term is **99% of the entire dense
   network's cost**. No amount of downstream sparsity can remove it.
2. **Membrane updates.** Every neuron does a decay multiply and an add every
   timestep, whether or not it spiked. That is `T × n_neurons` MACs, charged
   even to a completely silent network, and it alone exceeds the spike-driven
   saving.

The measured hidden spike rate was **44.5%** of neuron-timesteps, well above
the ~20% breakeven, so even the spike-driven portion was not operating in the
regime the argument assumes.

### The counterfactual worth naming

If the input were rate-coded into spikes rather than injected as analog
current, the first layer would become spike-driven too. At MNIST's mean pixel
intensity of 0.133 over 25 timesteps that is about 2,597 input spikes, giving
**326.7 nJ, or 0.70x the dense network**.

That is the configuration the energy argument actually assumes, and in it the
spiking network does win. It was not run as an arm here because rate coding is
stochastic and costs accuracy, which would then be charged to "spiking" when it
belongs to the encoder. It is reported as an estimate, clearly labelled, rather
than quietly folded into the headline.

## Was the spiking arm handicapped?

Matched hyperparameters are what make an A/B an A/B, but they can also hide a
better result for one arm. Two settings could be doing that: the shared
learning rate, and the single fixed `T`. Both were swept (one seed, 20,000
training images, so this is a check on the conclusion rather than a second
experiment).

**Learning rate**, against a dense reference of 96.49% on the same subset:

| learning rate | spiking test accuracy |
|---|---|
| 3e-4 | 95.26% |
| 1e-3 (shared) | 96.00% |
| **3e-3** | **96.26%** |

The spiking arm does prefer a higher learning rate than the shared one, and
tuning recovers about half the gap. It does not close it. **The gap survives
tuning**, so it is not an artefact of the matched setting.

**Timesteps**, which is the accuracy/energy dial the whole design turns on:

| T | test accuracy | spikes/image | energy | vs dense |
|---|---|---|---|---|
| 5 | 96.03% | 299 | 467.3 nJ | 1.00x |
| 10 | 95.99% | 602 | 472.9 nJ | 1.01x |
| 25 | 96.00% | 1,519 | 490.0 nJ | 1.05x |
| 50 | 95.97% | 3,016 | 518.2 nJ | 1.11x |

This was the surprise, and it is the sharpest result in the chapter.

**Accuracy is flat.** From T=5 to T=50, a tenfold increase in simulation
length and spike count, accuracy moves by 0.06%, which is well inside the
noise. Meanwhile energy rises monotonically. There is no trade-off curve here
to pick a point on: the extra timesteps buy nothing at all.

The reason is worth stating plainly. With a static image injected as constant
current, **every timestep sees exactly the same input**. There is no temporal
structure for the leak to integrate and nothing for spike timing to encode, so
the network converges to something close to a thresholded dense layer repeated
`T` times. The entire mechanism that distinguishes a spiking neuron from a
step function is dead weight on this task, and the measurement says so.

That also explains the 44.5% spike rate. Nothing in the objective rewarded
sparsity, and nothing in the data gave the temporal machinery anything to do,
so the network had no reason to become sparse.

## What this does and does not show

**It shows** that on static images, in a shallow network, with analog input
encoding, surrogate-gradient training works and costs about half a point of
accuracy, and that the energy argument does not survive honest accounting of
the input layer and the membrane updates.

**It does not show** that spiking networks are a bad idea. MNIST with a static
image and two layers is close to the least favourable setting available:

- there is no temporal structure for the leak to exploit, so the entire
  mechanism that distinguishes a spiking neuron is dead weight,
- the input is dense and analog, which is the case the encoding penalty is
  worst in (an event camera emits spikes natively, and the penalty vanishes),
- the network is too shallow for spike-driven layers to dominate the operation
  count. In a deep network, the static first layer amortises across many
  spike-driven ones, and the ratio moves.

The measured claim is narrow, and it is the one that was tested.

**What would change the answer**, in rough order of how much:

1. **Temporally structured input.** Event-camera or audio data, where the leak
   has something to integrate. The flat accuracy-versus-`T` curve above is the
   clearest possible evidence that static images give the mechanism nothing to
   work with.
2. **Spike-encoded input**, removing the 461.6 nJ analog first layer, worth
   0.70x dense on its own by the estimate above.
3. **A sparsity penalty in the loss.** At 44.5% activity the network is nowhere
   near the regime the energy argument assumes. This was deliberately not added,
   because changing one arm's objective would stop it being the same experiment,
   but it is the obvious next lever.
4. **Depth**, to amortise the input layer across more spike-driven ones.

None of these were run. They are named so the negative result is not mistaken
for a broader one than it is.

## Reproducing it

```bash
python experiments/exp_spiking_ab.py --seeds 3 --epochs 15 --n-train 60000
python experiments/exp_spiking_ab.py --sensitivity     # adds the sweeps below
```

Results land in `artifacts/experiments/spiking_ab.json`.

---

Previous: [22. A transformer](22-transformer.md)
