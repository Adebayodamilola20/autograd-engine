# 9. Activation functions

> Implemented in [`nabla/nn/activations.py`](../nabla/nn/activations.py).
> Measured in [chapter 20 §2](20-experiments.md#2-activations-and-why-nonlinearity-is-mandatory).

---

## 1. Why a nonlinearity is not optional

This is the single most important fact in the chapter, and it is provable in
two lines.

A linear layer is $f(x) = xW + b$. Compose two:

$$f_2(f_1(x)) = (xW_1 + b_1)W_2 + b_2 = x\underbrace{(W_1W_2)}_{W'} + \underbrace{(b_1W_2 + b_2)}_{b'}$$

That is **a single linear layer**. Stack a hundred and the argument repeats a
hundred times.

> Without a nonlinearity between them, depth buys **nothing**. Not "little" —
> exactly nothing.

Our 784→128→64→10 network has 109,386 parameters. With linear activations it
has the representational power of **7,850** of them: the size of a single
784→10 map. The other 101,536 are a redundant parameterisation of the same
small function class.

### The measured version

We ran it ([chapter 20](20-experiments.md)):

| activation | test accuracy |
|---|---|
| ReLU | **94.60% ± 0.15** |
| linear | **89.92% ± 0.68** |

**4.7 points, from a function with no parameters.**

And note the linear network still scores 89.92%. MNIST is nearly linearly
separable, so this failure **does not announce itself** — the loss curve is
smooth and plausible, the model trains, and it is quietly capped below what the
architecture on paper should reach. On a genuinely nonlinear problem it cannot
work at all: `examples/train_xor.py` shows the same model failing on four data
points.

## 2. The three that matter

Everything about an activation's behaviour during training is determined by its
**derivative**, because that is what the backward pass multiplies by.

### tanh

$$\tanh(x) = \frac{e^{2x}-1}{e^{2x}+1} \qquad \frac{d}{dx} = 1 - \tanh^2 x$$

- Output **(−1, 1)**, zero-centred
- Derivative peaks at **1.0** at `x = 0`, → 0 as `|x|` grows

Zero-centred output matters more than it sounds: if activations are all
positive, all gradients to a neuron's weights share a sign, and the weight
vector can only move diagonally. tanh avoids that.

Saturation is the weakness. A unit with a large input has derivative ≈ 0 and
learns nothing — visible concretely in [chapter 7](07-visualisation.md#3-the-canonical-example),
where `tanh(-5)` passes a gradient of 0.00018.

### sigmoid

$$\sigma(x) = \frac{1}{1+e^{-x}} \qquad \sigma' = \sigma(1-\sigma)$$

- Output **(0, 1)** — reads as a probability, which is its real use
- Derivative peaks at **0.25**

That 0.25 is the whole problem. Every sigmoid layer attenuates the backward
signal by up to a factor of 4 *before anything else happens*. Ten layers is a
factor of a million. This is the vanishing gradient problem in its original
form, and it is a property of the derivative, not a bug.

Measured across our three layers: sigmoid attenuates **32.9×** from last layer
to first, versus tanh's 6.8×, and reaches the first layer with a gradient less
than half of ReLU's.

**There is no reason to use sigmoid in a hidden layer.** tanh is strictly better
— same shape, 4× the peak derivative, zero-centred. Sigmoid belongs on an output
that must be a probability, and nowhere else.

### ReLU

$$\text{relu}(x) = \max(0, x) \qquad \frac{d}{dx} = \begin{cases}1 & x>0\\0 & x<0\end{cases}$$

The derivative is **exactly 1** on the active path. One multiplied by itself any
number of times is one, so **depth costs the gradient nothing**. That single
fact is most of why deep networks became trainable, and it is confirmed in our
depth sweep: the first-to-last gradient ratio stays around 14–20× whether the
network has 2 layers or 7.

It is also trivially cheap — a comparison, no `exp`.

The cost is real: for `x < 0` the gradient is exactly **zero**. A unit that
never activates receives no gradient, never updates, and never starts
activating. It is **dead permanently**. `leaky_relu` exists for this, passing
`0.01x` for negative inputs so the gradient is small but nonzero.

### Sparsity is a feature

A trained ReLU network has roughly **half its hidden units silent** on any given
input — measured live in the [web demo](../web/README.md), typically ~50% in
layer 0 and ~30% in layer 1.

That sounds wasteful and is not. Different inputs activate different subsets, so
the network uses a **sparse, input-dependent code**. Only the units that fire
participate, which makes representations more separable.

## 3. Choosing

| situation | use | because |
|---|---|---|
| hidden layers, default | **ReLU** | derivative 1, cheap, depth-friendly |
| dead units suspected | leaky ReLU | small nonzero negative gradient |
| shallow net, bounded output wanted | tanh | zero-centred, well-behaved |
| output, binary probability | sigmoid | maps to (0, 1) |
| output, multi-class | **none** — emit logits | softmax belongs in the loss ([chapter 10](10-losses.md)) |
| hidden layers | never sigmoid | tanh dominates it |

## 4. Implementation note: compose where possible

`leaky_relu` is not a new primitive:

$$\text{leaky}(x) = \text{relu}(x) - \alpha\,\text{relu}(-x)$$

Built from `relu`, `neg` and `sub` — all already derived and gradient-checked.
No new backward rule, so no new opportunity for a sign error. `softplus` is
likewise composed from `exp`, `log` and `add`.

The general principle from [chapter 3](03-operators.md#negation-subtraction-division--composed-not-new):
**every operation defined in terms of existing ones is an operation whose
derivative cannot be wrong.**

## 5. Pairing with initialisation

Activations and weight initialisation are not independent choices:

| activation | init | why |
|---|---|---|
| ReLU | **He**, variance $2/n_{in}$ | the 2 compensates for ReLU zeroing half its inputs |
| tanh / sigmoid | **Xavier** | balances forward and backward variance |
| linear | LeCun | variance $1/n_{in}$ |

The library selects automatically from the activation, because mismatching them
is subtle: the network still trains, just worse, and it looks like a learning
rate problem.

---

**Next:** [10. Loss functions](10-losses.md)
