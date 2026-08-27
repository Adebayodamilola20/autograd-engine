# 13. XOR

> Runnable: `python examples/train_xor.py`

Before MNIST, four data points.

```
(0,0) → 0
(0,1) → 1
(1,0) → 1
(1,1) → 0
```

XOR is small enough to reason about completely — you can hold the whole problem,
the whole network, and every parameter in your head at once. If the engine works
here, it works; if it does not, the failure is findable. Going straight to
60,000 images would mean debugging on a problem where "it trained badly" has
fifty possible causes.

---

## 1. Why XOR is the right test

```
    x2
     1 │  ●(0,1)=1        ○(1,1)=0
       │
     0 │  ○(0,0)=0        ●(1,0)=1
       └──────────────────────────  x1
          0                1
```

The two 1s sit on one diagonal, the two 0s on the other.

A single linear unit computes $w_1x_1 + w_2x_2 + b$ and can only split the plane
with a **straight line**. No straight line separates those diagonals. This is
Minsky and Papert's 1969 result, and it is the reason the first wave of neural
network research stalled.

A hidden layer plus a nonlinearity lets the network build intermediate features
— effectively OR and NAND — and combine them. **That is the whole argument for
depth, in four data points.**

## 2. Two controls that must fail

Good experiments include cases you expect to fail. If a broken engine could pass
your test, the test proves nothing.

### Control 1 — no hidden layer

```
  MLP(2 → 2), no hidden layer, 2000 steps
      x1   x2   target   pred     P(1)
       0    0        0      0   0.5000   ✓
       0    1        1      0   0.5000   ✗
       1    0        1      1   0.5000   ✓
       1    1        0      1   0.5000   ✗

    final loss 0.6931   accuracy 50%
```

Stuck at 50%, exactly as proved. **Not a bug** — a correct result about an
insufficient model.

### Control 2 — two layers, but no nonlinearity

```
  MLP(2 → 8 → 2) with activation='linear'
    final loss 0.6931   accuracy 75%
```

Eight hidden units and it still cannot do it, because
$W_2(W_1x + b_1) + b_2$ collapses to $W'x + b'$
([chapter 9 §1](09-activations.md#1-why-a-nonlinearity-is-not-optional)).
**Depth without nonlinearity is an illusion.**

The loss is exactly $\ln 2 = 0.6931$ — the model outputs P = 0.5 for every input
and has genuinely given up. The 75% "accuracy" is argmax breaking ties between
two identical logits, not the model knowing anything. Worth noticing: **an
accuracy number can be actively misleading**, and the loss is the honest signal
here.

## 3. The real thing

```
    MLP(2→8→2, 42 params), tanh hidden layer

    gradient check before training: PASS  (max rel err 6.27e-09)
    42 parameters verified against finite differences
```

The gradient check runs **before** training. If the engine were wrong, this
fails immediately with a clear message, instead of producing a model that trains
badly for reasons that look like a hyperparameter problem.

Then it learns XOR, 4/4.

## 4. What the hidden layer actually built

The most valuable output of the script. Each hidden unit's activation on each
of the four inputs:

```
                h0      h1      h2      h3      h4      h5      h6      h7
(1,1)→0      0.941   0.588   0.947   0.872  -0.950  -0.794  -1.000  -1.000
```

Each hidden unit has become a **different linear feature of the input**. The
output layer then finds a linear combination of *those* that separates the
classes — impossible in the original coordinates, easy in the representation the
hidden layer built.

**That is what "learning features" means, concretely.** Not a metaphor: the
network changed coordinate systems, and in the new coordinates the problem is
linearly separable.

`artifacts/xor_decision_boundary.png` shows the resulting boundary — visibly
curved, wrapping around the diagonals in a way no single line could.

## 5. Why this belongs before MNIST

Every mechanism MNIST needs is exercised here:

- graph construction across a multi-layer network
- backpropagation through a hidden layer
- gradient accumulation where parameters feed several outputs
- softmax cross-entropy
- an optimiser loop that converges
- gradient checking on real parameters

At a scale where **every number can be checked by hand**, and where the two
controls prove the test can fail.

Once XOR works, MNIST is the same code with bigger numbers.

---

**Next:** [14. MNIST](14-mnist.md)
