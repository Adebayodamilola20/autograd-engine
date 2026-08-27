# 14. MNIST

> Runnable: `python examples/train_mnist.py` — about 16 seconds.

```
784 → 128 → 64 → 10        109,386 parameters
Adam, lr=1e-3, batch 64, ReLU
14 epochs, 16.3 s

97.63% test accuracy    (9,763 of 10,000 unseen digits)
```

Every gradient in that run was computed by `nabla/core/tensor.py`. Every
parameter update by `nabla/optim/adam.py`. Every one of those backward rules was
verified against finite differences before we got here.

Reproducible: seed 0, same numbers every run.

---

## 1. The setup

28×28 grayscale images, flattened to 784 features, ten classes.

**Pixels are scaled to [0, 1].** Raw 0–255 values would hand the first layer
gradients 255× too large, and no learning rate works for both that layer and the
rest. This is the most common MNIST mistake and it presents as "the model won't
train".

The data is split train / validation / test. The **test set is touched exactly
once**, at the end ([chapter 12 §3](12-training.md#3-train-validation-test--three-sets-three-jobs)).

## 2. Results

```
    test loss      0.1027
    test accuracy  97.63%   (9,763 of 10,000 correct)
    best val acc   97.60%   (epoch 6)
    training time  16.3 s over 14 epochs
    throughput     46,439 samples/s
```

Validation peaked at epoch 6; early stopping ended the run at 14. Test accuracy
(97.63%) landing just above validation (97.60%) is a good sign — no leakage, and
the validation set was not overfit by the epoch-selection process.

### Per-class

| digit | accuracy | | digit | accuracy |
|---|---|---|---|---|
| 0 | 98.77% | | 5 | 97.54% |
| 1 | 98.94% | | 6 | 97.60% |
| 2 | 96.40% | | 7 | 97.11% |
| 3 | 97.51% | | 8 | 97.14% |
| **4** | **96.12%** | | 9 | 99.16% |

## 3. The mistakes are the interesting part

```
    4 misread as 9:   25 times
    3 misread as 9:   11 times
    2 misread as 7:   10 times
    8 misread as 9:    9 times
    7 misread as 9:    8 times
    5 misread as 3:    8 times
```

**These are not random.** 4/9 differ only by whether the top closes. 7/2 share a
horizontal stroke and a diagonal. 3/5 share an open right side. 8/9 share the
upper loop.

The model is failing on **genuinely ambiguous pairs** — the same ones a human
misreads on bad handwriting. That is the signature of a model that learned
something about shape, rather than one that memorised or found a spurious
shortcut. A confusion matrix scattered uniformly would be far more worrying than
this concentrated one.

Digit 4 has the lowest accuracy and 4→9 is the single most common error; the two
facts are the same fact.

### The most confident mistake

```
Most confident mistake: expected 6, predicted 1 with 100.0% confidence.
```

`artifacts/mnist_worst_mistake.png` shows the digit. **100% confidence, and
wrong.**

This is worth dwelling on. Softmax probabilities are *not* calibrated
uncertainty. The network was trained to minimise cross-entropy, which rewards
confident correct answers — and confidence is a byproduct of that objective, not
an estimate of reliability. A model can be certain and wrong, and nothing in the
training procedure discourages it.

Anyone deploying a classifier and treating `max(softmax)` as "how sure the model
is" should look at this image first.

## 4. What the first layer learned

`artifacts/mnist_weights.png` reshapes each first-layer weight column back to
28×28 and displays it. These are literally *what each hidden unit is looking
for*.

They are not digits. They are **stroke fragments** — short curves, edges,
oriented segments — plus regions of negative weight meaning "there should be no
ink here". Absence of ink is as informative as presence: what distinguishes a 4
from a 9 is a *gap*.

The 64-unit second layer combines these into more complex shapes, and the output
layer combines those into digit identity. That is a feature hierarchy, and it
was not designed — it fell out of gradient descent.

## 5. Artifacts

| file | shows |
|---|---|
| `mnist_training_curves.png` | train/val loss and accuracy per epoch |
| `mnist_predictions.png` | test digits with prediction and confidence |
| `mnist_mistakes.png` | every error, with expected vs predicted |
| `mnist_worst_mistake.png` | the 100%-confident failure, with its distribution |
| `mnist_confusion.png` | the full 10×10 matrix |
| `mnist_weights.png` | first-layer weights as images |
| `mnist_results.json` | full history and metrics |
| `mnist_checkpoints/best.json` | weights, optimiser state, architecture |

## 6. Honest context

**97.63% is a good result for a from-scratch MLP and is not state of the art.**
A convolutional network reaches 99.5%+, because convolution builds in the prior
that a stroke means the same thing wherever it appears — translation
equivariance our fully-connected model has to learn from scratch, separately, for
every position.

The point was never the number. The point is that **the number was produced
entirely by machinery we built**, and that every gradient behind it was checked
against finite differences.

## 7. Scalar vs tensor

The same training runs on the scalar `Value` engine:

```bash
python examples/train_mnist.py --engine scalar
```

It is **~254,000× slower per sample** and defaults to a small subset for that
reason. One sample builds a graph of **328,804 nodes with depth 1,000**.

That is not a failure of the scalar engine — it computes identical gradients.
It is the measurement that motivates [chapter 15](15-tensors.md), and running it
once is worth more than reading about it.

## 8. Try it yourself

```bash
python web/server.py     # then draw a digit at http://127.0.0.1:8000
```

The [web demo](../web/README.md) serves this exact checkpoint, shows the ten
class probabilities live, and displays per-layer activations — including the
~50% of hidden units that go silent on any given digit.

---

**Next:** [15. From scalars to tensors](15-tensors.md)
