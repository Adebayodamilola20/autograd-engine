# 12. The training loop

> Implemented in [`nabla/training/`](../nabla/training/) and
> [`nabla/data/`](../nabla/data/).
> Tested by [`tests/test_data.py`](../tests/test_data.py) (38 tests).

Everything is in place: a graph, gradients, a network, a loss, an optimiser.
Training is the loop that ties them together.

---

## 1. The four lines that matter

```python
for batch_x, batch_y in loader:
    optimizer.zero_grad()                    # 1. clear last step's gradients
    loss = loss_fn(model(batch_x), batch_y)  # 2. forward -- builds the graph
    loss.backward()                          # 3. backward -- fills every .grad
    optimizer.step()                         # 4. update every parameter
```

That is the whole algorithm. Everything else in `Trainer` — metrics,
validation, checkpoints, early stopping, logging — is bookkeeping around these
four lines.

**Step 1 is not optional.** `backward()` accumulates
([chapter 5 §4](05-backpropagation.md#4-accumulation-semantics)), so without
`zero_grad()` step *n* uses the sum of gradients from steps 1…*n*. The loss
still decreases at first, which is what makes this bug so hard to spot. We
reproduce PyTorch's semantics deliberately rather than hiding them.

**Step 2 builds a fresh graph every iteration.** The previous graph is garbage
collected. This is define-by-run: there is no graph to reuse, and that is why a
model can contain a Python `if` or loop.

## 2. Epochs and batches

An **epoch** is one pass over the training data. A **batch** is the group of
samples used for a single update.

The relationship people forget: with `N` samples and batch size `B`, one epoch
performs **N/B updates**. Doubling the batch **halves the number of parameter
updates**. Steps, not samples, are what move the weights — which is why a
fixed-learning-rate batch-size sweep mostly measures update count rather than
anything about batching. [Chapter 20 §5](20-experiments.md#5-batch-size) has
the numbers.

Shuffling every epoch is on by default. A fixed order lets the model exploit
correlations between adjacent samples, and if the data happens to be sorted by
class it is catastrophic.

## 3. Train, validation, test — three sets, three jobs

| set | used for | touched |
|---|---|---|
| **train** | computing gradients | every step |
| **validation** | choosing epochs, hyperparameters, early stopping | every epoch, never trained on |
| **test** | the final honest number | **exactly once** |

Training loss measures how well the model fits data it has already seen, which
it can always improve by memorising. Validation loss measures performance on
data it was never updated on, which is the only number that estimates
generalisation. **When training loss falls while validation loss rises, the
model has started memorising** — and you can only see that if you held data
back.

The test set exists because using validation accuracy to *choose* things makes
it optimistic: you have selected for whatever happened to work on those
particular samples. `examples/train_mnist.py` touches the test set once, at the
end, and prints it with that fact stated.

The split is **stratified** by default — class proportions preserved in both
halves. On small splits a plain random 10% can miss a class entirely, which
makes the validation accuracy meaningless.

## 4. What `Trainer` adds

```python
trainer = Trainer(model, optimizer, loss_fn, metric='accuracy', grad_clip=None)

history = trainer.fit(
    train_loader, val_loader,
    epochs=20,
    checkpoint_dir='artifacts/mnist_checkpoints',
    checkpoint_metric='val_acc',
    early_stopping_patience=8,
    lr_schedule=lambda epoch, lr: lr * 0.95,
    callbacks=[my_callback],
)
```

**Metrics.** Loss and accuracy on both sets, per epoch, plus wall-clock time and
gradient norm. The gradient norm is the cheapest health check available: a value
of ~0 means nothing is flowing, ~1e6 means the next step will overshoot badly.

**Checkpointing.** Writes `best.json` whenever the tracked metric improves, and
`last.json` every epoch. A checkpoint carries model state, **optimiser state**,
the epoch number, the metric history, and the architecture. Saving only the
weights and resuming resets Adam's moments to zero, producing a visible bump in
the loss curve as the buffers refill.

The architecture metadata matters more than it looks: matrix shapes let you
recover layer sizes, but **the activation function is not recoverable from the
shapes at all**. A checkpoint you cannot load without the script that wrote it
is not much of a checkpoint. The [web demo](../web/README.md) reconstructs the
model from exactly this.

**Early stopping.** Halts after N epochs without improvement. Guards against
wasting time once validation has turned around — the point at which the model
has started memorising.

**Engine abstraction.** `Trainer` works with both the scalar and tensor engines.
The only difference is one method: the tensor engine builds one graph per batch,
the scalar engine one per sample. Everything downstream — backward, the
optimiser step, the metrics — is shared.

## 5. The data pipeline

`Dataset` holds arrays. `DataLoader` yields batches. The properties that matter
are bookkeeping rather than mathematics, and each is pinned by a test:

- every sample appears **exactly once** per epoch
- features stay **paired with their labels** (shuffling them independently would
  be catastrophic *and silent* — the model would train happily and learn nothing)
- the order **changes between epochs**
- batches do **not alias** the dataset

`DataLoader` has an `as_arrays` flag. The default yields Python lists, which is
what the scalar engine wants; `as_arrays=True` yields NumPy arrays for the
tensor engine. That flag came out of [chapter 19](19-optimisation.md), where
profiling found the ndarray → list → ndarray round trip consuming **39% of every
epoch**. Fixing it made MNIST training 2.08× faster with bit-identical output.

The module had **no tests at all** before that, which is exactly why the bug
lived there. That is not a coincidence — it is what "untested" means in
practice.

## 6. Reproducibility

Every source of randomness takes a seed: weight initialisation, the train/val
split, and the shuffle. With `seed=0`, `examples/train_mnist.py` reproduces
**97.63%** exactly, run after run.

This is not a nicety. Without it you cannot tell whether a change helped or you
got a lucky initialisation — and [chapter 20](20-experiments.md) shows
configurations whose seed-to-seed spread exceeds 15 percentage points.

---

**Next:** [13. XOR](13-xor.md) — the smallest problem that proves it all works.
