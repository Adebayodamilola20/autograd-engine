# 21. What we learned

The project set out to answer one question: **what actually happens when you
call `.backward()`?**

Here is what came out of answering it properly.

---

## 1. Backpropagation is smaller than its reputation

The core algorithm is six lines:

```python
def backward(self):
    order = topological_sort(self)
    for node in order:
        if node._prev:
            node.reset_grad()
    self.grad += 1.0
    for node in reversed(order):
        node._backward()
```

Everything else — neurons, layers, losses, optimisers, MNIST at 97.63% — is
built on top of that. Backpropagation is not a special algorithm for neural
networks. It is **the chain rule, applied in reverse topological order, with
accumulation**. The neural network is not a special case; it is a large
expression.

Once that lands, a lot of the field stops looking like magic.

## 2. The interesting bugs are the ones that pass tests

Three real bugs from this project, and what they have in common:

**Gradient contamination across sweeps.** `.grad` was doing two jobs — storing
the answer *and* carrying gradient along edges during the sweep. Calling
`backward()` twice re-read pass-one residue and inflated the result: 18 where
12 was correct.

The existing accumulation test used `L = x*x`, which has **no intermediate node
between the leaf and the output** — nothing to contaminate. It passed against a
broken engine. A tensor test that happened to include one extra `.sum()` node
exposed it immediately.

> A test that exercises the minimum structure often cannot see the bug.

**39% of every epoch spent on type conversion.** The `DataLoader` converted each
batch to Python lists; the model converted them straight back. `tolist` and
`asarray` were the two most expensive entries in the profile — ahead of every
piece of real arithmetic. `nabla/data/` had **zero tests**, which is precisely
where the bug lived. Not a coincidence; that is what "untested" means.

**`ndarray @ tensor` silently dropping the graph.** NumPy's `__matmul__` ran
first, coerced our `Tensor` into a 0-d object array, and the autodiff graph was
never built.

All three produced *plausible* behaviour. None crashed. The common thread:
**correctness that looks correct is the dangerous kind.**

## 3. Verify against something that shares no reasoning

Unit tests written by the person who derived the formula are circular. If the
derivation was wrong, the test is wrong in the same way and passes.

Finite differences break that circle — computed from the definition of a
derivative, sharing no code and no reasoning with the analytic rules. 183 such
checks cover every operator, activation, loss and full-network parameter
gradient.

An unchecked derivative is a guess that happens to be typed in monospace.

## 4. Granularity beat everything else

| engine | forward + backward per sample | vs next |
|---|---|---|
| our scalar `Value` | 2.09 s | — |
| our tensor engine | 7.3 µs | **~285,000×** |
| PyTorch | 4.5 µs | 1.6× |

The step from our scalar engine to our tensor engine dwarfs the step from our
tensor engine to PyTorch — a project with thousands of contributors and a C++
core.

**Almost the entire performance story is what a node represents**, and we
captured nearly all of it ourselves. That is the real answer to "why are
frameworks built around tensors": not mathematical necessity — our scalar engine
computes identical gradients — but that **the unit of work must be big enough
that per-operation interpreter overhead stops mattering.**

## 5. Measure before optimising — the cliché is true and specific

The plausible story was: *"we're Python, PyTorch is C++, so make the engine
faster."* Every part of that was wrong, and acting on it would have meant
rewriting the one file that was fine.

The clue was an **internal contradiction**: a single training step was 1.4×
slower than PyTorch, but a whole epoch — which is nothing but a sequence of
steps — was 4.5× slower. 157 steps at 837 µs should be 131 ms; it took 510 ms.

One number in isolation is nearly useless. **Two numbers that cannot both be
true is a map to the bug.** Benchmark suites earn their keep through their
internal contradictions.

Result: MNIST training 56.2 s → 27.0 s → 16.3 s, **bit-identical output**. An
optimisation that changes the answer is not an optimisation; it is a bug with a
stopwatch.

## 6. Folk wisdom is cheap to test, and sometimes wrong

From [chapter 20](20-experiments.md), each about a minute of compute:

- **"Adam beats SGD"** — did not survive a tuned SGD baseline. SGD at lr=0.5
  scored 94.77% against Adam's 94.60%. Adam's product is *robustness to a
  hyperparameter*, not a better optimum.
- **"Large batches hurt accuracy"** — is really "few updates hurt accuracy".
  Batch 512 at fixed lr scored 91.95%; scale the lr by 8 and it recovers to
  94.55%, level with batch 64 at twice the speed.
- **"Deeper is better"** — the 6-layer network scored *below* the 3-layer one
  with *more* parameters. More capacity, worse result: an optimisation failure,
  not a representational one.

And the one that most needs stating: **report variance or report nothing.** Some
configurations vary by over 15 percentage points across seeds. Any single-run
comparison between nearby settings is noise dressed as a finding.

## 7. Numerical stability is a separate axis from optimisation stability

Pushing SGD to lr=1000 produced a first-epoch loss on the order of **1e57** —
and never a NaN. The run completed.

That is log-sum-exp doing its job: subtracting the row maximum makes the largest
exponent $e^0 = 1$, so overflow is **impossible by construction**, not merely
unlikely.

The model was destroyed; the arithmetic was fine. Those are different failures,
and conflating them is why "my loss went NaN" gets misdiagnosed as a
learning-rate problem when it is often a stability bug — or the reverse.

## 8. Good abstractions prove themselves by reuse

`graph.py` was written for `Value` in Phase 3 and runs on `Tensor` unmodified.
`Module`, `Parameter`, `SGD`, `Adam`, `Trainer` and the checkpoint code were all
reused across two completely different engines without a single change.

That was not luck, and it was not planned in detail either. It happened because
each layer was given the narrowest interface that could work: the sort only ever
needed `_prev`; the optimisers only ever needed `.data` and `.grad`.

If the optimiser had assumed gradients were floats, none of it would have
transferred.

---

## What I can now explain

The original goal was to be able to explain these. I can:

- what a computational graph is, and why running the forward pass *is* building it
- what automatic differentiation is, and how it differs from symbolic and numerical differentiation
- what reverse mode is, and why it is right for neural networks — many inputs, one output, one row of the Jacobian per sweep
- how the chain rule makes backpropagation possible, and why topological order is required rather than convenient
- why gradients accumulate, and what breaks when they do not
- how gradients flow backward through a network, and why ReLU's derivative of exactly 1 changed what was trainable
- how parameters are represented, and why zero-initialisation destroys a layer
- how gradient descent updates parameters, and what momentum and Adam add
- how a loss produces gradients, and why softmax + cross-entropy gives exactly $p - y$
- why frameworks are built around tensors, and why optimised libraries are faster
- the difference between scalar and tensor autodiff — and that it is granularity, not mathematics

---

## Future work

Honest about what is missing and roughly what each would take.

**Convolutions.** The clearest next step. MNIST at 99.5% needs translation
equivariance our fully-connected model must learn separately for every position.
`conv2d` forward is straightforward; the backward rule is where the learning is.

**More layer types.** Batch norm (whose backward rule is genuinely tricky —
the mean and variance depend on every sample in the batch), dropout, embeddings.

**A real Tensor API.** Indexing, slicing, concatenation, `stack`. Currently
enough for MLPs and not much more.

**Lazy gradient allocation.** Measured at ~4% and deliberately declined
([chapter 19 §6](19-optimisation.md#6-the-second-fix-and-one-we-declined)) — it
would need `None`-handling threaded through all ~30 backward closures, and those
closures are the educational point. Recorded so it is not re-derived.

**Forward mode.** We built reverse mode because neural networks need it. Forward
mode is more natural for many-outputs/few-inputs, and implementing both would
make the duality concrete rather than asserted.

**Second derivatives.** Making `backward()` itself differentiable — building the
backward pass out of `Value` operations rather than raw floats — would give
Hessian-vector products and is a genuinely interesting exercise in
self-application.

**A GPU backend.** Would mean writing kernels, and would teach why CUDA exists
in the way this project taught why tensors exist.

---

## The one-paragraph version

An autodiff engine is a data structure that remembers arithmetic, plus a
traversal that applies the chain rule backwards through it. The hard parts are
not the calculus — they are accumulating gradients correctly when a value is
reused, ordering the traversal so gradients are complete before they are used,
keeping the arithmetic stable when logits are large, and choosing a granularity
where the interpreter is not the bottleneck. Everything a framework does on top
of that is engineering, and it is worth knowing which is which.

---

**Back to:** [documentation index](README.md)
