# Phase 19 — Profiling and optimisation

> **Result:** full MNIST training went from **56.2 s to 27.0 s** — a 2.08x
> speedup — with **bit-identical output**: still 97.63% test accuracy, still 14
> epochs, still seed 0. Inference got 7.8x faster. Nothing in the autodiff
> engine changed.

This document is the method, not just the outcome. The method is the point:
we were wrong about where the time was going, and the only reason we found out
is that we measured instead of guessing.

---

## 1. Don't optimise. Measure, then optimise.

The rule everyone repeats and few follow. Here is what following it actually
looked like.

Before Phase 17 existed, the plausible story about our performance was: *"we're
a Python autodiff engine, PyTorch is C++, so we're slow because Python is slow
and the fix is to make the engine faster."* Every part of that is wrong, and it
would have sent us to rewrite `tensor.py` — the one file that turned out to be
fine.

## 2. The anomaly

Phase 17 produced this table:

| operation | nabla | PyTorch | PyTorch faster by |
|---|---|---|---|
| forward (batch of 64) | 127.9 µs | 89.9 µs | 1.4x |
| forward + backward | 469.3 µs | 290.9 µs | 1.6x |
| full training step | 837.1 µs | 605.2 µs | **1.4x** |
| one MNIST epoch | 510.1 ms | 114.3 ms | **4.5x** |
| inference over test set | 68.3 ms | 2.3 ms | **29.5x** |

Read the last three rows together. A single training step was only 1.4x slower
than PyTorch — genuinely close. But an epoch, which is *nothing but* a sequence
of training steps, was 4.5x slower.

That is arithmetically impossible unless something outside the steps is
expensive. An epoch of 157 steps at 837 µs should take ~131 ms. It took 510 ms.
**About 74% of the epoch was being spent somewhere other than training.**

Inference was worse still at 29.5x, and inference is the *simplest* thing the
library does. That is the shape of a fixed per-batch overhead: with no backward
pass to dilute it, overhead is almost the entire cost.

An anomaly you cannot explain is the most valuable thing a benchmark can give
you. It means your model of the system is wrong, and it tells you exactly where.

## 3. The profiler

`benchmarks/profile_training.py` runs a real epoch under `cProfile`.

A note on why a profiler rather than more timers: hand-placed timers can only
measure what you already suspect. We suspected the engine. The engine was
innocent. A profiler measures everything, including the thing you would never
have thought to time — which is where the time usually is.

The cost is that `cProfile` adds per-call overhead, so it inflates absolute
timings and exaggerates frequently-called functions. Use it for the *shape* of
the problem, then confirm with untainted wall-clock numbers.

**Before:**

```
    function                                       ncalls  tottime  cumtime
    ─────────────────────────────────────────────  ──────  ───────  ───────
    {built-in method numpy.asarray}                  4710    0.221    0.221
    {method 'tolist' of 'numpy.ndarray' objects}      314    0.113    0.113
    .../optim/adam.py:153(step)                       157    0.112    0.112
    .../core/tensor.py:364(_backward)                 471    0.071    0.075
    .../_core/numeric.py:97(zeros_like)              8321    0.037    0.041
    .../core/tensor.py:316(__matmul__)                471    0.031    0.036
```

The two most expensive entries in the entire epoch — `asarray` and `tolist` —
are **type conversions**. They compute nothing. They are ahead of `adam.step`,
ahead of the matmul backward pass, ahead of every piece of real arithmetic in
the library.

Together: 0.334 s of a 0.861 s epoch. **39% of training was spent converting
data between two representations of the same numbers.**

## 4. The bug

`DataLoader.__iter__` did this:

```python
xs = self.dataset.x[idx].tolist()   # ndarray  ->  list of lists
yield xs, ys.tolist()
```

and `TensorMLP.forward` did this:

```python
x = Tensor(np.asarray(x, dtype=np.float64))   # list of lists  ->  ndarray
```

So every batch made this journey:

```
    ndarray  --.tolist()-->  50,176 Python floats  --np.asarray()-->  ndarray
```

Every batch. Every epoch. 64 × 784 = 50,176 numbers boxed into Python objects
and immediately unboxed again, to arrive at a bitwise copy of where they
started.

Worse, `Tensor.__init__` *already* calls `np.asarray(data, dtype=np.float64)`,
so the conversion in `forward` was redundant even on its own terms.

**Why it was there:** it was correct for the engine it was written for. The
scalar `Value` engine genuinely wants Python floats — handing `Value` a
`np.float64` works, but drags NumPy scalar dispatch into every one of its
~300,000 operations per sample. The list conversion was the right decision for
Phase 2 that nobody revisited when the tensor engine arrived in Phase 18.

That is the most common shape of a real performance bug: not a stupid mistake,
but a reasonable decision that outlived its context.

## 5. The fix

Let the loader yield what the consumer wants:

```python
def __init__(self, ..., as_arrays: bool = False):
    ...

# in __iter__
xs = self.dataset.x[idx]
if self.as_arrays:
    yield xs, ys              # tensor engine: hand over the array
else:
    yield xs.tolist(), ys.tolist()   # scalar engine: Python floats
```

Plus deleting the redundant `np.asarray` in `TensorMLP.forward`.

Two design notes:

- **The list path stays the default.** Each engine gets the representation that
  suits it. Making arrays the default would have quietly slowed the scalar
  engine, which is the one that can least afford it.
- **No aliasing.** NumPy fancy indexing (`x[idx]` with an index array) always
  returns a copy, so batches remain independent of the dataset.
  `tests/test_data.py::test_batches_do_not_alias_the_dataset` pins that — if it
  ever became a view, mutating a batch would corrupt the dataset for every
  later epoch.

## 6. The second fix, and one we declined

**Taken — `np.zeros` over `np.zeros_like`.** `Tensor.__init__` allocates a zero
gradient per node, 8,321 times per epoch. Measured:

```
zeros_like         2330 ns
zeros(a.shape)     1430 ns     <- 1.6x faster
```

`zeros_like` inspects the source array's dtype, memory order and subclass.
We already know all three — `data` is float64 and C-contiguous, one line above.
So the inspection is pure waste. Identical result, ~3% of an epoch, no added
complexity.

**Declined — lazy gradient allocation.** The bigger version of the same idea is
to not allocate `.grad` until backward actually needs it, as PyTorch does. It
would save the remaining allocations, but it requires either a property (adding
attribute-access overhead to the hottest path in the library, possibly a net
loss) or threading `None`-handling through all ~30 backward closures.

We measured the prize at roughly 4% and judged the cost too high: it would make
every backward rule harder to read, and these backward rules are the entire
educational point of the repository. **Not sacrificing understanding for
optimisation is a real constraint, and it is allowed to lose an argument.**

Recording the rejected option matters as much as recording the taken one —
otherwise the next person re-derives it from scratch.

## 7. Results

**Profile after** — `asarray` and `tolist` are gone from the top entirely, and
total profiled time fell from 861 ms to 197 ms:

```
    function                                    ncalls  tottime  cumtime
    ──────────────────────────────────────────  ──────  ───────  ───────
    .../optim/adam.py:153(step)                    157    0.063    0.063
    .../core/tensor.py:364(_backward)              471    0.032    0.034
    .../_core/numeric.py:97(zeros_like)           8321    0.014    0.017
    .../core/tensor.py:316(__matmul__)             471    0.011    0.013
    .../data/dataset.py:182(__iter__)              158    0.008    0.008
```

The top entry is now `adam.step` — the optimiser doing real arithmetic — and
below it the matmul backward pass. **A healthy profile is one where the
expensive things are the things that compute.**

**Wall clock:**

| measurement | before | after | speedup |
|---|---|---|---|
| one MNIST epoch (10k samples) | 510 ms | 218 ms | **2.3x** |
| inference over test set | 68.3 ms | 8.5 ms | **7.8x** |
| **full MNIST training (14 epochs)** | **56.2 s** | **27.0 s** | **2.08x** |
| test accuracy | 97.63% | 97.63% | **identical** |

**Against PyTorch:**

| operation | before | after |
|---|---|---|
| full training step | 1.4x | 1.6x |
| one MNIST epoch | 4.5x | **1.2x** |
| inference | 29.5x | **4.4x** |

A full training epoch on our own engine is now within **1.2x** of PyTorch, at
matched dtype and thread count.

## 8. What this phase actually taught

**The bottleneck was not in the autodiff engine.** It was in the plumbing around
it. No amount of staring at `tensor.py` would have found it, and the intuitive
story ("Python is slow, rewrite the engine") pointed in exactly the wrong
direction.

**Untested code is where bugs live.** `nabla/data/` had *zero* tests before this
phase, while the engine had several hundred. The performance bug lived in the
untested module. That is not a coincidence — it is what "untested" means in
practice. Phase 19 added `tests/test_data.py`.

**The step-vs-epoch discrepancy was the whole clue.** One number in isolation is
almost useless. Two numbers that cannot both be true is a map to the bug.
Benchmark suites earn their keep through internal contradictions.

**Identical output is the standard.** Every optimisation here was verified to
produce bit-identical results: 97.63%, same epochs, same seed. An optimisation
that changes the answer is not an optimisation, it is a bug with a stopwatch.

**A rejected optimisation is a result.** We measured lazy gradient allocation at
~4% and declined it on readability grounds. That decision is documented so it
does not have to be rediscovered.

---

## Reproducing this

```bash
# the profile, both paths
python benchmarks/profile_training.py                  # current (fast) path
python benchmarks/profile_training.py --list-batches   # the old round trip

# the wall-clock effect, in isolation
python benchmarks/benchmark_autograd.py --skip-scalar
python benchmarks/benchmark_autograd.py --skip-scalar --list-batches

# against PyTorch
python benchmarks/compare.py
```

Every number in this document came from those commands on the machine recorded
in [`18-performance.md`](18-performance.md).
