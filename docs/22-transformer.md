# 22. A transformer, and why it needed no new autodiff

The claim this repository has been making since chapter 1 is that
backpropagation is one rule applied repeatedly, and that everything built on
top of it is arrangement rather than new mathematics. A transformer is the
strongest available test of that claim, because it is the architecture people
most often assume must require something special.

It does not. `nabla/nn/transformer.py` contains a complete GPT, and it adds
**zero backward rules** to the engine. Every gradient in it comes from
`core/tensor.py` unchanged.

## The two operations people expect to be special

### Embedding lookup

Turning a token id into a vector looks like table indexing rather than
arithmetic, so it seems like it should need its own derivative. Mathematically
it is a one-hot vector times a matrix:

```
one_hot(id) @ W        # (1, V) @ (V, C) -> (1, C)
```

Doing that literally, for a 50,000-token vocabulary, is fifty thousand
multiplications by zero to retrieve one row. `W[ids]` gives the identical
answer by reading the row directly.

The gradient is where it gets interesting. Only the rows that were used get a
gradient, and **a row used twice must receive the sum of both contributions**.
`Tensor.__getitem__` already implemented this, with `np.add.at`:

```python
def _backward() -> None:
    buffer = np.zeros(in_shape)
    np.add.at(buffer, index, out.grad)
    self.grad = self.grad + buffer
```

The obvious-looking `buffer[index] += out.grad` is wrong here, and wrong
silently. NumPy fancy indexing with repeated indices keeps only the last write,
so in a language model every frequently-occurring token would train on a
fraction of its true gradient. The model would still train. It would just be
worse, for no visible reason.

This is decision D6 (accumulate, never assign) reappearing in NumPy's clothing,
and it is the same rule that makes `Value.backward()` use `+=`.

### Attention

```
softmax(Q Kᵀ / √d + M) V
```

Two matrix multiplies with a softmax between them. The only thing that
distinguishes it from the `Linear` layer of chapter 8 is that the arrays carry
two leading dimensions, `(batch, head)`, instead of one.

`__matmul__` already handled that, because its backward rule was written with
`np.swapaxes(..., -1, -2)` rather than `.T`, and passes the result through
`_unbroadcast`:

```python
grad_self  = out.grad @ np.swapaxes(other.data, -1, -2)
grad_other = np.swapaxes(self.data, -1, -2) @ out.grad
```

`(B, H, T, d) @ (B, H, d, T)` therefore differentiates with exactly the same
eight lines as `(m, k) @ (k, n)`. Nothing was added.

## What attention actually computes

Read it as a soft dictionary lookup. Every position emits three vectors:

- a **query**: what am I looking for
- a **key**: what do I offer
- a **value**: what I will hand over if selected

Each query is compared against every key by dot product. Those scores go
through a softmax to become weights that sum to one, and the output is the
weighted average of the values. Positions relevant to each other end up with
large weights, and what "relevant" means is learned.

### Why divide by √d

A dot product of two `d`-dimensional vectors with unit-variance entries has
variance `d`. For `d = 64` the scores entering the softmax are spread about
eight times wider than the vectors themselves.

A wide softmax saturates: one weight goes to 1, the rest to 0, and the gradient
through it vanishes. Dividing by `√d` restores unit variance and keeps the
layer trainable.

Omitting it does not crash. It just stops learning for large heads, which is a
miserable bug to locate from symptoms.

### Why the mask is the most important line

The model predicts token `t+1` from tokens `0..t`. Attention as written lets
position `t` attend to *every* position, including future ones. Without a mask,
the model learns to read the answer off its own input, scores a beautiful
near-zero loss, and generates nothing but gibberish the moment you ask it to
write, because at generation time the future is not there to copy.

Adding `-1e9` above the diagonal before the softmax sends those weights to
exactly zero:

```python
scores = scores + Tensor(self._mask[:time, :time])
weights = scores.softmax(axis=-1)
```

`-1e9` rather than `-inf` because `-inf * 0` is `NaN`, and one `NaN` poisons
the entire batch. `-1e9` underflows to zero in the softmax and stays finite.

`tests/test_transformer.py::TestCausality` pins this down directly: change the
last token of a sequence, and every earlier position's logits must be
bit-for-bit identical.

## A gradient that is provably zero

While gradient-checking the attention layer, one parameter stood out:
`attn.k.b`, the bias on the key projection, had a gradient of `4.4e-16`.

That is not a bug, and it is not noise. It is a fact about the mathematics.

Scores are `s[i][j] = q_i · k_j`. Adding a bias `b` to every key sends
`s[i][j] → s[i][j] + q_i · b`, which is **a constant within each row `i`**.
Softmax is invariant to adding a constant to all the logits in a row, so the
bias cancels exactly. The key bias cannot affect the output, and the engine's
gradient says so to machine precision.

Confirmed from the other direction too: adding `7.0` to every key bias changes
the layer's output by `6.7e-16`.

This is a good property to have found, because it checks the engine against
mathematics rather than against itself. It is also why several published
transformers drop the key bias outright.

It did, however, break the first version of the test. Measuring *relative*
error between two arrays that both contain nothing but rounding noise gives a
meaningless ratio. The fix follows the convention `core/gradcheck` already
used: below a scale floor, report the absolute difference instead.

## Interpreting the gradient check

The whole-model check disagrees with finite differences by about `3.5e-7`,
not the `1e-12` a single operator achieves. That is expected, and it is worth
knowing how to tell "expected" from "broken" rather than tuning the tolerance
until the tests pass.

Central differences carry `O(ε²)` truncation error. So a **correct** derivative
shows an error that falls by 100x each time `ε` falls by 10x, while a **wrong**
backward rule leaves a constant floor that no `ε` can shift. Sweeping it:

| ε | max relative error |
|---|---|
| 1e-3 | 3.46e-03 |
| 1e-4 | 3.53e-05 |
| 1e-5 | 3.53e-07 |
| 1e-6 | 3.44e-09 |
| 1e-7 | 5.10e-08 (roundoff floor) |

Four clean factors of 100, then the roundoff floor takes over as `ε` gets small
enough that `f(x+ε) - f(x-ε)` starts losing significant digits. The
disagreement belongs to the measurement, not the engine.

This is the diagnostic worth remembering: **sweep epsilon before you touch the
tolerance.**

## Tokenization

A model never sees text, only integers, and the mapping is a real design
choice. `nabla/data/tokenizer.py` implements two.

`CharTokenizer` gives one id per character. Vocabulary about 65 for English,
nothing is ever out-of-vocabulary, and you can verify it by reading it. The
cost is that a `T`-token context covers only `T` characters, and attention cost
grows as `T²`.

`BPETokenizer` is byte-level Byte Pair Encoding, the algorithm behind GPT-2 and
Llama. Start from the 256 possible bytes, repeatedly merge the most frequent
adjacent pair into a new token. Common sequences like `the` collapse to one id
while rare words still decompose. Typically three to four characters per token,
so the same context window reaches three to four times further.

Starting from **bytes** rather than characters is what makes it total: any byte
sequence decodes, so emoji and accented text need no special handling and there
is no out-of-vocabulary token.

The subtlety is decoding. A multi-byte UTF-8 character can be split across two
tokens, so decoding token-by-token would fail on the halves. Bytes are
accumulated across the whole sequence and decoded once at the end.

## Where the parameters go

Per layer, with `C = d_model`:

| component | parameters |
|---|---|
| attention (Q, K, V, output) | 4C² |
| feed-forward (C→4C→C) | 8C² |
| two LayerNorms | 4C |

So roughly `12C²` per layer, and **two thirds of a transformer's weights are in
the feed-forward blocks, not in attention.** Attention gets the attention;
the MLP gets the parameters.

Weight tying shares the token embedding with the output projection, which
removes `V × C` parameters. For a large vocabulary that is often most of the
model. The shared parameter receives gradient from both uses, and the engine
adds them automatically, because accumulation is the rule everywhere.

## The learning rate is not the one that worked before

The MLPs elsewhere in this repository train happily at `lr=3e-3`, so that was
the first default here. It was wrong, and the way it was wrong is worth
recording because the symptom does not look like a bad learning rate.

Training did not diverge, produce `NaN`, or fail. It fell from chance to about
2.45 and then simply stopped improving, holding that value from step 500 to
step 2000. That reads like "the model has reached its capacity", which is the
wrong conclusion and sends you looking for more parameters.

Measured on tinyshakespeare, 4 layers, 700 steps, validation loss:

| learning rate | validation loss |
|---|---|
| 3e-3 | 2.527 |
| **1e-3** | **2.237** |
| 5e-4 | 2.267 |

At `1e-3` the model passes in 700 steps the plateau that `3e-3` could not clear
in 2000. The steps at `3e-3` are large enough to bounce across the minimum
rather than settle into it, and a deeper stack of residual blocks is more
sensitive to this than a two-layer MLP.

The general lesson: **a loss that stalls above chance but well short of good is
a learning-rate symptom at least as often as a capacity one**, and it is much
cheaper to test.

## What this does not buy you

The architecture is the same one GPT uses. The scale is not, and no amount of
architectural fidelity substitutes for it. See the measured ceiling in the
README: this trains models of roughly one to five million parameters in
practical time on a laptop CPU, which is enough to learn the structure of
English and generate text with real words and local grammar, and nowhere near
enough for conversation or factual recall.

That gap is not a missing feature. It is CPU NumPy versus a datacentre, and it
is honest to say so.

---

Previous: [21. Conclusions](21-conclusions.md)
