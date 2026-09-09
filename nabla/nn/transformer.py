r"""A GPT-style transformer, on the same engine as everything else.

What this is
------------
The decoder half of "Attention Is All You Need", which is the architecture
behind GPT. Every gradient in it comes from ``core/tensor.py``: there is no
new autodiff here, and that is the point of the file. A transformer turns out
to be a particular arrangement of matrix multiplies, softmax, and addition,
all of which the engine already differentiated.

Nothing below needed a single new backward rule. The two that people assume
are special are not:

* **Embedding lookup** is ``weight[ids]``. Its backward is a scatter-add,
  which ``Tensor.__getitem__`` already does with ``np.add.at`` -- and *must*
  do, because a token appearing twice in a batch has to accumulate both
  gradients rather than keep the last.
* **Attention** is two batched matmuls with a softmax between them. Batched
  matmul broadcasts its leading dimensions and ``_unbroadcast`` sums the
  gradient back down, so ``(B, H, T, d) @ (B, H, d, T)`` differentiates with
  the same eight lines that handle ``(m, k) @ (k, n)``.

The shape convention
--------------------
``(B, T, C)`` throughout: batch, time (position in the sequence), channels
(``d_model``). Attention temporarily splits ``C`` into ``(H, hd)`` and moves
heads next to the batch axis, giving ``(B, H, T, hd)``, so that one batched
matmul does every head of every sequence at once.

Why causal masking exists
-------------------------
The model is trained to predict token ``t+1`` from tokens ``0..t``. Attention
as written lets position ``t`` look at *every* position, including the future
ones, so without a mask the model would learn to read the answer off its own
input and score a near-zero loss that collapses the instant you ask it to
generate. The mask adds ``-inf`` above the diagonal before the softmax, so
those weights come out as exactly zero.

That is the single most important line in this file, and the one whose absence
produces a model that looks like it trained perfectly and cannot write.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..core.tensor import Tensor
from .module import Module
from .tensor_mlp import Linear, TensorParameter

__all__ = [
    "Embedding",
    "LayerNorm",
    "MultiHeadAttention",
    "FeedForward",
    "Block",
    "GPT",
    "gelu",
]


# ======================================================================
# activation
# ======================================================================


def gelu(x: Tensor) -> Tensor:
    r"""Gaussian Error Linear Unit, tanh approximation.

    .. math::
        \mathrm{GELU}(x) = 0.5x\left(1 + \tanh\left[
            \sqrt{2/\pi}\left(x + 0.044715x^3\right)\right]\right)

    ReLU makes a hard decision at zero: a slightly negative input is deleted
    entirely. GELU scales the input by roughly the probability that a standard
    normal falls below it, so small negatives are damped rather than erased
    and the function stays smooth. Transformers train more stably with it, and
    every GPT uses it.

    Built from ``tanh``, multiplication and addition, so its derivative comes
    out of the engine rather than being written by hand.
    """
    inner = (x + x * x * x * 0.044715) * 0.7978845608028654   # sqrt(2/pi)
    return x * (inner.tanh() + 1.0) * 0.5


# ======================================================================
# layers
# ======================================================================


class Embedding(Module):
    r"""A lookup table mapping integer ids to vectors.

    Mathematically this is a one-hot vector times a matrix. Doing it literally
    would multiply a ``(B, T, V)`` one-hot array by a ``(V, C)`` matrix, which
    for a 50,000-word vocabulary is billions of multiplications by zero. The
    lookup ``weight[ids]`` gives the identical answer by reading the rows
    directly.

    The gradient is the interesting half. Only the rows that were *used* get a
    gradient, and a row used twice must receive the **sum** of both
    contributions. ``Tensor.__getitem__`` handles this with ``np.add.at``;
    ordinary ``buffer[ids] += grad`` would silently keep only the last write
    for repeated indices, which in a language model means every common word
    trains on a fraction of its true gradient.

    Examples
    --------
    >>> import numpy as np
    >>> table = Embedding(10, 4, rng=np.random.default_rng(0))
    >>> table(np.array([[1, 2, 1]])).shape
    (1, 3, 4)
    """

    def __init__(
        self,
        n_embeddings: int,
        dim: int,
        *,
        rng: np.random.Generator | None = None,
        label: str = "",
    ) -> None:
        super().__init__()
        rng = rng or np.random.default_rng()
        self.n_embeddings = n_embeddings
        self.dim = dim
        # 0.02 is the GPT-2 initialisation. Small, because these vectors are
        # added to positional embeddings and then fed straight into a
        # LayerNorm; starting large just gets normalised away.
        self.weight = TensorParameter(
            rng.normal(0.0, 0.02, size=(n_embeddings, dim)), label=f"{label}emb"
        )

    def forward(self, ids: np.ndarray | Sequence[int]) -> Tensor:
        """``(...)`` integer ids -> ``(..., dim)``."""
        return self.weight[np.asarray(ids, dtype=np.intp)]


class LayerNorm(Module):
    r"""Normalise each position across its channels, then rescale.

    .. math::
        y = \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} \odot \gamma + \beta

    where :math:`\mu` and :math:`\sigma^2` are computed over the **last** axis
    only, per position, per sequence.

    Contrast with batch normalisation, which normalises across the batch: that
    makes one sequence's statistics depend on the others in its batch, which is
    wrong for language (batch composition is arbitrary) and unusable at
    generation time (batch of one). LayerNorm's statistics are per-token, so a
    sequence behaves identically whatever it is batched with. That property is
    why transformers use it.

    ``gamma`` starts at one and ``beta`` at zero, so the layer begins as exact
    normalisation and learns its way out if that helps.
    """

    def __init__(self, dim: int, *, eps: float = 1e-5, label: str = "") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.gamma = TensorParameter(np.ones(dim), label=f"{label}gamma")
        self.beta = TensorParameter(np.zeros(dim), label=f"{label}beta")

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(axis=-1, keepdims=True)
        centred = x - mean
        # Variance as mean(centred^2) rather than a separate pass: one fewer
        # traversal of the graph, and the algebra is identical.
        variance = (centred * centred).mean(axis=-1, keepdims=True)
        return centred / ((variance + self.eps) ** 0.5) * self.gamma + self.beta


class MultiHeadAttention(Module):
    r"""Causal self-attention over ``H`` heads.

    .. math::
        \mathrm{Attention}(Q,K,V) =
        \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V

    Read it as a soft dictionary lookup. Each position emits a **query** (what
    am I looking for), a **key** (what do I offer) and a **value** (what I will
    hand over). Every query is compared against every key by dot product, those
    scores become weights via softmax, and the output is the weighted average
    of the values. Positions that are relevant to each other end up with large
    weights, and the model learns what "relevant" means.

    Why divide by :math:`\sqrt{d_k}`
        A dot product of two ``d_k``-dimensional vectors of unit-variance
        entries has variance ``d_k``, so for ``d_k = 64`` the scores entering
        the softmax are spread about eight times wider than the vectors are.
        A wide softmax saturates: one weight goes to 1, the rest to 0, and the
        gradient through it vanishes. Scaling restores unit variance and keeps
        the layer trainable. Skipping it does not crash; it just stops learning
        for large heads, which is a miserable bug to locate.

    Why several heads
        One softmax can only average one way. Splitting the channels into ``H``
        independent heads lets the layer attend to several different things at
        once (a verb's subject and its object, say) and concatenate the
        answers. The split is free: ``H`` heads of size ``C/H`` cost exactly
        what one head of size ``C`` costs.

    ``M`` is the causal mask: ``0`` on and below the diagonal, ``-inf`` above.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        block_size: int,
        *,
        rng: np.random.Generator | None = None,
        label: str = "",
    ) -> None:
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_head={n_head}; "
                f"each head takes d_model/n_head={d_model / n_head:.2f} channels"
            )
        rng = rng or np.random.default_rng()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.block_size = block_size
        self.scale = 1.0 / np.sqrt(self.head_dim)

        self.q = Linear(d_model, d_model, rng=rng, label=f"{label}q.")
        self.k = Linear(d_model, d_model, rng=rng, label=f"{label}k.")
        self.v = Linear(d_model, d_model, rng=rng, label=f"{label}v.")
        self.proj = Linear(d_model, d_model, rng=rng, label=f"{label}proj.")

        # Built once, not per forward pass. It has no parameters, so it is a
        # plain array rather than anything the optimiser should see. -1e9
        # rather than -inf because -inf * 0 is NaN, and a fully-masked row
        # would poison the whole batch; -1e9 underflows to zero in the softmax
        # and stays finite.
        self._mask = np.triu(np.full((block_size, block_size), -1e9), k=1)

    def _split_heads(self, x: Tensor, batch: int, time: int) -> Tensor:
        """``(B, T, C) -> (B, H, T, hd)``, so one matmul covers every head."""
        return x.reshape(batch, time, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

    def forward(self, x: Tensor) -> Tensor:
        batch, time, _ = x.shape
        if time > self.block_size:
            raise ValueError(
                f"sequence length {time} exceeds block_size {self.block_size}; "
                "the causal mask and positional embeddings are only built that far"
            )

        q = self._split_heads(self.q(x), batch, time)
        k = self._split_heads(self.k(x), batch, time)
        v = self._split_heads(self.v(x), batch, time)

        # (B, H, T, hd) @ (B, H, hd, T) -> (B, H, T, T): every query against
        # every key, for every head, in one batched multiply.
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        # Slice the mask to the actual length so shorter prompts work during
        # generation without rebuilding it.
        scores = scores + Tensor(self._mask[:time, :time])
        weights = scores.softmax(axis=-1)

        out = weights @ v                                    # (B, H, T, hd)
        # Heads back together. `transpose` then `reshape` is the inverse of
        # the split, and is what "concatenate the heads" means in practice.
        out = out.transpose(0, 2, 1, 3).reshape(batch, time, self.d_model)
        return self.proj(out)


class FeedForward(Module):
    r"""Position-wise MLP: ``C -> 4C -> C``, with GELU between.

    Attention moves information *between* positions but applies no depth to any
    one of them: its output is a weighted average of values, which is linear in
    ``V``. This block supplies the nonlinearity, independently at every
    position, and is where most of a transformer's parameters live
    (:math:`8C^2` of the roughly :math:`12C^2` per layer).

    The 4x expansion is convention from the original paper, kept because it is
    what every published parameter count assumes.
    """

    def __init__(
        self,
        d_model: int,
        *,
        mult: int = 4,
        rng: np.random.Generator | None = None,
        label: str = "",
    ) -> None:
        super().__init__()
        rng = rng or np.random.default_rng()
        self.fc = Linear(d_model, mult * d_model, rng=rng, label=f"{label}fc.")
        self.proj = Linear(mult * d_model, d_model, rng=rng, label=f"{label}proj.")

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(gelu(self.fc(x)))


class Block(Module):
    r"""One transformer layer: attention, then feed-forward, both residual.

    .. math::
        \begin{aligned}
        x &\leftarrow x + \mathrm{Attention}(\mathrm{LayerNorm}(x)) \\
        x &\leftarrow x + \mathrm{FeedForward}(\mathrm{LayerNorm}(x))
        \end{aligned}

    Two design choices worth naming, because both are load-bearing:

    **Residual connections.** ``x + f(x)`` rather than ``f(x)``. The derivative
    of the sum with respect to ``x`` is :math:`1 + f'(x)`, so there is always a
    path carrying gradient with a factor of exactly one, however deep the
    stack. Without it the gradient is a long product of Jacobians and decays
    geometrically, which is precisely the vanishing-gradient problem that
    capped network depth for years.

    **Pre-norm.** The LayerNorm sits *inside* the residual branch, before the
    sublayer, rather than after the addition. Post-norm (as originally
    published) needs a learning-rate warmup to train at all at depth, because
    the normalisation sits directly on the residual path and interferes with
    that clean gradient route. Pre-norm is what GPT-2 onward use, and it trains
    without ceremony.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        block_size: int,
        *,
        mult: int = 4,
        rng: np.random.Generator | None = None,
        label: str = "",
    ) -> None:
        super().__init__()
        rng = rng or np.random.default_rng()
        self.ln1 = LayerNorm(d_model, label=f"{label}ln1.")
        self.attn = MultiHeadAttention(
            d_model, n_head, block_size, rng=rng, label=f"{label}attn."
        )
        self.ln2 = LayerNorm(d_model, label=f"{label}ln2.")
        self.ffn = FeedForward(d_model, mult=mult, rng=rng, label=f"{label}ffn.")

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


# ======================================================================
# the model
# ======================================================================


class GPT(Module):
    r"""A decoder-only transformer language model.

    Given a sequence of token ids, produces a distribution over the next token
    at *every* position at once. That last part is what makes the thing
    trainable at reasonable cost: a single forward pass over ``T`` tokens
    yields ``T`` supervised predictions, not one.

    Architecture
    ------------
    ``token embedding + positional embedding -> N blocks -> LayerNorm ->
    project to vocabulary``.

    Why positional embeddings are needed at all
        Attention is a weighted sum, and a sum does not care about order.
        Without positional information "dog bites man" and "man bites dog" have
        identical representations. Adding a learned per-position vector is the
        cheapest fix, and is what GPT-2 does.

    Weight tying
        The output projection reuses the token embedding matrix transposed.
        Both matrices relate tokens to vectors, so sharing them removes
        ``V * C`` parameters (for a large vocabulary, often most of the model)
        and consistently improves quality. The shared parameter receives
        gradient from both the lookup and the projection; the engine adds the
        two contributions automatically, because accumulation is the rule
        everywhere.

    Examples
    --------
    >>> model = GPT(vocab_size=32, block_size=8, d_model=16, n_head=2,
    ...             n_layer=2, seed=0)
    >>> import numpy as np
    >>> model(np.zeros((1, 8), dtype=int)).shape
    (1, 8, 32)
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        d_model: int = 128,
        n_head: int = 4,
        n_layer: int = 4,
        *,
        mult: int = 4,
        tie_weights: bool = True,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        rng = rng or np.random.default_rng(seed)

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer
        self.mult = mult
        self.tie_weights = tie_weights

        self.token_embedding = Embedding(vocab_size, d_model, rng=rng, label="tok.")
        self.position_embedding = Embedding(block_size, d_model, rng=rng, label="pos.")
        self.blocks = [
            Block(d_model, n_head, block_size, mult=mult, rng=rng, label=f"h{i}.")
            for i in range(n_layer)
        ]
        self.ln_f = LayerNorm(d_model, label="ln_f.")
        # Untied models need their own output matrix. Tied ones reuse the
        # embedding and must not register a second parameter, or the optimiser
        # would update a matrix nothing reads.
        self.head = (
            None if tie_weights
            else Linear(d_model, vocab_size, rng=rng, label="head.")
        )

    # ------------------------------------------------------------------

    def forward(self, ids: np.ndarray | Sequence[Sequence[int]]) -> Tensor:
        """``(B, T)`` token ids -> ``(B, T, vocab_size)`` logits.

        Logits, not probabilities: the softmax belongs in the loss, where it
        can be fused with the log for numerical stability. Same reason
        ``TensorMLP`` emits logits.
        """
        ids = np.asarray(ids, dtype=np.intp)
        if ids.ndim == 1:
            ids = ids[None, :]                    # a bare sequence is a batch of 1
        _, time = ids.shape
        if time > self.block_size:
            raise ValueError(
                f"sequence length {time} exceeds block_size {self.block_size}"
            )

        # Positions are the same for every sequence, so one (T, C) lookup
        # broadcasts across the batch.
        x = self.token_embedding(ids) + self.position_embedding(np.arange(time))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        if self.head is not None:
            return self.head(x)
        return x @ self.token_embedding.weight.transpose(1, 0)

    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: Sequence[int],
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> list[int]:
        """Sample a continuation, one token at a time.

        No gradients are needed here, so this reads ``.data`` off the logits
        and works in NumPy: building a graph we would immediately discard is
        pure waste, and generation is the one place this model is run in anger.

        Parameters
        ----------
        temperature
            Divides the logits. Below 1 sharpens the distribution (safer,
            repetitive); above 1 flattens it (more varied, less coherent).
            At 0 this degenerates to always taking the most likely token,
            which is handled explicitly rather than by dividing by zero.
        top_k
            Sample only from the ``k`` most likely tokens. The tail of a
            vocabulary distribution is long and mostly nonsense; without
            truncation, rare tokens are drawn far more often than their
            individual probabilities suggest, because there are so many of
            them.
        """
        rng = rng or np.random.default_rng()
        ids = list(prompt)
        if not ids:
            raise ValueError("prompt must contain at least one token")

        for _ in range(max_new_tokens):
            # Only the last block_size tokens are visible; the model has no
            # positional embedding beyond that.
            window = ids[-self.block_size:]
            logits = self.forward(np.array([window]))
            step = np.asarray(logits.data[0, len(window) - 1], dtype=np.float64)

            if temperature <= 0:
                ids.append(int(step.argmax()))
                continue

            step = step / temperature
            if top_k is not None and 0 < top_k < step.size:
                cutoff = np.partition(step, -top_k)[-top_k]
                step = np.where(step < cutoff, -np.inf, step)

            # Softmax with the max subtracted: exp of a large logit overflows,
            # and the shift cancels exactly in the ratio.
            step = np.exp(step - step.max())
            ids.append(int(rng.choice(step.size, p=step / step.sum())))

        return ids

    # ------------------------------------------------------------------

    def config(self) -> dict[str, Any]:
        """The arguments needed to rebuild this model, for a checkpoint."""
        return {
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "d_model": self.d_model,
            "n_head": self.n_head,
            "n_layer": self.n_layer,
            "mult": self.mult,
            "tie_weights": self.tie_weights,
        }

    def summary(self) -> str:
        counts = {
            "embeddings": self.token_embedding.num_parameters()
            + self.position_embedding.num_parameters(),
            "blocks": sum(b.num_parameters() for b in self.blocks),
            "final norm": self.ln_f.num_parameters(),
        }
        if self.head is not None:
            counts["head"] = self.head.num_parameters()

        lines = [
            f"GPT({self.vocab_size} vocab, {self.block_size} ctx, "
            f"{self.d_model} d_model, {self.n_head} heads, {self.n_layer} layers"
            f"{', tied' if self.tie_weights else ''})",
            f"{'component':<20} {'params':>12}",
            "-" * 34,
        ]
        for name, count in counts.items():
            lines.append(f"{name:<20} {count:>12,}")
        lines.append("-" * 34)
        lines.append(f"{'total':<20} {self.num_parameters():>12,}")
        return "\n".join(lines)
