"""Tests for the transformer.

The centre of gravity here is ``TestGradients``. Every other property in this
file could be checked by eye; the gradients could not, and they are the thing
that silently degrades. A transformer with a subtly wrong backward pass still
trains -- just worse -- so nothing but a finite-difference check against the
forward pass will catch it.

``TestCausality`` is second in importance. A model that can see the future
scores a beautiful training loss and generates gibberish, which is the most
expensive bug in this file to diagnose from symptoms alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from nabla.core.tensor import Tensor
from nabla.losses.tensor_losses import tensor_cross_entropy
from nabla.nn.transformer import (
    GPT,
    Block,
    Embedding,
    FeedForward,
    LayerNorm,
    MultiHeadAttention,
    gelu,
)


def numerical_gradient(loss_fn, tensor, *, eps=1e-5):
    """Central-difference gradient of ``loss_fn()`` w.r.t. every entry."""
    out = np.zeros_like(tensor.data)
    it = np.nditer(tensor.data, flags=["multi_index"])
    while not it.finished:
        index = it.multi_index
        original = tensor.data[index]
        tensor.data[index] = original + eps
        high = loss_fn()
        tensor.data[index] = original - eps
        low = loss_fn()
        tensor.data[index] = original
        out[index] = (high - low) / (2 * eps)
        it.iternext()
    return out


def max_relative_error(a, b):
    r"""Largest disagreement, measured against the scale of the whole tensor.

    .. math::
        \frac{\max_i |a_i - b_i|}
             {\max(\|a\|_\infty, \|b\|_\infty, \varepsilon_{\text{floor}})}

    The obvious alternative -- dividing entry by entry -- is wrong for a
    tensor, and misleadingly so. A parameter's gradient entries here span four
    orders of magnitude (the position embedding runs from 5.9e-3 to 17.5).
    Central differences commit roughly the same *absolute* truncation error at
    every entry, so a per-entry ratio judges the smallest gradient in the
    tensor against its own tiny magnitude and reports a failure that says
    nothing about the backward rule. Scoring the whole tensor against its own
    largest entry asks the question actually worth asking: is this gradient
    field accurate relative to its own size?

    This is the standard array gradient-check norm, and matches the spirit of
    ``core/gradcheck.relative_error``, which falls back to absolute difference
    below ``_SCALE_FLOOR`` for the same reason.

    That fallback is not hypothetical here. ``attn.k.b`` has an analytically
    zero gradient (see ``test_the_key_bias_is_a_mathematical_no_op``), so both
    arrays hold nothing but rounding noise, and a ratio of one noise floor to
    another is meaningless. Below the floor the absolute difference is the
    honest number.
    """
    difference = float(np.max(np.abs(a - b)))
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    if scale < 1e-8:
        return difference
    return difference / scale


# Not a fudge factor arrived at by raising the bar until the tests went green.
# Central differences carry O(eps^2) truncation error, so a *correct*
# derivative shows an error that falls by 100x each time eps falls by 10x,
# while a wrong backward rule leaves a constant floor no eps can shift.
# Sweeping eps over the whole-model check gives:
#
#     eps 1e-3 -> 3.46e-03      eps 1e-6 -> 3.44e-09
#     eps 1e-4 -> 3.53e-05      eps 1e-7 -> 5.10e-08  (roundoff floor)
#     eps 1e-5 -> 3.53e-07
#
# Four clean factors of 100, and the same pattern on every parameter tested.
# The residual disagreement belongs to the measurement, not the engine.
#
# 1e-5 also lands where ``core/gradcheck.relative_error`` documents it should:
# "fine for a deep or badly-scaled expression". A two-layer transformer is
# both, at roughly forty chained operations from weight to loss.
GRADIENT_TOLERANCE = 1e-5


# ======================================================================


class TestShapes:
    def test_embedding_maps_ids_to_vectors(self):
        table = Embedding(10, 4, rng=np.random.default_rng(0))
        assert table(np.array([[1, 2, 1]])).shape == (1, 3, 4)

    def test_layernorm_zeroes_mean_and_unit_variance(self):
        norm = LayerNorm(6)
        x = Tensor(np.random.default_rng(0).normal(size=(4, 6)) * 7 + 3)
        out = norm(x).data
        assert np.allclose(out.mean(axis=-1), 0.0, atol=1e-9)
        assert np.allclose(out.std(axis=-1), 1.0, atol=1e-4)

    def test_attention_preserves_shape(self):
        attn = MultiHeadAttention(16, 4, 8, rng=np.random.default_rng(0))
        x = Tensor(np.random.default_rng(0).normal(size=(2, 8, 16)))
        assert attn(x).shape == (2, 8, 16)

    def test_block_preserves_shape(self):
        block = Block(16, 4, 8, rng=np.random.default_rng(0))
        x = Tensor(np.random.default_rng(0).normal(size=(2, 8, 16)))
        assert block(x).shape == (2, 8, 16)

    def test_model_emits_logits_per_position(self):
        model = GPT(vocab_size=13, block_size=6, d_model=8, n_head=2,
                    n_layer=2, seed=0)
        assert model(np.zeros((3, 6), dtype=int)).shape == (3, 6, 13)

    def test_a_bare_sequence_is_treated_as_a_batch_of_one(self):
        model = GPT(vocab_size=13, block_size=6, d_model=8, n_head=2,
                    n_layer=1, seed=0)
        assert model(np.array([1, 2, 3])).shape == (1, 3, 13)

    def test_heads_must_divide_the_channels(self):
        with pytest.raises(ValueError, match="divisible"):
            MultiHeadAttention(10, 4, 8)

    def test_sequences_longer_than_the_context_are_rejected(self):
        model = GPT(vocab_size=13, block_size=4, d_model=8, n_head=2,
                    n_layer=1, seed=0)
        with pytest.raises(ValueError, match="exceeds block_size"):
            model(np.zeros((1, 5), dtype=int))


class TestCausality:
    """Position ``t`` must depend on ``0..t`` and nothing later."""

    def test_future_tokens_cannot_change_earlier_predictions(self):
        model = GPT(vocab_size=17, block_size=6, d_model=16, n_head=4,
                    n_layer=2, seed=0)
        a = np.array([[3, 1, 4, 1, 5, 9]])
        b = a.copy()
        b[0, -1] = 2                      # change only the final token

        first = model(a).data
        second = model(b).data

        # Every position before the change must be bit-for-bit unaffected.
        assert np.allclose(first[:, :-1], second[:, :-1], atol=1e-12)
        # And the changed position itself must actually differ, or the test
        # would pass on a model that ignores its input entirely.
        assert not np.allclose(first[:, -1], second[:, -1])

    def test_attention_weights_are_lower_triangular(self):
        attn = MultiHeadAttention(8, 2, 5, rng=np.random.default_rng(0))
        x = Tensor(np.random.default_rng(1).normal(size=(1, 5, 8)))
        q = attn._split_heads(attn.q(x), 1, 5)
        k = attn._split_heads(attn.k(x), 1, 5)
        scores = (q @ k.transpose(0, 1, 3, 2)) * attn.scale
        weights = (scores + Tensor(attn._mask[:5, :5])).softmax(axis=-1).data

        upper = np.triu(np.ones((5, 5), dtype=bool), k=1)
        assert np.allclose(weights[..., upper], 0.0, atol=1e-12)
        # Each row is still a probability distribution over the visible past.
        assert np.allclose(weights.sum(axis=-1), 1.0)


class TestGradients:
    """Finite differences against the analytic backward pass.

    This is the property the whole repository sells, applied to the
    architecture people assume needs a special-cased autodiff. It does not:
    these gradients come from ``core/tensor.py`` unchanged.
    """

    def _check(self, model, ids, targets, tol=GRADIENT_TOLERANCE):
        def loss_value():
            logits = model(ids)
            flat = logits.reshape(-1, logits.shape[-1])
            return tensor_cross_entropy(flat, targets.reshape(-1)).item()

        logits = model(ids)
        flat = logits.reshape(-1, logits.shape[-1])
        loss = tensor_cross_entropy(flat, targets.reshape(-1))
        model.zero_grad()
        loss.backward()

        worst = 0.0
        for name, parameter in model.named_parameters():
            analytic = parameter.grad.copy()
            numeric = numerical_gradient(loss_value, parameter)
            error = max_relative_error(analytic, numeric)
            assert error < tol, f"{name}: relative error {error:.2e}"
            worst = max(worst, error)
        return worst

    def test_gelu_matches_finite_differences(self):
        x = Tensor(np.array([-2.0, -0.5, 0.0, 0.5, 2.0]))
        out = gelu(x).sum()
        out.backward()
        analytic = x.grad.copy()

        numeric = numerical_gradient(lambda: gelu(x).sum().item(), x)
        assert max_relative_error(analytic, numeric) < GRADIENT_TOLERANCE

    def test_layernorm_gradients(self):
        norm = LayerNorm(5)
        x = Tensor(np.random.default_rng(0).normal(size=(3, 5)))

        def loss_value():
            return (norm(x) * norm(x)).sum().item()

        out = (norm(x) * norm(x)).sum()
        norm.zero_grad()
        out.backward()
        for name, parameter in norm.named_parameters():
            error = max_relative_error(
                parameter.grad, numerical_gradient(loss_value, parameter)
            )
            assert error < GRADIENT_TOLERANCE, f"{name}: {error:.2e}"

    def test_feedforward_gradients(self):
        ffn = FeedForward(6, rng=np.random.default_rng(0))
        x = Tensor(np.random.default_rng(1).normal(size=(2, 6)))

        def loss_value():
            return ffn(x).sum().item()

        out = ffn(x).sum()
        ffn.zero_grad()
        out.backward()
        for name, parameter in ffn.named_parameters():
            error = max_relative_error(
                parameter.grad, numerical_gradient(loss_value, parameter)
            )
            assert error < GRADIENT_TOLERANCE, f"{name}: {error:.2e}"

    def test_attention_gradients(self):
        attn = MultiHeadAttention(8, 2, 4, rng=np.random.default_rng(0))
        x = Tensor(np.random.default_rng(1).normal(size=(2, 4, 8)))

        def loss_value():
            return attn(x).sum().item()

        out = attn(x).sum()
        attn.zero_grad()
        out.backward()
        for name, parameter in attn.named_parameters():
            error = max_relative_error(
                parameter.grad, numerical_gradient(loss_value, parameter)
            )
            assert error < GRADIENT_TOLERANCE, f"{name}: {error:.2e}"

    def test_whole_model_gradients_tied(self):
        """The end-to-end check, with the embedding shared with the output."""
        model = GPT(vocab_size=7, block_size=4, d_model=8, n_head=2,
                    n_layer=2, seed=0)
        rng = np.random.default_rng(0)
        ids = rng.integers(0, 7, size=(2, 4))
        targets = rng.integers(0, 7, size=(2, 4))
        assert self._check(model, ids, targets) < GRADIENT_TOLERANCE

    def test_whole_model_gradients_untied(self):
        model = GPT(vocab_size=7, block_size=4, d_model=8, n_head=2,
                    n_layer=2, tie_weights=False, seed=0)
        rng = np.random.default_rng(0)
        ids = rng.integers(0, 7, size=(2, 4))
        targets = rng.integers(0, 7, size=(2, 4))
        assert self._check(model, ids, targets) < GRADIENT_TOLERANCE

    def test_the_key_bias_is_a_mathematical_no_op(self):
        r"""``attn.k.b`` provably cannot affect anything, and its gradient says so.

        Attention scores are :math:`s_{ij} = q_i \cdot k_j`. Adding a bias
        :math:`b` to every key sends :math:`s_{ij} \to s_{ij} + q_i \cdot b`,
        which is a constant *within each row* :math:`i`. Softmax is invariant
        to adding a constant to all the logits in a row, so the bias cancels
        exactly and the layer's output is unchanged.

        This is a good test to keep because it checks the engine against
        mathematics rather than against itself: the gradient is not merely
        small, it is zero to machine precision, and the forward pass is
        genuinely unmoved by a large perturbation. (It is also why several
        published transformers drop the key bias outright.)
        """
        attn = MultiHeadAttention(8, 2, 4, rng=np.random.default_rng(0))
        x = Tensor(np.random.default_rng(1).normal(size=(2, 4, 8)))

        out = attn(x).sum()
        attn.zero_grad()
        out.backward()
        assert np.max(np.abs(attn.k.b.grad)) < 1e-12

        # And the forward pass agrees: a large shift changes nothing.
        before = attn(x).data.copy()
        attn.k.b.data = attn.k.b.data + 7.0
        assert np.allclose(attn(x).data, before, atol=1e-12)

        # The query bias, by contrast, must genuinely matter, or the test
        # above would be proving that the whole layer ignores its biases.
        assert np.max(np.abs(attn.q.b.grad)) > 1e-3

    def test_a_repeated_token_accumulates_both_gradients(self):
        """The scatter-add property, which ``[] +=`` would silently break."""
        table = Embedding(5, 3, rng=np.random.default_rng(0))
        ids = np.array([[2, 2, 2]])                # same row, three times

        out = table(ids).sum()
        table.zero_grad()
        out.backward()

        # d(sum)/d(row 2) is 1 per use, so three uses means exactly 3.
        assert np.allclose(table.weight.grad[2], 3.0)
        # Unused rows must receive nothing at all.
        assert np.allclose(np.delete(table.weight.grad, 2, axis=0), 0.0)


class TestGeneration:
    def test_generate_extends_the_prompt(self):
        model = GPT(vocab_size=11, block_size=8, d_model=8, n_head=2,
                    n_layer=1, seed=0)
        out = model.generate([1, 2], 6, rng=np.random.default_rng(0))
        assert len(out) == 8
        assert out[:2] == [1, 2]
        assert all(0 <= t < 11 for t in out)

    def test_zero_temperature_is_deterministic(self):
        model = GPT(vocab_size=11, block_size=8, d_model=8, n_head=2,
                    n_layer=1, seed=0)
        first = model.generate([1, 2], 5, temperature=0.0)
        second = model.generate([1, 2], 5, temperature=0.0)
        assert first == second

    def test_top_k_restricts_the_sampled_set(self):
        """With k=1 sampling can only ever pick the argmax."""
        model = GPT(vocab_size=11, block_size=8, d_model=8, n_head=2,
                    n_layer=1, seed=0)
        greedy = model.generate([1, 2], 5, temperature=0.0)
        sampled = model.generate([1, 2], 5, top_k=1, rng=np.random.default_rng(3))
        assert greedy == sampled

    def test_generation_runs_past_the_context_window(self):
        """Prompts longer than block_size must slide, not crash."""
        model = GPT(vocab_size=11, block_size=4, d_model=8, n_head=2,
                    n_layer=1, seed=0)
        out = model.generate([1, 2, 3], 10, rng=np.random.default_rng(0))
        assert len(out) == 13

    def test_an_empty_prompt_is_rejected(self):
        model = GPT(vocab_size=11, block_size=4, d_model=8, n_head=2,
                    n_layer=1, seed=0)
        with pytest.raises(ValueError, match="at least one token"):
            model.generate([], 3)


class TestLearning:
    def test_the_model_can_memorise_a_short_sequence(self):
        """The end-to-end proof: loss must fall well below chance.

        Overfitting one sequence is the right smoke test for a language model.
        It isolates optimisation from generalisation: if a model with enough
        capacity cannot drive the loss on a single example toward zero,
        something is broken in the architecture or the gradients, and no amount
        of data would have saved it.
        """
        from nabla.optim import AdamW

        rng = np.random.default_rng(0)
        model = GPT(vocab_size=6, block_size=8, d_model=32, n_head=4,
                    n_layer=2, seed=0)
        ids = rng.integers(0, 6, size=(1, 8))
        targets = rng.integers(0, 6, size=(1, 8))

        optimiser = AdamW(model.parameters(), lr=3e-3)
        first = last = None
        for step in range(120):
            logits = model(ids)
            loss = tensor_cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
            )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            if step == 0:
                first = loss.item()
            last = loss.item()

        assert first > np.log(6) * 0.8          # started near chance
        assert last < 0.1                       # and memorised it
