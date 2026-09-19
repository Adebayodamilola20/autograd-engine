"""Phase 16 -- loss functions: correctness, gradients, and numerical stability.

The stability tests matter as much as the correctness ones. A loss that is
mathematically right and numerically fragile produces ``nan`` on the first
confident prediction, and a ``nan`` gradient poisons every parameter in one
backward pass.
"""

from __future__ import annotations

import math

import pytest

from nabla import Value
from nabla.core.gradcheck import check_gradients
from nabla.losses import (
    binary_cross_entropy,
    cross_entropy,
    huber_loss,
    log_softmax,
    log_sum_exp,
    mae_loss,
    mse_loss,
    nll_loss,
    softmax,
    softmax_cross_entropy,
    sse_loss,
)


def values(xs):
    return [Value(x) for x in xs]


# ======================================================================
# MSE family
# ======================================================================


class TestMSE:
    def test_forward(self):
        # ((2-1)² + (3-5)²) / 2 = (1 + 4)/2
        assert mse_loss(values([2.0, 3.0]), [1.0, 5.0]).data == pytest.approx(2.5)

    def test_zero_when_perfect(self):
        assert mse_loss(values([1.0, 2.0]), [1.0, 2.0]).data == 0.0

    def test_gradient_is_proportional_to_error(self):
        # d/dŷ [(ŷ-y)²/N] = 2(ŷ-y)/N
        preds = values([2.0, 3.0])
        mse_loss(preds, [1.0, 5.0]).backward()
        assert preds[0].grad == pytest.approx(2 * (2.0 - 1.0) / 2)
        assert preds[1].grad == pytest.approx(2 * (3.0 - 5.0) / 2)

    def test_single_value_form(self):
        assert mse_loss(Value(3.0), 1.0).data == pytest.approx(4.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="predictions"):
            mse_loss(values([1.0]), [1.0, 2.0])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            mse_loss([], [])

    def test_gradcheck(self):
        assert check_gradients(
            lambda a, b: mse_loss([a, b], [1.0, -0.5]), [0.3, 0.9]
        ).passed

    def test_sse_is_n_times_mse(self):
        preds = [0.3, 0.9]
        m = mse_loss(values(preds), [1.0, -0.5]).data
        s = sse_loss(values(preds), [1.0, -0.5]).data
        assert s == pytest.approx(2 * m)


class TestRobustLosses:
    def test_mae_forward(self):
        # (|2-1| + |3-5|)/2 = 1.5
        assert mae_loss(values([2.0, 3.0]), [1.0, 5.0]).data == pytest.approx(1.5)

    def test_mae_gradient_is_bounded(self):
        # ±1/N regardless of error size -- that's the robustness.
        small = values([1.1])
        mae_loss(small, [1.0]).backward()
        big = values([100.0])
        mae_loss(big, [1.0]).backward()
        assert abs(small[0].grad) == pytest.approx(abs(big[0].grad))

    def test_mse_gradient_is_not_bounded(self):
        small = values([1.1])
        mse_loss(small, [1.0]).backward()
        big = values([100.0])
        mse_loss(big, [1.0]).backward()
        assert abs(big[0].grad) > 100 * abs(small[0].grad)

    def test_huber_matches_mse_near_zero(self):
        # For |e| <= delta the Huber loss is e²/2.
        assert huber_loss(values([0.7]), [0.5], delta=1.0).data == pytest.approx(
            0.5 * 0.2**2
        )

    def test_huber_is_linear_in_the_tail(self):
        # delta(|e| - delta/2) = 1*(3.5 - 0.5)
        assert huber_loss(values([4.0]), [0.5], delta=1.0).data == pytest.approx(3.0)

    def test_huber_pieces_agree_at_the_boundary(self):
        """Value *and* slope must match at |e| = delta, or it isn't smooth."""
        below = huber_loss(values([1.5 - 1e-9]), [0.5], delta=1.0).data
        above = huber_loss(values([1.5 + 1e-9]), [0.5], delta=1.0).data
        assert below == pytest.approx(above, abs=1e-8)

        a = values([1.5 - 1e-6])
        huber_loss(a, [0.5], delta=1.0).backward()
        b = values([1.5 + 1e-6])
        huber_loss(b, [0.5], delta=1.0).backward()
        assert a[0].grad == pytest.approx(b[0].grad, abs=1e-5)

    @pytest.mark.parametrize("x", [0.7, 4.0, -3.0])
    def test_huber_gradcheck(self, x):
        assert check_gradients(lambda a: huber_loss([a], [0.5]), [x]).passed


# ======================================================================
# Softmax and cross-entropy
# ======================================================================


class TestSoftmax:
    def test_probabilities_sum_to_one(self):
        probs = [p.data for p in softmax(values([2.0, 1.0, 0.1]))]
        assert sum(probs) == pytest.approx(1.0)
        assert all(0.0 < p < 1.0 for p in probs)

    def test_matches_the_definition(self):
        logits = [2.0, 1.0, 0.1]
        expected = [math.exp(z) / sum(math.exp(v) for v in logits) for z in logits]
        got = [p.data for p in softmax(values(logits))]
        assert got == pytest.approx(expected)

    def test_shift_invariance(self):
        """softmax(z + c) == softmax(z). The property that makes it stable."""
        base = [p.data for p in softmax(values([2.0, 1.0, 0.1]))]
        shifted = [p.data for p in softmax(values([102.0, 101.0, 100.1]))]
        assert base == pytest.approx(shifted)

    def test_preserves_ordering(self):
        # Monotonic, so argmax(logits) == argmax(softmax(logits)).
        probs = [p.data for p in softmax(values([0.5, 3.0, -1.0]))]
        assert probs.index(max(probs)) == 1

    def test_uniform_logits_give_uniform_probabilities(self):
        probs = [p.data for p in softmax(values([5.0] * 4))]
        assert probs == pytest.approx([0.25] * 4)

    def test_log_softmax_equals_log_of_softmax(self):
        logits = values([2.0, 1.0, 0.1])
        direct = [lp.data for lp in log_softmax(logits)]
        indirect = [math.log(p.data) for p in softmax(values([2.0, 1.0, 0.1]))]
        assert direct == pytest.approx(indirect)

    def test_log_sum_exp_matches_the_naive_form_when_safe(self):
        logits = [1.0, 2.0, 3.0]
        assert log_sum_exp(values(logits)).data == pytest.approx(
            math.log(sum(math.exp(z) for z in logits))
        )

    def test_log_sum_exp_of_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            log_sum_exp([])


class TestNumericalStability:
    """The tests that separate a working loss from one that returns nan."""

    def test_huge_logits_do_not_overflow(self):
        # exp(1000) is inf. The log-sum-exp trick keeps every exponent <= 0.
        loss = cross_entropy(values([1000.0, 999.0, 1.0]), 0)
        assert math.isfinite(loss.data)
        assert loss.data == pytest.approx(math.log(1 + math.exp(-1) + math.exp(-999)))

    def test_tiny_probabilities_do_not_underflow_to_negative_infinity(self):
        # p ≈ e^-1000 underflows to exactly 0.0; log(0) is -inf.
        # Computing log-softmax directly gives a perfectly ordinary -1000.
        loss = cross_entropy(values([-1000.0, 0.0]), 0)
        assert math.isfinite(loss.data)
        assert loss.data == pytest.approx(1000.0, rel=1e-6)

    def test_gradients_stay_finite_at_extremes(self):
        logits = values([1000.0, -1000.0])
        cross_entropy(logits, 1).backward()
        assert all(math.isfinite(v.grad) for v in logits)

    def test_softmax_of_huge_logits(self):
        probs = [p.data for p in softmax(values([800.0, 801.0]))]
        assert sum(probs) == pytest.approx(1.0)
        assert all(math.isfinite(p) for p in probs)

    def test_binary_cross_entropy_from_logits_is_stable(self):
        for z in (-800.0, -50.0, 0.0, 50.0, 800.0):
            for y in (0.0, 1.0):
                loss = binary_cross_entropy(Value(z), y)
                assert math.isfinite(loss.data), f"z={z}, y={y}"


class TestCrossEntropy:
    def test_forward_is_negative_log_probability(self):
        logits = [2.0, 1.0, 0.1]
        p0 = math.exp(2.0) / sum(math.exp(z) for z in logits)
        assert cross_entropy(values(logits), 0).data == pytest.approx(-math.log(p0))

    def test_perfect_prediction_gives_near_zero_loss(self):
        assert cross_entropy(values([100.0, 0.0, 0.0]), 0).data == pytest.approx(0.0, abs=1e-30)

    def test_confidently_wrong_is_punished_without_bound(self):
        mild = cross_entropy(values([0.0, 1.0]), 0).data
        severe = cross_entropy(values([-50.0, 50.0]), 0).data
        assert severe > 40 * mild

    def test_uniform_prediction_gives_log_k(self):
        # No information: loss is ln(K).
        for k in (2, 5, 10):
            assert cross_entropy(values([0.0] * k), 0).data == pytest.approx(math.log(k))

    def test_the_gradient_is_p_minus_y(self):
        r"""The identity that makes softmax + cross-entropy the right pair.

        ∂L/∂zᵢ = pᵢ - yᵢ, with no σ' factor -- the 1/p from the log cancels
        the p from the softmax exactly. Derived by the engine here, and
        compared against the closed form.
        """
        logits = values([2.0, 1.0, 0.1])
        cross_entropy(logits, 0).backward()

        probs = [p.data for p in softmax(values([2.0, 1.0, 0.1]))]
        for i, (v, p) in enumerate(zip(logits, probs)):
            expected = p - (1.0 if i == 0 else 0.0)
            assert v.grad == pytest.approx(expected)

    def test_gradients_sum_to_zero(self):
        # Σ(pᵢ - yᵢ) = 1 - 1 = 0, because both sum to one.
        logits = values([1.5, -0.5, 2.0, 0.3])
        cross_entropy(logits, 2).backward()
        assert sum(v.grad for v in logits) == pytest.approx(0.0, abs=1e-12)

    def test_correct_class_gets_negative_gradient(self):
        # Push its logit up; push every other one down.
        logits = values([1.0, 2.0, 0.5])
        cross_entropy(logits, 0).backward()
        assert logits[0].grad < 0
        assert logits[1].grad > 0 and logits[2].grad > 0

    def test_target_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            cross_entropy(values([1.0, 2.0]), 5)

    def test_equals_nll_of_log_softmax(self):
        logits = values([1.0, 2.0, 0.5])
        a = cross_entropy(logits, 1).data
        b = nll_loss(log_softmax(values([1.0, 2.0, 0.5])), 1).data
        assert a == pytest.approx(b)

    @pytest.mark.parametrize("target", [0, 1, 2])
    def test_gradcheck(self, target):
        assert check_gradients(
            lambda a, b, c: cross_entropy([a, b, c], target), [0.5, -1.2, 2.0]
        ).passed


class TestBatchCrossEntropy:
    def test_averages_over_the_batch(self):
        batch = [values([2.0, 1.0]), values([0.0, 3.0])]
        individual = [
            cross_entropy(values([2.0, 1.0]), 0).data,
            cross_entropy(values([0.0, 3.0]), 1).data,
        ]
        got = softmax_cross_entropy(batch, [0, 1]).data
        assert got == pytest.approx(sum(individual) / 2)

    def test_gradient_magnitude_is_batch_size_independent(self):
        """Averaging (not summing) is why a learning rate transfers between
        batch sizes."""
        one = values([2.0, 1.0])
        softmax_cross_entropy([one], [0]).backward()

        many = [values([2.0, 1.0]) for _ in range(8)]
        softmax_cross_entropy(many, [0] * 8).backward()

        assert many[0][0].grad == pytest.approx(one[0].grad / 8)
        total = sum(row[0].grad for row in many)
        assert total == pytest.approx(one[0].grad)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="targets"):
            softmax_cross_entropy([values([1.0, 2.0])], [0, 1])

    def test_empty_batch_raises(self):
        with pytest.raises(ValueError, match="empty batch"):
            softmax_cross_entropy([], [])


class TestLabelRangeIsChecked:
    """A negative class label must not be accepted as an index.

    ``out[rows, targets] = 1.0`` raises for a label at or above ``n_classes``,
    but a negative label is a perfectly legal NumPy index counting from the
    end, so ``-1`` silently one-hotted the last class. The guard was
    asymmetric: too high was caught, too low was not.

    It matters because ``-1`` is the conventional sentinel for "unlabelled"
    or "ignore this row", so the most natural way to mark a row as having no
    label was also the way to silently train it against the final class.
    """

    def _logits(self):
        import numpy as np

        from nabla.core.tensor import Tensor

        # Strongly predicts class 2, so a wrapped -1 looks like a perfect
        # prediction and the bug produces a *flattering* number.
        return Tensor(np.array([[0.0, 0.0, 9.0]]))

    def test_cross_entropy_rejects_a_negative_label(self):
        from nabla.losses.tensor_losses import tensor_cross_entropy

        with pytest.raises(ValueError, match=r"-1 is outside \[0, 2\]"):
            tensor_cross_entropy(self._logits(), [-1])

    def test_the_message_explains_the_sentinel_trap(self):
        from nabla.losses.tensor_losses import one_hot_array

        with pytest.raises(ValueError, match="not an 'ignore' sentinel"):
            one_hot_array([-1], 3)

    def test_accuracy_rejects_a_negative_label(self):
        """Loss and accuracy must not disagree about the same batch.

        Before the fix, ``-1`` gave a near-zero loss (one-hot wrapped to
        class 2, which the model predicted) *and* zero accuracy (``argmax``
        returns 2, which is not -1). Two metrics contradicting each other is
        far harder to debug than one clear error.
        """
        from nabla.losses.tensor_losses import tensor_accuracy

        with pytest.raises(ValueError, match=r"-1 is outside"):
            tensor_accuracy(self._logits(), [-1])

    def test_both_one_hot_helpers_agree(self):
        from nabla.data.dataset import one_hot
        from nabla.losses.tensor_losses import one_hot_array

        for helper in (one_hot, one_hot_array):
            with pytest.raises(ValueError, match="outside"):
                helper([-1], 3)

    def test_a_too_large_label_is_still_rejected(self):
        from nabla.losses.tensor_losses import one_hot_array

        with pytest.raises(ValueError, match=r"5 is outside \[0, 2\]"):
            one_hot_array([5], 3)

    def test_accuracy_rejects_a_target_count_mismatch(self):
        from nabla.losses.tensor_losses import tensor_accuracy

        with pytest.raises(ValueError, match="1 rows of logits but 2 targets"):
            tensor_accuracy(self._logits(), [0, 1])

    def test_valid_labels_are_untouched(self):
        from nabla.data.dataset import one_hot
        from nabla.losses.tensor_losses import (
            one_hot_array,
            tensor_accuracy,
            tensor_cross_entropy,
        )

        assert one_hot_array([0, 2], 3).tolist() == [[1, 0, 0], [0, 0, 1]]
        assert one_hot([0, 2], 3).tolist() == [[1, 0, 0], [0, 0, 1]]
        assert tensor_accuracy(self._logits(), [2]) == 1.0
        assert tensor_cross_entropy(self._logits(), [2]).item() == pytest.approx(
            0.000247, abs=1e-5
        )

    def test_an_empty_batch_is_not_an_error(self):
        """``min``/``max`` of an empty array raise, so size is checked first."""
        from nabla.losses.tensor_losses import one_hot_array

        assert one_hot_array([], 3).shape == (0, 3)


class TestBinaryCrossEntropy:
    @pytest.mark.parametrize("z, y", [(1.3, 1.0), (-2.1, 0.0), (0.0, 1.0), (0.0, 0.0)])
    def test_gradient_is_sigmoid_minus_target(self, z, y):
        v = Value(z)
        binary_cross_entropy(v, y).backward()
        assert v.grad == pytest.approx(1.0 / (1.0 + math.exp(-z)) - y)

    def test_forward_matches_the_definition(self):
        for z, y in ((1.3, 1.0), (-0.7, 0.0), (2.0, 1.0)):
            p = 1.0 / (1.0 + math.exp(-z))
            expected = -(y * math.log(p) + (1 - y) * math.log(1 - p))
            assert binary_cross_entropy(Value(z), y).data == pytest.approx(expected)

    def test_from_probabilities_rejects_out_of_range(self):
        with pytest.raises(ValueError, match="probability"):
            binary_cross_entropy(Value(1.5), 1.0, from_logits=False)

    @pytest.mark.parametrize("z", [-3.0, -0.5, 0.5, 3.0])
    @pytest.mark.parametrize("y", [0.0, 1.0])
    def test_gradcheck(self, z, y):
        assert check_gradients(lambda v: binary_cross_entropy(v, y), [z]).passed


class TestWhyMSEIsWrongForClassification:
    r"""The claim in ``losses/mse.py``, measured rather than asserted.

    With a sigmoid output and MSE, the gradient is :math:`2(\sigma - y)\sigma'`.
    When the model is confidently wrong, :math:`\sigma' \approx 0` kills it.
    Cross-entropy's gradient is :math:`\sigma - y`, with no such factor.
    """

    def test_mse_gradient_vanishes_when_confidently_wrong(self):
        z = Value(-10.0)  # says "class 0" with ~99.995% confidence
        mse_loss([z.sigmoid()], [1.0]).backward()  # but the truth is class 1
        assert abs(z.grad) < 1e-4  # essentially no correction signal

    def test_cross_entropy_gradient_stays_strong(self):
        z = Value(-10.0)
        binary_cross_entropy(z, 1.0).backward()
        assert abs(z.grad) == pytest.approx(1.0, abs=1e-4)

    def test_the_ratio_is_four_orders_of_magnitude(self):
        a = Value(-10.0)
        mse_loss([a.sigmoid()], [1.0]).backward()
        b = Value(-10.0)
        binary_cross_entropy(b, 1.0).backward()
        assert abs(b.grad) / abs(a.grad) > 10_000
