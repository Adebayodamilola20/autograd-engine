"""Tests for ``training/metrics.py``.

Every test here is about an input that should be refused and was not. The
module had no test file, and the bugs it contained were all the same shape: a
malformed call returned a plausible number instead of raising, so the mistake
surfaced as a suspiciously good metric rather than as a failure.

That shape is worth naming, because it is the expensive one. A crash gets
fixed in minutes. A validation accuracy that is silently too high gets
believed.
"""

from __future__ import annotations

import pytest

from nabla.training.metrics import (
    EpochMetrics,
    RunningAverage,
    accuracy,
    argmax,
    confusion_counts,
)


class TestArgmax:
    def test_returns_the_index_of_the_largest(self):
        assert argmax([2.0, 1.0, 5.0]) == 2
        assert argmax([7.0]) == 0

    def test_ties_take_the_first(self):
        """Documented behaviour, and what `>` rather than `>=` produces."""
        assert argmax([1.0, 3.0, 3.0]) == 1

    def test_an_empty_sequence_raises(self):
        """It used to return 0.

        The loop starts at index 1, so an empty input fell straight through to
        the initial ``best = 0``. That is indistinguishable from confidently
        predicting class 0, which is a wrong answer wearing the costume of a
        real one.
        """
        with pytest.raises(ValueError, match="empty sequence"):
            argmax([])


class TestAccuracy:
    def test_counts_exact_matches(self):
        assert accuracy([0, 1, 1], [0, 1, 2]) == pytest.approx(2 / 3)

    def test_empty_is_zero_not_an_error(self):
        assert accuracy([], []) == 0.0

    def test_mismatched_lengths_raise(self):
        """It used to report 100%.

        ``zip`` stops at the shorter sequence, so three predictions against
        two targets scored the first two and divided by 2. A dropped batch or
        an off-by-one in the evaluation loop showed up as an improbably good
        number instead of an error.
        """
        with pytest.raises(ValueError, match="3 predictions but 2 targets"):
            accuracy([0, 1, 2], [0, 1])

    def test_the_truncation_case_would_have_scored_one(self):
        """Pins the specific input that used to return a perfect score."""
        with pytest.raises(ValueError):
            accuracy([0, 1, 9], [0, 1])


class TestConfusionCounts:
    def test_builds_true_by_predicted(self):
        assert confusion_counts([0, 1], [0, 1], 2) == [[1, 0], [0, 1]]

    def test_off_diagonal_records_the_mistake(self):
        # true 0, predicted 1
        assert confusion_counts([1], [0], 2) == [[0, 1], [0, 0]]

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="predictions but"):
            confusion_counts([0, 1, 2], [0, 1], 3)

    def test_a_negative_target_raises(self):
        """It used to count into the last row.

        ``matrix[-1][0] += 1`` is a legal Python index, so a ``-1``
        "unlabelled" sentinel silently inflated the final class. A label at or
        above ``n_classes`` already raised IndexError; both directions now
        behave the same.
        """
        with pytest.raises(ValueError, match=r"target -1 is outside \[0, 2\]"):
            confusion_counts([0], [-1], 3)

    def test_a_negative_prediction_raises(self):
        with pytest.raises(ValueError, match=r"prediction -1 is outside"):
            confusion_counts([-1], [0], 3)

    def test_a_too_large_label_raises_with_a_message(self):
        """Previously an IndexError with no mention of which label."""
        with pytest.raises(ValueError, match=r"target 5 is outside \[0, 2\]"):
            confusion_counts([0], [5], 3)

    def test_row_sums_equal_the_class_counts(self):
        matrix = confusion_counts([0, 1, 1, 2], [0, 1, 2, 2], 3)
        assert [sum(row) for row in matrix] == [1, 1, 2]


class TestRunningAverage:
    def test_weights_by_batch_size(self):
        """A final partial batch must not count as much as a full one."""
        average = RunningAverage()
        average.update(1.0, 32)
        average.update(5.0, 8)
        assert average.value == pytest.approx((1.0 * 32 + 5.0 * 8) / 40)

    def test_empty_is_zero(self):
        assert RunningAverage().value == 0.0

    def test_a_zero_weight_contributes_nothing(self):
        average = RunningAverage()
        average.update(9.0, 0)
        assert average.value == 0.0

    def test_a_negative_weight_raises(self):
        """It used to return the value itself, which looks entirely normal.

        ``update(9.0, -2)`` left ``total=-18, count=-2``; the signs cancel to
        exactly ``9.0``. Nothing downstream could distinguish that from a real
        mean.
        """
        with pytest.raises(ValueError, match="non-negative"):
            RunningAverage().update(9.0, -2)

    def test_reset_clears_both_accumulators(self):
        average = RunningAverage()
        average.update(4.0, 2)
        average.reset()
        assert average.value == 0.0 and average.count == 0


class TestEpochMetrics:
    def test_samples_per_second(self):
        assert EpochMetrics(1, samples=100, seconds=2.0).samples_per_second == 50.0

    def test_zero_seconds_does_not_divide_by_zero(self):
        assert EpochMetrics(1, samples=100, seconds=0.0).samples_per_second == 0.0

    def test_format_line_mentions_the_epoch(self):
        line = EpochMetrics(3, train_loss=0.5).format_line(10)
        assert "3" in line and "10" in line
