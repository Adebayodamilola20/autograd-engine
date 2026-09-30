"""Tests for the experiment harness in ``experiments/common.py``.

The harness produces the numbers every experiment reports, so a defect here
is invisible in the code and highly visible in the conclusions. It had no
tests.

The focus is ``aggregate``, which turns several seeded runs into the
"mean ± std" line that the experiment tables print.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import RunResult, aggregate, mean_curve  # noqa: E402


def run(label: str, seed: int = 0, **overrides) -> RunResult:
    fields = {"label": label, "config": {}, "seed": seed}
    fields.update(overrides)
    return RunResult(**fields)


class TestAggregate:
    def test_groups_by_label_preserving_first_seen_order(self):
        results = [run("b"), run("a"), run("b"), run("a")]
        assert [a.label for a in aggregate(results)] == ["b", "a"]
        assert [a.runs for a in aggregate(results)] == [2, 2]

    def test_empty_input_gives_no_groups(self):
        assert aggregate([]) == []

    def test_reports_the_sample_standard_deviation(self):
        r"""``ddof=1``, not NumPy's default of 0.

        These runs are a *sample* from the distribution of outcomes a seed can
        produce, and the reported spread is an estimate of that distribution's
        spread, so Bessel's correction applies. The default understates it by
        :math:`\sqrt{n/(n-1)}`: 18% at three seeds, 41% at two.

        Every experiment in the repository reports "mean ± std" over three
        seeds, so the uncorrected value made every error bar in every table
        too narrow by the same factor.
        """
        accuracies = [0.970, 0.975, 0.968]
        runs = [run("v", seed=i, test_acc=a) for i, a in enumerate(accuracies)]

        got = aggregate(runs)[0].test_acc_std
        assert got == pytest.approx(float(np.std(accuracies, ddof=1)))
        assert got > float(np.std(accuracies)), "correction must widen the bar"

    def test_a_single_run_reports_zero_not_nan(self):
        """``ddof=1`` on one sample divides by zero and numpy yields nan.

        One run has no spread to estimate, so 0.0 is the honest answer and a
        nan would poison the formatted table.
        """
        std = aggregate([run("v", test_acc=0.97)])[0].test_acc_std
        assert std == 0.0
        assert not np.isnan(std)

    def test_the_mean_is_unaffected(self):
        runs = [run("v", seed=i, test_acc=a) for i, a in enumerate([0.9, 1.0])]
        assert aggregate(runs)[0].test_acc_mean == pytest.approx(0.95)

    def test_diverged_runs_are_counted(self):
        runs = [run("v", seed=0, diverged=True), run("v", seed=1)]
        assert aggregate(runs)[0].diverged == 1

    def test_all_diverged_reports_as_such(self):
        runs = [run("v", seed=0, diverged=True)]
        assert aggregate(runs)[0].acc_text() == "diverged"

    def test_a_non_finite_final_loss_does_not_poison_the_mean(self):
        """A diverged run's inf must not make the whole group's mean inf."""
        # `final_train_loss` is derived from the last entry of `train_loss`.
        runs = [
            run("v", seed=0, train_loss=[1.0, float("inf")]),
            run("v", seed=1, train_loss=[1.0, 0.2]),
        ]
        assert aggregate(runs)[0].final_loss_mean == pytest.approx(0.2)


class TestMeanCurve:
    def test_truncates_to_the_shortest_run(self):
        """Averaging ragged curves would pad the short one with garbage."""
        runs = [
            run("v", seed=0, train_loss=[1.0, 0.5, 0.2]),
            run("v", seed=1, train_loss=[2.0, 1.5]),
        ]
        curve = mean_curve(runs, "v", "train_loss")
        assert curve.tolist() == pytest.approx([1.5, 1.0])

    def test_an_unknown_label_gives_an_empty_curve(self):
        assert mean_curve([run("v", train_loss=[1.0])], "other", "train_loss").size == 0

    def test_ignores_runs_from_other_labels(self):
        runs = [
            run("keep", seed=0, train_loss=[1.0, 1.0]),
            run("drop", seed=1, train_loss=[9.0, 9.0]),
        ]
        assert mean_curve(runs, "keep", "train_loss").tolist() == [1.0, 1.0]
