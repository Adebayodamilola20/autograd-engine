"""Phase 17 -- tests for the benchmark harness.

The benchmarks produce the numbers we make claims about, so the machinery that
produces them needs to be correct. A timing harness that quietly reports the
mean when it says "best", or a formatter that drops a factor of 1000, would
make every number in ``docs/18-performance.md`` wrong in a way nobody would
notice by reading it.

We do not test *how fast* anything is -- that is machine-dependent and would
make the suite flaky. We test that the harness measures and reports honestly.
"""

from __future__ import annotations

import json
import time

import pytest

from benchmarks.common import (
    BenchmarkReport,
    Timing,
    describe_environment,
    human_time,
    measure,
    peak_rss_mb,
    table,
)


class TestTiming:
    def test_best_per_op_divides_by_the_operation_count(self):
        t = Timing(label="x", best=1.0, median=1.5, mean=1.6, runs=3, per_call=100)
        assert t.best_per_op == pytest.approx(0.01)

    def test_ops_per_second_uses_the_best_run(self):
        t = Timing(label="x", best=0.5, median=1.0, mean=2.0, runs=3, per_call=100)
        assert t.ops_per_second == pytest.approx(200.0)

    def test_per_call_defaults_to_one(self):
        t = Timing(label="x", best=2.0, median=2.0, mean=2.0, runs=1)
        assert t.best_per_op == pytest.approx(2.0)
        assert t.ops_per_second == pytest.approx(0.5)

    def test_zero_duration_does_not_divide_by_zero(self):
        """A run too fast for the clock must not crash the report."""
        t = Timing(label="x", best=0.0, median=0.0, mean=0.0, runs=1)
        assert t.ops_per_second == float("inf")

    def test_as_dict_is_json_serialisable(self):
        t = Timing(label="x", best=1.0, median=1.0, mean=1.0, runs=1, per_call=2)
        restored = json.loads(json.dumps(t.as_dict()))
        assert restored["label"] == "x"
        assert restored["best_per_op"] == pytest.approx(0.5)


class TestMeasure:
    def test_runs_the_function_warmup_plus_repeat_times(self):
        calls = []
        measure(lambda: calls.append(1), label="t", repeat=4, warmup=2)
        assert len(calls) == 6

    def test_warmup_results_are_discarded(self):
        """The warm-up run is the slow one; it must not reach the statistics."""
        durations = iter([0.05, 0.001, 0.001, 0.001])

        def fn() -> None:
            time.sleep(next(durations))

        t = measure(fn, label="t", repeat=3, warmup=1)
        assert t.runs == 3
        assert t.best < 0.03  # the 50 ms warm-up was thrown away

    def test_best_is_never_greater_than_median_or_mean(self):
        t = measure(lambda: sum(range(200)), label="t", repeat=8)
        assert t.best <= t.median
        assert t.best <= t.mean

    def test_records_the_label_and_per_call(self):
        t = measure(lambda: None, label="named", repeat=2, per_call=32)
        assert t.label == "named"
        assert t.per_call == 32


class TestHumanTime:
    @pytest.mark.parametrize(
        "seconds,expected_unit",
        [
            (5e-10, "ns"),
            (5e-7, "ns"),
            (5e-5, "µs"),
            (5e-2, "ms"),
            (5.0, "s"),
            (300.0, "min"),
            (7200.0, "h"),
        ],
    )
    def test_picks_a_readable_unit(self, seconds, expected_unit):
        assert human_time(seconds).endswith(expected_unit)

    def test_boundaries_do_not_produce_absurd_numbers(self):
        """Every formatted value should have a small mantissa."""
        for exponent in range(-10, 5):
            text = human_time(10.0**exponent)
            mantissa = float(text.split()[0].replace(",", ""))
            assert 0 <= mantissa < 10000, f"{text} for 1e{exponent}"

    def test_zero_is_handled(self):
        assert human_time(0.0).endswith("ns")


class TestTable:
    def test_every_row_has_the_same_width(self):
        text = table(
            [["a", "1", "2"], ["much longer label", "30000", "4"]],
            ["name", "x", "y"],
        )
        lines = text.splitlines()
        assert len({len(line) for line in lines}) == 1

    def test_includes_headers_and_all_rows(self):
        text = table([["alpha", "1"], ["beta", "2"]], ["name", "count"])
        for token in ("name", "count", "alpha", "beta"):
            assert token in text

    def test_first_column_left_aligned_rest_right_aligned(self):
        text = table([["a", "1"]], ["name", "count"])
        row = text.splitlines()[2]
        assert row.strip().startswith("a")
        assert row.rstrip().endswith("1")


class TestBenchmarkReport:
    def test_add_stores_the_timing_under_its_label(self):
        report = BenchmarkReport(engine="test")
        report.add(Timing(label="op", best=1.0, median=1.0, mean=1.0, runs=1))
        assert "op" in report.timings
        assert report.timings["op"]["best"] == pytest.approx(1.0)

    def test_add_returns_the_timing_so_it_can_be_used_inline(self):
        report = BenchmarkReport(engine="test")
        t = Timing(label="op", best=1.0, median=1.0, mean=1.0, runs=1)
        assert report.add(t) is t

    def test_finish_records_memory_and_environment(self):
        report = BenchmarkReport(engine="test").finish()
        assert report.peak_rss_mb > 0
        assert "python" in report.environment

    def test_save_round_trips_through_json(self, tmp_path):
        report = BenchmarkReport(engine="test", parameters=42)
        report.add(Timing(label="op", best=1.0, median=1.0, mean=1.0, runs=1))
        path = report.save(tmp_path / "nested" / "bench.json")

        restored = json.loads(path.read_text())
        assert restored["engine"] == "test"
        assert restored["parameters"] == 42
        assert restored["timings"]["op"]["best"] == pytest.approx(1.0)

    def test_save_creates_missing_parent_directories(self, tmp_path):
        path = BenchmarkReport(engine="t").save(tmp_path / "a" / "b" / "c.json")
        assert path.exists()


class TestEnvironment:
    def test_peak_rss_is_plausible(self):
        """Catches the macOS-bytes vs Linux-kilobytes trap.

        Getting the unit wrong is a silent 1024x error. Any Python process
        running NumPy uses more than 1 MiB and less than 1 TiB, so a wrong
        divisor lands outside this window.
        """
        assert 1.0 < peak_rss_mb() < 1_000_000.0

    def test_environment_names_the_machine_and_numpy(self):
        env = describe_environment()
        assert env["python"]
        assert env["platform"]
        assert "numpy" in env
