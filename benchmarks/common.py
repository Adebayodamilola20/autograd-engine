"""Phase 17 -- shared measurement machinery for the benchmarks.

Benchmarking is easy to do badly, and a benchmark nobody trusts is worse than
no benchmark. The rules this module enforces:

**Warm up first.** The first call to anything pays for import-time work, lazy
allocation, and a cold instruction cache. On our engine it also pays for
NumPy's first BLAS dispatch. Warm-up runs are executed and thrown away.

**Report the minimum, not the mean.** A benchmark measures ``true cost +
noise``, and OS noise is strictly non-negative -- the scheduler can only ever
*slow* a run down. The minimum is therefore the closest estimate of the real
cost, while the mean drifts upward with whatever else the machine was doing.
We report the median too, so a suspicious gap between them is visible.

**Measure processes separately for memory.** PyTorch allocates most of its
memory in C++, where ``tracemalloc`` cannot see it. Peak RSS from the OS
counts both, so each engine is benchmarked in its own process and reports its
own peak RSS. Comparing ``tracemalloc`` numbers across the two would flatter
us enormously and mean nothing.

**Pin the seed and the data.** Both engines get the same architecture, the same
sample count, the same batch size and the same seed. Anything else is
comparing two different experiments.
"""

from __future__ import annotations

import json
import platform
import resource
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

# The one architecture every benchmark uses, so all numbers are comparable.
ARCHITECTURE = [784, 128, 64, 10]
BATCH_SIZE = 64
SEED = 0


# ======================================================================
# timing
# ======================================================================


@dataclass
class Timing:
    """The result of timing one operation repeatedly."""

    label: str
    best: float  #: fastest run, seconds -- our estimate of true cost
    median: float
    mean: float
    runs: int
    per_call: int = 1  #: operations folded into a single timed run

    @property
    def best_per_op(self) -> float:
        return self.best / self.per_call

    @property
    def ops_per_second(self) -> float:
        return self.per_call / self.best if self.best > 0 else float("inf")

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["best_per_op"] = self.best_per_op
        d["ops_per_second"] = self.ops_per_second
        return d


def measure(
    fn: Callable[[], Any],
    *,
    label: str,
    repeat: int = 5,
    warmup: int = 1,
    per_call: int = 1,
) -> Timing:
    """Time ``fn`` ``repeat`` times after ``warmup`` discarded runs.

    ``per_call`` says how many logical operations one ``fn()`` performs, so a
    loop over a whole epoch can report per-sample cost without the loop
    overhead being attributed to the operation.
    """
    for _ in range(warmup):
        fn()

    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)

    return Timing(
        label=label,
        best=min(samples),
        median=statistics.median(samples),
        mean=statistics.fmean(samples),
        runs=repeat,
        per_call=per_call,
    )


# ======================================================================
# memory
# ======================================================================


def peak_rss_mb() -> float:
    """Peak resident set size for this process, in MiB.

    Counts every allocation the process has made, including the ones PyTorch
    makes in C++ that ``tracemalloc`` is blind to. ``ru_maxrss`` is bytes on
    macOS and kilobytes on Linux -- a genuinely annoying platform difference
    that silently produces 1024x errors if you ignore it.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return raw / divisor


# ======================================================================
# results
# ======================================================================


@dataclass
class BenchmarkReport:
    """Everything one engine measured, ready to serialise and merge."""

    engine: str
    architecture: list[int] = field(default_factory=lambda: list(ARCHITECTURE))
    parameters: int = 0
    batch_size: int = BATCH_SIZE
    seed: int = SEED
    timings: dict[str, dict[str, Any]] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)
    peak_rss_mb: float = 0.0

    def add(self, timing: Timing) -> Timing:
        self.timings[timing.label] = timing.as_dict()
        return timing

    def finish(self) -> "BenchmarkReport":
        self.peak_rss_mb = peak_rss_mb()
        self.environment = describe_environment()
        return self

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path


def describe_environment() -> dict[str, str]:
    """Record the machine, because a benchmark without one is unfalsifiable."""
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "machine": platform.machine(),
    }
    try:
        import numpy as np

        env["numpy"] = np.__version__
    except ImportError:  # pragma: no cover
        pass
    try:  # pragma: no cover - only present in the torch benchmark's process
        import torch

        env["torch"] = torch.__version__
        env["torch_threads"] = str(torch.get_num_threads())
    except ImportError:
        pass
    return env


# ======================================================================
# presentation
# ======================================================================


def rule(title: str, width: int = 78) -> None:
    print(f"\n\033[1m{title}\033[0m\n" + "─" * width)


def human_time(seconds: float) -> str:
    """Format a duration at a sensible scale.

    Benchmarks here span nanoseconds (one scalar multiply) to hours (the scalar
    engine on full MNIST), so a fixed unit would be unreadable at one end.
    """
    if seconds < 1e-6:
        return f"{seconds * 1e9:,.0f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:,.1f} µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:,.1f} ms"
    if seconds < 90:
        return f"{seconds:,.2f} s"
    if seconds < 3600:
        return f"{seconds / 60:,.1f} min"
    return f"{seconds / 3600:,.1f} h"


def table(rows: list[list[str]], headers: list[str], indent: str = "    ") -> str:
    """Render a fixed-width table. Right-aligns everything but the first column."""
    widths = [
        max(len(str(r[i])) for r in [headers, *rows]) for i in range(len(headers))
    ]

    def line(cells: list[str]) -> str:
        out = [str(cells[0]).ljust(widths[0])]
        out += [str(c).rjust(w) for c, w in zip(cells[1:], widths[1:])]
        return indent + "  ".join(out)

    sep = indent + "  ".join("─" * w for w in widths)
    return "\n".join([line(headers), sep, *(line(r) for r in rows)])
