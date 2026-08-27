"""Phase 19 -- find out where the time actually goes, before changing anything.

Run:
    python benchmarks/profile_training.py
    python benchmarks/profile_training.py --sort tottime --limit 30
    python benchmarks/profile_training.py --engine scalar --n-train 200

Phase 17 produced a suspicion: a single training step is only ~1.4x slower than
PyTorch, but a whole epoch is ~4.5x slower, so something *between* the steps is
expensive. A suspicion is not a finding. This script profiles a real epoch and
reads the answer off the call counts.

Why cProfile and not a timer
----------------------------
Hand-placed timers only measure what you already suspected. A profiler measures
everything, including the thing you would never have thought to time -- which is
usually where the time is. The cost is that cProfile adds per-call overhead, so
it *distorts* absolute timings and exaggerates functions called very often. Use
it to find the shape of the problem, then confirm the fix with the untainted
wall-clock benchmarks from Phase 17.

Two views are printed:
  * **tottime** -- time in the function itself, excluding callees. This is where
    the CPU actually is.
  * **cumtime** -- time including callees. This tells you which high-level phase
    owns the cost.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from benchmarks.common import (  # noqa: E402
    ARCHITECTURE,
    ARTIFACTS,
    BATCH_SIZE,
    SEED,
    human_time,
    rule,
    table,
)
from nabla.data import DataLoader  # noqa: E402
from nabla.data.mnist import load_mnist  # noqa: E402
from nabla.losses import softmax_cross_entropy  # noqa: E402
from nabla.losses.tensor_losses import tensor_cross_entropy  # noqa: E402
from nabla.nn import MLP  # noqa: E402
from nabla.nn.tensor_mlp import TensorMLP  # noqa: E402
from nabla.optim import Adam  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile a training epoch.")
    p.add_argument("--engine", choices=["tensor", "scalar"], default="tensor")
    p.add_argument("--n-train", type=int, default=10000)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--limit", type=int, default=18, help="rows per view")
    p.add_argument("--sort", default="both", choices=["tottime", "cumtime", "both"])
    p.add_argument(
        "--list-batches",
        action="store_true",
        help="force the pre-Phase-19 list round trip, to see what it cost",
    )
    p.add_argument("--out", default=str(ARTIFACTS / "profile_training.txt"))
    return p.parse_args()


def _split(arch: list[int]) -> tuple[int, list[int], int]:
    return arch[0], list(arch[1:-1]), arch[-1]


def top_rows(stats: pstats.Stats, sort: str, limit: int) -> list[list[str]]:
    """Extract the top ``limit`` functions as table rows."""
    buffer = io.StringIO()
    stats.stream = buffer  # type: ignore[attr-defined]
    stats.sort_stats(sort).print_stats(limit)
    text = buffer.getvalue()

    rows: list[list[str]] = []
    started = False
    for line in text.splitlines():
        if line.strip().startswith("ncalls"):
            started = True
            continue
        if not started or not line.strip():
            continue
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        ncalls, tottime, _percall, cumtime, _percall2, where = parts
        # Trim absolute paths down to something readable.
        if "/" in where:
            where = ".../" + "/".join(Path(where.split("(")[0]).parts[-2:])
            where = where.split(":")[0] + ":" + line.split(":")[-1].strip()
        rows.append([where[:52], ncalls, tottime, cumtime])
        if len(rows) >= limit:
            break
    return rows


def main() -> int:
    args = parse_args()

    train, _ = load_mnist(n_train=args.n_train, n_test=10, seed=SEED)
    # The scalar engine wants Python floats; the tensor engine wants arrays.
    # --list-batches forces the slow path, to profile what Phase 19 removed.
    as_arrays = args.engine == "tensor" and not args.list_batches
    loader = DataLoader(
        train, batch_size=BATCH_SIZE, seed=SEED, as_arrays=as_arrays
    )

    if args.engine == "tensor":
        model = TensorMLP(*_split(ARCHITECTURE), activation="relu", seed=SEED)
        loss_fn = tensor_cross_entropy
    else:
        model = MLP(784, [32], 10, activation="relu", seed=SEED)
        loss_fn = softmax_cross_entropy

    optimizer = Adam(model.parameters(), lr=1e-3)

    def epoch() -> None:
        for bx, by in loader:
            optimizer.zero_grad()
            if args.engine == "tensor":
                loss = loss_fn(model(bx), by)
            else:
                loss = loss_fn([model(x) for x in bx], by)
            loss.backward()
            optimizer.step()

    epoch()  # warm up: first-call costs are not what we are looking for

    rule(f"Profiling {args.epochs} epoch(s), {args.engine} engine, {len(train):,} samples")

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(args.epochs):
        epoch()
    profiler.disable()

    stats = pstats.Stats(profiler)
    total = stats.total_tt

    print(f"    total profiled time: {human_time(total)}")
    print(
        "    (inflated by profiler overhead -- compare shapes, not absolutes)\n"
    )

    views = ["tottime", "cumtime"] if args.sort == "both" else [args.sort]
    for sort in views:
        heading = {
            "tottime": "By tottime -- where the CPU actually is",
            "cumtime": "By cumtime -- which phase owns the cost",
        }[sort]
        rule(heading)
        rows = top_rows(stats, sort, args.limit)
        print(table(rows, ["function", "ncalls", "tottime", "cumtime"]))

    # ══════════════════════════════════════════════════════════════════
    rule("Reading the result")
    print(
        "    Look for two things:\n\n"
        "    1. A function with a huge *ncalls* relative to the work it does.\n"
        "       That is Python overhead, and the fix is to do the same work in\n"
        "       fewer, larger calls.\n"
        "    2. A function high in *tottime* that is not arithmetic. Time spent\n"
        "       anywhere other than a BLAS call is time not spent training.\n\n"
        "    If `tolist`, `asarray` or list comprehensions in the data path rank\n"
        "    highly, the bottleneck is marshalling, not mathematics -- exactly\n"
        "    what the Phase 17 step-vs-epoch gap predicted."
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    stats.stream = buffer  # type: ignore[attr-defined]
    stats.sort_stats("tottime").print_stats(60)
    out.write_text(buffer.getvalue())
    print(f"\n    full profile written to {out}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
