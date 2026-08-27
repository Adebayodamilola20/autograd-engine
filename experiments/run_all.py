"""Phase 22 -- run every experiment and collect the results.

Run:
    python experiments/run_all.py                    # full sweep, several minutes
    python experiments/run_all.py --quick            # fast, for checking it works
    python experiments/run_all.py --only learning_rate activations

Each experiment is a standalone script and can be run on its own; this is just
the driver that runs them in order and reports which succeeded. They execute in
subprocesses so that one failure cannot take down the rest, and so each starts
from a clean interpreter state.

Written up in `docs/20-experiments.md`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = [
    ("learning_rate", "exp_learning_rate.py", "Learning rate: too high, too low, just right"),
    ("activations", "exp_activations.py", "Activations, and why nonlinearity is mandatory"),
    ("architecture", "exp_architecture.py", "Depth and width"),
    ("optimizers", "exp_optimizers.py", "SGD vs momentum vs Adam"),
    ("batch_size", "exp_batch_size.py", "Batch size and the linear scaling rule"),
]


def main() -> int:
    p = argparse.ArgumentParser(description="Run the Phase 22 experiment suite.")
    p.add_argument("--quick", action="store_true", help="2 seeds, 4 epochs, small subset")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--n-train", type=int, default=8000)
    p.add_argument("--only", nargs="*", help="run only these experiment names")
    args = p.parse_args()

    if args.quick:
        args.seeds, args.epochs, args.n_train = 2, 4, 4000

    selected = [
        e for e in EXPERIMENTS if not args.only or e[0] in args.only
    ]
    if not selected:
        print(f"No experiment matched {args.only}. Known: "
              f"{', '.join(e[0] for e in EXPERIMENTS)}")
        return 1

    flags = [
        "--seeds", str(args.seeds),
        "--epochs", str(args.epochs),
        "--n-train", str(args.n_train),
    ]

    print(f"\n\033[1mPhase 22 -- experiments\033[0m")
    print("─" * 78)
    print(f"    {len(selected)} experiment(s), {args.seeds} seeds, {args.epochs} epochs, "
          f"{args.n_train:,} samples")
    print(f"    output: artifacts/experiments/\n")

    outcomes = []
    started_all = time.perf_counter()

    for name, script, description in selected:
        print(f"\n\033[1m▶ {name}\033[0m -- {description}")
        started = time.perf_counter()
        result = subprocess.run(
            [sys.executable, str(ROOT / "experiments" / script), *flags],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        elapsed = time.perf_counter() - started
        ok = result.returncode == 0
        outcomes.append((name, ok, elapsed))

        if ok:
            print(f"  ✓ done in {elapsed:.1f}s")
        else:
            print(f"  ✗ FAILED (exit {result.returncode}) after {elapsed:.1f}s")
            print(result.stdout[-1500:])
            print(result.stderr[-1500:], file=sys.stderr)

    total = time.perf_counter() - started_all

    print(f"\n\033[1mSummary\033[0m")
    print("─" * 78)
    for name, ok, elapsed in outcomes:
        mark = "✓" if ok else "✗"
        print(f"    {mark}  {name:<20} {elapsed:>7.1f}s")
    passed = sum(1 for _, ok, _ in outcomes if ok)
    print(f"\n    {passed}/{len(outcomes)} succeeded in {total:.1f}s total")
    print(f"    results in artifacts/experiments/, written up in docs/20-experiments.md\n")

    return 0 if passed == len(outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
