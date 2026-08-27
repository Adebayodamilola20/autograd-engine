"""Phase 17 -- the same network in PyTorch, as something to be measured against.

Run:
    python benchmarks/benchmark_pytorch.py
    python benchmarks/benchmark_pytorch.py --n-train 10000 --epochs 1

Writes ``artifacts/bench_torch.json``.

PyTorch appears in this repository **only here**. Nothing in ``nabla/`` imports
it, and none of these results feed back into the engine. It is the ruler, not
the thing being built.

Keeping the comparison honest
-----------------------------
A benchmark is only meaningful if the two sides are doing the same work, so
this script deliberately mirrors ``benchmark_autograd.py``:

* identical architecture (784→128→64→10, ReLU) and identical parameter count
* identical batch size, seed, optimiser (Adam, lr=1e-3) and loss
* the same measurement protocol from ``common.py`` -- warm up, report the
  minimum
* **float64**, matching our engine. This one costs PyTorch something real:
  it is tuned for float32 and would be meaningfully faster there. Letting it
  use float32 while we use float64 would be measuring the dtype, not the
  engine, so we hold it fixed and note it.
* single-threaded by default (``--threads 1``), because our engine is
  single-threaded. Pass ``--threads 0`` to let PyTorch use every core, which
  is the honest number for "what would I actually use in practice" and is
  reported separately in the writeup.

The remaining differences -- fused kernels, a C++ autograd graph, no Python
object per operation -- are exactly what we want to measure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from benchmarks.common import (  # noqa: E402
    ARCHITECTURE,
    ARTIFACTS,
    BATCH_SIZE,
    SEED,
    BenchmarkReport,
    human_time,
    measure,
    rule,
    table,
)
from nabla.data import DataLoader  # noqa: E402
from nabla.data.mnist import load_mnist  # noqa: E402

DTYPE = torch.float64


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark PyTorch as a baseline.")
    p.add_argument("--n-train", type=int, default=10000)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument(
        "--threads",
        type=int,
        default=1,
        help="torch CPU threads; 0 means 'use all', for the practical number",
    )
    p.add_argument("--out", default=str(ARTIFACTS / "bench_torch.json"))
    return p.parse_args()


def build_model() -> nn.Sequential:
    """784→128→64→10 with ReLU, matching ``TensorMLP`` exactly.

    ``nn.Linear`` stores its weight transposed relative to ours -- ``(out, in)``
    where we use ``(in, out)`` -- because it computes ``xWᵀ + b``. Same
    mathematics, same parameter count; a memory-layout choice, and a good
    reminder that "the weight matrix" has no canonical orientation.
    """
    sizes = ARCHITECTURE
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1], dtype=DTYPE))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def main() -> int:
    args = parse_args()
    torch.manual_seed(SEED)
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    report = BenchmarkReport(engine="pytorch", batch_size=BATCH_SIZE, seed=SEED)
    report.notes["dtype"] = str(DTYPE)
    report.notes["threads"] = torch.get_num_threads()

    rng = np.random.default_rng(SEED)
    model = build_model()
    report.parameters = sum(p.numel() for p in model.parameters())

    x_batch = torch.tensor(rng.random((BATCH_SIZE, 784)), dtype=DTYPE)
    y_batch = torch.zeros(BATCH_SIZE, dtype=torch.long)
    loss_fn = nn.CrossEntropyLoss()

    # ══════════════════════════════════════════════════════════════════
    rule("1. The floor: one tensor operation")

    a = torch.tensor(2.0, dtype=DTYPE, requires_grad=True)
    b = torch.tensor(3.0, dtype=DTYPE, requires_grad=True)

    t_mul = report.add(
        measure(lambda: a * b, label="scalar_multiply", repeat=args.repeat * 200)
    )

    def op_and_backward() -> None:
        x = torch.tensor(2.0, dtype=DTYPE, requires_grad=True)
        y = torch.tensor(3.0, dtype=DTYPE, requires_grad=True)
        (x * y).backward()

    t_mulback = report.add(
        measure(
            op_and_backward, label="scalar_multiply_backward", repeat=args.repeat * 200
        )
    )

    print(f"    build one node          {human_time(t_mul.best):>12}")
    print(f"    build + backward        {human_time(t_mulback.best):>12}")
    print(
        "\n    Note this is *not* PyTorch's strength. A one-element tensor pays\n"
        "    full dispatch cost -- type promotion, device selection, autograd\n"
        "    bookkeeping -- to multiply two numbers. Our scalar Value is\n"
        "    competitive here, and only here. Everything below is the rout."
    )

    # ══════════════════════════════════════════════════════════════════
    rule("3-5. Forward, backward, and a full training step")

    def forward() -> None:
        with torch.no_grad():
            model(x_batch)

    def forward_backward() -> None:
        model.zero_grad(set_to_none=False)
        loss_fn(model(x_batch), y_batch).backward()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def training_step() -> None:
        optimizer.zero_grad()
        loss_fn(model(x_batch), y_batch).backward()
        optimizer.step()

    t_fwd = report.add(
        measure(forward, label="tensor_forward_batch", repeat=args.repeat * 4,
                per_call=BATCH_SIZE)
    )
    t_fb = report.add(
        measure(forward_backward, label="tensor_forward_backward_batch",
                repeat=args.repeat * 4, per_call=BATCH_SIZE)
    )
    t_step = report.add(
        measure(training_step, label="tensor_training_step", repeat=args.repeat * 4,
                per_call=BATCH_SIZE)
    )

    print(
        table(
            [
                ["forward", human_time(t_fwd.best), human_time(t_fwd.best_per_op),
                 f"{t_fwd.ops_per_second:,.0f}"],
                ["fwd+backward", human_time(t_fb.best), human_time(t_fb.best_per_op),
                 f"{t_fb.ops_per_second:,.0f}"],
                ["full step", human_time(t_step.best), human_time(t_step.best_per_op),
                 f"{t_step.ops_per_second:,.0f}"],
            ],
            ["operation", f"per batch({BATCH_SIZE})", "per sample", "samples/s"],
        )
    )

    ratio = t_fb.best / t_fwd.best
    report.notes["backward_to_forward_ratio"] = ratio
    print(f"\n    forward+backward / forward = \033[1m{ratio:.2f}x\033[0m -- the same ~3x")
    print("    structural ratio our engine shows. The constant differs; the")
    print("    shape of the computation does not.")

    # ══════════════════════════════════════════════════════════════════
    rule("6. A real training epoch on MNIST")

    train, test = load_mnist(n_train=args.n_train, n_test=args.n_test, seed=SEED)
    loader = DataLoader(train, batch_size=BATCH_SIZE, seed=SEED)

    # Pre-convert once, so we time the engine and not the list→tensor marshalling.
    batches = [
        (torch.tensor(np.asarray(bx), dtype=DTYPE), torch.tensor(by, dtype=torch.long))
        for bx, by in loader
    ]
    x_test = torch.tensor(test.x, dtype=DTYPE)
    y_test = torch.tensor(test.y, dtype=torch.long)

    epoch_model = build_model()
    epoch_opt = torch.optim.Adam(epoch_model.parameters(), lr=1e-3)

    def one_epoch() -> None:
        for bx, by in batches:
            epoch_opt.zero_grad()
            loss_fn(epoch_model(bx), by).backward()
            epoch_opt.step()

    t_epoch = report.add(
        measure(one_epoch, label="mnist_epoch", repeat=args.epochs, warmup=0,
                per_call=len(train))
    )

    with torch.no_grad():
        accuracy = (epoch_model(x_test).argmax(dim=1) == y_test).double().mean().item()

    def inference_pass() -> None:
        with torch.no_grad():
            for start in range(0, len(test), 256):
                epoch_model(x_test[start : start + 256]).argmax(dim=1)

    t_infer = report.add(
        measure(inference_pass, label="mnist_inference", repeat=args.repeat,
                per_call=len(test))
    )

    report.metrics.update(
        {
            "train_samples": len(train),
            "test_samples": len(test),
            "epochs_timed": args.epochs,
            "accuracy_after_epochs": accuracy,
            "seconds_per_epoch": t_epoch.best,
            "train_samples_per_second": t_epoch.ops_per_second,
            "inference_samples_per_second": t_infer.ops_per_second,
        }
    )

    print(
        table(
            [
                [f"train ({args.epochs} epoch(s))", human_time(t_epoch.best),
                 f"{t_epoch.ops_per_second:,.0f}"],
                ["inference", human_time(t_infer.best),
                 f"{t_infer.ops_per_second:,.0f}"],
            ],
            ["phase", "wall time", "samples/s"],
        )
    )
    print(
        f"\n    accuracy after {args.epochs} epoch(s) on {len(train):,} samples: "
        f"\033[1m{accuracy * 100:.2f}%\033[0m"
    )
    print(
        "\n    Accuracy should land close to ours. If it does, the comparison is\n"
        "    fair: two engines doing the same job, one much faster. If it does\n"
        "    not, something is misconfigured and the timings mean nothing."
    )

    # ══════════════════════════════════════════════════════════════════
    rule("Summary")

    report.finish()
    path = report.save(args.out)
    print(f"    architecture   {'→'.join(map(str, ARCHITECTURE))}")
    print(f"    parameters     {report.parameters:,}")
    print(f"    dtype          {DTYPE}  (matched to ours; float32 would be faster)")
    print(f"    threads        {torch.get_num_threads()}")
    print(f"    peak RSS       {report.peak_rss_mb:,.1f} MiB")
    print(f"    written to     {path}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
