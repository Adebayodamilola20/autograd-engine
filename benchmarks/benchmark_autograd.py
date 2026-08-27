"""Phase 17 -- measure our own engine.

Run:
    python benchmarks/benchmark_autograd.py
    python benchmarks/benchmark_autograd.py --n-train 10000 --epochs 3
    python benchmarks/benchmark_autograd.py --skip-scalar     # the slow one

Writes ``artifacts/bench_nabla.json``. Run ``benchmarks/compare.py`` to put it
side by side with PyTorch.

What is measured, and why each one is here
------------------------------------------
1. **One scalar op.** The floor. Every cost above is some multiple of this.
2. **Graph size.** The number that explains everything else.
3. **Forward pass**, scalar engine vs tensor engine, same architecture.
4. **Forward + backward.** The ratio to (3) should be roughly 2-3x -- the
   backward pass of a matmul is two more matmuls.
5. **A full training step**, including the optimiser, so nothing hides.
6. **A training epoch** on real MNIST, which is the number that matters.
7. **Inference throughput** on the test set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

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
from nabla import Value, graph_size  # noqa: E402
from nabla.core.tensor import Tensor  # noqa: E402
from nabla.data import DataLoader  # noqa: E402
from nabla.data.mnist import load_mnist  # noqa: E402
from nabla.losses import softmax_cross_entropy  # noqa: E402
from nabla.losses.tensor_losses import tensor_cross_entropy  # noqa: E402
from nabla.nn import MLP  # noqa: E402
from nabla.nn.tensor_mlp import TensorMLP  # noqa: E402
from nabla.optim import Adam  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark the nabla engine.")
    p.add_argument("--n-train", type=int, default=10000)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument(
        "--scalar-samples",
        type=int,
        default=8,
        help="samples to time on the scalar engine (it is ~1000x slower)",
    )
    p.add_argument("--skip-scalar", action="store_true")
    p.add_argument("--out", default=str(ARTIFACTS / "bench_nabla.json"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    report = BenchmarkReport(engine="nabla", batch_size=BATCH_SIZE, seed=SEED)

    rng = np.random.default_rng(SEED)
    arch_label = "→".join(map(str, ARCHITECTURE))

    # ══════════════════════════════════════════════════════════════════
    rule("1. The floor: one scalar operation")

    a, b = Value(2.0), Value(3.0)
    t_mul = report.add(
        measure(lambda: a * b, label="scalar_multiply", repeat=args.repeat * 200)
    )

    def scalar_op_and_backward() -> None:
        x, y = Value(2.0), Value(3.0)
        (x * y).backward()

    t_mulback = report.add(
        measure(
            scalar_op_and_backward,
            label="scalar_multiply_backward",
            repeat=args.repeat * 200,
        )
    )

    print(f"    build one node          {human_time(t_mul.best):>12}")
    print(f"    build + backward        {human_time(t_mulback.best):>12}")
    print(
        "\n    A Python float multiply is ~30 ns. Every Value we build costs an\n"
        "    object allocation, a tuple, and a closure -- which is the entire\n"
        "    story of why a scalar engine is slow, measured rather than asserted."
    )

    # ══════════════════════════════════════════════════════════════════
    rule("2. Graph size: the number that explains the rest")

    tensor_model = TensorMLP(*_split(ARCHITECTURE), activation="relu", seed=SEED)
    report.parameters = tensor_model.num_parameters()

    x_batch = rng.random((BATCH_SIZE, 784))
    tensor_out = tensor_model(x_batch)
    tensor_loss = tensor_cross_entropy(tensor_out, [0] * BATCH_SIZE)
    tensor_graph = graph_size(tensor_loss)
    tensor_nodes = tensor_graph["nodes"]

    scalar_graph = None
    scalar_nodes = None
    if not args.skip_scalar:
        scalar_model = MLP(*_split(ARCHITECTURE), activation="relu", seed=SEED)
        one_out = scalar_model(x_batch[0].tolist())
        scalar_graph = graph_size(softmax_cross_entropy([one_out], [0]))
        scalar_nodes = scalar_graph["nodes"]

    rows = [
        [
            f"tensor (batch of {BATCH_SIZE})",
            f"{tensor_nodes:,}",
            f"{tensor_graph['edges']:,}",
            f"{tensor_graph['depth']:,}",
            f"{tensor_nodes / BATCH_SIZE:,.1f}",
        ]
    ]
    if scalar_graph:
        rows.append(
            [
                "scalar (1 sample)",
                f"{scalar_nodes:,}",
                f"{scalar_graph['edges']:,}",
                f"{scalar_graph['depth']:,}",
                f"{scalar_nodes:,.1f}",
            ]
        )
    print(table(rows, ["engine", "nodes", "edges", "depth", "nodes/sample"]))

    report.notes["tensor_graph"] = tensor_graph
    report.notes["scalar_graph_per_sample"] = scalar_graph

    if scalar_nodes:
        ratio = scalar_nodes * BATCH_SIZE / tensor_nodes
        report.notes["node_ratio"] = ratio
        print(
            f"\n    Same architecture, same arithmetic, same gradients.\n"
            f"    The scalar engine allocates \033[1m{ratio:,.0f}x\033[0m more Python objects\n"
            f"    to compute them. backward() then walks every one of those nodes,\n"
            f"    calling a Python closure at each.\n\n"
            f"    Note the \033[1mdepth\033[0m column too. Depth is the length of the longest\n"
            f"    chain the gradient must flow along, and it is *inherently serial* --\n"
            f"    no amount of hardware parallelism shortens it. Width can be\n"
            f"    vectorised away; depth cannot. That is the real structural\n"
            f"    difference between the two engines, not just the node count."
        )

    # ══════════════════════════════════════════════════════════════════
    rule("3-5. Forward, backward, and a full training step")

    def tensor_forward() -> None:
        tensor_model(x_batch)

    def tensor_forward_backward() -> None:
        out = tensor_model(x_batch)
        tensor_cross_entropy(out, [0] * BATCH_SIZE).backward()

    optimizer = Adam(tensor_model.parameters(), lr=1e-3)

    def tensor_step() -> None:
        optimizer.zero_grad()
        out = tensor_model(x_batch)
        tensor_cross_entropy(out, [0] * BATCH_SIZE).backward()
        optimizer.step()

    t_fwd = report.add(
        measure(
            tensor_forward,
            label="tensor_forward_batch",
            repeat=args.repeat * 4,
            per_call=BATCH_SIZE,
        )
    )
    t_fb = report.add(
        measure(
            tensor_forward_backward,
            label="tensor_forward_backward_batch",
            repeat=args.repeat * 4,
            per_call=BATCH_SIZE,
        )
    )
    t_step = report.add(
        measure(
            tensor_step,
            label="tensor_training_step",
            repeat=args.repeat * 4,
            per_call=BATCH_SIZE,
        )
    )

    rows = [
        [
            "tensor  forward",
            human_time(t_fwd.best),
            human_time(t_fwd.best_per_op),
            f"{t_fwd.ops_per_second:,.0f}",
        ],
        [
            "tensor  fwd+backward",
            human_time(t_fb.best),
            human_time(t_fb.best_per_op),
            f"{t_fb.ops_per_second:,.0f}",
        ],
        [
            "tensor  full step",
            human_time(t_step.best),
            human_time(t_step.best_per_op),
            f"{t_step.ops_per_second:,.0f}",
        ],
    ]

    if not args.skip_scalar:
        n = args.scalar_samples
        scalar_xs = [x_batch[i].tolist() for i in range(n)]

        def scalar_forward() -> None:
            for xs in scalar_xs:
                scalar_model(xs)

        def scalar_forward_backward() -> None:
            outs = [scalar_model(xs) for xs in scalar_xs]
            softmax_cross_entropy(outs, [0] * n).backward()

        t_sfwd = report.add(
            measure(
                scalar_forward,
                label="scalar_forward",
                repeat=max(2, args.repeat // 2),
                warmup=0,
                per_call=n,
            )
        )
        t_sfb = report.add(
            measure(
                scalar_forward_backward,
                label="scalar_forward_backward",
                repeat=max(2, args.repeat // 2),
                warmup=0,
                per_call=n,
            )
        )
        rows = [
            [
                "scalar  forward",
                human_time(t_sfwd.best_per_op * BATCH_SIZE),
                human_time(t_sfwd.best_per_op),
                f"{t_sfwd.ops_per_second:,.1f}",
            ],
            [
                "scalar  fwd+backward",
                human_time(t_sfb.best_per_op * BATCH_SIZE),
                human_time(t_sfb.best_per_op),
                f"{t_sfb.ops_per_second:,.1f}",
            ],
            *rows,
        ]

    print(table(rows, ["operation", f"per batch({BATCH_SIZE})", "per sample", "samples/s"]))

    backward_ratio = t_fb.best / t_fwd.best
    report.notes["backward_to_forward_ratio"] = backward_ratio
    print(
        f"\n    forward+backward / forward = \033[1m{backward_ratio:.2f}x\033[0m\n"
        "    Expected: the backward pass of Y = XW is two more matmuls\n"
        "    (X̄ = ȲWᵀ and W̄ = XᵀȲ), so a training step costs roughly 3x a\n"
        "    forward pass. This is why inference is cheap and training is not."
    )

    if not args.skip_scalar:
        speedup = t_sfb.best_per_op / (t_fb.best_per_op)
        report.notes["tensor_speedup_over_scalar"] = speedup
        print(
            f"\n    The tensor engine is \033[1m{speedup:,.0f}x\033[0m faster per sample than the\n"
            "    scalar engine, on identical mathematics. That gap is Phase 19."
        )

    # ══════════════════════════════════════════════════════════════════
    rule("6. A real training epoch on MNIST")

    train, test = load_mnist(n_train=args.n_train, n_test=args.n_test, seed=SEED)
    loader = DataLoader(train, batch_size=BATCH_SIZE, seed=SEED)
    test_loader = DataLoader(test, batch_size=256, shuffle=False)

    epoch_model = TensorMLP(*_split(ARCHITECTURE), activation="relu", seed=SEED)
    epoch_opt = Adam(epoch_model.parameters(), lr=1e-3)

    def one_epoch() -> None:
        for bx, by in loader:
            epoch_opt.zero_grad()
            out = epoch_model(bx)
            tensor_cross_entropy(out, by).backward()
            epoch_opt.step()

    t_epoch = report.add(
        measure(
            one_epoch,
            label="mnist_epoch",
            repeat=args.epochs,
            warmup=0,
            per_call=len(train),
        )
    )

    correct = 0
    for bx, by in test_loader:
        correct += int((epoch_model.predict(np.asarray(bx)) == np.asarray(by)).sum())
    accuracy = correct / len(test)

    def inference_pass() -> None:
        for bx, _ in test_loader:
            epoch_model.predict(np.asarray(bx))

    t_infer = report.add(
        measure(
            inference_pass, label="mnist_inference", repeat=args.repeat, per_call=len(test)
        )
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
                [
                    f"train ({args.epochs} epoch(s))",
                    human_time(t_epoch.best),
                    f"{t_epoch.ops_per_second:,.0f}",
                ],
                ["inference", human_time(t_infer.best), f"{t_infer.ops_per_second:,.0f}"],
            ],
            ["phase", "wall time", "samples/s"],
        )
    )
    print(
        f"\n    accuracy after {args.epochs} epoch(s) on {len(train):,} samples: "
        f"\033[1m{accuracy * 100:.2f}%\033[0m"
    )

    if not args.skip_scalar and scalar_nodes:
        projected = t_sfb.best_per_op * len(train) * args.epochs
        report.notes["scalar_projected_epoch_seconds"] = projected
        print(
            f"\n    Projected cost of the same work on the scalar engine: "
            f"\033[1m{human_time(projected)}\033[0m\n"
            "    (measured per-sample cost x sample count -- an extrapolation,\n"
            "    not a measurement, but the per-sample figure above is real.)"
        )

    # ══════════════════════════════════════════════════════════════════
    rule("Summary")

    report.finish()
    path = report.save(args.out)
    print(f"    architecture   {arch_label}")
    print(f"    parameters     {report.parameters:,}")
    print(f"    peak RSS       {report.peak_rss_mb:,.1f} MiB")
    print(f"    written to     {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")
    print()
    return 0


def _split(arch: list[int]) -> tuple[int, list[int], int]:
    """``[784, 128, 64, 10] -> (784, [128, 64], 10)``, the MLP constructor shape."""
    return arch[0], list(arch[1:-1]), arch[-1]


if __name__ == "__main__":
    raise SystemExit(main())
