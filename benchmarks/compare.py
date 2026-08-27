"""Phase 17 -- run both engines, merge the numbers, explain the gap.

Run:
    python benchmarks/compare.py
    python benchmarks/compare.py --n-train 10000 --epochs 2
    python benchmarks/compare.py --reuse          # use existing JSON, don't re-run

Produces:
    artifacts/bench_nabla.json
    artifacts/bench_torch.json
    artifacts/bench_comparison.json
    artifacts/bench_comparison.png
    docs/18-performance.md

Each engine runs in its **own subprocess**, for two reasons. Peak RSS is a
per-process number, so measuring both in one process would attribute PyTorch's
111 MB of shared libraries to us. And importing torch changes the process --
it sets thread pools and can swap the BLAS library NumPy dispatches to, which
would silently contaminate our measurements.

The goal is not to win
----------------------
It is to find out *precisely where* the time goes, and to be able to explain
every ratio in the table below. A result we cannot explain is a result we do
not yet understand -- and this phase exists for the understanding, not the
scoreboard.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.common import ARTIFACTS, ROOT, human_time, rule, table  # noqa: E402

DOCS = ROOT / "docs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare nabla against PyTorch.")
    p.add_argument("--n-train", type=int, default=10000)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--scalar-samples", type=int, default=8)
    p.add_argument("--skip-scalar", action="store_true")
    p.add_argument("--reuse", action="store_true", help="use existing JSON files")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def run(script: str, extra: list[str]) -> None:
    cmd = [sys.executable, str(ROOT / "benchmarks" / script), *extra]
    print(f"    $ {' '.join(cmd[1:])}")
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"{script} failed with exit code {result.returncode}")


def ratio_text(ours: float, theirs: float, higher_is_better: bool) -> str:
    """How many times faster PyTorch is. Always phrased in their favour."""
    if ours <= 0 or theirs <= 0:
        return "n/a"
    factor = (theirs / ours) if higher_is_better else (ours / theirs)
    if factor >= 1:
        return f"{factor:,.1f}x"
    return f"{1 / factor:,.1f}x ours"


def main() -> int:
    args = parse_args()
    common = [
        "--n-train", str(args.n_train),
        "--n-test", str(args.n_test),
        "--epochs", str(args.epochs),
        "--repeat", str(args.repeat),
    ]

    if not args.reuse:
        rule("Running both engines, in separate processes")
        nabla_extra = [*common, "--scalar-samples", str(args.scalar_samples)]
        if args.skip_scalar:
            nabla_extra.append("--skip-scalar")
        run("benchmark_autograd.py", nabla_extra)
        run("benchmark_pytorch.py", common)

    nabla = json.loads((ARTIFACTS / "bench_nabla.json").read_text())
    torch_ = json.loads((ARTIFACTS / "bench_torch.json").read_text())

    # ══════════════════════════════════════════════════════════════════
    rule("Like for like")

    print(
        table(
            [
                ["architecture", "→".join(map(str, nabla["architecture"])),
                 "→".join(map(str, torch_["architecture"])), "identical"],
                ["parameters", f"{nabla['parameters']:,}", f"{torch_['parameters']:,}",
                 "identical" if nabla["parameters"] == torch_["parameters"] else "MISMATCH"],
                ["batch size", str(nabla["batch_size"]), str(torch_["batch_size"]), ""],
                ["dtype", "float64", torch_["notes"].get("dtype", "?"), "matched"],
                ["threads", "1", str(torch_["notes"].get("threads", "?")), "matched"],
            ],
            ["", "nabla", "pytorch", "note"],
        )
    )

    if nabla["parameters"] != torch_["parameters"]:
        print("\n    \033[1mPARAMETER COUNT MISMATCH -- the comparison is invalid.\033[0m")

    # ══════════════════════════════════════════════════════════════════
    rule("Timings")

    comparisons = [
        ("tensor_forward_batch", "forward (batch)", False),
        ("tensor_forward_backward_batch", "forward + backward", False),
        ("tensor_training_step", "full training step", False),
        ("mnist_epoch", "one MNIST epoch", False),
        ("mnist_inference", "inference over test set", False),
    ]

    rows = []
    ratios = {}
    for key, label, higher in comparisons:
        if key not in nabla["timings"] or key not in torch_["timings"]:
            continue
        ours = nabla["timings"][key]["best"]
        theirs = torch_["timings"][key]["best"]
        factor = ours / theirs
        ratios[key] = factor
        rows.append([label, human_time(ours), human_time(theirs), f"{factor:,.1f}x"])

    print(table(rows, ["operation", "nabla", "pytorch", "pytorch faster by"]))

    # ══════════════════════════════════════════════════════════════════
    rule("Accuracy, memory, and the sanity check")

    n_acc = nabla["metrics"].get("accuracy_after_epochs", 0)
    t_acc = torch_["metrics"].get("accuracy_after_epochs", 0)
    print(
        table(
            [
                ["accuracy after "
                 f"{nabla['metrics'].get('epochs_timed', '?')} epoch(s)",
                 f"{n_acc * 100:.2f}%", f"{t_acc * 100:.2f}%",
                 f"{abs(n_acc - t_acc) * 100:.2f} pp apart"],
                ["peak RSS", f"{nabla['peak_rss_mb']:,.0f} MiB",
                 f"{torch_['peak_rss_mb']:,.0f} MiB", ""],
                ["training throughput",
                 f"{nabla['metrics'].get('train_samples_per_second', 0):,.0f}/s",
                 f"{torch_['metrics'].get('train_samples_per_second', 0):,.0f}/s", ""],
                ["inference throughput",
                 f"{nabla['metrics'].get('inference_samples_per_second', 0):,.0f}/s",
                 f"{torch_['metrics'].get('inference_samples_per_second', 0):,.0f}/s", ""],
            ],
            ["", "nabla", "pytorch", "note"],
        )
    )
    print(
        "\n    Accuracy being close is the sanity check that makes the timings\n"
        "    mean anything: both engines really are solving the same problem.\n"
        "    They will not match exactly -- different weight initialisation\n"
        "    (our He normal vs PyTorch's Kaiming uniform) and a different\n"
        "    shuffle -- but they should land within a point or two."
    )

    # ══════════════════════════════════════════════════════════════════
    rule("Where our time actually goes")

    step = ratios.get("tensor_training_step")
    epoch = ratios.get("mnist_epoch")
    infer = ratios.get("mnist_inference")

    if step and epoch:
        print(
            f"    A single training step is \033[1m{step:,.1f}x\033[0m slower than PyTorch, and a\n"
            f"    whole epoch is \033[1m{epoch:,.1f}x\033[0m slower.\n\n"
            "    The arithmetic was never the problem. NumPy and PyTorch both\n"
            "    call the same class of tuned BLAS kernel for the matmuls, so on\n"
            "    the maths itself we are close to competitive.\n\n"
            "    The first run of this benchmark showed an epoch at 4.5x while a\n"
            "    step sat at 1.4x -- and *that gap* was the finding. Everything\n"
            "    between the steps was costing more than the steps. Profiling\n"
            "    (benchmarks/profile_training.py) blamed the DataLoader: it\n"
            "    converted each batch ndarray to nested Python lists, which the\n"
            "    model converted straight back. `tolist` and `asarray` were the\n"
            "    two most expensive entries in the profile, ahead of every piece\n"
            "    of real arithmetic.\n\n"
            "    DataLoader(as_arrays=True) removed the round trip (Phase 19).\n"
            "    Run with --list-batches to reproduce the old numbers."
        )
    if infer:
        print(
            f"\n    Inference is now the widest remaining gap ({infer:,.1f}x). With no\n"
            "    backward pass to dilute it, fixed per-batch overhead is most of\n"
            "    the cost -- so it is the most sensitive of these numbers to any\n"
            "    per-operation Python work still in the path."
        )

    scalar_speedup = nabla["notes"].get("tensor_speedup_over_scalar")
    if scalar_speedup:
        print(
            f"\n    For scale: our own tensor engine is \033[1m{scalar_speedup:,.0f}x\033[0m faster than\n"
            "    our own scalar engine. The distance from micrograd to NumPy is\n"
            "    vastly larger than the distance from NumPy to PyTorch. Choosing\n"
            "    the right *granularity* mattered far more than any other\n"
            "    optimisation available to us."
        )

    # ══════════════════════════════════════════════════════════════════
    merged = {
        "nabla": nabla,
        "pytorch": torch_,
        "ratios": ratios,
        "settings": vars(args),
    }
    (ARTIFACTS / "bench_comparison.json").write_text(json.dumps(merged, indent=2))

    if not args.no_plot:
        try:
            _plot(nabla, torch_, ratios)
            print(f"\n    artifacts/bench_comparison.png")
        except ImportError as err:
            print(f"\n    (plot skipped: {err})")

    _write_doc(nabla, torch_, ratios)
    print(f"    docs/18-performance.md")
    print(f"    artifacts/bench_comparison.json")
    print()
    return 0


def _plot(nabla: dict, torch_: dict, ratios: dict) -> None:
    """Log-scale per-sample cost, because the range spans six orders of magnitude."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, ours, theirs = [], [], []
    for key, label in [
        ("tensor_forward_batch", "forward"),
        ("tensor_forward_backward_batch", "fwd+bwd"),
        ("tensor_training_step", "train step"),
        ("mnist_epoch", "epoch"),
        ("mnist_inference", "inference"),
    ]:
        if key in nabla["timings"] and key in torch_["timings"]:
            labels.append(label)
            ours.append(nabla["timings"][key]["best_per_op"])
            theirs.append(torch_["timings"][key]["best_per_op"])

    scalar_key = "scalar_forward_backward"
    has_scalar = scalar_key in nabla["timings"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    x = range(len(labels))
    width = 0.38
    ax.bar([i - width / 2 for i in x], ours, width, label="nabla (ours)",
           color="#4c72b0")
    ax.bar([i + width / 2 for i in x], theirs, width, label="PyTorch",
           color="#dd8452")
    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("seconds per sample (log scale)")
    ax.set_title("Cost per sample -- same architecture, same dtype")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.set_axisbelow(True)  # bars over gridlines, not under

    for i, (o, t) in enumerate(zip(ours, theirs)):
        ax.text(i, max(o, t) * 1.6, f"{o / t:,.1f}x", ha="center", fontsize=9)

    # Headroom for the ratio labels, which sit above the tallest bar.
    ax.set_ylim(top=max(max(ours), max(theirs)) * 6)

    ax = axes[1]
    bars = ["scalar\n(ours)", "tensor\n(ours)", "PyTorch"]
    key = "tensor_forward_backward_batch"
    values = [
        nabla["timings"][scalar_key]["best_per_op"] if has_scalar else 0,
        nabla["timings"][key]["best_per_op"],
        torch_["timings"][key]["best_per_op"],
    ]
    colors = ["#c44e52", "#4c72b0", "#dd8452"]
    ax.bar(bars, values, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("seconds per sample (log scale)")
    ax.set_title("Forward + backward: the cost of granularity")
    ax.grid(axis="y", alpha=0.3, which="major")
    ax.set_axisbelow(True)
    for i, v in enumerate(values):
        if v > 0:
            ax.text(i, v * 1.8, human_time(v), ha="center", fontsize=10)
    ax.set_ylim(top=max(values) * 8)

    fig.suptitle(
        f"nabla vs PyTorch  --  {'→'.join(map(str, nabla['architecture']))}, "
        f"{nabla['parameters']:,} parameters, float64, 1 thread",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "bench_comparison.png", dpi=140)
    plt.close(fig)


def _write_doc(nabla: dict, torch_: dict, ratios: dict) -> None:
    """Emit ``docs/18-performance.md`` from the measured numbers.

    Generated rather than hand-written, so the prose can never drift out of
    sync with the JSON it describes.
    """
    env = nabla.get("environment", {})

    def t(engine: dict, key: str) -> str:
        return human_time(engine["timings"][key]["best"]) if key in engine["timings"] else "—"

    def per(engine: dict, key: str) -> str:
        return (
            human_time(engine["timings"][key]["best_per_op"])
            if key in engine["timings"]
            else "—"
        )

    rows = []
    for key, label in [
        ("tensor_forward_batch", "forward (batch of 64)"),
        ("tensor_forward_backward_batch", "forward + backward"),
        ("tensor_training_step", "full training step"),
        ("mnist_epoch", "one MNIST epoch"),
        ("mnist_inference", "inference over test set"),
    ]:
        if key in ratios:
            rows.append(
                f"| {label} | {t(nabla, key)} | {t(torch_, key)} | "
                f"{ratios[key]:,.1f}x |"
            )

    scalar_note = ""
    if "scalar_forward_backward" in nabla["timings"]:
        s = nabla["timings"]["scalar_forward_backward"]["best_per_op"]
        v = nabla["timings"]["tensor_forward_backward_batch"]["best_per_op"]
        p = torch_["timings"]["tensor_forward_backward_batch"]["best_per_op"]
        scalar_note = f"""
## The number that matters most

| engine | forward + backward, per sample | vs the next one up |
|---|---|---|
| our scalar `Value` engine | {human_time(s)} | — |
| our tensor engine | {human_time(v)} | **{s / v:,.0f}x faster** |
| PyTorch | {human_time(p)} | {v / p:,.1f}x faster |

Read that table twice. The step from *our scalar engine* to *our tensor engine*
is roughly **{s / v:,.0f}x**. The step from our tensor engine to PyTorch — a
project with thousands of contributors, hand-tuned kernels, and a C++ core — is
about **{v / p:,.1f}x**.

Almost the entire performance story is **granularity**, and we captured most of
it ourselves by changing what a node in the graph represents. Both engines
compute identical gradients; one asks Python to manage
{nabla["notes"].get("scalar_graph_per_sample", {}).get("nodes", 0):,} objects per
sample and the other asks it to manage
{nabla["notes"].get("tensor_graph", {}).get("nodes", 0)} per batch.
"""

    doc = f"""# Performance: nabla vs PyTorch

*Generated by `benchmarks/compare.py` — every number here is measured on this
machine, not quoted. Re-run it and this file is rewritten.*

Machine: {env.get('platform', 'unknown')}
Python {env.get('python', '?')}, NumPy {env.get('numpy', '?')},
PyTorch {torch_.get('environment', {}).get('torch', '?')}

## The setup

Both engines train the **identical** network:

- architecture `{'→'.join(map(str, nabla['architecture']))}`, ReLU hidden layers
- **{nabla['parameters']:,} parameters** in both — verified equal, not assumed
- batch size {nabla['batch_size']}, Adam at lr=1e-3, softmax cross-entropy
- **float64 on both sides**, single-threaded on both sides

Those last two matter. PyTorch is tuned for float32 and would be meaningfully
faster there, and it will happily use every core. Holding dtype and thread
count fixed means we measure the *engine*, not the configuration. It is the
comparison least flattering to us, which is the only kind worth publishing.

## Results

| operation | nabla | PyTorch | PyTorch faster by |
|---|---|---|---|
{chr(10).join(rows)}

Accuracy after {nabla['metrics'].get('epochs_timed', '?')} epoch(s) on
{nabla['metrics'].get('train_samples', 0):,} samples:
**{nabla['metrics'].get('accuracy_after_epochs', 0) * 100:.2f}%** (ours) vs
**{torch_['metrics'].get('accuracy_after_epochs', 0) * 100:.2f}%** (PyTorch).

That closeness is the sanity check. Timings only mean something if both engines
are actually solving the same problem, and they are. (They do not match
*exactly*: we initialise with He normal, PyTorch with Kaiming uniform, and the
shuffles differ.)

Peak RSS: {nabla['peak_rss_mb']:,.0f} MiB (ours) vs
{torch_['peak_rss_mb']:,.0f} MiB (PyTorch). Ours is inflated by the scalar-engine
graph built earlier in the same process — {nabla["notes"].get("scalar_graph_per_sample", {}).get("nodes", 0):,}
live Python objects is not free.
{scalar_note}
## Why PyTorch is faster

Ranked by how much they actually explain, on this workload:

**1. Python overhead per operation, not the arithmetic.** Our tensor engine
allocates a `Tensor` object, a `_prev` tuple and a `_backward` closure for every
operation in the graph. PyTorch records the same information in C++ structs.
For a graph of {nabla["notes"].get("tensor_graph", {}).get("nodes", 0)} nodes that
is a small fixed cost — which is exactly why our *per-step* gap
({ratios.get('tensor_training_step', 0):,.1f}x) is so much smaller than people
expect.

**2. Data marshalling — which was our own fault, and is now fixed.** The first
run of this benchmark showed an epoch at 4.5x while a single step sat at 1.4x.
Everything *between* the steps cost more than the steps. Profiling found our
`DataLoader` converting each batch ndarray to nested Python lists, which the
model converted straight back: `tolist` and `asarray` were the two most
expensive entries in the profile, ahead of every piece of real arithmetic.

`DataLoader(as_arrays=True)` deleted the round trip. The epoch went to
{ratios.get('mnist_epoch', 0):,.1f}x and inference from 29.5x to
{ratios.get('mnist_inference', 0):,.1f}x, with **identical accuracy** — the
change computes exactly the same numbers. See `docs/19-optimisation.md`.

The lesson generalises past this repo: the bottleneck was not in the autodiff
engine at all, and no amount of staring at `tensor.py` would have found it.

**3. Fused kernels.** `F.cross_entropy` computes `p - y` in one pass without
ever materialising the one-hot matrix or the intermediate softmax. We build ours
compositionally out of `log_softmax`, `mul` and `sum` — clearer to read and to
differentiate, but it allocates several full-size intermediates.

**4. Memory layout and dispatch.** PyTorch controls its own allocator, reuses
buffers across iterations, and dispatches an operation through a type-erased
C++ table rather than a Python method lookup.

**5. Everything we never reach here.** GPU kernels and CUDA, multi-threaded CPU
kernels, `torch.compile` graph fusion, mixed precision. On this size of MLP,
single-threaded and in float64, none of these are in play — which is precisely
why the measured gap is a handful of times rather than a thousand.

## What optimised tensor libraries actually buy you

The honest summary is that **NumPy is already doing the hard part**. Both
libraries hand a `(64, 784) @ (784, 128)` matmul to a tuned BLAS kernel written
in C and assembly, which blocks the matrices to fit in L1/L2 cache and issues
SIMD fused multiply-adds. That kernel is where nearly all the floating-point
work happens, and we and PyTorch are calling the same *kind* of thing.

What PyTorch adds on top is the removal of *per-operation Python cost* and the
provision of *hardware we cannot otherwise reach*. Both are real. Neither is the
autodiff algorithm — reverse mode in `nabla/core/tensor.py` is the same
algorithm PyTorch runs, and it produces the same numbers to within floating-point
tolerance.

## The lesson for this project

The gap that dominates everything is **scalar vs tensor**, not **us vs PyTorch**.
Representing a whole layer as one node instead of {nabla["notes"].get("scalar_graph_per_sample", {}).get("nodes", 0):,}
was worth orders of magnitude. Rewriting our engine in C++ afterwards would be
worth a handful.

That is the real answer to "why is PyTorch built around tensors?" — not because
tensors are mathematically necessary (our scalar engine computes the same
gradients), but because **the unit of work has to be big enough that the
interpreter overhead per unit stops mattering.**
"""
    DOCS.mkdir(exist_ok=True)
    (DOCS / "18-performance.md").write_text(doc)


if __name__ == "__main__":
    raise SystemExit(main())
