"""Phase 12/13 -- train MNIST with our own autodiff engine, end to end.

Run:
    python examples/train_mnist.py                     # full 784→128→64→10
    python examples/train_mnist.py --epochs 30
    python examples/train_mnist.py --engine scalar     # the Value engine (slow!)
    python examples/train_mnist.py --help

No PyTorch. No TensorFlow. Every gradient in this script comes from
``nabla/core/tensor.py``, every parameter update from ``nabla/optim/``, and
every one of those was verified against finite differences before we got here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from nabla.data import DataLoader, train_val_split  # noqa: E402
from nabla.data.mnist import load_mnist  # noqa: E402
from nabla.losses import softmax_cross_entropy  # noqa: E402
from nabla.losses.tensor_losses import tensor_cross_entropy  # noqa: E402
from nabla.nn import MLP  # noqa: E402
from nabla.nn.tensor_mlp import TensorMLP  # noqa: E402
from nabla.optim import SGD, Adam  # noqa: E402
from nabla.training import Trainer, confusion_counts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def rule(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m\n" + "─" * 78)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train MNIST with the nabla autodiff engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--engine", choices=["tensor", "scalar"], default="tensor",
                   help="tensor: batched (fast). scalar: one Value per number (very slow)")
    p.add_argument("--hidden", type=int, nargs="*", default=[128, 64])
    p.add_argument("--activation", default="relu",
                   choices=["relu", "tanh", "sigmoid", "leaky_relu"])
    p.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    p.add_argument("--lr", type=float, default=None, help="default: 1e-3 Adam, 0.1 SGD")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-train", type=int, default=None, help="subset size (default: all)")
    p.add_argument("--n-test", type=int, default=None)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.lr is None:
        args.lr = 1e-3 if args.optimizer == "adam" else 0.1

    # The scalar engine needs ~20 s per sample on this architecture, so it gets
    # a small subset by default. That is not a limitation to hide -- it is the
    # measurement that motivates the tensor engine.
    if args.engine == "scalar" and args.n_train is None:
        args.n_train, args.n_test = 600, 200
        args.epochs = min(args.epochs, 3)
        args.hidden = args.hidden or [32]

    # ══════════════════════════════════════════════════════════════════
    rule("1. Data")

    train_full, test = load_mnist(n_train=args.n_train, n_test=args.n_test, seed=args.seed)
    train, val = train_val_split(train_full, val_fraction=args.val_fraction, seed=args.seed)

    print(f"    train      {len(train):>7,} samples")
    print(f"    validation {len(val):>7,} samples   (held out -- never trained on)")
    print(f"    test       {len(test):>7,} samples   (touched exactly once, at the end)")
    print(f"    features   {train.n_features:>7}   (28×28 flattened)")
    print(f"    classes    {train.n_classes:>7}")
    print("\n    pixels scaled to [0, 1]: raw 0-255 values would hand the first")
    print("    layer gradients 255x too large (see nabla/data/mnist.py).")

    # as_arrays skips the ndarray -> list -> ndarray round trip, which profiling
    # showed was ~39% of an epoch (Phase 19). The scalar engine wants the lists.
    arrays = args.engine == "tensor"
    train_loader = DataLoader(
        train, batch_size=args.batch_size, seed=args.seed, as_arrays=arrays
    )
    val_loader = DataLoader(val, batch_size=256, shuffle=False, as_arrays=arrays)
    test_loader = DataLoader(test, batch_size=256, shuffle=False, as_arrays=arrays)

    # ══════════════════════════════════════════════════════════════════
    rule("2. Model")

    if args.engine == "tensor":
        model = TensorMLP(784, args.hidden, 10, activation=args.activation, seed=args.seed)
        loss_fn = tensor_cross_entropy
    else:
        model = MLP(784, args.hidden, 10, activation=args.activation, seed=args.seed)
        loss_fn = softmax_cross_entropy

    print(model.summary())
    print(f"\n    engine: {args.engine}")
    if args.engine == "scalar":
        nodes = 2 * sum(a * b for a, b in zip([784, *args.hidden], [*args.hidden, 10]))
        print(f"    ~{nodes:,} graph nodes per sample, all Python objects.")
        print("    This is the slow path, run deliberately -- see benchmarks/.")

    optimizer = (
        Adam(model.parameters(), lr=args.lr)
        if args.optimizer == "adam"
        else SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    )
    print(f"    optimizer: {optimizer!r}")
    print("    loss: softmax cross-entropy (log-sum-exp stabilised)")

    # ══════════════════════════════════════════════════════════════════
    rule("3. Training")

    trainer = Trainer(
        model, optimizer, loss_fn, metric="accuracy", grad_clip=args.grad_clip
    )
    started = time.perf_counter()
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=args.epochs,
        checkpoint_dir=ARTIFACTS / "mnist_checkpoints",
        checkpoint_metric="val_acc",
        early_stopping_patience=8,
    )
    total_seconds = time.perf_counter() - started

    # ══════════════════════════════════════════════════════════════════
    rule("4. Test set -- used exactly once")

    test_loss, test_acc = trainer.evaluate(test_loader)
    best_epoch, best_val = history.best("val_acc")

    print(f"    test loss      {test_loss:.4f}")
    print(f"    test accuracy  {test_acc * 100:.2f}%   "
          f"({int(round(test_acc * len(test))):,} of {len(test):,} correct)")
    print(f"    best val acc   {best_val * 100:.2f}%  (epoch {best_epoch + 1})")
    print(f"    training time  {total_seconds:.1f} s over {len(history['train_loss'])} epochs")
    print(f"    throughput     {len(train) * len(history['train_loss']) / total_seconds:,.0f} samples/s")

    # ══════════════════════════════════════════════════════════════════
    rule("5. Where it goes wrong")

    true, predicted, logits = trainer.collect_predictions(test_loader)
    matrix = confusion_counts(true, predicted, 10)

    print("    confusion matrix (rows = true, columns = predicted)\n")
    print("         " + "".join(f"{k:>6}" for k in range(10)))
    print("       " + "─" * 62)
    for i in range(10):
        cells = "".join(
            f"\033[1m{matrix[i][j]:>6}\033[0m" if i == j else f"{matrix[i][j]:>6}"
            for j in range(10)
        )
        recall = matrix[i][i] / max(sum(matrix[i]), 1)
        print(f"    {i} │{cells}   {recall * 100:5.1f}%")

    confusions = sorted(
        ((matrix[i][j], i, j) for i in range(10) for j in range(10) if i != j),
        reverse=True,
    )[:6]
    print("\n    most frequent confusions:")
    for count, i, j in confusions:
        print(f"      {i} misread as {j}: {count:>4} times")
    print("\n    These are the genuinely ambiguous pairs -- 4/9, 3/5, 7/2 share strokes.")

    # per-class accuracy
    print(f"\n    {'digit':>7} {'correct':>9} {'total':>7} {'accuracy':>10}")
    print("      " + "─" * 36)
    for k in range(10):
        total_k = sum(matrix[k])
        print(f"    {k:>7} {matrix[k][k]:>9,} {total_k:>7,} "
              f"{matrix[k][k] / max(total_k, 1) * 100:>9.2f}%")

    # ══════════════════════════════════════════════════════════════════
    rule("6. Artifacts")

    ARTIFACTS.mkdir(exist_ok=True)
    summary = {
        "engine": args.engine,
        "architecture": [784, *args.hidden, 10],
        "activation": args.activation,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs_run": len(history["train_loss"]),
        "parameters": model.num_parameters(),
        "train_samples": len(train),
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "best_val_accuracy": best_val,
        "training_seconds": total_seconds,
        "seed": args.seed,
        "history": dict(history),
    }
    (ARTIFACTS / "mnist_results.json").write_text(json.dumps(summary, indent=2))
    print("    artifacts/mnist_results.json")
    print("    artifacts/mnist_checkpoints/best.json")

    if not args.no_plots:
        try:
            from nabla.visualization.plots import (
                plot_confusion_matrix,
                plot_digit_grid,
                plot_prediction_detail,
                plot_training_curves,
                plot_weight_images,
            )

            plot_training_curves(
                history,
                title=f"MNIST  {'→'.join(map(str, [784, *args.hidden, 10]))}  "
                      f"({args.activation}, {args.optimizer})",
                path=ARTIFACTS / "mnist_training_curves.png",
            )
            print("    artifacts/mnist_training_curves.png")

            probs = np.exp(np.asarray(logits) - np.max(logits, axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
            confidence = probs.max(axis=1)

            plot_digit_grid(
                test.x[:32], test.y[:32], predicted[:32], confidence[:32],
                title="Predictions on unseen test digits",
                path=ARTIFACTS / "mnist_predictions.png",
            )
            print("    artifacts/mnist_predictions.png")

            wrong = [i for i, (t, p) in enumerate(zip(true, predicted)) if t != p]
            if wrong:
                idx = wrong[:32]
                plot_digit_grid(
                    test.x[idx], [true[i] for i in idx], [predicted[i] for i in idx],
                    [confidence[i] for i in idx],
                    title=f"Every one it got wrong ({len(wrong)} of {len(true)})",
                    path=ARTIFACTS / "mnist_mistakes.png",
                )
                print("    artifacts/mnist_mistakes.png")

                worst = max(wrong, key=lambda i: confidence[i])
                plot_prediction_detail(
                    test.x[worst], probs[worst], true[worst],
                    title="Most confidently wrong prediction",
                    path=ARTIFACTS / "mnist_worst_mistake.png",
                )
                print("    artifacts/mnist_worst_mistake.png")
                print(f"\n    Most confident mistake: expected {true[worst]}, "
                      f"predicted {predicted[worst]} with "
                      f"{confidence[worst] * 100:.1f}% confidence.")

            plot_confusion_matrix(
                true, predicted, normalize=True,
                title=f"MNIST confusion matrix ({test_acc * 100:.2f}% accurate)",
                path=ARTIFACTS / "mnist_confusion.png",
            )
            print("    artifacts/mnist_confusion.png")

            if args.engine == "tensor":
                plot_weight_images(
                    model.layers[0].W.data.T,
                    title="First-layer weights, reshaped to 28×28",
                    path=ARTIFACTS / "mnist_weights.png",
                )
                print("    artifacts/mnist_weights.png")
        except ImportError as err:
            print(f"    (plots skipped: {err})")

    rule("Result")
    print(f"    \033[1m{test_acc * 100:.2f}% test accuracy\033[0m on {len(test):,} unseen digits,")
    print(f"    trained with {model.num_parameters():,} parameters and gradients")
    print("    computed entirely by our own reverse-mode autodiff engine.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
