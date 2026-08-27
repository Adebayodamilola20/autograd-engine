"""Phase 6/13 -- matplotlib figures for training, data and diagnostics.

Every function takes plain Python/NumPy data and returns a Matplotlib figure,
optionally saving it. Nothing here knows about ``Value``, ``Trainer`` or
``MLP`` -- keeping the plotting layer ignorant of the engine means the figures
work just as well for the tensor engine, the PyTorch baseline, or anything
else we want to compare against.

The palette is deliberately small and consistent: one hue for training, one for
validation, one accent for highlights, and a diverging map only where a
quantity really is signed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:  # matplotlib is an optional extra; the engine itself never needs it
    import matplotlib

    matplotlib.use("Agg")  # headless-safe: write files, never open a window
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    HAVE_MPL = True
except ImportError:  # pragma: no cover
    HAVE_MPL = False
    Figure = Any  # type: ignore[misc,assignment]

__all__ = [
    "plot_training_curves",
    "plot_digit_grid",
    "plot_prediction_detail",
    "plot_confusion_matrix",
    "plot_activations",
    "plot_gradient_histogram",
    "plot_weight_images",
    "plot_decision_boundary",
    "save",
]

TRAIN = "#2f6fd0"
VAL = "#e07a3f"
ACCENT = "#4c9a6a"
BAD = "#c94f4f"
GRID = {"alpha": 0.25, "linewidth": 0.7}


def _require_mpl() -> None:
    if not HAVE_MPL:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for plotting: pip install matplotlib"
        )


def _style(ax) -> None:
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig: Figure, path: str | Path, *, dpi: int = 140) -> Path:
    """Write a figure and close it, creating parent directories as needed."""
    _require_mpl()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


# ====================================================================
# training
# ====================================================================


def plot_training_curves(
    history: Mapping[str, Sequence[float]],
    *,
    title: str = "Training",
    path: str | Path | None = None,
) -> Figure:
    """Loss and accuracy against epoch, train vs validation.

    ``history`` is the dict returned by ``Trainer.fit()``: keys such as
    ``train_loss``, ``val_loss``, ``train_acc``, ``val_acc``. Missing keys are
    skipped, so this works for regression runs with no accuracy too.

    Reading the result is most of what experiment analysis is:

    * both losses falling together -> still learning,
    * training loss falling while validation rises -> overfitting,
    * loss flat from step one -> learning rate too low, or gradients not flowing,
    * loss oscillating or NaN -> learning rate too high.
    """
    _require_mpl()
    has_acc = "train_acc" in history or "val_acc" in history
    fig, axes = plt.subplots(1, 2 if has_acc else 1, figsize=(11 if has_acc else 6, 4))
    axes = np.atleast_1d(axes)

    ax = axes[0]
    if "train_loss" in history:
        ax.plot(history["train_loss"], color=TRAIN, lw=2, label="train")
    if "val_loss" in history:
        ax.plot(history["val_loss"], color=VAL, lw=2, label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Loss")
    ax.legend(frameon=False)
    _style(ax)

    if has_acc:
        ax = axes[1]
        if "train_acc" in history:
            ax.plot(
                np.asarray(history["train_acc"]) * 100, color=TRAIN, lw=2, label="train"
            )
        if "val_acc" in history:
            ax.plot(
                np.asarray(history["val_acc"]) * 100, color=VAL, lw=2, label="validation"
            )
            best = float(np.max(history["val_acc"])) * 100
            ax.axhline(best, color=ACCENT, ls="--", lw=1, alpha=0.8)
            ax.annotate(
                f"best {best:.2f}%",
                xy=(0.02, best),
                xycoords=("axes fraction", "data"),
                va="bottom",
                fontsize=9,
                color=ACCENT,
            )
        ax.set_xlabel("epoch")
        ax.set_ylabel("accuracy (%)")
        ax.set_title("Accuracy")
        ax.legend(frameon=False, loc="lower right")
        _style(ax)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    if path:
        save(fig, path)
    return fig


# ====================================================================
# data and predictions
# ====================================================================


def plot_digit_grid(
    images: np.ndarray,
    labels: Sequence[int] | None = None,
    predictions: Sequence[int] | None = None,
    confidences: Sequence[float] | None = None,
    *,
    rows: int = 4,
    cols: int = 8,
    title: str = "MNIST samples",
    path: str | Path | None = None,
) -> Figure:
    """A grid of digits, optionally annotated with true and predicted labels.

    When ``predictions`` is supplied, correct cells are titled in green and
    incorrect ones in red with ``true -> predicted``. This is the figure that
    makes a model's failure modes obvious at a glance -- 4s read as 9s, 5s as
    3s, and so on.
    """
    _require_mpl()
    images = np.asarray(images)
    if images.ndim == 2 and images.shape[1] == 784:
        images = images.reshape(-1, 28, 28)

    n = min(rows * cols, len(images))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.35, rows * 1.55))
    axes = np.atleast_1d(axes).ravel()

    for i in range(len(axes)):
        ax = axes[i]
        ax.axis("off")
        if i >= n:
            continue
        ax.imshow(images[i], cmap="gray_r", interpolation="nearest", vmin=0, vmax=1)

        if predictions is not None and labels is not None:
            ok = int(predictions[i]) == int(labels[i])
            colour = ACCENT if ok else BAD
            text = f"{int(predictions[i])}" if ok else f"{int(labels[i])}→{int(predictions[i])}"
            if confidences is not None:
                text += f"\n{confidences[i] * 100:.0f}%"
            ax.set_title(text, fontsize=8.5, color=colour, pad=3)
        elif labels is not None:
            ax.set_title(str(int(labels[i])), fontsize=9, pad=3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    if path:
        save(fig, path)
    return fig


def plot_prediction_detail(
    image: np.ndarray,
    probabilities: Sequence[float],
    true_label: int | None = None,
    *,
    title: str = "",
    path: str | Path | None = None,
) -> Figure:
    """One digit beside its full probability distribution over the 10 classes.

    Shows what the model actually produced, not just its argmax -- the
    difference between a confident 99% answer and a 34%/31% coin-flip is the
    most useful thing to see when a prediction is wrong.
    """
    _require_mpl()
    image = np.asarray(image).reshape(28, 28)
    probs = np.asarray(probabilities, dtype=float)
    pred = int(np.argmax(probs))

    fig, (ax_img, ax_bar) = plt.subplots(
        1, 2, figsize=(8, 3.2), gridspec_kw={"width_ratios": [1, 2.1]}
    )

    ax_img.imshow(image, cmap="gray_r", interpolation="nearest", vmin=0, vmax=1)
    ax_img.axis("off")
    if true_label is not None:
        correct = pred == int(true_label)
        ax_img.set_title(
            f"true {int(true_label)}   predicted {pred}",
            fontsize=10,
            color=ACCENT if correct else BAD,
            fontweight="bold",
        )
    else:
        ax_img.set_title(f"predicted {pred}", fontsize=10, fontweight="bold")

    colours = [TRAIN] * 10
    colours[pred] = ACCENT if (true_label is None or pred == true_label) else BAD
    if true_label is not None and pred != true_label:
        colours[int(true_label)] = ACCENT

    ax_bar.bar(range(10), probs * 100, color=colours)
    ax_bar.set_xticks(range(10))
    ax_bar.set_xlabel("class")
    ax_bar.set_ylabel("probability (%)")
    ax_bar.set_ylim(0, 100)
    for k, p in enumerate(probs):
        if p > 0.03:
            ax_bar.text(k, p * 100 + 1.5, f"{p * 100:.0f}", ha="center", fontsize=8)
    _style(ax_bar)

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    if path:
        save(fig, path)
    return fig


def plot_confusion_matrix(
    true_labels: Sequence[int],
    predicted: Sequence[int],
    *,
    n_classes: int = 10,
    normalize: bool = False,
    title: str = "Confusion matrix",
    path: str | Path | None = None,
) -> Figure:
    """Which classes get mistaken for which.

    The diagonal is correct predictions; off-diagonal cells name the specific
    confusions. On MNIST the usual offenders are 4/9, 3/5, 7/2 and 8/3 -- pairs
    that genuinely share strokes.
    """
    _require_mpl()
    cm = np.zeros((n_classes, n_classes), dtype=float)
    for t, p in zip(true_labels, predicted):
        cm[int(t), int(p)] += 1

    counts = cm.copy()
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=12, fontweight="bold")

    hi = cm.max() if cm.max() > 0 else 1.0
    for i in range(n_classes):
        for j in range(n_classes):
            if counts[i, j] == 0:
                continue
            text = f"{cm[i, j]:.2f}" if normalize else f"{int(counts[i, j])}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if cm[i, j] > hi * 0.55 else "#333333",
            )
    fig.tight_layout()
    if path:
        save(fig, path)
    return fig


# ====================================================================
# diagnostics
# ====================================================================


def plot_activations(
    functions: Mapping[str, Any],
    *,
    span: float = 5.0,
    path: str | Path | None = None,
) -> Figure:
    """Activation functions and their derivatives, side by side.

    ``functions`` maps a name to a callable taking and returning a ``Value``.
    Both curves are produced *by the engine itself* -- the derivative panel is
    the output of ``backward()``, not an analytic formula typed in separately.
    So this figure doubles as a visual gradient check.

    The derivative panel is the important one. It shows directly why deep
    sigmoid stacks fail (peak slope 0.25), why tanh is better but still
    saturates, and why ReLU's flat 1.0 lets gradient survive depth.
    """
    _require_mpl()
    from ..core.value import Value

    xs = np.linspace(-span, span, 601)
    fig, (ax_f, ax_d) = plt.subplots(1, 2, figsize=(11, 4))
    cmap = plt.get_cmap("tab10")

    for k, (name, fn) in enumerate(functions.items()):
        ys, dys = [], []
        for x in xs:
            v = Value(float(x))
            out = fn(v)
            out.backward()
            ys.append(out.data)
            dys.append(v.grad)
        colour = cmap(k)
        ax_f.plot(xs, ys, lw=2, color=colour, label=name)
        ax_d.plot(xs, dys, lw=2, color=colour, label=name)

    ax_f.set_title("Activation  f(x)")
    ax_d.set_title("Derivative  f'(x)   (computed by backward())")
    for ax in (ax_f, ax_d):
        ax.axhline(0, color="#999999", lw=0.8)
        ax.axvline(0, color="#999999", lw=0.8)
        ax.set_xlabel("x")
        ax.legend(frameon=False)
        _style(ax)

    fig.tight_layout()
    if path:
        save(fig, path)
    return fig


def plot_gradient_histogram(
    grads_by_layer: Mapping[str, Sequence[float]],
    *,
    title: str = "Gradient magnitude by layer",
    path: str | Path | None = None,
) -> Figure:
    """Log-scale gradient distributions, one row per layer.

    The single most useful diagnostic for a network that will not train.
    Healthy layers overlap within an order of magnitude or two. Gradients that
    shrink by 10x per layer as you move away from the loss are *vanishing*;
    ones that grow are *exploding*. Both are visible instantly here and
    invisible in the loss curve.
    """
    _require_mpl()
    names = list(grads_by_layer)
    fig, axes = plt.subplots(
        len(names), 1, figsize=(7, 1.7 * len(names)), sharex=True, squeeze=False
    )

    for ax, name in zip(axes[:, 0], names):
        g = np.abs(np.asarray(grads_by_layer[name], dtype=float))
        g = g[g > 0]
        if len(g) == 0:
            ax.text(0.5, 0.5, "all gradients zero", ha="center", transform=ax.transAxes)
        else:
            ax.hist(np.log10(g), bins=40, color=TRAIN, alpha=0.85)
            ax.axvline(np.log10(np.median(g)), color=VAL, lw=1.6, ls="--")
        ax.set_ylabel(name, fontsize=9, rotation=0, ha="right", va="center")
        _style(ax)

    axes[-1, 0].set_xlabel("log₁₀ |gradient|")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    if path:
        save(fig, path)
    return fig


def plot_weight_images(
    weights: np.ndarray,
    *,
    rows: int = 4,
    cols: int = 8,
    title: str = "First-layer weights",
    path: str | Path | None = None,
) -> Figure:
    """Reshape first-layer weights back to 28x28 and view them as images.

    Each hidden unit's weight vector lives in the same space as the input, so
    it can be looked at directly. A trained network shows stroke-like and
    blob-like detectors; an untrained one shows pure noise. Watching noise turn
    into structure is the most direct evidence that learning happened.

    Diverging colour map, symmetric about zero: red is positive evidence, blue
    negative.
    """
    _require_mpl()
    w = np.asarray(weights)
    if w.shape[-1] != 784:
        w = w.T
    n = min(rows * cols, w.shape[0])

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.25))
    axes = np.atleast_1d(axes).ravel()
    for i in range(len(axes)):
        axes[i].axis("off")
        if i >= n:
            continue
        img = w[i].reshape(28, 28)
        lim = np.abs(img).max() or 1.0
        axes[i].imshow(img, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    if path:
        save(fig, path)
    return fig


def plot_decision_boundary(
    predict,
    points: np.ndarray,
    labels: Sequence[int],
    *,
    resolution: int = 120,
    padding: float = 0.35,
    title: str = "Decision boundary",
    path: str | Path | None = None,
) -> Figure:
    """Colour the input plane by the model's output. 2-D problems only.

    Used for XOR (Phase 21), where the whole point is visible in one picture:
    a linear model can only draw a straight boundary and must fail, while a
    network with one hidden layer bends the plane into the four correct
    regions.
    """
    _require_mpl()
    points = np.asarray(points, dtype=float)
    x_min, x_max = points[:, 0].min() - padding, points[:, 0].max() + padding
    y_min, y_max = points[:, 1].min() - padding, points[:, 1].max() + padding
    gx, gy = np.meshgrid(
        np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
    )
    grid = np.c_[gx.ravel(), gy.ravel()]
    zs = np.asarray([predict(p) for p in grid], dtype=float).reshape(gx.shape)

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    mesh = ax.contourf(gx, gy, zs, levels=40, cmap="RdBu_r", alpha=0.85, vmin=0, vmax=1)
    ax.contour(gx, gy, zs, levels=[0.5], colors="#222222", linewidths=1.6)
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label="model output")

    labels = np.asarray(labels)
    for cls, marker, colour in ((0, "o", "#2f6fd0"), (1, "s", "#c94f4f")):
        sel = labels == cls
        ax.scatter(
            points[sel, 0],
            points[sel, 1],
            marker=marker,
            s=170,
            c=colour,
            edgecolors="white",
            linewidths=2,
            zorder=3,
            label=f"class {cls}",
        )
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    if path:
        save(fig, path)
    return fig
