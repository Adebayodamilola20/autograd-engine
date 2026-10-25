"""The matplotlib figures used by the examples and experiments.

Each plot is drawn headless (Agg) from small synthetic inputs and written to a
temp directory. The checks are about behaviour the rest of the project relies
on: files are written, panels appear for the data that exists and not for the
data that doesn't, and the derivative panel of plot_activations really is the
engine's own backward() output.
"""

from __future__ import annotations

import numpy as np
import pytest

plots = pytest.importorskip("nabla.visualization.plots")
if not plots.HAVE_MPL:  # pragma: no cover
    pytest.skip("matplotlib not installed", allow_module_level=True)

import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _written(path):
    assert path.exists() and path.stat().st_size > 1000, f"{path} was not written"


def test_save_creates_missing_directories(tmp_path):
    fig, _ = plt.subplots()
    out = plots.save(fig, tmp_path / "deep" / "er" / "fig.png")
    _written(out)


def test_training_curves_with_and_without_accuracy(tmp_path):
    full = {"train_loss": [1.0, 0.6, 0.4], "val_loss": [1.1, 0.7, 0.5],
            "train_acc": [0.5, 0.7, 0.8], "val_acc": [0.45, 0.65, 0.75]}
    fig = plots.plot_training_curves(full, path=tmp_path / "full.png")
    _written(tmp_path / "full.png")
    assert len(fig.axes) == 2                       # loss panel + accuracy panel

    regression = {"train_loss": [3.0, 2.0, 1.5]}    # no accuracy keys at all
    fig = plots.plot_training_curves(regression, path=tmp_path / "reg.png")
    _written(tmp_path / "reg.png")


def test_digit_grid_accepts_flat_images_and_marks_mistakes(tmp_path):
    rng = np.random.default_rng(0)
    images = rng.random((6, 784))                    # flat rows are reshaped to 28x28
    labels = [0, 1, 2, 3, 4, 5]
    predictions = [0, 1, 9, 3, 4, 6]                 # two mistakes
    fig = plots.plot_digit_grid(images, labels, predictions, [0.9] * 6,
                                rows=2, cols=4, path=tmp_path / "grid.png")
    _written(tmp_path / "grid.png")
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert any("2→9" in t for t in titles)
    assert any("5→6" in t for t in titles)
    assert len(titles) == 6                           # the two spare cells stay blank


def test_prediction_detail(tmp_path):
    probs = np.full(10, 0.02)
    probs[7] = 0.82
    plots.plot_prediction_detail(np.zeros((28, 28)), probs, true_label=7,
                                 title="a 7", path=tmp_path / "detail.png")
    _written(tmp_path / "detail.png")


@pytest.mark.parametrize("normalize", [False, True])
def test_confusion_matrix(tmp_path, normalize):
    true = [0, 0, 1, 1, 2, 2]
    pred = [0, 1, 1, 1, 2, 0]
    plots.plot_confusion_matrix(true, pred, n_classes=3, normalize=normalize,
                                path=tmp_path / "cm.png")
    _written(tmp_path / "cm.png")


def test_activations_derivative_panel_comes_from_backward(tmp_path):
    fig = plots.plot_activations({"relu": lambda v: v.relu(), "tanh": lambda v: v.tanh()},
                                 path=tmp_path / "act.png")
    _written(tmp_path / "act.png")
    # the derivative of tanh peaks at 1 at x = 0; if the panel were not fed by
    # backward() this curve would not exist
    derivative_axes = fig.axes[1]
    ys = np.concatenate([line.get_ydata() for line in derivative_axes.get_lines()])
    assert ys.max() == pytest.approx(1.0, abs=1e-2)


def test_gradient_histogram(tmp_path):
    rng = np.random.default_rng(0)
    grads = {"layer 1": rng.normal(0, 1e-3, 200), "layer 2": rng.normal(0, 1e-2, 200)}
    plots.plot_gradient_histogram(grads, path=tmp_path / "grads.png")
    _written(tmp_path / "grads.png")


@pytest.mark.parametrize("shape", [(784, 8), (8, 784)])
def test_weight_images_accepts_either_orientation(tmp_path, shape):
    w = np.random.default_rng(0).normal(size=shape)
    fig = plots.plot_weight_images(w, rows=2, cols=4, path=tmp_path / "w.png")
    _written(tmp_path / "w.png")
    assert sum(1 for ax in fig.axes if ax.images) == 8


def test_decision_boundary_for_xor(tmp_path):
    points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    labels = [0, 1, 1, 0]
    xor = lambda p: float((p[0] > 0.5) != (p[1] > 0.5))  # noqa: E731
    plots.plot_decision_boundary(xor, points, labels, resolution=25,
                                 path=tmp_path / "xor.png")
    _written(tmp_path / "xor.png")
