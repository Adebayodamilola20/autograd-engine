"""The Trainer: the loop that every example and the CLI share.

These pin its behaviour end to end on problems small enough to run in
milliseconds: it must actually learn, record history, write checkpoints that
load back, stop early, apply a learning-rate schedule, call callbacks, and do
all of it identically on the scalar and tensor engines.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from nabla.data import DataLoader, Dataset
from nabla.losses import mse_loss, softmax_cross_entropy
from nabla.losses.tensor_losses import tensor_cross_entropy
from nabla.nn import MLP
from nabla.nn.tensor_mlp import TensorMLP
from nabla.optim import SGD, Adam
from nabla.training import Trainer
from nabla.training.checkpoint import load_checkpoint


def blobs(n=60, seed=0):
    """Two well-separated clusters: any working trainer reaches ~100% here."""
    rng = np.random.default_rng(seed)
    a = rng.normal([-2.0, -2.0], 0.5, size=(n // 2, 2))
    b = rng.normal([2.0, 2.0], 0.5, size=(n // 2, 2))
    return Dataset(np.vstack([a, b]), np.array([0] * (n // 2) + [1] * (n // 2)), name="blobs")


def scalar_trainer(**kwargs):
    model = MLP(2, [4], 2, activation="tanh", seed=0)
    return Trainer(model, Adam(model.parameters(), lr=0.05), softmax_cross_entropy,
                   verbose=False, **kwargs)


def tensor_trainer(**kwargs):
    model = TensorMLP(2, [8], 2, activation="relu", seed=0)
    return Trainer(model, Adam(model.parameters(), lr=0.05), tensor_cross_entropy,
                   verbose=False, **kwargs)


# ---------------------------------------------------------------- learning


@pytest.mark.parametrize("make, arrays", [(scalar_trainer, False), (tensor_trainer, True)])
def test_fit_learns_a_separable_problem(make, arrays):
    data = blobs()
    trainer = make()
    loader = DataLoader(data, batch_size=10, seed=0, as_arrays=arrays)
    history = trainer.fit(loader, DataLoader(data, batch_size=30, shuffle=False, as_arrays=arrays), epochs=8)

    assert len(history["train_loss"]) == 8
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert history["val_acc"][-1] >= 0.95


def test_engine_is_detected_from_the_model():
    assert scalar_trainer().engine == "scalar"
    assert tensor_trainer().engine == "tensor"


def test_evaluate_does_not_change_the_weights():
    trainer = scalar_trainer()
    before = [p.data for p in trainer.model.parameters()]
    loss, acc = trainer.evaluate(DataLoader(blobs(), batch_size=20, shuffle=False))
    after = [p.data for p in trainer.model.parameters()]
    assert before == after
    assert loss > 0 and 0.0 <= acc <= 1.0


def test_regression_has_no_accuracy():
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, size=(40, 1))
    data = Dataset(x, 2.0 * x[:, 0] + 0.5, name="line")
    model = MLP(1, [4], 1, activation="tanh", seed=0)
    trainer = Trainer(
        model, SGD(model.parameters(), lr=0.1),
        lambda outs, ys: mse_loss([o[0] for o in outs], list(ys)),
        metric=None, verbose=False,
    )
    history = trainer.fit(DataLoader(data, batch_size=8, seed=0), epochs=5)
    assert "train_acc" not in history          # no accuracy column for regression
    assert history["train_loss"][-1] < history["train_loss"][0]


def test_grad_clip_limits_how_far_each_step_moves():
    # With plain SGD a step moves the weights by lr * |clipped gradient|, so a
    # tiny clip bounds the total movement over the epoch.
    model = MLP(2, [4], 2, activation="tanh", seed=0)
    trainer = Trainer(model, SGD(model.parameters(), lr=0.05), softmax_cross_entropy,
                      grad_clip=1e-3, verbose=False)
    before = np.array([p.data for p in model.parameters()])
    trainer.train_epoch(DataLoader(blobs(), batch_size=10, seed=0))      # 6 steps
    moved = np.linalg.norm(np.array([p.data for p in model.parameters()]) - before)
    assert moved <= 6 * 0.05 * 1e-3 + 1e-12


# ---------------------------------------------------------------- bookkeeping


def test_history_best_finds_the_right_epoch():
    trainer = scalar_trainer()
    data = blobs()
    trainer.fit(DataLoader(data, batch_size=10, seed=0),
                DataLoader(data, batch_size=30, shuffle=False), epochs=5)
    epoch, score = trainer.history.best("val_acc", "max")
    assert 1 <= epoch <= 5
    assert score == max(trainer.history["val_acc"])
    loss_epoch, loss = trainer.history.best("val_loss", "min")
    assert loss == min(trainer.history["val_loss"])


def test_callbacks_see_every_epoch():
    seen = []
    trainer = scalar_trainer()
    trainer.fit(DataLoader(blobs(), batch_size=10, seed=0), epochs=3,
                callbacks=[lambda m: seen.append(m.epoch)])
    assert seen == [1, 2, 3]


def test_lr_schedule_is_applied_each_epoch():
    trainer = scalar_trainer()
    lrs = []
    trainer.fit(DataLoader(blobs(), batch_size=10, seed=0), epochs=3,
                lr_schedule=lambda epoch, lr: lr * 0.5,
                callbacks=[lambda m: lrs.append(m.learning_rate)])
    assert lrs == pytest.approx([0.025, 0.0125, 0.00625])


def test_log_fn_receives_progress_lines():
    lines = []
    model = MLP(2, [4], 2, activation="tanh", seed=0)
    trainer = Trainer(model, Adam(model.parameters(), lr=0.05), softmax_cross_entropy,
                      log_fn=lines.append)
    data = blobs()
    trainer.fit(DataLoader(data, batch_size=10, seed=0),
                DataLoader(data, batch_size=30, shuffle=False), epochs=2)
    assert lines[0].startswith("training ")
    assert any("best val_acc" in line for line in lines)


def test_quiet_trainer_logs_nothing():
    lines = []
    trainer = scalar_trainer()
    trainer.log = lines.append
    trainer.fit(DataLoader(blobs(), batch_size=10, seed=0), epochs=2)
    assert lines == []


# ---------------------------------------------------------------- checkpoints and stopping


def test_checkpoints_are_written_and_load_back(tmp_path):
    trainer = scalar_trainer()
    data = blobs()
    trainer.fit(DataLoader(data, batch_size=10, seed=0),
                DataLoader(data, batch_size=30, shuffle=False), epochs=3,
                checkpoint_dir=tmp_path)
    assert (tmp_path / "best.json").exists() and (tmp_path / "last.json").exists()

    best = json.loads((tmp_path / "best.json").read_text())
    assert best["metadata"]["best_metric"] == "val_acc"
    assert best["metadata"]["architecture"]["sizes"] == [2, 4, 2]

    fresh = MLP(2, [4], 2, activation="tanh", seed=123)
    payload = load_checkpoint(tmp_path / "last.json", model=fresh)
    assert payload["epoch"] == 3
    assert [p.data for p in fresh.parameters()] == [p.data for p in trainer.model.parameters()]


def test_early_stopping_stops_when_nothing_improves():
    trainer = scalar_trainer()
    data = blobs()
    # val_loss of a constant cannot improve if we never train: lr=0
    trainer.optimizer.lr = 0.0
    trainer.fit(DataLoader(data, batch_size=10, seed=0),
                DataLoader(data, batch_size=30, shuffle=False), epochs=20,
                checkpoint_metric="val_loss", early_stopping_patience=2)
    assert len(trainer.history["train_loss"]) == 3   # best at 1, then 2 without improvement


def test_exploding_loss_raises_a_helpful_error():
    model = MLP(2, [4], 2, activation="relu", seed=0)
    trainer = Trainer(model, SGD(model.parameters(), lr=1e30), softmax_cross_entropy, verbose=False)
    with pytest.raises(RuntimeError, match="learning rate"):
        trainer.fit(DataLoader(blobs(), batch_size=10, seed=0), epochs=5)


# ---------------------------------------------------------------- inspection


@pytest.mark.parametrize("make, arrays", [(scalar_trainer, False), (tensor_trainer, True)])
def test_collect_predictions_lines_up_with_the_data(make, arrays):
    data = blobs(20)
    trainer = make()
    true, predicted, logits = trainer.collect_predictions(
        DataLoader(data, batch_size=7, shuffle=False, as_arrays=arrays))
    assert true == [int(y) for y in data.y]
    assert len(predicted) == len(logits) == 20
    assert all(len(row) == 2 for row in logits)
    assert predicted == [int(np.argmax(row)) for row in logits]
