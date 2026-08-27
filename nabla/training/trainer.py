"""Phase 11 -- the training loop.

The loop itself is six lines, and they are worth memorising because every
deep-learning framework is an elaboration of them::

    for batch_x, batch_y in loader:
        optimizer.zero_grad()          # 1. clear last step's gradients
        loss = loss_fn(model(batch_x), batch_y)   # 2. forward
        loss.backward()                # 3. reverse-mode autodiff
        optimizer.step()               # 4. move against the gradient

Everything else in this file -- metrics, validation, checkpoints, early
stopping, logging -- is bookkeeping around those four lines. Keeping them
visible is deliberate.

Order matters
-------------
``zero_grad()`` must come before ``backward()``, because ``backward()``
accumulates (decision D6). ``step()`` must come after ``backward()``, because
it reads the gradients that ``backward()`` wrote. Get either wrong and the
model still trains, badly, with no error -- which is why the order is stated
explicitly rather than left to be inferred.

Design
------
The ``Trainer`` knows nothing about MNIST, or about images. It takes a model
that maps a feature list to an output list, a loss function, and a data
loader. Swapping in a different dataset requires no change here -- which was
the brief's requirement, and is a good test of whether the abstraction is real.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from ..core.value import Value
from ..data.dataset import DataLoader
from .checkpoint import save_checkpoint
from .metrics import EpochMetrics, RunningAverage, Timer, accuracy, argmax

__all__ = ["Trainer", "TrainingHistory"]

LossFn = Callable[[Any, Sequence[Any]], Value]
Callback = Callable[[EpochMetrics], None]


def _resolve_engine(engine: str, model: Any) -> str:
    """Decide whether the model consumes one sample or a whole batch."""
    if engine in ("scalar", "tensor"):
        return engine
    if engine != "auto":
        raise ValueError(f"engine must be 'scalar', 'tensor' or 'auto', got {engine!r}")
    # A tensor model's parameters hold arrays; a scalar model's hold floats.
    params = model.parameters()
    if params and hasattr(params[0].data, "shape"):
        return "tensor"
    return "scalar"


def _scalar(loss: Any) -> float:
    """Extract a plain float from a ``Value`` or a 0-d ``Tensor`` loss."""
    data = loss.data
    return float(data) if not hasattr(data, "reshape") else float(data.reshape(()))


class TrainingHistory(dict):
    """Metric lists keyed by name, plus the per-epoch records.

    Subclasses ``dict`` so it plugs straight into ``plot_training_curves`` and
    into ``json.dump`` without conversion.
    """

    def __init__(self) -> None:
        super().__init__()
        self.epochs: list[EpochMetrics] = []

    def record(self, metrics: EpochMetrics) -> None:
        self.epochs.append(metrics)
        for key, value in metrics.to_dict().items():
            if key in ("epoch", "samples"):
                continue
            self.setdefault(key, []).append(value)

    def best(self, key: str = "val_acc", mode: str = "max") -> tuple[int, float]:
        """``(epoch_index, value)`` of the best entry for ``key``."""
        values = self.get(key)
        if not values:
            raise KeyError(f"no history recorded for {key!r}")
        pick = max if mode == "max" else min
        best_value = pick(values)
        return values.index(best_value), best_value


class Trainer:
    """Runs training epochs, tracks metrics, and handles checkpoints.

    Parameters
    ----------
    model
        Anything callable on a feature list returning a list of ``Value``,
        with ``parameters()`` and ``zero_grad()``.
    optimizer
        Anything with ``step()`` and ``zero_grad()``.
    loss_fn
        ``(batch_outputs, batch_targets) -> Value``. Receives the whole batch
        so it can average, which keeps gradient magnitude independent of batch
        size.
    metric
        ``'accuracy'`` for classification, ``None`` for regression.
    grad_clip
        Clip the global gradient norm to this value. The standard remedy for
        exploding gradients; ``None`` disables it.
    engine
        ``'scalar'`` calls the model once per sample and builds one graph per
        sample; ``'tensor'`` passes the whole batch as an array and builds one
        graph for the batch. ``'auto'`` detects which by inspecting the model.
        The four-line training loop is *identical* either way -- only the
        granularity of the forward call changes, which is the clearest possible
        demonstration that scalar and tensor autodiff are the same algorithm.
    verbose
        Print one line per epoch.
    log_fn
        Where those lines go. Defaults to ``print``; the experiment scripts
        redirect it.

    Examples
    --------
    >>> from nabla.nn import MLP
    >>> from nabla.optim import Adam
    >>> from nabla.losses import softmax_cross_entropy
    >>> model = MLP(4, [8], 3, activation='relu', seed=0)
    >>> trainer = Trainer(model, Adam(model.parameters()), softmax_cross_entropy)
    """

    def __init__(
        self,
        model: Any,
        optimizer: Any,
        loss_fn: LossFn,
        *,
        metric: str | None = "accuracy",
        grad_clip: float | None = None,
        engine: str = "auto",
        verbose: bool = True,
        log_fn: Callable[[str], None] = print,
    ) -> None:
        self.model = model
        self.engine = _resolve_engine(engine, model)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.grad_clip = grad_clip
        self.verbose = verbose
        self.log = log_fn
        self.history = TrainingHistory()

    # ------------------------------------------------------------------
    # engine abstraction
    # ------------------------------------------------------------------

    def _forward_and_loss(self, batch_x, batch_y):
        """Run the model over a batch and return ``(loss, predicted_labels)``.

        The only place the two engines differ. Everything downstream -- the
        backward call, the optimiser step, the metrics -- is shared.
        """
        if self.engine == "tensor":
            outputs = self.model(batch_x)            # one graph for the batch
            loss = self.loss_fn(outputs, batch_y)
            predicted = (
                np.asarray(outputs.data).argmax(axis=-1).tolist()
                if self.metric == "accuracy"
                else []
            )
            return loss, predicted

        outputs = [self.model(x) for x in batch_x]   # one graph per sample
        loss = self.loss_fn(outputs, batch_y)
        predicted = (
            [argmax([v.data for v in out]) for out in outputs]
            if self.metric == "accuracy"
            else []
        )
        return loss, predicted

    # ------------------------------------------------------------------
    # the loop
    # ------------------------------------------------------------------

    def train_epoch(self, loader: DataLoader) -> tuple[float, float | None, float]:
        """One pass over the training data. Returns ``(loss, acc, grad_norm)``."""
        self.model.train()
        loss_meter = RunningAverage()
        correct = 0
        seen = 0
        last_grad_norm = 0.0

        for batch_x, batch_y in loader:
            # ---- 1. clear gradients from the previous step ------------
            # backward() accumulates; without this, step n uses the sum of
            # gradients from steps 1..n.
            self.optimizer.zero_grad()

            # ---- 2. forward: build a fresh graph for this batch --------
            loss, predicted = self._forward_and_loss(batch_x, batch_y)

            # ---- 3. backward: reverse-mode autodiff -------------------
            loss.backward()

            # ---- 3b. optional gradient clipping -----------------------
            if self.grad_clip is not None:
                last_grad_norm = self.optimizer.clip_grad_norm(self.grad_clip)
            else:
                last_grad_norm = self.optimizer.gradient_norm()

            # ---- 4. update the parameters -----------------------------
            self.optimizer.step()

            # ---- bookkeeping ------------------------------------------
            n = len(batch_x)
            loss_value = _scalar(loss)
            loss_meter.update(loss_value, n)
            seen += n
            if self.metric == "accuracy":
                correct += sum(
                    int(p == int(t)) for p, t in zip(predicted, batch_y)
                )

            if not math.isfinite(loss_value):
                raise RuntimeError(
                    f"loss became {loss_value} at step {self.optimizer.step_count}. "
                    "Almost always a learning rate that is too high -- try "
                    "lowering it or enabling grad_clip."
                )

        acc = correct / seen if (self.metric == "accuracy" and seen) else None
        return loss_meter.value, acc, last_grad_norm

    def evaluate(self, loader: DataLoader) -> tuple[float, float | None]:
        """Loss and accuracy on data the model is not being updated on.

        No ``torch.no_grad()`` equivalent exists here, and that is worth
        noticing: in a define-by-run scalar engine the graph *is* the
        computation, so evaluation builds the same graph a training step would
        and simply never calls ``backward()``. The wasted allocation is one of
        the costs quantified in Phase 17 -- and precisely what
        ``torch.no_grad()`` was invented to avoid.
        """
        self.model.eval()
        loss_meter = RunningAverage()
        correct = 0
        seen = 0

        for batch_x, batch_y in loader:
            loss, predicted = self._forward_and_loss(batch_x, batch_y)
            n = len(batch_x)
            loss_meter.update(_scalar(loss), n)
            seen += n
            if self.metric == "accuracy":
                correct += sum(
                    int(p == int(t)) for p, t in zip(predicted, batch_y)
                )

        acc = correct / seen if (self.metric == "accuracy" and seen) else None
        return loss_meter.value, acc

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        *,
        epochs: int = 10,
        checkpoint_dir: str | Path | None = None,
        checkpoint_metric: str = "val_acc",
        early_stopping_patience: int | None = None,
        callbacks: Iterable[Callback] = (),
        lr_schedule: Callable[[int, float], float] | None = None,
    ) -> TrainingHistory:
        """Train for ``epochs``, evaluating and logging each one.

        Parameters
        ----------
        checkpoint_dir
            If set, writes ``best.json`` whenever ``checkpoint_metric``
            improves, and ``last.json`` every epoch.
        early_stopping_patience
            Stop after this many epochs without improvement. Guards against
            wasting time once validation performance has turned around -- the
            point at which the model has started memorising.
        lr_schedule
            ``(epoch, current_lr) -> new_lr``, applied at the start of each
            epoch. Decaying the learning rate late in training lets the
            optimiser settle into a minimum instead of bouncing around it.
        """
        # Architecture travels with the weights. A checkpoint whose shape you
        # have to reverse-engineer from matrix dimensions is a checkpoint that
        # cannot be loaded without the script that wrote it -- and the
        # activation function is not recoverable from the shapes at all.
        # ``web/server.py`` reconstructs the model from exactly this.
        architecture = {
            "engine": self.engine,
            "class": type(self.model).__name__,
            "sizes": list(getattr(self.model, "sizes", [])),
            "activation": getattr(self.model, "activation_name", None),
            "output_activation": getattr(self.model, "output_activation_name", None),
            "parameters": self.model.num_parameters(),
        }

        maximise = not checkpoint_metric.endswith("loss")
        best_score = -math.inf if maximise else math.inf
        best_epoch = -1
        since_improvement = 0

        if self.verbose:
            self.log(
                f"training {self.model} for {epochs} epochs on "
                f"{len(train_loader.dataset):,} samples "
                f"({len(train_loader)} batches of {train_loader.batch_size})"
            )

        for epoch in range(1, epochs + 1):
            if lr_schedule is not None:
                self.optimizer.lr = lr_schedule(epoch, self.optimizer.lr)

            timer = Timer()
            with timer:
                train_loss, train_acc, grad_norm = self.train_epoch(train_loader)
                val_loss, val_acc = (
                    self.evaluate(val_loader) if val_loader is not None else (None, None)
                )

            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                seconds=timer.elapsed,
                samples=len(train_loader.dataset),
                grad_norm=grad_norm,
                learning_rate=getattr(self.optimizer, "lr", None),
            )
            self.history.record(metrics)

            if self.verbose:
                self.log(metrics.format_line(epochs))
            for callback in callbacks:
                callback(metrics)

            # ---- checkpointing and early stopping ---------------------
            score = metrics.to_dict().get(checkpoint_metric)
            if score is not None:
                improved = score > best_score if maximise else score < best_score
                if improved:
                    best_score, best_epoch = score, epoch
                    since_improvement = 0
                    if checkpoint_dir:
                        save_checkpoint(
                            Path(checkpoint_dir) / "best.json",
                            model=self.model,
                            optimizer=self.optimizer,
                            epoch=epoch,
                            history=dict(self.history),
                            metadata={
                                "best_metric": checkpoint_metric,
                                "score": score,
                                "architecture": architecture,
                            },
                        )
                else:
                    since_improvement += 1

            if checkpoint_dir:
                save_checkpoint(
                    Path(checkpoint_dir) / "last.json",
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    history=dict(self.history),
                    metadata={"architecture": architecture},
                )

            if (
                early_stopping_patience is not None
                and since_improvement >= early_stopping_patience
            ):
                if self.verbose:
                    self.log(
                        f"early stopping: no {checkpoint_metric} improvement for "
                        f"{since_improvement} epochs (best {best_score:.4f} "
                        f"at epoch {best_epoch})"
                    )
                break

        if self.verbose and best_epoch > 0:
            self.log(f"best {checkpoint_metric}: {best_score:.4f} at epoch {best_epoch}")
        return self.history

    # ------------------------------------------------------------------
    # inspection
    # ------------------------------------------------------------------

    def predict_batch(self, xs: Sequence[Sequence[float]]) -> list[list[float]]:
        """Raw outputs (logits) for each input row."""
        self.model.eval()
        if self.engine == "tensor":
            return np.asarray(self.model(xs).data).tolist()
        return [[v.data for v in self.model(x)] for x in xs]

    def collect_predictions(
        self, loader: DataLoader
    ) -> tuple[list[int], list[int], list[list[float]]]:
        """``(true_labels, predicted_labels, logits)`` over a whole loader.

        The input to the confusion matrix and the misclassification gallery in
        Phase 13. Note the loader must be unshuffled for the returned order to
        line up with the dataset.
        """
        self.model.eval()
        true: list[int] = []
        predicted: list[int] = []
        all_logits: list[list[float]] = []

        for batch_x, batch_y in loader:
            rows = self.predict_batch(batch_x)
            for logits, y in zip(rows, batch_y):
                all_logits.append(list(logits))
                predicted.append(argmax(logits))
                true.append(int(y))
        return true, predicted, all_logits
