"""Phase 23 -- saving and restoring models.

Format: plain JSON. Not because it is efficient (it is not -- a 109k-parameter
model is ~2 MB of text), but because it is *inspectable*. You can open a
checkpoint in an editor and read the weights. For a project whose purpose is
understanding, that beats a binary format that requires the library that wrote
it. ``save_npz`` is available when size actually matters.

What belongs in a checkpoint
----------------------------
Not just weights. Restoring only ``model.state_dict()`` and continuing training
resets the optimiser's momentum and Adam moments to zero, which produces a
visible bump in the loss curve as the buffers refill. A complete checkpoint
carries model state, optimiser state, the epoch number and the metric history.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["save_checkpoint", "load_checkpoint", "save_npz", "load_npz"]


def save_checkpoint(
    path: str | Path,
    *,
    model: Any,
    optimizer: Any | None = None,
    epoch: int = 0,
    history: dict[str, list[float]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a complete training state to JSON.

    Parameters
    ----------
    model
        Anything with ``state_dict()``.
    optimizer
        Optional; include it to resume training cleanly.
    metadata
        Free-form: architecture, seed, hyperparameters, dataset name. Worth
        filling in -- a checkpoint whose architecture you have to guess is
        nearly useless six months later.
    """
    payload: dict[str, Any] = {
        "format": "nabla-checkpoint-v1",
        "epoch": epoch,
        "model": model.state_dict(),
        "metadata": metadata or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if history is not None:
        payload["history"] = {k: list(map(float, v)) for k, v in history.items()}

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def load_checkpoint(
    path: str | Path,
    *,
    model: Any | None = None,
    optimizer: Any | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Read a checkpoint, optionally loading it into a model and optimiser.

    Returns the full payload, so ``epoch``, ``history`` and ``metadata`` are
    available whether or not a model was passed.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != "nabla-checkpoint-v1":
        raise ValueError(
            f"unrecognised checkpoint format {payload.get('format')!r}"
        )
    if model is not None:
        model.load_state_dict(payload["model"], strict=strict)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def save_npz(path: str | Path, arrays: dict[str, np.ndarray]) -> Path:
    """Compressed binary save, for when size matters (e.g. the web demo)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **arrays)
    return p


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path)) as data:
        return {k: data[k] for k in data.files}
