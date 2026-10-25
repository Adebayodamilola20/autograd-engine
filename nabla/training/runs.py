"""Config-driven runs and a record of every result.

A number in a README is only as good as your ability to get it again. This
module makes that mechanical:

* ``load_config`` reads a JSON file of hyperparameters (``configs/*.json``)
  and refuses keys the script doesn't know, so a typo can't silently fall back
  to a default;
* ``write_run_record`` saves what was run and what came out -- the config,
  its hash, the git commit, library versions and the final metrics -- to
  ``runs/<name>_<timestamp>.json``.

Two records with the same ``config_hash`` and ``git_commit`` were the same
experiment; if their metrics differ, something is not deterministic.

    >>> cfg = {"seed": 0, "lr": 0.001}
    >>> config_hash(cfg) == config_hash({"lr": 0.001, "seed": 0})
    True
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

__all__ = ["load_config", "config_hash", "write_run_record"]


def load_config(path: str | Path, allowed: Iterable[str] | None = None) -> dict[str, Any]:
    """Read a JSON config. With ``allowed``, unknown keys raise ``ValueError``."""
    p = Path(path)
    try:
        config = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"config file not found: {p}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{p} is not valid JSON: {exc}") from None
    if not isinstance(config, dict):
        raise ValueError(f"{p} must contain a JSON object, got {type(config).__name__}")
    if allowed is not None:
        unknown = sorted(set(config) - set(allowed))
        if unknown:
            raise ValueError(f"{p}: unknown setting(s) {', '.join(unknown)}")
    return config


def config_hash(config: Mapping[str, Any]) -> str:
    """Short, order-independent fingerprint of a config."""
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _git_commit(cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=cwd,
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def write_run_record(
    name: str,
    config: Mapping[str, Any],
    results: Mapping[str, Any],
    *,
    out_dir: str | Path = "runs",
    now: datetime | None = None,
) -> Path:
    """Write ``<out_dir>/<name>_<UTC timestamp>.json`` and return its path."""
    now = now or datetime.now(timezone.utc)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    record = {
        "name": name,
        "timestamp": now.isoformat(timespec="seconds"),
        "config": dict(config),
        "config_hash": config_hash(config),
        "git_commit": _git_commit(out.resolve().parent),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "results": dict(results),
    }
    path = out / f"{name}_{now.strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return path
