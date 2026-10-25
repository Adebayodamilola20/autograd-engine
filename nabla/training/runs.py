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

CI uses the command-line check to hold a result to a threshold:

    python -m nabla.training.runs check runs --name mnist --metric test_accuracy --min 0.95

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
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = ["load_config", "config_hash", "write_run_record", "latest_record", "check_record"]


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


def latest_record(out_dir: str | Path, name: str) -> dict[str, Any]:
    """The most recent ``<name>_*.json`` record in ``out_dir``."""
    records = sorted(Path(out_dir).glob(f"{name}_*.json"))
    if not records:
        raise FileNotFoundError(f"no {name}_*.json run records in {out_dir}")
    return json.loads(records[-1].read_text(encoding="utf-8"))


def check_record(record: Mapping[str, Any], metric: str,
                 minimum: float | None = None, maximum: float | None = None) -> str:
    """Raise ``ValueError`` if ``record['results'][metric]`` is outside the bounds."""
    results = record.get("results", {})
    if metric not in results:
        raise ValueError(f"run has no result named {metric!r} (has: {', '.join(results)})")
    value = float(results[metric])
    if minimum is not None and value < minimum:
        raise ValueError(f"{metric} = {value:.4f}, below the required {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{metric} = {value:.4f}, above the allowed {maximum}")
    return f"{record.get('name')}: {metric} = {value:.4f} (config {record.get('config_hash')}) OK"


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m nabla.training.runs")
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check", help="hold the latest run record to a threshold")
    c.add_argument("out_dir")
    c.add_argument("--name", required=True)
    c.add_argument("--metric", required=True)
    c.add_argument("--min", type=float, default=None, dest="minimum")
    c.add_argument("--max", type=float, default=None, dest="maximum")
    args = parser.parse_args(argv)
    try:
        print(check_record(latest_record(args.out_dir, args.name), args.metric, args.minimum, args.maximum))
    except (FileNotFoundError, ValueError) as exc:
        print(f"check failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
