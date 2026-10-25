"""Logging for training runs: readable by default, machine-readable on request.

Library code logs to the ``nabla`` logger and never prints progress itself.
With no setup, ``ensure_configured()`` attaches a plain handler that writes just
the message to stdout, so scripts look exactly as they always did. For long or
unattended runs, switch to JSON lines, one object per event with its fields:

    NABLA_LOG_FORMAT=json python examples/train_mnist.py
    nabla --log-format json train data.csv
    nabla --quiet train data.csv                  # warnings only

The environment variables are how settings reach the example scripts that the
CLI runs as subprocesses.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import IO, Any

__all__ = ["get_logger", "configure", "ensure_configured", "JsonFormatter"]

ROOT = "nabla"
_HANDLER_FLAG = "_nabla_handler"


class JsonFormatter(logging.Formatter):
    """One JSON object per line: time, level, logger, message, plus any ``fields``."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        entry.update(getattr(record, "fields", None) or {})
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def get_logger(name: str) -> logging.Logger:
    """``get_logger("training")`` -> the ``nabla.training`` logger."""
    return logging.getLogger(f"{ROOT}.{name}")


def configure(
    level: str | int | None = None,
    fmt: str | None = None,
    stream: IO[str] | None = None,
) -> logging.Logger:
    """Set up the ``nabla`` logger. Defaults come from NABLA_LOG_LEVEL / NABLA_LOG_FORMAT."""
    level = level or os.environ.get("NABLA_LOG_LEVEL", "INFO")
    fmt = (fmt or os.environ.get("NABLA_LOG_FORMAT", "plain")).lower()
    if fmt not in ("plain", "json"):
        raise ValueError(f"log format must be 'plain' or 'json', got {fmt!r}")

    logger = logging.getLogger(ROOT)
    for h in [h for h in logger.handlers if getattr(h, _HANDLER_FLAG, False)]:
        logger.removeHandler(h)
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == "json" else logging.Formatter("%(message)s"))
    setattr(handler, _HANDLER_FLAG, True)
    logger.addHandler(handler)
    logger.setLevel(level.upper() if isinstance(level, str) else level)
    logger.propagate = False
    return logger


def ensure_configured() -> None:
    """Configure with defaults unless the application already set logging up."""
    if not logging.getLogger(ROOT).handlers and not logging.getLogger().handlers:
        configure()
