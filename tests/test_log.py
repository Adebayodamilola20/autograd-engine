"""Structured logging (nabla/log.py) and how the Trainer and CLI use it."""

from __future__ import annotations

import io
import json
import logging

import numpy as np
import pytest

from nabla.data import DataLoader, Dataset
from nabla.log import JsonFormatter, configure, ensure_configured, get_logger
from nabla.losses import softmax_cross_entropy
from nabla.nn import MLP
from nabla.optim import Adam
from nabla.training import Trainer


@pytest.fixture(autouse=True)
def _reset_nabla_logger(monkeypatch):
    # setenv first so monkeypatch restores the original state even when the
    # code under test writes these variables itself (the CLI does)
    for var in ("NABLA_LOG_LEVEL", "NABLA_LOG_FORMAT"):
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    logger = logging.getLogger("nabla")
    saved = (list(logger.handlers), logger.level, logger.propagate)
    logger.handlers.clear()
    yield
    logger.handlers[:] = saved[0]
    logger.setLevel(saved[1])
    logger.propagate = saved[2]


def _fit(epochs=2):
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(-2, 0.5, (20, 2)), rng.normal(2, 0.5, (20, 2))])
    data = Dataset(x, np.array([0] * 20 + [1] * 20))
    model = MLP(2, [4], 2, activation="tanh", seed=0)
    trainer = Trainer(model, Adam(model.parameters(), lr=0.05), softmax_cross_entropy)
    trainer.fit(DataLoader(data, batch_size=10, seed=0),
                DataLoader(data, batch_size=40, shuffle=False), epochs=epochs)


def test_json_lines_carry_every_epoch_metric():
    out = io.StringIO()
    configure(fmt="json", stream=out)
    _fit(epochs=2)
    events = [json.loads(line) for line in out.getvalue().splitlines()]
    epochs = [e for e in events if e.get("event") == "epoch"]
    assert [e["epoch"] for e in epochs] == [1, 2]
    for e in epochs:
        assert e["logger"] == "nabla.training" and e["level"] == "info"
        assert {"train_loss", "val_acc", "seconds"} <= set(e)
    assert events[-1]["event"] == "best"


def test_plain_format_is_just_the_message():
    out = io.StringIO()
    configure(fmt="plain", stream=out)
    _fit(epochs=1)
    first = out.getvalue().splitlines()[0]
    assert first.startswith("training ") and "{" not in first


def test_quiet_level_hides_progress():
    out = io.StringIO()
    configure(level="WARNING", stream=out)
    _fit(epochs=1)
    assert out.getvalue() == ""


def test_environment_variables_choose_the_defaults(monkeypatch):
    monkeypatch.setenv("NABLA_LOG_FORMAT", "json")
    monkeypatch.setenv("NABLA_LOG_LEVEL", "DEBUG")
    out = io.StringIO()
    logger = configure(stream=out)
    assert logger.level == logging.DEBUG
    get_logger("x").debug("hello", extra={"fields": {"k": 1}})
    assert json.loads(out.getvalue())["k"] == 1


def test_reconfiguring_replaces_rather_than_duplicates():
    out = io.StringIO()
    configure(stream=out)
    configure(stream=out)
    get_logger("x").info("once")
    assert out.getvalue().count("once") == 1


def test_bad_format_is_rejected():
    with pytest.raises(ValueError, match="plain"):
        configure(fmt="xml")


def test_ensure_configured_respects_an_existing_setup():
    mine = logging.StreamHandler(io.StringIO())
    logging.getLogger("nabla").addHandler(mine)
    ensure_configured()
    handlers = logging.getLogger("nabla").handlers
    assert mine in handlers
    # pytest may attach its own capture handler; what matters is we added none
    assert not any(getattr(h, "_nabla_handler", False) for h in handlers)


def test_json_formatter_includes_exceptions():
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        record = logging.LogRecord("nabla.t", logging.ERROR, __file__, 1, "failed", None,
                                   exc_info=__import__("sys").exc_info())
    entry = json.loads(JsonFormatter().format(record))
    assert entry["message"] == "failed" and "boom" in entry["exception"]


def test_cli_flags_set_the_environment(monkeypatch, capsys):
    import os

    from nabla.cli.main import main
    monkeypatch.setattr("nabla.cli.main.configure", lambda: None)
    assert main(["--quiet", "--log-format", "json", "grad", "x*x", "--at", "x=3"]) == 0
    assert os.environ["NABLA_LOG_LEVEL"] == "WARNING"
    assert os.environ["NABLA_LOG_FORMAT"] == "json"
    assert "6" in capsys.readouterr().out
