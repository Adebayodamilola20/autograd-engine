"""Config files and run records (nabla/training/runs.py)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from nabla.training.runs import (
    check_record,
    config_hash,
    latest_record,
    load_config,
    main,
    write_run_record,
)


def test_load_config_reads_json(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"seed": 3, "lr": 0.01}))
    assert load_config(p) == {"seed": 3, "lr": 0.01}


def test_unknown_keys_are_rejected_not_ignored(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"seed": 3, "learnig_rate": 0.01}))   # typo
    with pytest.raises(ValueError, match="learnig_rate"):
        load_config(p, allowed={"seed", "lr"})


@pytest.mark.parametrize("text, message", [("{not json", "not valid JSON"), ("[1, 2]", "JSON object")])
def test_bad_files_explain_themselves(tmp_path, text, message):
    p = tmp_path / "c.json"
    p.write_text(text)
    with pytest.raises(ValueError, match=message):
        load_config(p)


def test_missing_file_names_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.json"):
        load_config(tmp_path / "nope.json")


def test_config_hash_ignores_key_order_but_not_values():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})
    assert len(config_hash({})) == 12


def test_run_record_contains_everything_needed_to_repeat_it(tmp_path):
    when = datetime(2026, 10, 25, 12, 0, 0, tzinfo=timezone.utc)
    path = write_run_record("mnist", {"seed": 0, "lr": 0.001}, {"test_accuracy": 0.9763},
                            out_dir=tmp_path / "runs", now=when)
    assert path.name == "mnist_20261025T120000Z.json"
    record = json.loads(path.read_text())
    assert record["config"] == {"seed": 0, "lr": 0.001}
    assert record["config_hash"] == config_hash({"seed": 0, "lr": 0.001})
    assert record["results"]["test_accuracy"] == 0.9763
    assert record["timestamp"] == "2026-10-25T12:00:00+00:00"
    for key in ("python", "numpy", "platform", "git_commit"):
        assert key in record


def _two_runs(tmp_path):
    out = tmp_path / "runs"
    write_run_record("mnist", {"seed": 0}, {"test_accuracy": 0.90}, out_dir=out,
                     now=datetime(2026, 10, 1, tzinfo=timezone.utc))
    write_run_record("mnist", {"seed": 0}, {"test_accuracy": 0.97}, out_dir=out,
                     now=datetime(2026, 10, 2, tzinfo=timezone.utc))
    write_run_record("gpt", {"seed": 0}, {"perplexity": 5.5}, out_dir=out,
                     now=datetime(2026, 10, 3, tzinfo=timezone.utc))
    return out


def test_latest_record_picks_the_newest_of_that_name(tmp_path):
    out = _two_runs(tmp_path)
    assert latest_record(out, "mnist")["results"]["test_accuracy"] == 0.97
    with pytest.raises(FileNotFoundError):
        latest_record(out, "snake")


def test_check_record_bounds():
    record = {"name": "mnist", "config_hash": "abc", "results": {"test_accuracy": 0.96}}
    assert "OK" in check_record(record, "test_accuracy", minimum=0.95)
    with pytest.raises(ValueError, match="below"):
        check_record(record, "test_accuracy", minimum=0.97)
    with pytest.raises(ValueError, match="above"):
        check_record(record, "test_accuracy", maximum=0.9)
    with pytest.raises(ValueError, match="no result named"):
        check_record(record, "loss")


def test_check_command_exit_codes(tmp_path, capsys):
    out = str(_two_runs(tmp_path))
    assert main(["check", out, "--name", "mnist", "--metric", "test_accuracy", "--min", "0.95"]) == 0
    assert "OK" in capsys.readouterr().out
    assert main(["check", out, "--name", "mnist", "--metric", "test_accuracy", "--min", "0.99"]) == 1
    assert "below" in capsys.readouterr().err
    assert main(["check", out, "--name", "gpt", "--metric", "perplexity", "--max", "6"]) == 0
