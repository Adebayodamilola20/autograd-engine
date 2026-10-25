"""Config files and run records (nabla/training/runs.py)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from nabla.training.runs import config_hash, load_config, write_run_record


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
