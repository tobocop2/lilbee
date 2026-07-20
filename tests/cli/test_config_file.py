"""Tests for agent config-file load/atomic-write helpers."""

from __future__ import annotations

import json

import pytest
import typer

from lilbee.cli.agent_configs import config_file


def test_load_returns_empty_when_absent(tmp_path):
    out = config_file.load_config_dict(
        tmp_path / "missing.json", parse=json.loads, parse_error=json.JSONDecodeError, label="x"
    )
    assert out == {}


def test_load_returns_parsed_mapping(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"a": 1}))
    out = config_file.load_config_dict(
        path, parse=json.loads, parse_error=json.JSONDecodeError, label="x"
    )
    assert out == {"a": 1}


def test_load_returns_empty_for_non_mapping(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps([1, 2, 3]))
    out = config_file.load_config_dict(
        path, parse=json.loads, parse_error=json.JSONDecodeError, label="x"
    )
    assert out == {}


def test_load_refuses_corrupt_file(tmp_path):
    path = tmp_path / "c.json"
    path.write_text("{ not valid")
    with pytest.raises(typer.Exit) as exc:
        config_file.load_config_dict(
            path, parse=json.loads, parse_error=json.JSONDecodeError, label="opencode config"
        )
    assert exc.value.exit_code == 1
    assert path.read_text() == "{ not valid"  # untouched
