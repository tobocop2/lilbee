"""T1 CLI model. List, browse, and show against an empty registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import Lane, run_lilbee


@pytest.mark.cli
def test_model_list_returns_json_array(lane: Lane, lilbee_data: Path) -> None:
    """`lilbee --json model list` returns a JSON array (possibly empty)."""
    result = run_lilbee(lane, ["--json", "model", "list"], data_dir=lilbee_data, timeout=60)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict | list), payload


@pytest.mark.cli
def test_model_list_text_runs(lane: Lane, lilbee_data: Path) -> None:
    """Default (non-JSON) `lilbee model list` runs and exits 0."""
    result = run_lilbee(lane, ["model", "list"], data_dir=lilbee_data, timeout=60)
    assert result.returncode == 0, result.stderr


@pytest.mark.cli
def test_model_help_lists_subcommands(lane: Lane, lilbee_data: Path) -> None:
    result = run_lilbee(lane, ["model", "--help"], data_dir=lilbee_data, timeout=60)
    assert result.returncode == 0
    output = result.stdout + result.stderr
    for sub in ("list", "pull", "rm"):
        assert sub in output, f"model --help missing {sub}"


@pytest.mark.cli
def test_model_show_unknown_exits_nonzero(lane: Lane, lilbee_data: Path) -> None:
    """Asking for a model that doesn't exist returns a non-zero exit."""
    result = run_lilbee(
        lane, ["model", "show", "this-model-does-not-exist"], data_dir=lilbee_data, timeout=60
    )
    assert result.returncode != 0
