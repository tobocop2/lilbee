"""T0 smoke. Gates publish.

Minimal CLI checks against the artifact-under-test that don't require any LLM
or embedding model to be installed. Sync / search / ask scenarios live in T1
under proper model-fixture setup. Anything failing here fails the gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import Lane, run_lilbee


@pytest.mark.smoke
def test_version_runs_and_prints_lilbee(lane: Lane, lilbee_data: Path) -> None:
    result = run_lilbee(lane, ["--version"], data_dir=lilbee_data, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "lilbee" in (result.stdout + result.stderr).lower()


@pytest.mark.smoke
def test_help_succeeds(lane: Lane, lilbee_data: Path) -> None:
    result = run_lilbee(lane, ["--help"], data_dir=lilbee_data, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "Usage" in result.stdout or "USAGE" in result.stdout.upper()


@pytest.mark.smoke
def test_top_level_help_lists_core_commands(lane: Lane, lilbee_data: Path) -> None:
    """The flagship commands must remain in --help. A regression here is loud."""
    result = run_lilbee(lane, ["--help"], data_dir=lilbee_data, timeout=15)
    assert result.returncode == 0, result.stderr
    output = result.stdout
    for command in ("search", "sync", "ask", "chat", "status", "serve", "mcp"):
        assert command in output, f"missing top-level command in --help: {command}"


@pytest.mark.smoke
def test_status_returns_json_on_empty(lane: Lane, lilbee_data: Path) -> None:
    """`lilbee --json status` parses cleanly against an empty data dir."""
    result = run_lilbee(lane, ["--json", "status"], data_dir=lilbee_data, timeout=30)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict), f"--json status was not a dict: {payload!r}"
    assert payload.get("command") == "status", payload


@pytest.mark.smoke
def test_status_reports_zero_chunks_initially(lane: Lane, lilbee_data: Path) -> None:
    """A fresh data dir reports zero indexed chunks and an empty sources list."""
    result = run_lilbee(lane, ["--json", "status"], data_dir=lilbee_data, timeout=30)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload.get("total_chunks", -1) == 0, payload
    assert payload.get("sources") == [], payload


@pytest.mark.smoke
def test_unknown_subcommand_exits_nonzero(lane: Lane, lilbee_data: Path) -> None:
    """Negative path. Unknown subcommands fail loudly, not silently."""
    result = run_lilbee(lane, ["this-command-does-not-exist"], data_dir=lilbee_data, timeout=15)
    assert result.returncode != 0
    combined = (result.stdout + result.stderr).lower()
    assert "no such command" in combined or "usage" in combined or "error" in combined
