"""T1 CLI negative paths. Unknown commands, malformed args, no-arg failure modes."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import Lane, run_lilbee


@pytest.mark.cli
def test_unknown_top_level_command_fails(lane: Lane, lilbee_data: Path) -> None:
    result = run_lilbee(lane, ["this-is-not-a-command"], data_dir=lilbee_data, timeout=60)
    assert result.returncode != 0
    combined = (result.stdout + result.stderr).lower()
    assert any(needle in combined for needle in ("no such command", "usage", "error"))


@pytest.mark.cli
def test_unknown_wiki_subcommand_fails(lane: Lane, lilbee_data: Path) -> None:
    result = run_lilbee(lane, ["wiki", "not-a-subcommand"], data_dir=lilbee_data, timeout=60)
    assert result.returncode != 0


@pytest.mark.cli
def test_search_without_query_fails(lane: Lane, lilbee_data: Path) -> None:
    """`lilbee search` without a query is a usage error, not a silent zero-result."""
    result = run_lilbee(lane, ["search"], data_dir=lilbee_data, timeout=60)
    assert result.returncode != 0


@pytest.mark.cli
def test_invalid_global_flag_fails(lane: Lane, lilbee_data: Path) -> None:
    """Unknown global flags are rejected by Typer with non-zero exit."""
    result = run_lilbee(
        lane, ["--this-flag-does-not-exist", "status"], data_dir=lilbee_data, timeout=60
    )
    assert result.returncode != 0
