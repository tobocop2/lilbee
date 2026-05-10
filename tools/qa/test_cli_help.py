"""T1 CLI help. Every top-level command and wiki subcommand renders --help."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import CLI_FAST_TIMEOUT, Lane, run_lilbee

_TOP_LEVEL_COMMANDS = [
    "search",
    "sync",
    "rebuild",
    "add",
    "chunks",
    "remove",
    "ask",
    "chat",
    "version",
    "self-check",
    "self-check-extras",
    "status",
    "reset",
    "init",
    "serve",
    "token",
    "topics",
    "login",
    "mcp",
    "setup",
    "wiki",
    "model",
]

_WIKI_SUBCOMMANDS = [
    "lint",
    "citations",
    "status",
    "synthesize",
    "prune",
    "build",
    "update",
    "drafts",
]

_DRAFTS_SUBCOMMANDS = [
    "list",
    "diff",
    "accept",
    "reject",
]


@pytest.mark.cli
@pytest.mark.parametrize("command", _TOP_LEVEL_COMMANDS, ids=_TOP_LEVEL_COMMANDS)
def test_top_level_command_renders_help(command: str, lane: Lane, lilbee_data: Path) -> None:
    """Every top-level command must respond to --help with usage text and exit 0."""
    result = run_lilbee(lane, [command, "--help"], data_dir=lilbee_data, timeout=CLI_FAST_TIMEOUT)
    assert result.returncode == 0, f"{command} --help failed:\n{result.stderr}"
    output = result.stdout + result.stderr
    assert "Usage" in output, f"{command} --help missing usage text"


@pytest.mark.cli
@pytest.mark.parametrize("subcommand", _WIKI_SUBCOMMANDS, ids=_WIKI_SUBCOMMANDS)
def test_wiki_subcommand_renders_help(subcommand: str, lane: Lane, lilbee_data: Path) -> None:
    """Every wiki subcommand renders --help."""
    result = run_lilbee(
        lane, ["wiki", subcommand, "--help"], data_dir=lilbee_data, timeout=CLI_FAST_TIMEOUT
    )
    assert result.returncode == 0, f"wiki {subcommand} --help failed:\n{result.stderr}"
    assert "Usage" in (result.stdout + result.stderr)


@pytest.mark.cli
@pytest.mark.parametrize("subcommand", _DRAFTS_SUBCOMMANDS, ids=_DRAFTS_SUBCOMMANDS)
def test_wiki_drafts_subcommand_renders_help(
    subcommand: str, lane: Lane, lilbee_data: Path
) -> None:
    """Every wiki drafts subcommand renders --help."""
    result = run_lilbee(
        lane,
        ["wiki", "drafts", subcommand, "--help"],
        data_dir=lilbee_data,
        timeout=CLI_FAST_TIMEOUT,
    )
    assert result.returncode == 0, f"wiki drafts {subcommand} --help failed:\n{result.stderr}"
    assert "Usage" in (result.stdout + result.stderr)


@pytest.mark.cli
def test_setup_crawler_renders_help(lane: Lane, lilbee_data: Path) -> None:
    """The setup crawler subcommand exists and renders --help."""
    result = run_lilbee(
        lane, ["setup", "crawler", "--help"], data_dir=lilbee_data, timeout=CLI_FAST_TIMEOUT
    )
    assert result.returncode == 0, result.stderr
    assert "Usage" in (result.stdout + result.stderr)
