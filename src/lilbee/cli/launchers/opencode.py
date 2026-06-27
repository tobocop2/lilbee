"""Opencode launcher: wires the inline config, installs the skill, runs opencode."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import typer

from lilbee.cli.agent_configs import config_file
from lilbee.cli.agent_configs.merge import deep_merge, prune_lilbee
from lilbee.cli.agent_configs.opencode import opencode_config
from lilbee.cli.launchers.launcher import LILBEE_TOKEN_ENV_VAR, run_launcher
from lilbee.cli.launchers.server import LOOPBACK, served_chat_ctx
from lilbee.cli.launchers.skill_install import install_bundled_skill
from lilbee.core.config import cfg

_OPENCODE_INSTALL_HINT = "opencode binary not found on PATH. Install it from https://opencode.ai/."
_TOKEN_REF = "{env:" + LILBEE_TOKEN_ENV_VAR + "}"
_SETUP_MARKER_NAME = "opencode-setup.json"
_MCP_CONTAINER_KEY = "mcp"


def _opencode_config_path() -> Path:
    return Path.home() / ".config" / "opencode" / "opencode.json"


def _opencode_skill_dest() -> Path:
    return Path.home() / ".config" / "opencode" / "skills" / "lilbee-mcp"


def _setup_marker_path() -> Path:
    """lilbee's record that opencode setup already ran (so launch doesn't re-prompt)."""
    return cfg.data_dir / "launchers" / _SETUP_MARKER_NAME


def _setup_recorded() -> bool:
    return _setup_marker_path().exists()


def _record_setup() -> None:
    """Persist that the user accepted opencode setup; idempotent (atomic write)."""
    path = _setup_marker_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as tmp:
            tmp_name = tmp.name
            tmp.write(json.dumps({"accepted": True}).encode("utf-8"))
        os.replace(tmp_name, path)
    except BaseException:
        if tmp_name is not None:
            Path(tmp_name).unlink(missing_ok=True)
        raise


def _print_setup_plan() -> None:
    """Tell the user exactly which files the first-run setup writes."""
    typer.secho("First-time opencode setup will write:", fg=typer.colors.CYAN)
    typer.echo(f"  - lilbee provider + MCP entry -> {_opencode_config_path()}")
    typer.echo(f"  - lilbee MCP skill -> {_opencode_skill_dest()}")
    typer.echo(
        "Only the `lilbee` keys and the active model are written; your other "
        "providers and settings are preserved. The token is referenced by env, "
        "never written as a literal. To undo, remove the `lilbee` entries and the skill dir."
    )


def _is_interactive() -> bool:
    """True when stdin is a TTY, so a confirmation prompt can be answered."""
    return sys.stdin.isatty()


def _confirm_setup(assume_yes: bool) -> bool:
    """Prompt before the first opencode setup; True means proceed.

    Skipped when already recorded, when *assume_yes* is set, or when stdin is
    not a TTY (scripts/CI: invoking ``launch opencode`` is the consent there).
    The choice is remembered so later launches don't re-prompt.
    """
    if _setup_recorded():
        return True
    _print_setup_plan()
    if assume_yes or not _is_interactive():
        _record_setup()
        return True
    if not typer.confirm("Proceed with opencode setup?", default=True):
        typer.secho("Skipped opencode setup.", fg=typer.colors.YELLOW)
        return False
    _record_setup()
    return True


class OpencodeLauncher:
    """``Launcher`` implementation for opencode (https://opencode.ai/)."""

    name = "opencode"
    install_hint = _OPENCODE_INSTALL_HINT

    def __init__(self, *, assume_yes: bool = False, include_mcp: bool = True) -> None:
        self._assume_yes = assume_yes
        self._include_mcp = include_mcp

    def find_binary(self) -> str | None:
        return shutil.which("opencode")

    def prepare(
        self, *, token: str, port: int, model_refs: list[str]
    ) -> tuple[list[str], dict[str, str]]:
        if not _confirm_setup(self._assume_yes):
            raise typer.Exit(0)
        # The lilbee-mcp guidance skill only helps when the MCP tool is wired in;
        # skip it when MCP is disabled (a previously-installed skill is left alone).
        if self._include_mcp:
            install_bundled_skill(_opencode_skill_dest())
        # The token is referenced via {env:...}; opencode expands it at load, so the
        # written config never holds the literal. The launcher sets it in the env.
        block = opencode_config(
            base_url=f"http://{LOOPBACK}:{port}",
            api_key=_TOKEN_REF,
            model_refs=model_refs,
            chat_ctx=served_chat_ctx(port),
            default_ref=str(cfg.chat_model),
            include_mcp=self._include_mcp,
        )
        config = config_file.load_config_dict(
            _opencode_config_path(),
            parse=json.loads,
            parse_error=json.JSONDecodeError,
            label="opencode config (opencode.json)",
        )
        deep_merge(config, block)
        if not self._include_mcp:
            prune_lilbee(config, _MCP_CONTAINER_KEY)
        config_file.atomic_write_text(_opencode_config_path(), json.dumps(config, indent=2))
        return ([], {**os.environ, LILBEE_TOKEN_ENV_VAR: token})


def opencode_cmd(
    yes: bool = typer.Option(
        False,
        "--no-prompt",
        "--yes",
        "-y",
        help="Proceed with first-run setup without the interactive prompt (for scripts/CI).",
    ),
    mcp: bool | None = typer.Option(
        None,
        "--mcp/--no-mcp",
        help="Inject lilbee's MCP search tool into opencode. Defaults to the "
        "agent_mcp_enabled config; --mcp/--no-mcp overrides it for this launch.",
    ),
) -> None:
    """Launch opencode with lilbee as its model provider."""
    include_mcp = cfg.agent_mcp_enabled if mcp is None else mcp
    run_launcher(OpencodeLauncher(assume_yes=yes, include_mcp=include_mcp))
