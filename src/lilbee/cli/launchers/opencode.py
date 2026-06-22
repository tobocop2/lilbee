"""Opencode launcher: wires the inline config, installs the skill, runs opencode."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from importlib import resources
from pathlib import Path

import typer

from lilbee.cli.agent_configs.opencode import opencode_config
from lilbee.cli.launchers.launcher import run_launcher
from lilbee.cli.launchers.server import LOOPBACK, served_chat_ctx
from lilbee.core.config import cfg

_OPENCODE_INSTALL_HINT = "opencode binary not found on PATH. Install it from https://opencode.ai/."
_SKILL_PACKAGE = "lilbee.skills.lilbee_mcp"
_OPENCODE_CONFIG_ENV_VAR = "OPENCODE_CONFIG_CONTENT"
_SETUP_MARKER_NAME = "opencode-setup.json"


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
    typer.echo(f"  - lilbee MCP skill -> {_opencode_skill_dest()}")
    typer.echo(
        "The write is skipped if already present; everything else (provider, "
        "MCP, model pin) is passed to opencode per session and persists nowhere. "
        "To undo, delete the skill dir."
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


def _install_lilbee_skill() -> Path | None:
    """Copy the bundled lilbee MCP skill into opencode's global skills dir.

    Skip when the destination already exists so user customizations are
    preserved. Returns the destination path on a fresh copy, else ``None``.
    """
    dest = _opencode_skill_dest()
    if dest.exists():
        return None
    source = resources.files(_SKILL_PACKAGE)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Build in a temp dir and atomically rename, so a failed/partial copy never
    # leaves a half-written skill dir that exists() would then skip forever.
    staging = Path(tempfile.mkdtemp(dir=dest.parent, prefix=".lilbee-mcp-"))
    try:
        for entry in source.iterdir():
            if entry.is_file() and not entry.name.startswith("__"):
                (staging / entry.name).write_bytes(entry.read_bytes())
        os.replace(staging, dest)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return dest


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
            _install_lilbee_skill()
        # The block carries the session's ephemeral port and token; never persist
        # it into user config.
        block = opencode_config(
            base_url=f"http://{LOOPBACK}:{port}",
            api_key=token,
            model_refs=model_refs,
            chat_ctx=served_chat_ctx(port),
            default_ref=str(cfg.chat_model),
            include_mcp=self._include_mcp,
        )
        env = {**os.environ, _OPENCODE_CONFIG_ENV_VAR: json.dumps(block)}
        return ([], env)


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
