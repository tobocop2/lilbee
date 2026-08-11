"""Claude Code launcher: Anthropic env wiring, MCP config, skill, exec."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import typer

from lilbee.cli.agent_configs import config_file
from lilbee.cli.agent_configs.claude import claude_mcp_config
from lilbee.cli.launchers.launcher import LILBEE_TOKEN_ENV_VAR, run_launcher
from lilbee.cli.launchers.server import LOOPBACK, client_chat_ctx
from lilbee.cli.launchers.setup_gate import confirm_first_run_setup
from lilbee.cli.launchers.skill_install import install_bundled_skill
from lilbee.core.config import cfg

_CLAUDE_INSTALL_HINT = (
    "claude binary not found on PATH. Install Claude Code from https://claude.com/claude-code."
)
_SETUP_MARKER_NAME = "claude-setup.json"
_MCP_CONFIG_NAME = "claude-mcp.json"
# The written MCP config references the token by env var (Claude Code expands
# ${...} at load), so the file never holds the literal.
_TOKEN_REF = "${" + LILBEE_TOKEN_ENV_VAR + "}"


def _mcp_config_path() -> Path:
    """Launcher-generated MCP config, kept inside lilbee's own data dir.

    Passed via ``--mcp-config`` instead of merging into ``~/.claude.json`` so a
    launch never rewrites Claude Code's own settings.
    """
    return cfg.data_dir / "launchers" / _MCP_CONFIG_NAME


def _claude_skill_dest() -> Path:
    return Path.home() / ".claude" / "skills" / "lilbee-mcp"


def _find_claude_binary() -> str | None:
    """The claude binary on PATH, or its two conventional install locations."""
    found = shutil.which("claude")
    if found:
        return found
    for candidate in (
        Path.home() / ".claude" / "local" / "claude",
        Path.home() / ".local" / "bin" / "claude",
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _print_setup_plan() -> None:
    """Tell the user exactly which files the first-run setup writes."""
    typer.secho("First-time Claude Code setup will write:", fg=typer.colors.CYAN)
    typer.echo(f"  - lilbee MCP config -> {_mcp_config_path()}")
    typer.echo(f"  - lilbee MCP skill -> {_claude_skill_dest()}")
    typer.echo(
        "Claude Code's own settings are not touched; the MCP config is passed "
        "per launch via --mcp-config. The token is referenced by env, never "
        "written as a literal. To undo, remove the two paths above."
    )


class ClaudeLauncher:
    """``Launcher`` implementation for Claude Code (https://claude.com/claude-code)."""

    name = "claude"
    install_hint = _CLAUDE_INSTALL_HINT

    def __init__(self, *, assume_yes: bool = False, include_mcp: bool = True) -> None:
        self._assume_yes = assume_yes
        self._include_mcp = include_mcp

    def find_binary(self) -> str | None:
        return _find_claude_binary()

    def prepare(
        self, *, token: str, port: int, model_refs: list[str]
    ) -> tuple[list[str], dict[str, str]]:
        from lilbee.catalog import agent_model_id

        base_url = f"http://{LOOPBACK}:{port}"
        model_id = agent_model_id(str(cfg.chat_model))
        env = {
            **os.environ,
            LILBEE_TOKEN_ENV_VAR: token,
            "ANTHROPIC_BASE_URL": base_url,
            # Claude Code sends the auth token as a bearer Authorization
            # header, which is exactly what lilbee's /v1 auth validates.
            "ANTHROPIC_AUTH_TOKEN": token,
            # Cleared so a real Anthropic key in the shell can't ride along to
            # the local server (or shadow the auth token).
            "ANTHROPIC_API_KEY": "",
            # Every tier Claude Code reaches for resolves to the one model
            # lilbee serves, subagents included.
            "ANTHROPIC_DEFAULT_OPUS_MODEL": model_id,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model_id,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model_id,
            "CLAUDE_CODE_SUBAGENT_MODEL": model_id,
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
            "DISABLE_ERROR_REPORTING": "1",
        }
        ctx = client_chat_ctx(port)
        if ctx is not None:
            # Compact before the local window overflows; Claude Code's default
            # assumes a frontier-sized context.
            env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(ctx)

        extra_args = ["--model", model_id]
        if self._include_mcp:
            if not confirm_first_run_setup(
                marker_name=_SETUP_MARKER_NAME,
                client_name="Claude Code",
                print_plan=_print_setup_plan,
                assume_yes=self._assume_yes,
            ):
                raise typer.Exit(0)
            block = claude_mcp_config(base_url=base_url, api_key=_TOKEN_REF)
            config_file.atomic_write_text(_mcp_config_path(), json.dumps(block, indent=2))
            extra_args.extend(["--mcp-config", str(_mcp_config_path())])
            install_bundled_skill(_claude_skill_dest())
        return (extra_args, env)


def claude_cmd(
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
        help="Inject lilbee's MCP search tool into Claude Code. Defaults to the "
        "agent_mcp_enabled config; --mcp/--no-mcp overrides it for this launch.",
    ),
) -> None:
    """Launch Claude Code with lilbee as its model backend."""
    include_mcp = cfg.agent_mcp_enabled if mcp is None else mcp
    run_launcher(ClaudeLauncher(assume_yes=yes, include_mcp=include_mcp))
