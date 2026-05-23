"""`lilbee agent-config <client>`, print a paste-ready config block."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import typer

from lilbee.cli.agent_configs import litellm, opencode
from lilbee.cli.launchers.server import (
    LOOPBACK,
    installed_chat_model_refs,
    running_server_session,
)

agent_config_app = typer.Typer(help="Print a paste-ready config block for an AI client.")

_SERVE_HINT = "Start `lilbee serve --port 8080` first, then re-run this command."


_JsonBuilder = Callable[..., dict[str, Any]]
_TextBuilder = Callable[..., str]


def _emit_block(builder: _JsonBuilder | _TextBuilder, **kwargs: Any) -> None:
    session = running_server_session()
    if session is None:
        typer.secho(_SERVE_HINT, err=True, fg=typer.colors.RED)
        raise typer.Exit(1)
    token, port = session
    block = builder(
        base_url=f"http://{LOOPBACK}:{port}",
        api_key=token,
        model_refs=installed_chat_model_refs(),
        **kwargs,
    )
    if isinstance(block, str):
        # Use typer.echo (no Rich word-wrap) so YAML stays parseable when
        # piped to a file or wrapped in narrow test terminals.
        typer.echo(block, nl=False)
    else:
        typer.echo(json.dumps(block, indent=2))


@agent_config_app.command("opencode")
def _opencode_cmd() -> None:
    """Print an opencode.json block (OpenAI-compatible provider + MCP server).

    The MCP server points at the running daemon's streamable-http endpoint so
    opencode shares its warm models instead of spawning a second process.
    """
    _emit_block(opencode.opencode_config, mcp_remote=True)


@agent_config_app.command("litellm")
def _litellm_cmd() -> None:
    """Print a LiteLLM `config.yaml` snippet routing model names to lilbee."""
    _emit_block(litellm.litellm_config)
