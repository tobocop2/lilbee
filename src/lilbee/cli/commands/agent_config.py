"""`lilbee agent-config <client>`, print a paste-ready config block."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

from lilbee.cli.agent_configs import litellm, opencode
from lilbee.cli.app import apply_overrides, data_dir_option, global_option
from lilbee.cli.launchers.server import (
    LOOPBACK,
    installed_chat_model_refs,
    running_server_session,
)
from lilbee.core.config import cfg
from lilbee.providers.model_ref import with_configured_remote_chat

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
        # Include a remote-configured chat model the native registry lacks, so the
        # emitted config lists the model lilbee serves (matching launch + /v1/models).
        model_refs=with_configured_remote_chat(installed_chat_model_refs(), cfg.chat_model),
        **kwargs,
    )
    if isinstance(block, str):
        # Use typer.echo (no Rich word-wrap) so YAML stays parseable when
        # piped to a file or wrapped in narrow test terminals.
        typer.echo(block, nl=False)
    else:
        typer.echo(json.dumps(block, indent=2))


@agent_config_app.command("opencode")
def _opencode_cmd(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Print an opencode.json block (OpenAI-compatible provider + MCP server)."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    _emit_block(opencode.opencode_config)


@agent_config_app.command("litellm")
def _litellm_cmd(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Print a LiteLLM `config.yaml` snippet routing model names to lilbee."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    _emit_block(litellm.litellm_config)
