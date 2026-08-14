"""`lilbee agent-config <client>`, print a paste-ready config block."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from lilbee.app.agent_configs.document import (
    AgentClient,
    AgentConfigDocument,
    ConfigFormat,
    build_agent_config,
    client_serves_models,
)
from lilbee.app.agent_configs.litellm import litellm_config
from lilbee.app.agent_configs.window import AGENT_CHAT_CTX_FLOOR
from lilbee.app.models import installed_chat_model_refs
from lilbee.cli.app import apply_overrides, data_dir_option, global_option
from lilbee.cli.launchers.hermes_mcp import MCP_EXTRA_HINT
from lilbee.cli.launchers.server import (
    LOOPBACK,
    client_chat_ctx,
    running_server_session,
)
from lilbee.core.config import cfg
from lilbee.providers.model_ref import with_configured_remote_chat

agent_config_app = typer.Typer(help="Print a paste-ready config block for an AI client.")

_SERVE_HINT = "Start `lilbee serve --port 8080` first, then re-run this command."

_STDIO_HINT = (
    "To have the client start lilbee itself instead of using the running "
    "server, register this block instead:"
)


def _session_or_exit() -> tuple[str, int]:
    """The running server's ``(token, port)``; exits non-zero when none is up."""
    session = running_server_session()
    if session is None:
        typer.secho(_SERVE_HINT, err=True, fg=typer.colors.RED)
        raise typer.Exit(1)
    return session


def _served_chat_models() -> list[str]:
    """Chat refs to advertise: the installed ones plus a remote-configured model."""
    return with_configured_remote_chat(installed_chat_model_refs(), cfg.chat_model)


def _warn_on_small_agent_window(chat_ctx: int | None) -> None:
    """Say when the served window cannot hold an agent's first turn, with the remedy.

    ``lilbee launch`` sizes the window itself; this paste path runs against a
    server the user started, whose window is fixed at boot, so the printed
    config would otherwise carry a window the first message overflows. An
    unknown window (no chat engine yet) stays silent.
    """
    if chat_ctx is None or chat_ctx >= AGENT_CHAT_CTX_FLOOR:
        return
    typer.secho(
        f"Warning: the server serves a {chat_ctx:,}-token context window, but an "
        f"agent's first turn (system prompt plus tool schemas) needs about "
        f"{AGENT_CHAT_CTX_FLOOR:,}, so the first message can overflow. To raise it: "
        f"stop the server, run 'lilbee engine stop', set chat_n_ctx_target to "
        f"{AGENT_CHAT_CTX_FLOOR} (or start with "
        f"LILBEE_CHAT_N_CTX_TARGET={AGENT_CHAT_CTX_FLOOR}), then run 'lilbee serve' "
        "again. If the window stays small, the model's trained context or device "
        "memory is the limit; use a longer-context model or a smaller quantization.",
        err=True,
        fg=typer.colors.YELLOW,
    )


def _build(client: AgentClient) -> AgentConfigDocument:
    """Build *client*'s document from the running server's port and token."""
    token, port = _session_or_exit()
    serves_models = client_serves_models(client)
    chat_ctx = client_chat_ctx(port) if serves_models else None
    _warn_on_small_agent_window(chat_ctx)
    return build_agent_config(
        client,
        base_url=f"http://{LOOPBACK}:{port}",
        api_key=token,
        model_refs=_served_chat_models() if serves_models else None,
        # Match the launchers: pin the served model as default and pass the
        # context window, so the pasted config opens on a lilbee model and trims
        # history to the right limit.
        default_ref=str(cfg.chat_model) if serves_models else None,
        chat_ctx=chat_ctx,
    )


def _emit(document: AgentConfigDocument) -> None:
    """Print the block on stdout, with any client-specific note on stderr."""
    if document.format is ConfigFormat.YAML:
        # Use typer.echo (no Rich word-wrap) so YAML stays parseable when
        # piped to a file or wrapped in narrow test terminals.
        typer.echo(document.content, nl=False)
    else:
        typer.echo(json.dumps(document.config, indent=2))
    if document.stdio_config is not None:
        typer.secho(_STDIO_HINT, err=True, fg=typer.colors.YELLOW)
        typer.echo(json.dumps(document.stdio_config, indent=2), err=True)
    if document.client is AgentClient.HERMES:
        # Parity with `lilbee launch hermes`: the MCP block only works once
        # hermes has its `mcp` extra. The paste path can't install it, so
        # surface the same hint (to stderr, keeping the YAML pipe-clean).
        typer.secho(MCP_EXTRA_HINT, err=True, fg=typer.colors.YELLOW)


@agent_config_app.command("claude")
def _claude_cmd(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Print a Claude Code mcpServers block registering lilbee's MCP tools."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    _emit(_build(AgentClient.CLAUDE))


@agent_config_app.command("opencode")
def _opencode_cmd(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Print an opencode.json block (OpenAI-compatible provider + MCP server)."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    _emit(_build(AgentClient.OPENCODE))


@agent_config_app.command("hermes")
def _hermes_cmd(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Print a hermes config.yaml block (OpenAI-compatible provider + MCP server)."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    _emit(_build(AgentClient.HERMES))


@agent_config_app.command("litellm")
def _litellm_cmd(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Print a LiteLLM `config.yaml` snippet routing model names to lilbee."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    token, port = _session_or_exit()
    snippet = litellm_config(
        base_url=f"http://{LOOPBACK}:{port}",
        api_key=token,
        model_refs=_served_chat_models(),
    )
    typer.echo(snippet, nl=False)
