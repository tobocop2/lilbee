"""`lilbee agent-config <client>`, print a paste-ready config block."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import typer

from lilbee.app.services import get_services
from lilbee.catalog.types import ModelTask
from lilbee.cli.agent_configs import opencode
from lilbee.cli.app import console
from lilbee.core.config import cfg
from lilbee.server.auth import server_json_path

agent_config_app = typer.Typer(help="Print a paste-ready config block for an AI client.")

_LOCAL_HOST = "127.0.0.1"
_MCP_COMMAND = ["lilbee", "mcp"]
_SERVE_HINT = "Start `lilbee serve --port 8080` first, then re-run this command."


def _server_session() -> tuple[str, int] | None:
    """Return `(token, port)` for the running server, or `None` if none is running."""
    session_path = server_json_path()
    port_path = cfg.data_dir / "server.port"
    if not session_path.exists() or not port_path.exists():
        return None
    try:
        data = json.loads(session_path.read_text())
        token = data.get("token")
        port = int(port_path.read_text().strip())
    except (json.JSONDecodeError, OSError, ValueError):
        return None
    if not isinstance(token, str) or not token:
        return None
    return token, port


def _chat_model_refs() -> list[str]:
    registry = get_services().registry
    return sorted(m.ref for m in registry.list_installed() if m.task == ModelTask.CHAT)


_JsonBuilder = Callable[..., dict[str, Any]]
_TextBuilder = Callable[..., str]


def _emit_block(builder: _JsonBuilder | _TextBuilder, **kwargs: Any) -> None:
    session = _server_session()
    if session is None:
        typer.secho(_SERVE_HINT, err=True, fg=typer.colors.RED)
        raise typer.Exit(1)
    token, port = session
    block = builder(
        base_url=f"http://{_LOCAL_HOST}:{port}",
        api_key=token,
        model_refs=_chat_model_refs(),
        **kwargs,
    )
    if isinstance(block, str):
        console.print(block, end="")
    else:
        console.print(json.dumps(block, indent=2))


@agent_config_app.command("opencode")
def _opencode_cmd() -> None:
    """Print an opencode.json block (OpenAI-compatible provider + MCP server)."""
    _emit_block(opencode.opencode_config, mcp_command=_MCP_COMMAND)
