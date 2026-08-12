"""Claude Code MCP registration builder: lilbee's tools over stdio or streamable http."""

from __future__ import annotations

from typing import Any

from lilbee.app.agent_configs.merge import LILBEE_PROVIDER_KEY
from lilbee.app.endpoints import MCP_PATH

_MCP_CONTAINER_KEY = "mcpServers"
_CLI_COMMAND = "lilbee"
_MCP_SUBCOMMAND = "mcp"


def claude_stdio_config() -> dict[str, Any]:
    """Return the mcpServers block that has Claude Code spawn `lilbee mcp` itself.

    Nothing here expires, so this is the form to persist: it starts its own
    process on demand instead of pointing at a daemon that may not be up."""
    return {
        _MCP_CONTAINER_KEY: {
            LILBEE_PROVIDER_KEY: {"command": _CLI_COMMAND, "args": [_MCP_SUBCOMMAND]}
        }
    }


def claude_http_config(*, base_url: str, api_key: str) -> dict[str, Any]:
    """Return the mcpServers block pointing Claude Code at a running lilbee server.

    Sharing the daemon means sharing its already-warm models. The bearer token
    is minted per server boot, so a persisted copy needs refreshing after a
    restart."""
    return {
        _MCP_CONTAINER_KEY: {
            LILBEE_PROVIDER_KEY: {
                "type": "http",
                "url": f"{base_url}{MCP_PATH}",
                "headers": {"Authorization": f"Bearer {api_key}"},
            }
        }
    }
