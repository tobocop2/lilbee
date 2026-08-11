"""Claude Code MCP config builder (``--mcp-config`` / ``.mcp.json`` shape)."""

from __future__ import annotations

from typing import Any


def claude_mcp_config(
    *,
    base_url: str,
    api_key: str,
    model_refs: list[str] | None = None,
) -> dict[str, Any]:
    """Return the Claude Code MCP block wiring lilbee's ``/mcp`` endpoint.

    Claude Code is not an OpenAI-compatible-provider consumer, so unlike the
    opencode/hermes builders there is no provider or model section here --
    models reach Claude Code through the launcher's ``ANTHROPIC_*`` env vars
    and the ``/v1/messages`` route. ``model_refs`` is accepted for builder
    signature parity with the other clients and unused.

    ``api_key`` lands in the header verbatim; the launcher passes the
    ``${LILBEE_TOKEN}`` env reference (Claude Code expands it at load) so the
    written file never holds the literal token.
    """
    del model_refs
    return {
        "mcpServers": {
            "lilbee": {
                "type": "http",
                "url": f"{base_url}/mcp",
                "headers": {"Authorization": f"Bearer {api_key}"},
            }
        }
    }
