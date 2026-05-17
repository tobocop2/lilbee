"""opencode.json builder."""

from __future__ import annotations

from typing import Any


def opencode_config(
    *,
    base_url: str,
    api_key: str,
    model_refs: list[str],
    mcp_command: list[str] | None = None,
) -> dict[str, Any]:
    """Return the opencode.json block wiring lilbee as a provider.

    ``base_url`` is the lilbee server origin (e.g. ``http://127.0.0.1:8080``);
    the chat-completions ``/v1`` suffix is appended here so callers pass a
    single canonical URL. When ``mcp_command`` is given, the block also
    registers a lilbee MCP server that runs that command.
    """
    block: dict[str, Any] = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "lilbee": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "lilbee",
                "options": {
                    "baseURL": f"{base_url}/v1",
                    "apiKey": api_key,
                },
                "models": {ref: {"name": ref} for ref in sorted(model_refs)},
            }
        },
    }
    if mcp_command is not None:
        block["mcp"] = {
            "lilbee": {
                "type": "local",
                "command": list(mcp_command),
                "enabled": True,
            }
        }
    return block
