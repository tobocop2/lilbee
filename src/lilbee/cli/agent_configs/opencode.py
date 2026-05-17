"""opencode.json template (https://opencode.ai/docs/configuration/)."""

from __future__ import annotations

from typing import Any


def opencode_config(
    *,
    base_url: str,
    api_key: str,
    model_refs: list[str],
    mcp_command: list[str],
) -> dict[str, Any]:
    """Return the opencode.json block exposing lilbee as an OpenAI-compatible provider.

    *base_url* is the lilbee server's origin (e.g. ``http://127.0.0.1:8080``); the
    OpenAI ``/v1`` suffix is appended here so callers pass a single canonical URL.
    """
    return {
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
        "mcp": {
            "lilbee": {
                "type": "local",
                "command": list(mcp_command),
                "enabled": True,
            }
        },
    }
