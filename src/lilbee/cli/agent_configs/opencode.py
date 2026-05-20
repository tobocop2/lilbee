"""opencode.json builder."""

from __future__ import annotations

import re
from typing import Any

_GGUF_SUFFIX = ".gguf"
_NATIVE_REF_PARTS = 3
"""Number of slash-separated segments in a native GGUF ref: ``<org>/<repo>/<file>``."""

_QUANT_TRAILER = re.compile(
    r"[-.](?P<quant>I?Q\d+(?:_[A-Z0-9]+)*|F16|F32|BF16)$",
    re.IGNORECASE,
)


def display_name(ref: str) -> str:
    """Render a chat-model ref as a short label for opencode's model picker.

    A native GGUF ref has the shape ``<org>/<repo>/<filename>.gguf``. The
    picker showing the full path is noisy because the filename and repo
    name are largely redundant. This helper extracts the filename, strips
    the ``.gguf`` extension, and turns ``Model-Q4_K_M`` (or
    ``Model.Q8_0``) into ``Model Q4_K_M``. Non-native refs (Ollama, hosted
    providers) and unrecognised filename shapes pass through unchanged so
    the picker still shows a meaningful identifier.
    """
    parts = ref.split("/")
    if len(parts) != _NATIVE_REF_PARTS or not parts[2].endswith(_GGUF_SUFFIX):
        return ref
    stem = parts[2].removesuffix(_GGUF_SUFFIX)
    match = _QUANT_TRAILER.search(stem)
    if match is None:
        return stem
    return f"{stem[: match.start()]} {match.group('quant')}"


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
                "models": {ref: {"name": display_name(ref)} for ref in sorted(model_refs)},
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
