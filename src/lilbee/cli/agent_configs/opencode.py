"""opencode.json builder."""

from __future__ import annotations

import re
from typing import Any

_GGUF_SUFFIX = ".gguf"
_NATIVE_REF_PARTS = 3
"""Number of slash-separated segments in a native GGUF ref: ``<org>/<repo>/<file>``."""

_OUTPUT_TOKEN_LIMIT = 8192
"""Per-response output cap reported to opencode (it reserves this from the context)."""

_MCP_TIMEOUT_MS = 120_000
"""Remote-MCP request timeout. opencode defaults to 5000 ms, which the first
``lilbee_search`` can exceed while the embedding model cold-loads."""

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


def _model_entry(ref: str, chat_ctx: int | None) -> dict[str, Any]:
    """One opencode model entry, carrying the served window so opencode trims to it.

    opencode's schema requires both ``limit.context`` and ``limit.output`` once a
    ``limit`` is present, so when the window is known emit both -- a limit with only
    ``context`` makes opencode reject the whole config and fail to start.
    """
    entry: dict[str, Any] = {"name": display_name(ref)}
    if chat_ctx is not None:
        entry["limit"] = {
            "context": chat_ctx,
            "output": max(1, min(chat_ctx // 2, _OUTPUT_TOKEN_LIMIT)),
        }
    return entry


def opencode_config(
    *,
    base_url: str,
    api_key: str,
    model_refs: list[str],
    chat_ctx: int | None = None,
    default_ref: str | None = None,
) -> dict[str, Any]:
    """Return the opencode.json block wiring lilbee as a provider.

    ``base_url`` is the lilbee server origin (e.g. ``http://127.0.0.1:8080``);
    the chat-completions ``/v1`` and MCP ``/mcp`` suffixes are appended here.
    The MCP block points opencode at the daemon's streamable-http endpoint with
    the bearer token, so retrieval shares the daemon's warm models instead of
    spawning a second process. ``chat_ctx`` is the active model's served window;
    when set it becomes each model's ``limit.context`` so opencode trims history
    to fit instead of overflowing on a long agentic session. ``default_ref``
    pins opencode's startup model via the top-level ``model`` key
    (``provider/model-id`` form).
    """
    config: dict[str, Any] = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "lilbee": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "lilbee",
                "options": {
                    "baseURL": f"{base_url}/v1",
                    "apiKey": api_key,
                },
                "models": {ref: _model_entry(ref, chat_ctx) for ref in sorted(model_refs)},
            }
        },
        "mcp": {
            "lilbee": {
                "type": "remote",
                "url": f"{base_url}/mcp",
                "enabled": True,
                "headers": {"Authorization": f"Bearer {api_key}"},
                "timeout": _MCP_TIMEOUT_MS,
            }
        },
    }
    if default_ref is not None:
        config["model"] = f"lilbee/{default_ref}"
    return config
