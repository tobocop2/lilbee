"""opencode.json template (https://opencode.ai/docs/configuration/)."""

from __future__ import annotations

from typing import Any

LILBEE_PRIMING = """# Working with lilbee

You have access to a local knowledge-base via the `lilbee` MCP server. The
tools to know:

- **`lilbee_search(query, top_k=5)`**: returns relevant chunks from the user's
  indexed library with their source file paths and line ranges. Call this
  before answering any question that might be in the library (their code,
  notes, docs, project files). Prefer it to guessing.
- **`lilbee_add(paths)`**: adds files or directories to the library.
- **`lilbee_status()`**: shows what is currently indexed and the active model
  configuration.

When you cite information returned by `lilbee_search`, include the file path
and line range so the user can verify. If the search returns nothing relevant,
say so explicitly rather than inventing an answer.
"""


def opencode_config(
    *,
    base_url: str,
    api_key: str,
    model_refs: list[str],
    mcp_command: list[str],
    instructions_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Return the opencode.json block wiring lilbee as a provider plus MCP server.

    *base_url* is the lilbee server's origin (e.g. ``http://127.0.0.1:8080``); the
    chat-completions ``/v1`` suffix is appended here so callers pass a single
    canonical URL. *instructions_paths* lists files opencode should load as
    extra instructions; the launcher uses this to point at a priming
    ``AGENTS.md`` so opencode sessions know about lilbee from turn one.
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
        "mcp": {
            "lilbee": {
                "type": "local",
                "command": list(mcp_command),
                "enabled": True,
            }
        },
    }
    if instructions_paths:
        block["instructions"] = list(instructions_paths)
    return block
