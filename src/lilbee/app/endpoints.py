"""Paths the lilbee server serves its agent-facing surfaces at.

The server's own mounts and the client config blocks lilbee hands out read
these, so a path cannot drift between what lilbee serves and what it tells a
client to call.
"""

from __future__ import annotations

MCP_PATH = "/mcp"
"""Streamable-http MCP endpoint."""

OPENAI_PATH = "/v1"
"""OpenAI-compatible chat completions and model list."""
