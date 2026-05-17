"""OpenAI SSE encoder: ``data: {json}\\n\\n`` frames plus ``[DONE]`` terminator."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any


async def encode_completions_sse(
    chunks: AsyncIterator[dict[str, Any]],
) -> AsyncIterator[bytes]:
    """Frame each chunk as an SSE ``data:`` event and append the ``[DONE]`` sentinel."""
    async for chunk in chunks:
        yield f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n".encode()
    yield b"data: [DONE]\n\n"
