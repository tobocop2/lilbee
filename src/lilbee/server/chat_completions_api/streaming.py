"""SSE encoder: ``data: {json}\\n\\n`` frames plus ``[DONE]`` terminator."""

from __future__ import annotations

from collections.abc import AsyncIterator

from lilbee.server.chat_completions_api.models import CompletionsStreamChunk


async def encode_completions_sse(
    chunks: AsyncIterator[CompletionsStreamChunk],
) -> AsyncIterator[bytes]:
    """Frame each chunk as an SSE ``data:`` event and append the ``[DONE]`` sentinel."""
    async for chunk in chunks:
        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n".encode()
    yield b"data: [DONE]\n\n"
