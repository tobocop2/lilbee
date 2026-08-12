"""SSE encoder: ``data: {json}\\n\\n`` frames plus ``[DONE]`` terminator."""

from __future__ import annotations

from collections.abc import AsyncIterator

from lilbee.server.chat_completions_api.models import CompletionsStreamChunk
from lilbee.server.handlers.sse import frames_with_keepalive

# Cadence for SSE comment frames emitted when no chat token has arrived. Keeps
# clients (notably opencode) from tripping their idle-stream timeout during
# slow first-token latency on local models.
_KEEPALIVE_INTERVAL_S = 5.0
_KEEPALIVE_FRAME = b": keepalive\n\n"


async def encode_completions_sse(
    chunks: AsyncIterator[CompletionsStreamChunk],
) -> AsyncIterator[bytes]:
    """Frame each chunk as an SSE ``data:`` event and append the ``[DONE]`` sentinel."""

    async def _frames() -> AsyncIterator[bytes]:
        async for chunk in chunks:
            yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n".encode()

    async for frame in frames_with_keepalive(
        _frames(), keepalive=_KEEPALIVE_FRAME, interval_s=_KEEPALIVE_INTERVAL_S
    ):
        yield frame
    yield b"data: [DONE]\n\n"
