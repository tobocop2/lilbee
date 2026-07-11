"""SSE encoder: ``data: {json}\\n\\n`` frames plus ``[DONE]`` terminator."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

from lilbee.server.chat_completions_api.models import CompletionsStreamChunk

# Cadence for SSE comment frames emitted when no chat token has arrived. Keeps
# clients (notably opencode) from tripping their idle-stream timeout during
# slow first-token latency on local models, which would otherwise fire a
# retry storm against the chat lock.
_KEEPALIVE_INTERVAL_S = 5.0
_KEEPALIVE_FRAME = b": keepalive\n\n"


async def encode_completions_sse(
    chunks: AsyncIterator[CompletionsStreamChunk],
) -> AsyncIterator[bytes]:
    """Frame each chunk as an SSE ``data:`` event and append the ``[DONE]`` sentinel.

    While the upstream generator is slow to produce the next chunk, the
    encoder yields ``: keepalive`` SSE comment frames every
    ``_KEEPALIVE_INTERVAL_S`` seconds so clients see traffic on the wire and
    don't trip their idle-stream timeout. The in-progress ``__anext__`` task
    is held across keepalive emissions; only on real completion or error is
    the task replaced.
    """
    iterator = chunks.__aiter__()
    pending = asyncio.ensure_future(iterator.__anext__())
    try:
        while True:
            done, _ = await asyncio.wait({pending}, timeout=_KEEPALIVE_INTERVAL_S)
            if pending not in done:
                yield _KEEPALIVE_FRAME
                continue
            try:
                chunk = pending.result()
            except StopAsyncIteration:
                break
            yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n".encode()
            pending = asyncio.ensure_future(iterator.__anext__())
    finally:
        if not pending.done():
            pending.cancel()
            with contextlib.suppress(BaseException):
                await pending
    yield b"data: [DONE]\n\n"
