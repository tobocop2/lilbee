"""SSE encoder for Anthropic's tagged event-stream framing."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any


async def encode_messages_sse(
    events: AsyncIterator[tuple[str, dict[str, Any]]],
) -> AsyncIterator[bytes]:
    """Frame each ``(event_name, payload)`` as an SSE ``event:``/``data:`` block.

    Anthropic streams end with the ``message_stop`` event; this encoder
    never emits a ``data: [DONE]`` terminator (that is OpenAI's protocol).
    """
    async for name, payload in events:
        yield f"event: {name}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()
