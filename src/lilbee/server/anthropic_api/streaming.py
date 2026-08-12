"""SSE encoder for the Anthropic Messages stream format."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from lilbee.server.anthropic_api.models import AnthropicEventType
from lilbee.server.handlers.sse import frames_with_keepalive

# Cadence for ping events while no token has arrived (same rationale as the
# completions keepalive comment frames).
_KEEPALIVE_INTERVAL_S = 5.0
_PING_FRAME = b'event: ping\ndata: {"type": "ping"}\n\n'


def encode_anthropic_event(event_type: AnthropicEventType, payload: dict[str, Any]) -> bytes:
    """Frame one event as Anthropic SSE: ``event: <type>`` + ``data: <json>``."""
    body = json.dumps(payload, separators=(",", ":"))
    return f"event: {event_type}\ndata: {body}\n\n".encode()


async def encode_anthropic_sse(
    events: AsyncIterator[tuple[AnthropicEventType, dict[str, Any]]],
) -> AsyncIterator[bytes]:
    """Frame each ``(type, payload)`` pair as SSE, pinging while upstream is slow.

    There is no ``[DONE]`` sentinel in the Anthropic format; the translator's
    trailing ``message_stop`` terminates the stream.
    """

    async def _frames() -> AsyncIterator[bytes]:
        async for event_type, payload in events:
            yield encode_anthropic_event(event_type, payload)

    async for frame in frames_with_keepalive(
        _frames(), keepalive=_PING_FRAME, interval_s=_KEEPALIVE_INTERVAL_S
    ):
        yield frame
