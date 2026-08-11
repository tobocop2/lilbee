"""SSE encoder for the Anthropic Messages stream format."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from lilbee.server.anthropic_api.models import AnthropicEventType

log = logging.getLogger(__name__)

# Cadence for ping events while no token has arrived; keeps clients from
# tripping their idle-stream timeout during slow first-token latency on local
# models (same rationale as the completions keepalive).
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
    iterator = events.__aiter__()
    pending = asyncio.ensure_future(iterator.__anext__())
    try:
        while True:
            done, _ = await asyncio.wait({pending}, timeout=_KEEPALIVE_INTERVAL_S)
            if pending not in done:
                yield _PING_FRAME
                continue
            try:
                event_type, payload = pending.result()
            except StopAsyncIteration:
                break
            yield encode_anthropic_event(event_type, payload)
            pending = asyncio.ensure_future(iterator.__anext__())
    finally:
        if not pending.done():
            pending.cancel()
            try:
                await pending
            except asyncio.CancelledError:
                # Expected: the future we just cancelled. By name, not
                # BaseException, which also swallowed a Ctrl-C landing here.
                pass
            except Exception:
                # The upstream failed as it was torn down. The request is
                # already unwinding, so record it rather than mask the unwind.
                log.debug("upstream anthropic stream errored during cleanup", exc_info=True)
