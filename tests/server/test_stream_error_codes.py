"""The RAG chat stream must carry the same typed error codes as its non-stream sibling.

The non-streaming /api/chat maps a dispatch failure to a distinct status, but
the streaming path flattened every failure into one code-less message, so a
client could not tell "pull the model" from "shorten the prompt".
"""

from __future__ import annotations

import asyncio
import json

import pytest

from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.server.chat_dispatch.concurrency import ChatSlotGuard
from lilbee.server.chat_dispatch.dispatch import (
    ModelDoesNotSupportToolsError,
    ModelNotFoundError,
)
from lilbee.server.routes.search import _gated_stream


def _payload(frame: str) -> dict:
    """Parse the data: line out of an SSE frame."""
    for line in frame.splitlines():
        if line.startswith("data: "):
            return json.loads(line.removeprefix("data: "))
    raise AssertionError(f"no data line in {frame!r}")


async def _collect(exc: BaseException) -> dict:
    async def _boom():
        raise exc
        yield  # pragma: no cover -- makes this an async generator

    frames = [frame async for frame in _gated_stream(_boom(), ChatSlotGuard())]
    assert len(frames) == 1
    return _payload(frames[0])


class TestStreamErrorCodes:
    @pytest.mark.parametrize(
        ("exc", "code"),
        [
            (ModelNotFoundError("nope"), "model_not_found"),
            (ModelDoesNotSupportToolsError("nope"), "model_does_not_support_tools"),
            (
                ProviderError("too long", kind=ProviderErrorKind.CONTEXT_OVERFLOW),
                "context_length_exceeded",
            ),
        ],
    )
    async def test_a_typed_dispatch_failure_carries_its_code(self, exc, code) -> None:
        assert (await _collect(exc))["code"] == code

    async def test_a_backend_failure_stays_generic(self) -> None:
        """Same redaction as the completions surface: the fleet's message names
        loopback ports and engine paths, so it is logged rather than streamed."""
        payload = await _collect(
            ProviderError("bind 127.0.0.1:8137 failed", kind=ProviderErrorKind.CONNECTION)
        )
        assert "127.0.0.1" not in payload["message"]

    async def test_an_unclassified_failure_stays_code_less(self) -> None:
        payload = await _collect(RuntimeError("something else"))
        assert "code" not in payload


class TestStalledConsumerStopsTheProducer:
    """Chat and RAG tokens are in the always-deliver class, so a generation
    streaming to a client that has stopped reading filled the queue with
    events nothing was permitted to shed, and it grew until the generation
    ended."""

    async def test_a_full_undroppable_queue_cancels_the_producer(self) -> None:
        from lilbee.server.handlers.sse import SseStream

        sse = SseStream()
        for index in range(sse.queue._max_events + 1):
            sse.put_threadsafe(f"token {index}")
        # put_threadsafe hands the work to the loop; let it run.
        await asyncio.sleep(0)
        assert sse.cancel.is_set()
        assert sse.queue.qsize() <= sse.queue._max_events + 1

    async def test_a_reading_consumer_is_never_treated_as_stalled(self) -> None:
        from lilbee.server.handlers.sse import SseStream

        sse = SseStream()
        for index in range(sse.queue._max_events * 2):
            sse.put_threadsafe(f"token {index}")
            await asyncio.sleep(0)
            sse.queue.get_nowait()
        assert not sse.cancel.is_set()


class TestTerminalFrameIsSharedByEveryStream:
    """Five streaming handlers ended with the same five lines, and the copies
    had drifted: the crawler-setup one skipped the cancel check, so it emitted
    a done frame to a client that had already disconnected."""

    async def _finished(self, result=None, exc=None):
        async def _run():
            if exc is not None:
                raise exc
            return result

        task = asyncio.ensure_future(_run())
        await asyncio.sleep(0)
        return task

    async def test_a_cancelled_stream_emits_nothing(self) -> None:
        from lilbee.server.handlers.sse import SseStream

        sse = SseStream()
        task = await self._finished(result={"ok": True})
        sse.cancel.set()
        assert sse.terminal_frame(task, lambda r: r) is None

    async def test_a_completed_producer_emits_its_done_payload(self) -> None:
        from lilbee.server.handlers.sse import SseStream

        sse = SseStream()
        task = await self._finished(result={"files_written": ["a.md"]})
        frame = sse.terminal_frame(task, lambda r: r)
        assert frame is not None
        assert "event: done" in frame
        assert "a.md" in frame

    async def test_a_failed_producer_emits_an_error(self) -> None:
        from lilbee.server.handlers.sse import SseStream

        sse = SseStream()
        task = await self._finished(exc=RuntimeError("network unreachable"))
        frame = sse.terminal_frame(task, lambda r: r)
        assert frame is not None
        assert "event: error" in frame
        assert "network unreachable" in frame

    async def test_an_unfinished_producer_emits_nothing(self) -> None:
        from lilbee.server.handlers.sse import SseStream

        sse = SseStream()
        pending = asyncio.ensure_future(asyncio.sleep(60))
        try:
            assert sse.terminal_frame(pending, lambda r: r) is None
        finally:
            pending.cancel()
