"""Tests for the OpenAI SSE encoder."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator

from lilbee.server.chat_completions_api.models import (
    CompletionsStreamChoice,
    CompletionsStreamChunk,
    CompletionsStreamDelta,
)
from lilbee.server.chat_completions_api.streaming import _KEEPALIVE_FRAME, encode_completions_sse


def _chunk(
    *,
    id: str = "x",
    model: str = "m",
    created: int = 0,
    delta: CompletionsStreamDelta | None = None,
    finish_reason: str | None = None,
) -> CompletionsStreamChunk:
    return CompletionsStreamChunk(
        id=id,
        created=created,
        model=model,
        choices=[
            CompletionsStreamChoice(
                index=0,
                delta=delta if delta is not None else CompletionsStreamDelta(),
                finish_reason=finish_reason,
            )
        ],
    )


async def _async_chunks(
    items: list[CompletionsStreamChunk],
) -> AsyncIterator[CompletionsStreamChunk]:
    for item in items:
        yield item


async def _drain(it: AsyncIterator[bytes]) -> bytes:
    return b"".join([chunk async for chunk in it])


class TestEncodeCompletionsSse:
    async def test_single_chunk_followed_by_done_terminator(self) -> None:
        body = await _drain(
            encode_completions_sse(
                _async_chunks(
                    [_chunk(delta=CompletionsStreamDelta(content="hi"))],
                )
            )
        )
        text = body.decode()
        lines = text.split("\n\n")
        assert lines[0].startswith("data: ")
        assert '"content":"hi"' in lines[0]
        assert lines[1] == "data: [DONE]"
        assert text.endswith("\n\n")

    async def test_each_chunk_becomes_a_data_frame(self) -> None:
        body = await _drain(
            encode_completions_sse(
                _async_chunks(
                    [
                        _chunk(id="a", delta=CompletionsStreamDelta(content="0")),
                        _chunk(id="b", delta=CompletionsStreamDelta(content="1")),
                        _chunk(id="c", delta=CompletionsStreamDelta(content="2")),
                    ]
                )
            )
        )
        frames = body.decode().split("\n\n")
        for index, ident in enumerate(("a", "b", "c")):
            payload = frames[index].removeprefix("data: ")
            assert f'"id":"{ident}"' in payload
            assert f'"content":"{index}"' in payload
        assert frames[3] == "data: [DONE]"

    async def test_empty_stream_still_emits_done_terminator(self) -> None:
        body = await _drain(encode_completions_sse(_async_chunks([])))
        assert body == b"data: [DONE]\n\n"

    async def test_chunks_omit_none_fields(self) -> None:
        # ``encode_completions_sse`` uses ``model_dump_json(exclude_none=True)``;
        # role-less, finish-less content frames must not leak ``"role":null`` etc.
        body = await _drain(
            encode_completions_sse(
                _async_chunks([_chunk(delta=CompletionsStreamDelta(content="hi"))])
            )
        )
        first_frame = body.decode().split("\n\n")[0]
        payload = first_frame.removeprefix("data: ")
        assert "null" not in payload
        assert '"role":' not in payload
        assert '"finish_reason":' not in payload
        assert '"tool_calls":' not in payload

    async def test_finish_chunk_includes_finish_reason(self) -> None:
        body = await _drain(
            encode_completions_sse(
                _async_chunks(
                    [
                        _chunk(
                            delta=CompletionsStreamDelta(),
                            finish_reason="stop",
                        )
                    ]
                )
            )
        )
        first_frame = body.decode().split("\n\n")[0]
        assert '"finish_reason":"stop"' in first_frame

    async def test_pending_task_is_cancelled_when_consumer_closes_during_wait(
        self, monkeypatch
    ) -> None:
        """The finally cleanup cancels a still-pending ``__anext__`` task when
        the consumer closes the encoder while it's mid-wait on a slow upstream.
        Verifies the awaitable doesn't leak on client disconnect.
        """
        import asyncio as _asyncio

        from lilbee.server.chat_completions_api import streaming as streaming_mod

        # Keepalive way longer than the test runs so the wait() block is the
        # one suspended when we aclose.
        monkeypatch.setattr(streaming_mod, "_KEEPALIVE_INTERVAL_S", 5.0)
        cancelled = _asyncio.Event()

        async def _never_yields() -> AsyncIterator[CompletionsStreamChunk]:
            try:
                await _asyncio.sleep(3600)
            except _asyncio.CancelledError:
                cancelled.set()
                raise
            yield _chunk(delta=CompletionsStreamDelta(content="never"))  # pragma: no cover

        encoder = encode_completions_sse(_never_yields())
        consumer = _asyncio.create_task(encoder.__anext__())
        await _asyncio.sleep(0.02)  # let the encoder enqueue the pending __anext__
        consumer.cancel()
        with contextlib.suppress(_asyncio.CancelledError):
            await consumer
        await encoder.aclose()
        # After aclose runs its finally, the pending upstream task has been
        # cancelled and awaited; the upstream generator saw its CancelledError.
        assert cancelled.is_set()

    async def test_idle_stream_emits_keepalive_comment(self, monkeypatch) -> None:
        """When the upstream chat is slow to emit its first token, the encoder
        must yield SSE comment frames so clients (opencode) don't trip their
        idle-stream timeout and fire a retry storm.
        """
        from lilbee.server.chat_completions_api import streaming as streaming_mod

        # Tight keepalive cadence so the test runs in milliseconds; the
        # production constant stays at 5s.
        monkeypatch.setattr(streaming_mod, "_KEEPALIVE_INTERVAL_S", 0.02)

        import asyncio as _asyncio

        async def _slow_chunks() -> AsyncIterator[CompletionsStreamChunk]:
            # Two keepalive intervals pass before any chunk arrives.
            await _asyncio.sleep(0.07)
            yield _chunk(delta=CompletionsStreamDelta(content="finally"))

        body = await _drain(encode_completions_sse(_slow_chunks()))
        keepalive = _KEEPALIVE_FRAME.decode()
        text = body.decode()
        assert keepalive in text, "keepalive frame missing during idle gap"
        assert text.count(keepalive) >= 2
        assert "finally" in text
        assert text.endswith("data: [DONE]\n\n")
