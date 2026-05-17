"""Tests for the OpenAI SSE encoder."""

from __future__ import annotations

from collections.abc import AsyncIterator

from lilbee.server.chat_completions_api.models import (
    CompletionsStreamChoice,
    CompletionsStreamChunk,
    CompletionsStreamDelta,
)
from lilbee.server.chat_completions_api.streaming import encode_completions_sse


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

    async def test_output_is_bytes_not_str(self) -> None:
        async for chunk in encode_completions_sse(
            _async_chunks([_chunk(delta=CompletionsStreamDelta(content="hi"))])
        ):
            assert isinstance(chunk, bytes)

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
