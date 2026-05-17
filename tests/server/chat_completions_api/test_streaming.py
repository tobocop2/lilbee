"""Tests for the OpenAI SSE encoder."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from lilbee.server.chat_completions_api.streaming import encode_completions_sse


async def _async_chunks(items: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
    for item in items:
        yield item


async def _drain(it: AsyncIterator[bytes]) -> bytes:
    return b"".join([chunk async for chunk in it])


class TestEncodeCompletionsSse:
    async def test_single_chunk_followed_by_done_terminator(self) -> None:
        body = await _drain(
            encode_completions_sse(
                _async_chunks(
                    [
                        {
                            "id": "x",
                            "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": {"content": "hi"}}],
                        }
                    ]
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
                        {"i": 0},
                        {"i": 1},
                        {"i": 2},
                    ]
                )
            )
        )
        frames = body.decode().split("\n\n")
        # Three data frames + done frame + trailing empty (from final \n\n split)
        assert frames[0] == 'data: {"i":0}'
        assert frames[1] == 'data: {"i":1}'
        assert frames[2] == 'data: {"i":2}'
        assert frames[3] == "data: [DONE]"

    async def test_empty_stream_still_emits_done_terminator(self) -> None:
        body = await _drain(encode_completions_sse(_async_chunks([])))
        assert body == b"data: [DONE]\n\n"

    async def test_chunks_serialize_with_compact_separators(self) -> None:
        body = await _drain(encode_completions_sse(_async_chunks([{"a": 1, "b": [2, 3]}])))
        # No spaces between keys/values: separators=(",", ":") in encoder.
        assert b'data: {"a":1,"b":[2,3]}\n\n' in body

    async def test_output_is_bytes_not_str(self) -> None:
        async for chunk in encode_completions_sse(_async_chunks([{"x": 1}])):
            assert isinstance(chunk, bytes)
