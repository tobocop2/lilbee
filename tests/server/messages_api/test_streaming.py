"""Tests for the Anthropic event-stream SSE encoder."""

from __future__ import annotations

from lilbee.server.messages_api.streaming import encode_messages_sse


def _aiter(items):
    async def gen():
        for x in items:
            yield x

    return gen()


async def _collect_bytes(stream) -> bytes:
    out = bytearray()
    async for chunk in stream:
        out.extend(chunk)
    return bytes(out)


class TestEncodeMessagesSse:
    async def test_each_event_has_event_and_data_lines(self) -> None:
        events = _aiter(
            [
                ("message_start", {"type": "message_start", "id": "msg_1"}),
                ("message_stop", {"type": "message_stop"}),
            ]
        )
        out = await _collect_bytes(encode_messages_sse(events))
        assert out == (
            b'event: message_start\ndata: {"type":"message_start","id":"msg_1"}\n\n'
            b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
        )

    async def test_no_done_terminator_emitted(self) -> None:
        events = _aiter([("message_stop", {"type": "message_stop"})])
        out = await _collect_bytes(encode_messages_sse(events))
        assert b"[DONE]" not in out

    async def test_compact_json_no_spaces(self) -> None:
        events = _aiter(
            [
                (
                    "content_block_delta",
                    {"index": 0, "delta": {"type": "text_delta", "text": "hi"}},
                )
            ]
        )
        out = await _collect_bytes(encode_messages_sse(events))
        assert b", " not in out
        assert b": " not in out.split(b"data: ", 1)[1]

    async def test_full_anthropic_sequence(self) -> None:
        events = _aiter(
            [
                ("message_start", {"type": "message_start"}),
                ("content_block_start", {"type": "content_block_start", "index": 0}),
                ("content_block_delta", {"type": "content_block_delta", "index": 0}),
                ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                ("message_delta", {"type": "message_delta"}),
                ("message_stop", {"type": "message_stop"}),
            ]
        )
        out = (await _collect_bytes(encode_messages_sse(events))).decode()
        names = [line[len("event: ") :] for line in out.splitlines() if line.startswith("event: ")]
        assert names == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]

    async def test_empty_iterator_yields_no_bytes(self) -> None:
        events = _aiter([])
        out = await _collect_bytes(encode_messages_sse(events))
        assert out == b""
