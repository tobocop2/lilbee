"""Tests for Anthropic Messages translation to and from canonical."""

from __future__ import annotations

import base64

import pytest

from lilbee.server.chat_dispatch.canonical import (
    CanonicalMessage,
    CanonicalResponse,
    CanonicalTool,
    CanonicalToolChoice,
    CanonicalUsage,
    ContentBlockDelta,
    ContentBlockStart,
    ContentBlockStop,
    ImageBlock,
    MessageDelta,
    MessageStart,
    MessageStop,
    StopReason,
    TextBlock,
    TextDelta,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseDelta,
)
from lilbee.server.messages_api.translate import (
    canonical_stream_to_messages_events,
    canonical_to_messages_response,
    messages_to_canonical_request,
)


def _aiter(items):
    async def gen():
        for x in items:
            yield x

    return gen()


async def _collect(events):
    out = []
    async for ev in events:
        out.append(ev)
    return out


class TestRequestTranslation:
    def test_minimal_string_content(self) -> None:
        req = messages_to_canonical_request(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16,
            }
        )
        assert req.model == "m"
        assert req.messages == [CanonicalMessage(role="user", content=[TextBlock(text="hi")])]
        assert req.system is None
        assert req.tools is None
        assert req.tool_choice is None
        assert req.max_tokens == 16
        assert req.stream is False

    def test_system_field_lifted(self) -> None:
        req = messages_to_canonical_request(
            {
                "model": "m",
                "system": "you are X",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            }
        )
        assert req.system == "you are X"

    def test_block_list_content(self) -> None:
        req = messages_to_canonical_request(
            {
                "model": "m",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "first"},
                            {"type": "text", "text": "second"},
                        ],
                    }
                ],
                "max_tokens": 1,
            }
        )
        assert req.messages[0].content == [
            TextBlock(text="first"),
            TextBlock(text="second"),
        ]

    def test_image_block_translates_to_canonical(self) -> None:
        raw = b"\x89PNG\r\n"
        encoded = base64.b64encode(raw).decode()
        req = messages_to_canonical_request(
            {
                "model": "m",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": encoded,
                                },
                            }
                        ],
                    }
                ],
                "max_tokens": 1,
            }
        )
        assert req.messages[0].content == [ImageBlock(media_type="image/png", data=raw)]

    def test_tools_translation(self) -> None:
        req = messages_to_canonical_request(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "name": "search",
                        "description": "Find stuff",
                        "input_schema": {"type": "object"},
                    }
                ],
                "max_tokens": 1,
            }
        )
        assert req.tools == [
            CanonicalTool(
                name="search",
                description="Find stuff",
                input_schema={"type": "object"},
            )
        ]

    def test_tool_without_description_defaults_blank(self) -> None:
        req = messages_to_canonical_request(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"name": "x", "input_schema": {}}],
                "max_tokens": 1,
            }
        )
        assert req.tools is not None
        assert req.tools[0].description == ""

    @pytest.mark.parametrize(
        ("choice", "expected_mode", "expected_name"),
        [
            ({"type": "auto"}, "auto", None),
            ({"type": "any"}, "any", None),
            ({"type": "none"}, "none", None),
            ({"type": "tool", "name": "search"}, "tool", "search"),
        ],
    )
    def test_tool_choice_shapes(self, choice, expected_mode, expected_name) -> None:
        req = messages_to_canonical_request(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "tool_choice": choice,
                "max_tokens": 1,
            }
        )
        assert req.tool_choice == CanonicalToolChoice(mode=expected_mode, tool_name=expected_name)

    def test_tool_choice_tool_without_name_raises(self) -> None:
        with pytest.raises(ValueError, match="tool_choice"):
            messages_to_canonical_request(
                {
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                    "tool_choice": {"type": "tool"},
                    "max_tokens": 1,
                }
            )

    def test_tool_choice_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="tool_choice"):
            messages_to_canonical_request(
                {
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                    "tool_choice": {"type": "bogus"},
                    "max_tokens": 1,
                }
            )

    def test_assistant_tool_use_round_trips(self) -> None:
        req = messages_to_canonical_request(
            {
                "model": "m",
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "t1",
                                "name": "search",
                                "input": {"q": "foo"},
                            }
                        ],
                    }
                ],
                "max_tokens": 1,
            }
        )
        assert req.messages[0].content == [ToolUseBlock(id="t1", name="search", input={"q": "foo"})]

    def test_user_tool_result_with_string_content(self) -> None:
        req = messages_to_canonical_request(
            {
                "model": "m",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t1",
                                "content": "result text",
                            }
                        ],
                    }
                ],
                "max_tokens": 1,
            }
        )
        assert req.messages[0].content == [
            ToolResultBlock(
                tool_use_id="t1",
                content=[TextBlock(text="result text")],
                is_error=False,
            )
        ]

    def test_user_tool_result_with_block_list_and_error(self) -> None:
        req = messages_to_canonical_request(
            {
                "model": "m",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t2",
                                "content": [{"type": "text", "text": "boom"}],
                                "is_error": True,
                            }
                        ],
                    }
                ],
                "max_tokens": 1,
            }
        )
        assert req.messages[0].content == [
            ToolResultBlock(
                tool_use_id="t2",
                content=[TextBlock(text="boom")],
                is_error=True,
            )
        ]

    def test_unknown_block_type_raises(self) -> None:
        with pytest.raises(ValueError, match="content block"):
            messages_to_canonical_request(
                {
                    "model": "m",
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "video", "url": "x"}],
                        }
                    ],
                    "max_tokens": 1,
                }
            )

    def test_image_non_base64_source_raises(self) -> None:
        with pytest.raises(ValueError, match="image source"):
            messages_to_canonical_request(
                {
                    "model": "m",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "url",
                                        "url": "http://example/x.png",
                                    },
                                }
                            ],
                        }
                    ],
                    "max_tokens": 1,
                }
            )

    def test_sampling_and_stream_fields(self) -> None:
        req = messages_to_canonical_request(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.3,
                "top_p": 0.7,
                "top_k": 30,
                "stop_sequences": ["END"],
                "max_tokens": 32,
                "stream": True,
            }
        )
        assert req.temperature == 0.3
        assert req.top_p == 0.7
        assert req.top_k == 30
        assert req.stop == ["END"]
        assert req.max_tokens == 32
        assert req.stream is True

    def test_missing_model_raises(self) -> None:
        with pytest.raises(ValueError, match="model"):
            messages_to_canonical_request({"messages": [], "max_tokens": 1})

    def test_missing_messages_raises(self) -> None:
        with pytest.raises(ValueError, match="messages"):
            messages_to_canonical_request({"model": "m", "max_tokens": 1})

    def test_missing_max_tokens_raises(self) -> None:
        with pytest.raises(ValueError, match="max_tokens"):
            messages_to_canonical_request(
                {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
            )

    def test_tool_use_round_trip_to_canonical(self) -> None:
        """tool_use + matching tool_result pair stays well-typed."""
        req = messages_to_canonical_request(
            {
                "model": "m",
                "messages": [
                    {"role": "user", "content": "find stuff"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "tu1",
                                "name": "search",
                                "input": {"q": "x"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tu1",
                                "content": "found x",
                            }
                        ],
                    },
                ],
                "max_tokens": 1,
            }
        )
        assert len(req.messages) == 3
        assert isinstance(req.messages[1].content[0], ToolUseBlock)
        assert isinstance(req.messages[2].content[0], ToolResultBlock)


class TestResponseTranslation:
    def test_text_only_response(self) -> None:
        resp = CanonicalResponse(
            id="msg_1",
            model="m",
            content=[TextBlock(text="hello")],
            stop_reason=StopReason.END_TURN,
            usage=CanonicalUsage(input_tokens=3, output_tokens=5),
        )
        body = canonical_to_messages_response(resp)
        assert body == {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "m",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 3, "output_tokens": 5},
        }

    def test_response_with_tool_use_block(self) -> None:
        resp = CanonicalResponse(
            id="msg_2",
            model="m",
            content=[
                TextBlock(text="I will call a tool"),
                ToolUseBlock(id="tu1", name="search", input={"q": "foo"}),
            ],
            stop_reason=StopReason.TOOL_USE,
            usage=CanonicalUsage(input_tokens=0, output_tokens=0),
        )
        body = canonical_to_messages_response(resp)
        assert body["content"] == [
            {"type": "text", "text": "I will call a tool"},
            {
                "type": "tool_use",
                "id": "tu1",
                "name": "search",
                "input": {"q": "foo"},
            },
        ]
        assert body["stop_reason"] == "tool_use"

    @pytest.mark.parametrize(
        ("canonical", "expected"),
        [
            (StopReason.END_TURN, "end_turn"),
            (StopReason.MAX_TOKENS, "max_tokens"),
            (StopReason.STOP_SEQUENCE, "stop_sequence"),
            (StopReason.TOOL_USE, "tool_use"),
            (StopReason.ERROR, "end_turn"),
        ],
    )
    def test_stop_reason_mapping(self, canonical, expected) -> None:
        resp = CanonicalResponse(
            id="x",
            model="m",
            content=[],
            stop_reason=canonical,
            usage=CanonicalUsage(input_tokens=0, output_tokens=0),
        )
        body = canonical_to_messages_response(resp)
        assert body["stop_reason"] == expected


class TestStreamEventTranslation:
    async def test_full_event_sequence(self) -> None:
        events = _aiter(
            [
                MessageStart(id="msg_1", model="m"),
                ContentBlockStart(index=0, block=TextBlock(text="")),
                ContentBlockDelta(index=0, delta=TextDelta(text="hel")),
                ContentBlockDelta(index=0, delta=TextDelta(text="lo")),
                ContentBlockStop(index=0),
                MessageDelta(
                    stop_reason=StopReason.END_TURN,
                    usage=CanonicalUsage(input_tokens=1, output_tokens=2),
                ),
                MessageStop(),
            ]
        )
        out = await _collect(canonical_stream_to_messages_events(events))
        names = [name for name, _ in out]
        assert names == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]

        ms = out[0][1]
        assert ms["type"] == "message_start"
        assert ms["message"]["id"] == "msg_1"
        assert ms["message"]["model"] == "m"
        assert ms["message"]["type"] == "message"
        assert ms["message"]["role"] == "assistant"
        assert ms["message"]["content"] == []
        assert ms["message"]["stop_reason"] is None
        assert ms["message"]["usage"] == {"input_tokens": 0, "output_tokens": 0}

        cbs = out[1][1]
        assert cbs == {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }

        cbd = out[2][1]
        assert cbd == {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hel"},
        }

        md = out[5][1]
        assert md == {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"input_tokens": 1, "output_tokens": 2},
        }

        stop = out[6][1]
        assert stop == {"type": "message_stop"}

    async def test_tool_use_content_block_start_and_delta(self) -> None:
        events = _aiter(
            [
                ContentBlockStart(
                    index=0,
                    block=ToolUseBlock(id="tu1", name="search", input={}),
                ),
                ContentBlockDelta(
                    index=0,
                    delta=ToolUseDelta(partial_json='{"q":'),
                ),
                ContentBlockDelta(
                    index=0,
                    delta=ToolUseDelta(partial_json='"foo"}'),
                ),
                ContentBlockStop(index=0),
            ]
        )
        out = await _collect(canonical_stream_to_messages_events(events))
        assert out[0] == (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "tu1",
                    "name": "search",
                    "input": {},
                },
            },
        )
        assert out[1] == (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"q":'},
            },
        )
        assert out[3] == (
            "content_block_stop",
            {"type": "content_block_stop", "index": 0},
        )

    async def test_message_delta_with_only_stop_reason(self) -> None:
        events = _aiter(
            [
                MessageDelta(stop_reason=StopReason.MAX_TOKENS, usage=None),
            ]
        )
        out = await _collect(canonical_stream_to_messages_events(events))
        assert out[0][1] == {
            "type": "message_delta",
            "delta": {"stop_reason": "max_tokens", "stop_sequence": None},
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

    async def test_tool_result_in_content_block_start(self) -> None:
        events = _aiter(
            [
                ContentBlockStart(
                    index=0,
                    block=ToolResultBlock(
                        tool_use_id="tu1",
                        content=[TextBlock(text="ok")],
                        is_error=False,
                    ),
                )
            ]
        )
        out = await _collect(canonical_stream_to_messages_events(events))
        assert out[0][1]["content_block"] == {
            "type": "tool_result",
            "tool_use_id": "tu1",
            "content": [{"type": "text", "text": "ok"}],
            "is_error": False,
        }

    async def test_image_in_content_block_start(self) -> None:
        events = _aiter(
            [
                ContentBlockStart(
                    index=0,
                    block=ImageBlock(media_type="image/png", data=b"\x89PNG"),
                )
            ]
        )
        out = await _collect(canonical_stream_to_messages_events(events))
        block = out[0][1]["content_block"]
        assert block["type"] == "image"
        assert block["source"]["type"] == "base64"
        assert block["source"]["media_type"] == "image/png"
        assert base64.b64decode(block["source"]["data"]) == b"\x89PNG"
