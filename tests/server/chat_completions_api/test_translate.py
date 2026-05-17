"""Tests for OpenAI <-> canonical translation."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from lilbee.server.chat_completions_api.translate import (
    canonical_stream_to_completions_chunks,
    canonical_to_completions_response,
    completions_to_canonical_request,
)
from lilbee.server.chat_dispatch.canonical import (
    CanonicalChatRequest,
    CanonicalMessage,
    CanonicalResponse,
    CanonicalStreamEvent,
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
    ToolUseBlock,
    ToolUseDelta,
)


class TestCompletionsToCanonicalRequest:
    def test_minimal_text_request(self) -> None:
        req = completions_to_canonical_request(
            {
                "model": "vendor/model::Q4",
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
        assert req.model == "vendor/model::Q4"
        assert req.system is None
        assert req.stream is False
        assert req.tools is None
        assert req.tool_choice is None
        assert len(req.messages) == 1
        assert req.messages[0].role == "user"
        assert req.messages[0].content == [TextBlock(text="hi")]

    def test_stream_flag_is_carried_through(self) -> None:
        req = completions_to_canonical_request(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "stream": True,
            }
        )
        assert req.stream is True

    def test_sampling_options_are_normalized(self) -> None:
        req = completions_to_canonical_request(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "temperature": 0.2,
                "top_p": 0.9,
                "max_tokens": 64,
                "stop": ["</s>"],
            }
        )
        assert req.temperature == 0.2
        assert req.top_p == 0.9
        assert req.max_tokens == 64
        assert req.stop == ["</s>"]

    def test_stop_can_be_a_single_string(self) -> None:
        req = completions_to_canonical_request(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "stop": "<|end|>",
            }
        )
        assert req.stop == ["<|end|>"]

    def test_system_message_is_lifted_out(self) -> None:
        req = completions_to_canonical_request(
            {
                "model": "m",
                "messages": [
                    {"role": "system", "content": "be terse"},
                    {"role": "user", "content": "hi"},
                ],
            }
        )
        assert req.system == "be terse"
        assert len(req.messages) == 1
        assert req.messages[0].role == "user"

    def test_multiple_system_messages_are_concatenated(self) -> None:
        req = completions_to_canonical_request(
            {
                "model": "m",
                "messages": [
                    {"role": "system", "content": "be terse"},
                    {"role": "system", "content": "no apologies"},
                    {"role": "user", "content": "hi"},
                ],
            }
        )
        assert req.system == "be terse\n\nno apologies"

    def test_multi_content_user_message_with_text_and_image(self) -> None:
        req = completions_to_canonical_request(
            {
                "model": "m",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,aGVsbG8="},
                            },
                        ],
                    }
                ],
            }
        )
        assert len(req.messages[0].content) == 2
        assert req.messages[0].content[0] == TextBlock(text="describe")
        image = req.messages[0].content[1]
        assert isinstance(image, ImageBlock)
        assert image.media_type == "image/png"
        assert image.data == b"hello"

    def test_image_url_with_http_url_keeps_url_in_data(self) -> None:
        req = completions_to_canonical_request(
            {
                "model": "m",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example/cat.png"},
                            }
                        ],
                    }
                ],
            }
        )
        image = req.messages[0].content[0]
        assert isinstance(image, ImageBlock)
        assert image.media_type == "image/url"
        assert image.data == b"https://example/cat.png"

    def test_assistant_message_with_tool_calls(self) -> None:
        req = completions_to_canonical_request(
            {
                "model": "m",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "ok",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"q":"foo"}',
                                },
                            }
                        ],
                    }
                ],
            }
        )
        msg = req.messages[0]
        assert msg.role == "assistant"
        # First a text block ("ok"), then a tool_use block.
        assert msg.content[0] == TextBlock(text="ok")
        tool_use = msg.content[1]
        assert isinstance(tool_use, ToolUseBlock)
        assert tool_use.id == "c1"
        assert tool_use.name == "search"
        assert tool_use.input == {"q": "foo"}

    def test_assistant_message_with_only_tool_calls_omits_empty_text(self) -> None:
        req = completions_to_canonical_request(
            {
                "model": "m",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "x", "arguments": "{}"},
                            }
                        ],
                    }
                ],
            }
        )
        msg = req.messages[0]
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], ToolUseBlock)

    def test_assistant_with_null_content_and_tool_calls(self) -> None:
        req = completions_to_canonical_request(
            {
                "model": "m",
                "messages": [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "x", "arguments": "{}"},
                            }
                        ],
                    }
                ],
            }
        )
        assert len(req.messages[0].content) == 1
        assert isinstance(req.messages[0].content[0], ToolUseBlock)

    def test_tool_role_message_becomes_tool_result_block(self) -> None:
        req = completions_to_canonical_request(
            {
                "model": "m",
                "messages": [
                    {
                        "role": "tool",
                        "tool_call_id": "c1",
                        "content": "result text",
                    }
                ],
            }
        )
        from lilbee.server.chat_dispatch.canonical import ToolResultBlock

        msg = req.messages[0]
        assert msg.role == "tool"
        assert len(msg.content) == 1
        block = msg.content[0]
        assert isinstance(block, ToolResultBlock)
        assert block.tool_use_id == "c1"
        assert block.content == [TextBlock(text="result text")]

    def test_tools_become_canonical_tools(self) -> None:
        req = completions_to_canonical_request(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search",
                            "description": "Find docs",
                            "parameters": {
                                "type": "object",
                                "properties": {"q": {"type": "string"}},
                            },
                        },
                    }
                ],
            }
        )
        assert req.tools == [
            CanonicalTool(
                name="search",
                description="Find docs",
                input_schema={
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            )
        ]

    def test_tool_without_description_defaults_to_empty(self) -> None:
        req = completions_to_canonical_request(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "x", "parameters": {}},
                    }
                ],
            }
        )
        assert req.tools is not None
        assert req.tools[0].description == ""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("auto", CanonicalToolChoice(mode="auto")),
            ("none", CanonicalToolChoice(mode="none")),
            ("required", CanonicalToolChoice(mode="any")),
        ],
    )
    def test_tool_choice_string_modes(self, raw, expected) -> None:
        req = completions_to_canonical_request(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "tool_choice": raw,
            }
        )
        assert req.tool_choice == expected

    def test_tool_choice_function_dict(self) -> None:
        req = completions_to_canonical_request(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "search"},
                },
            }
        )
        assert req.tool_choice == CanonicalToolChoice(mode="tool", tool_name="search")

    def test_unknown_string_tool_choice_raises(self) -> None:
        with pytest.raises(ValueError):
            completions_to_canonical_request(
                {
                    "model": "m",
                    "messages": [{"role": "user", "content": "x"}],
                    "tool_choice": "bogus",
                }
            )

    def test_malformed_tool_choice_dict_raises(self) -> None:
        with pytest.raises(ValueError):
            completions_to_canonical_request(
                {
                    "model": "m",
                    "messages": [{"role": "user", "content": "x"}],
                    "tool_choice": {"type": "function", "function": {}},
                }
            )

    def test_missing_model_raises(self) -> None:
        with pytest.raises(ValueError):
            completions_to_canonical_request({"messages": [{"role": "user", "content": "x"}]})

    def test_missing_messages_raises(self) -> None:
        with pytest.raises(ValueError):
            completions_to_canonical_request({"model": "m"})

    def test_unknown_content_block_type_raises(self) -> None:
        with pytest.raises(ValueError):
            completions_to_canonical_request(
                {
                    "model": "m",
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "audio", "audio": "..."}],
                        }
                    ],
                }
            )

    def test_unknown_role_raises(self) -> None:
        with pytest.raises(ValueError):
            completions_to_canonical_request(
                {
                    "model": "m",
                    "messages": [{"role": "developer", "content": "x"}],
                }
            )

    def test_system_with_list_content_concatenates_text_parts(self) -> None:
        req = completions_to_canonical_request(
            {
                "model": "m",
                "messages": [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "a"},
                            {"type": "text", "text": "b"},
                        ],
                    },
                    {"role": "user", "content": "x"},
                ],
            }
        )
        assert req.system == "ab"

    def test_system_with_non_string_non_list_content_raises(self) -> None:
        with pytest.raises(ValueError):
            completions_to_canonical_request(
                {
                    "model": "m",
                    "messages": [
                        {"role": "system", "content": 42},
                        {"role": "user", "content": "x"},
                    ],
                }
            )

    def test_user_content_dict_raises(self) -> None:
        with pytest.raises(ValueError):
            completions_to_canonical_request(
                {
                    "model": "m",
                    "messages": [{"role": "user", "content": {"foo": "bar"}}],
                }
            )

    def test_tool_role_message_with_list_content(self) -> None:
        from lilbee.server.chat_dispatch.canonical import ToolResultBlock

        req = completions_to_canonical_request(
            {
                "model": "m",
                "messages": [
                    {
                        "role": "tool",
                        "tool_call_id": "c1",
                        "content": [{"type": "text", "text": "ok"}],
                    }
                ],
            }
        )
        block = req.messages[0].content[0]
        assert isinstance(block, ToolResultBlock)
        assert block.content == [TextBlock(text="ok")]

    def test_tool_role_message_with_non_string_non_list_content_stringifies(self) -> None:
        from lilbee.server.chat_dispatch.canonical import ToolResultBlock

        req = completions_to_canonical_request(
            {
                "model": "m",
                "messages": [
                    {"role": "tool", "tool_call_id": "c1", "content": 42},
                ],
            }
        )
        block = req.messages[0].content[0]
        assert isinstance(block, ToolResultBlock)
        assert block.content == [TextBlock(text="42")]

    def test_tool_role_message_with_null_content_yields_empty_text(self) -> None:
        from lilbee.server.chat_dispatch.canonical import ToolResultBlock

        req = completions_to_canonical_request(
            {
                "model": "m",
                "messages": [
                    {"role": "tool", "tool_call_id": "c1", "content": None},
                ],
            }
        )
        block = req.messages[0].content[0]
        assert isinstance(block, ToolResultBlock)
        assert block.content == [TextBlock(text="")]

    def test_assistant_tool_call_with_malformed_json_args_falls_back_to_raw(self) -> None:
        req = completions_to_canonical_request(
            {
                "model": "m",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "x", "arguments": "not json{"},
                            }
                        ],
                    }
                ],
            }
        )
        tool_use = req.messages[0].content[0]
        assert isinstance(tool_use, ToolUseBlock)
        assert tool_use.input == {"_raw": "not json{"}

    def test_assistant_tool_call_with_array_args_falls_back_to_raw(self) -> None:
        req = completions_to_canonical_request(
            {
                "model": "m",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "x", "arguments": "[1,2]"},
                            }
                        ],
                    }
                ],
            }
        )
        tool_use = req.messages[0].content[0]
        assert isinstance(tool_use, ToolUseBlock)
        assert tool_use.input == {"_raw": "[1,2]"}

    def test_tool_choice_unsupported_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            completions_to_canonical_request(
                {
                    "model": "m",
                    "messages": [{"role": "user", "content": "x"}],
                    "tool_choice": 42,
                }
            )

    def test_stop_unsupported_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            completions_to_canonical_request(
                {
                    "model": "m",
                    "messages": [{"role": "user", "content": "x"}],
                    "stop": {"foo": "bar"},
                }
            )


class TestCanonicalToCompletionsResponse:
    def _resp(self, **overrides) -> CanonicalResponse:
        base = {
            "id": "msg_abc",
            "model": "vendor/model::Q4",
            "content": [TextBlock(text="hello")],
            "stop_reason": StopReason.END_TURN,
            "usage": CanonicalUsage(input_tokens=0, output_tokens=0),
        }
        base.update(overrides)
        return CanonicalResponse(**base)

    def test_text_only_response(self) -> None:
        body = canonical_to_completions_response(self._resp())
        assert body["id"] == "msg_abc"
        assert body["object"] == "chat.completion"
        assert body["model"] == "vendor/model::Q4"
        assert isinstance(body["created"], int)
        assert body["choices"] == [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello",
                },
                "finish_reason": "stop",
            }
        ]
        assert body["usage"] == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def test_response_with_tool_calls(self) -> None:
        body = canonical_to_completions_response(
            self._resp(
                content=[
                    ToolUseBlock(id="c1", name="search", input={"q": "foo"}),
                ],
                stop_reason=StopReason.TOOL_USE,
            )
        )
        choice = body["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["content"] is None
        assert choice["message"]["tool_calls"] == [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"q": "foo"}'},
            }
        ]

    def test_response_with_text_and_tool_call(self) -> None:
        body = canonical_to_completions_response(
            self._resp(
                content=[
                    TextBlock(text="ok"),
                    ToolUseBlock(id="c1", name="x", input={}),
                ],
                stop_reason=StopReason.TOOL_USE,
            )
        )
        msg = body["choices"][0]["message"]
        assert msg["content"] == "ok"
        assert len(msg["tool_calls"]) == 1

    @pytest.mark.parametrize(
        "stop_reason,expected",
        [
            (StopReason.END_TURN, "stop"),
            (StopReason.MAX_TOKENS, "length"),
            (StopReason.STOP_SEQUENCE, "stop"),
            (StopReason.TOOL_USE, "tool_calls"),
            (StopReason.ERROR, "stop"),
        ],
    )
    def test_finish_reason_mapping(self, stop_reason, expected) -> None:
        body = canonical_to_completions_response(self._resp(stop_reason=stop_reason))
        assert body["choices"][0]["finish_reason"] == expected

    def test_usage_passes_through_canonical_counts_honestly(self) -> None:
        body = canonical_to_completions_response(
            self._resp(usage=CanonicalUsage(input_tokens=5, output_tokens=7))
        )
        assert body["usage"] == {
            "prompt_tokens": 5,
            "completion_tokens": 7,
            "total_tokens": 12,
        }


async def _drain(it: AsyncIterator[dict]) -> list[dict]:
    return [chunk async for chunk in it]


async def _async_iter(items: list[CanonicalStreamEvent]) -> AsyncIterator[CanonicalStreamEvent]:
    for item in items:
        yield item


class TestCanonicalStreamToCompletionsChunks:
    async def test_text_only_stream_emits_role_then_content_then_finish(self) -> None:
        events: list[CanonicalStreamEvent] = [
            MessageStart(id="msg_x", model="m"),
            ContentBlockStart(index=0, block=TextBlock(text="")),
            ContentBlockDelta(index=0, delta=TextDelta(text="he")),
            ContentBlockDelta(index=0, delta=TextDelta(text="llo")),
            ContentBlockStop(index=0),
            MessageDelta(stop_reason=StopReason.END_TURN),
            MessageStop(),
        ]
        chunks = await _drain(
            canonical_stream_to_completions_chunks(
                _async_iter(events), model="m", response_id="msg_x"
            )
        )
        # First chunk: role
        assert chunks[0]["id"] == "msg_x"
        assert chunks[0]["object"] == "chat.completion.chunk"
        assert chunks[0]["model"] == "m"
        assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
        # Then two content deltas
        assert chunks[1]["choices"][0]["delta"] == {"content": "he"}
        assert chunks[2]["choices"][0]["delta"] == {"content": "llo"}
        # Finish chunk
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        assert chunks[-1]["choices"][0]["delta"] == {}

    async def test_message_start_alone_emits_nothing(self) -> None:
        events: list[CanonicalStreamEvent] = [
            MessageStart(id="msg_x", model="m"),
            MessageStop(),
        ]
        chunks = await _drain(
            canonical_stream_to_completions_chunks(
                _async_iter(events), model="m", response_id="msg_x"
            )
        )
        assert chunks == []

    async def test_tool_call_stream_emits_function_chunks(self) -> None:
        events: list[CanonicalStreamEvent] = [
            MessageStart(id="msg_x", model="m"),
            ContentBlockStart(index=0, block=ToolUseBlock(id="c1", name="search", input={})),
            ContentBlockDelta(index=0, delta=ToolUseDelta(partial_json='{"q":')),
            ContentBlockDelta(index=0, delta=ToolUseDelta(partial_json='"foo"}')),
            ContentBlockStop(index=0),
            MessageDelta(stop_reason=StopReason.TOOL_USE),
            MessageStop(),
        ]
        chunks = await _drain(
            canonical_stream_to_completions_chunks(
                _async_iter(events), model="m", response_id="msg_x"
            )
        )
        # First: open the tool call with id/name and empty args
        first = chunks[0]["choices"][0]["delta"]
        assert first["tool_calls"] == [
            {
                "index": 0,
                "id": "c1",
                "type": "function",
                "function": {"name": "search", "arguments": ""},
            }
        ]
        # Then two argument-fragment chunks
        assert chunks[1]["choices"][0]["delta"]["tool_calls"] == [
            {"index": 0, "function": {"arguments": '{"q":'}}
        ]
        assert chunks[2]["choices"][0]["delta"]["tool_calls"] == [
            {"index": 0, "function": {"arguments": '"foo"}'}}
        ]
        assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"

    async def test_text_then_tool_call_carries_consistent_indexing(self) -> None:
        events: list[CanonicalStreamEvent] = [
            MessageStart(id="msg_x", model="m"),
            ContentBlockStart(index=0, block=TextBlock(text="")),
            ContentBlockDelta(index=0, delta=TextDelta(text="thinking")),
            ContentBlockStop(index=0),
            ContentBlockStart(index=1, block=ToolUseBlock(id="c1", name="x", input={})),
            ContentBlockDelta(index=1, delta=ToolUseDelta(partial_json="{}")),
            ContentBlockStop(index=1),
            MessageDelta(stop_reason=StopReason.TOOL_USE),
            MessageStop(),
        ]
        chunks = await _drain(
            canonical_stream_to_completions_chunks(
                _async_iter(events), model="m", response_id="msg_x"
            )
        )
        # role chunk, content chunk, tool-open chunk (index 0 in tool_calls list,
        # because it is the first tool call), arg chunk, finish chunk.
        assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
        assert chunks[1]["choices"][0]["delta"] == {"content": "thinking"}
        tool_open = chunks[2]["choices"][0]["delta"]["tool_calls"][0]
        assert tool_open["index"] == 0
        assert tool_open["id"] == "c1"
        assert chunks[3]["choices"][0]["delta"]["tool_calls"][0] == {
            "index": 0,
            "function": {"arguments": "{}"},
        }
        assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"

    async def test_content_block_stop_does_not_emit_chunk(self) -> None:
        events: list[CanonicalStreamEvent] = [
            MessageStart(id="msg_x", model="m"),
            ContentBlockStart(index=0, block=TextBlock(text="")),
            ContentBlockDelta(index=0, delta=TextDelta(text="x")),
            ContentBlockStop(index=0),
            MessageDelta(stop_reason=StopReason.END_TURN),
            MessageStop(),
        ]
        chunks = await _drain(
            canonical_stream_to_completions_chunks(
                _async_iter(events), model="m", response_id="msg_x"
            )
        )
        # role + content + finish, three total. No stand-alone "stop" chunk.
        assert len(chunks) == 3

    @pytest.mark.parametrize(
        "stop_reason,expected",
        [
            (StopReason.END_TURN, "stop"),
            (StopReason.MAX_TOKENS, "length"),
            (StopReason.STOP_SEQUENCE, "stop"),
            (StopReason.TOOL_USE, "tool_calls"),
        ],
    )
    async def test_finish_reason_mapping(self, stop_reason, expected) -> None:
        events: list[CanonicalStreamEvent] = [
            MessageStart(id="msg_x", model="m"),
            MessageDelta(stop_reason=stop_reason),
            MessageStop(),
        ]
        chunks = await _drain(
            canonical_stream_to_completions_chunks(
                _async_iter(events), model="m", response_id="msg_x"
            )
        )
        assert chunks[-1]["choices"][0]["finish_reason"] == expected


def test_canonical_request_with_minimal_payload_uses_request_helper() -> None:
    """Spot-check that the dataclass roundtrips for the simplest payload."""
    req = completions_to_canonical_request(
        {"model": "m", "messages": [{"role": "user", "content": "x"}]}
    )
    assert isinstance(req, CanonicalChatRequest)
    assert isinstance(req.messages[0], CanonicalMessage)
