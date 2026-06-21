"""Tests for OpenAI <-> canonical translation."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from lilbee.server.chat_completions_api.models import (
    CompletionsRequest,
    CompletionsResponse,
    CompletionsStreamChunk,
)
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

_RESPONSE_ID = "chatcmpl-test-response-id"


def _translate(payload: dict[str, Any]) -> CanonicalChatRequest:
    """Validate ``payload`` into a CompletionsRequest then translate it."""
    return completions_to_canonical_request(CompletionsRequest.model_validate(payload))


class TestCompletionsToCanonicalRequest:
    def test_minimal_text_request(self) -> None:
        req = _translate(
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
        req = _translate(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "stream": True,
            }
        )
        assert req.stream is True

    def test_sampling_options_are_normalized(self) -> None:
        req = _translate(
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
        req = _translate(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "stop": "<|end|>",
            }
        )
        assert req.stop == ["<|end|>"]

    def test_system_message_is_lifted_out(self) -> None:
        req = _translate(
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
        req = _translate(
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

    def test_image_content_in_user_message_is_rejected(self) -> None:
        # The dispatcher cannot route image content to the chat provider
        # today. Translate raises ValueError so the route layer returns 400
        # rather than silently dropping the image and returning a text-only
        # completion that ignored the visual input.
        with pytest.raises(ValueError, match="Image content is not supported"):
            _translate(
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

    def test_assistant_message_with_tool_calls(self) -> None:
        req = _translate(
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
        req = _translate(
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
        req = _translate(
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
        req = _translate(
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
        msg = req.messages[0]
        assert msg.role == "tool"
        assert len(msg.content) == 1
        block = msg.content[0]
        assert isinstance(block, ToolResultBlock)
        assert block.tool_use_id == "c1"
        assert block.content == [TextBlock(text="result text")]

    def test_tools_become_canonical_tools(self) -> None:
        req = _translate(
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
        req = _translate(
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
        req = _translate(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "tool_choice": raw,
            }
        )
        assert req.tool_choice == expected

    def test_tool_choice_function_dict(self) -> None:
        req = _translate(
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

    def test_unknown_string_tool_choice_rejected_at_validation(self) -> None:
        # The ``ToolChoiceMode`` enum on the request model blocks any
        # string that is not ``auto`` / ``none`` / ``required``.
        with pytest.raises(ValueError):
            CompletionsRequest.model_validate(
                {
                    "model": "m",
                    "messages": [{"role": "user", "content": "x"}],
                    "tool_choice": "bogus",
                }
            )

    def test_malformed_tool_choice_dict_raises(self) -> None:
        # Missing ``name`` inside the function-choice nested model is a
        # pydantic-level validation error.
        with pytest.raises(ValueError):
            CompletionsRequest.model_validate(
                {
                    "model": "m",
                    "messages": [{"role": "user", "content": "x"}],
                    "tool_choice": {"type": "function", "function": {}},
                }
            )

    def test_missing_model_raises(self) -> None:
        with pytest.raises(ValueError):
            CompletionsRequest.model_validate({"messages": [{"role": "user", "content": "x"}]})

    def test_missing_messages_raises(self) -> None:
        with pytest.raises(ValueError):
            CompletionsRequest.model_validate({"model": "m"})

    def test_empty_messages_list_raises(self) -> None:
        # ``messages`` carries ``min_length=1``; an empty list must fail.
        with pytest.raises(ValueError):
            CompletionsRequest.model_validate({"model": "m", "messages": []})

    def test_empty_model_string_raises(self) -> None:
        # ``model`` carries ``min_length=1``.
        with pytest.raises(ValueError):
            CompletionsRequest.model_validate(
                {"model": "", "messages": [{"role": "user", "content": "x"}]}
            )

    def test_unknown_content_block_type_raises(self) -> None:
        # The discriminated union on content parts rejects unknown types.
        with pytest.raises(ValueError):
            CompletionsRequest.model_validate(
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
        # ``role`` is a Literal of the four OpenAI roles.
        with pytest.raises(ValueError):
            CompletionsRequest.model_validate(
                {
                    "model": "m",
                    "messages": [{"role": "developer", "content": "x"}],
                }
            )

    def test_unknown_extra_top_level_fields_are_ignored(self) -> None:
        # Pydantic's default config is ``extra="ignore"``; unknown fields
        # round-trip without raising and are absent from the model.
        req = CompletionsRequest.model_validate(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "frequency_penalty": 0.3,
                "user": "tobias",
            }
        )
        assert not hasattr(req, "frequency_penalty")
        assert not hasattr(req, "user")

    def test_unknown_extra_message_fields_are_ignored(self) -> None:
        # Same tolerance applies to the nested ``CompletionsMessage``.
        req = CompletionsRequest.model_validate(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "x", "extra_field": "ignored"}],
            }
        )
        assert req.messages[0].role == "user"
        assert not hasattr(req.messages[0], "extra_field")

    def test_system_with_list_content_concatenates_text_parts(self) -> None:
        req = _translate(
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
        # ``content`` validates as ``str | list | None`` at the pydantic
        # boundary; a bare int is rejected.
        with pytest.raises(ValueError):
            CompletionsRequest.model_validate(
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
            CompletionsRequest.model_validate(
                {
                    "model": "m",
                    "messages": [{"role": "user", "content": {"foo": "bar"}}],
                }
            )

    def test_tool_role_message_with_list_content(self) -> None:
        req = _translate(
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

    def test_tool_role_message_with_null_content_yields_empty_text(self) -> None:
        req = _translate(
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

    def test_system_with_null_content_yields_empty_system_prompt(self) -> None:
        # ``content: None`` on a system message validates fine but
        # contributes the empty string to the joined system prompt.
        req = _translate(
            {
                "model": "m",
                "messages": [
                    {"role": "system", "content": None},
                    {"role": "user", "content": "x"},
                ],
            }
        )
        assert req.system == ""

    def test_assistant_tool_call_with_malformed_json_args_falls_back_to_raw(self) -> None:
        req = _translate(
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
        req = _translate(
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
        # ``tool_choice`` is ``str | CompletionsNamedToolChoice | None``;
        # an int does not satisfy that union.
        with pytest.raises(ValueError):
            CompletionsRequest.model_validate(
                {
                    "model": "m",
                    "messages": [{"role": "user", "content": "x"}],
                    "tool_choice": 42,
                }
            )

    def test_stop_unsupported_shape_raises(self) -> None:
        # ``stop`` is ``str | list[str] | None``; a dict does not satisfy that.
        with pytest.raises(ValueError):
            CompletionsRequest.model_validate(
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
        body = canonical_to_completions_response(self._resp(), response_id=_RESPONSE_ID)
        assert isinstance(body, CompletionsResponse)
        assert body.id == _RESPONSE_ID
        assert body.object == "chat.completion"
        assert body.model == "vendor/model::Q4"
        assert isinstance(body.created, int)
        assert len(body.choices) == 1
        choice = body.choices[0]
        assert choice.index == 0
        assert choice.message.role == "assistant"
        assert choice.message.content == "hello"
        assert choice.message.tool_calls is None
        assert choice.finish_reason == "stop"
        assert body.usage.prompt_tokens == 0
        assert body.usage.completion_tokens == 0
        assert body.usage.total_tokens == 0

    def test_response_id_is_carried_through(self) -> None:
        body = canonical_to_completions_response(self._resp(), response_id="chatcmpl-explicit-123")
        assert body.id == "chatcmpl-explicit-123"

    def test_response_with_tool_calls(self) -> None:
        body = canonical_to_completions_response(
            self._resp(
                content=[
                    ToolUseBlock(id="c1", name="search", input={"q": "foo"}),
                ],
                stop_reason=StopReason.TOOL_USE,
            ),
            response_id=_RESPONSE_ID,
        )
        choice = body.choices[0]
        assert choice.finish_reason == "tool_calls"
        assert choice.message.content is None
        assert choice.message.tool_calls is not None
        assert len(choice.message.tool_calls) == 1
        call = choice.message.tool_calls[0]
        assert call.id == "c1"
        assert call.type == "function"
        assert call.function.name == "search"
        assert call.function.arguments == '{"q": "foo"}'

    def test_response_with_text_and_tool_call(self) -> None:
        body = canonical_to_completions_response(
            self._resp(
                content=[
                    TextBlock(text="ok"),
                    ToolUseBlock(id="c1", name="x", input={}),
                ],
                stop_reason=StopReason.TOOL_USE,
            ),
            response_id=_RESPONSE_ID,
        )
        msg = body.choices[0].message
        assert msg.content == "ok"
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1

    @pytest.mark.parametrize(
        "stop_reason,expected",
        [
            (StopReason.END_TURN, "stop"),
            (StopReason.MAX_TOKENS, "length"),
            (StopReason.TOOL_USE, "tool_calls"),
        ],
    )
    def test_finish_reason_mapping(self, stop_reason, expected) -> None:
        body = canonical_to_completions_response(
            self._resp(stop_reason=stop_reason), response_id=_RESPONSE_ID
        )
        assert body.choices[0].finish_reason == expected

    def test_usage_passes_through_canonical_counts_honestly(self) -> None:
        body = canonical_to_completions_response(
            self._resp(usage=CanonicalUsage(input_tokens=5, output_tokens=7)),
            response_id=_RESPONSE_ID,
        )
        assert body.usage.prompt_tokens == 5
        assert body.usage.completion_tokens == 7
        assert body.usage.total_tokens == 12


async def _drain(
    it: AsyncIterator[CompletionsStreamChunk],
) -> list[CompletionsStreamChunk]:
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
        # All chunks are typed StreamChunk instances.
        assert all(isinstance(c, CompletionsStreamChunk) for c in chunks)
        # First chunk: role
        assert chunks[0].id == "msg_x"
        assert chunks[0].object == "chat.completion.chunk"
        assert chunks[0].model == "m"
        assert chunks[0].choices[0].delta.role == "assistant"
        assert chunks[0].choices[0].delta.content is None
        # Then two content deltas
        assert chunks[1].choices[0].delta.content == "he"
        assert chunks[2].choices[0].delta.content == "llo"
        # Finish chunk
        assert chunks[-1].choices[0].finish_reason == "stop"
        assert chunks[-1].choices[0].delta.role is None
        assert chunks[-1].choices[0].delta.content is None
        assert chunks[-1].choices[0].delta.tool_calls is None

    async def test_tool_call_first_stream_emits_role_on_first_chunk(self) -> None:
        """A response that opens with a tool call (no leading text) must still carry
        role:assistant on the first chunk, or OpenAI-SDK delta accumulation breaks."""
        events: list[CanonicalStreamEvent] = [
            MessageStart(id="msg_x", model="m"),
            ContentBlockStart(
                index=0, block=ToolUseBlock(id="call_1", name="get_weather", input={})
            ),
            ContentBlockDelta(index=0, delta=ToolUseDelta(partial_json='{"city":"SF"}')),
            ContentBlockStop(index=0),
            MessageDelta(stop_reason=StopReason.TOOL_USE),
            MessageStop(),
        ]
        chunks = await _drain(
            canonical_stream_to_completions_chunks(
                _async_iter(events), model="m", response_id="msg_x"
            )
        )
        first = chunks[0].choices[0].delta
        assert first.role == "assistant"
        assert first.tool_calls is not None
        assert first.tool_calls[0].function.name == "get_weather"

    async def test_message_delta_usage_emits_final_usage_chunk(self) -> None:
        """A MessageDelta carrying usage produces a trailing usage chunk with an
        empty choices list and populated totals (include_usage shape). (F4)"""
        events: list[CanonicalStreamEvent] = [
            MessageStart(id="msg_x", model="m"),
            ContentBlockStart(index=0, block=TextBlock(text="")),
            ContentBlockDelta(index=0, delta=TextDelta(text="hi")),
            ContentBlockStop(index=0),
            MessageDelta(
                stop_reason=StopReason.END_TURN,
                usage=CanonicalUsage(input_tokens=6, output_tokens=2),
            ),
            MessageStop(),
        ]
        chunks = await _drain(
            canonical_stream_to_completions_chunks(
                _async_iter(events), model="m", response_id="msg_x"
            )
        )
        usage_chunk = chunks[-1]
        assert usage_chunk.choices == []
        assert usage_chunk.usage is not None
        assert usage_chunk.usage.prompt_tokens == 6
        assert usage_chunk.usage.completion_tokens == 2
        assert usage_chunk.usage.total_tokens == 8

    async def test_message_delta_without_usage_emits_no_usage_chunk(self) -> None:
        events: list[CanonicalStreamEvent] = [
            MessageStart(id="msg_x", model="m"),
            ContentBlockStart(index=0, block=TextBlock(text="")),
            ContentBlockDelta(index=0, delta=TextDelta(text="hi")),
            ContentBlockStop(index=0),
            MessageDelta(stop_reason=StopReason.END_TURN),
            MessageStop(),
        ]
        chunks = await _drain(
            canonical_stream_to_completions_chunks(
                _async_iter(events), model="m", response_id="msg_x"
            )
        )
        assert all(c.usage is None for c in chunks)

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

    async def test_second_text_block_does_not_re_emit_role(self) -> None:
        # Two consecutive text blocks share one role-emit chunk;
        # the second block's start yields nothing.
        events: list[CanonicalStreamEvent] = [
            ContentBlockStart(index=0, block=TextBlock(text="")),
            ContentBlockStop(index=0),
            ContentBlockStart(index=1, block=TextBlock(text="")),
            ContentBlockDelta(index=1, delta=TextDelta(text="ok")),
            MessageDelta(stop_reason=StopReason.END_TURN),
        ]
        chunks = await _drain(
            canonical_stream_to_completions_chunks(
                _async_iter(events), model="m", response_id="msg_x"
            )
        )
        role_chunks = [c for c in chunks if c.choices[0].delta.role == "assistant"]
        assert len(role_chunks) == 1

    async def test_message_delta_without_stop_reason_emits_stop(self) -> None:
        events: list[CanonicalStreamEvent] = [
            MessageDelta(stop_reason=None),
        ]
        chunks = await _drain(
            canonical_stream_to_completions_chunks(
                _async_iter(events), model="m", response_id="msg_x"
            )
        )
        assert chunks[-1].choices[0].finish_reason == "stop"

    async def test_unrecognized_content_block_is_silently_skipped(self) -> None:
        # Tool-result blocks do not appear in assistant responses; the chunker
        # drops them rather than emitting a malformed chunk.
        events: list[CanonicalStreamEvent] = [
            ContentBlockStart(
                index=0, block=ToolResultBlock(tool_use_id="t1", content=[TextBlock(text="r")])
            ),
        ]
        chunks = await _drain(
            canonical_stream_to_completions_chunks(
                _async_iter(events), model="m", response_id="msg_x"
            )
        )
        assert chunks == []

    async def test_unrecognized_content_block_delta_is_silently_skipped(self) -> None:
        # The mapper guards against a future canonical delta variant that
        # is not TextDelta or ToolUseDelta. Construct one ad-hoc to cover
        # the defensive ``return None`` branch.
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class _UnknownDelta:
            pass

        events: list[CanonicalStreamEvent] = [
            ContentBlockDelta(index=0, delta=_UnknownDelta()),  # type: ignore[arg-type]
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
        first = chunks[0].choices[0].delta
        assert first.tool_calls is not None
        assert len(first.tool_calls) == 1
        opened = first.tool_calls[0]
        assert opened.index == 0
        assert opened.id == "c1"
        assert opened.type == "function"
        assert opened.function is not None
        assert opened.function.name == "search"
        assert opened.function.arguments == ""
        # Then two argument-fragment chunks
        for fragment_chunk, expected_args in (
            (chunks[1], '{"q":'),
            (chunks[2], '"foo"}'),
        ):
            tool_calls = fragment_chunk.choices[0].delta.tool_calls
            assert tool_calls is not None
            assert len(tool_calls) == 1
            assert tool_calls[0].index == 0
            assert tool_calls[0].id is None
            assert tool_calls[0].function is not None
            assert tool_calls[0].function.arguments == expected_args
        assert chunks[-1].choices[0].finish_reason == "tool_calls"

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
        assert chunks[0].choices[0].delta.role == "assistant"
        assert chunks[1].choices[0].delta.content == "thinking"
        tool_open_calls = chunks[2].choices[0].delta.tool_calls
        assert tool_open_calls is not None
        tool_open = tool_open_calls[0]
        assert tool_open.index == 0
        assert tool_open.id == "c1"
        arg_calls = chunks[3].choices[0].delta.tool_calls
        assert arg_calls is not None
        assert arg_calls[0].index == 0
        assert arg_calls[0].function is not None
        assert arg_calls[0].function.arguments == "{}"
        assert chunks[-1].choices[0].finish_reason == "tool_calls"

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
        assert chunks[-1].choices[0].finish_reason == expected


def test_canonical_request_with_minimal_payload_uses_request_helper() -> None:
    """Spot-check that the dataclass roundtrips for the simplest payload."""
    req = _translate({"model": "m", "messages": [{"role": "user", "content": "x"}]})
    assert isinstance(req, CanonicalChatRequest)
    assert isinstance(req.messages[0], CanonicalMessage)


def test_message_from_request_rejects_system_role_defensively() -> None:
    """The private message converter guards against being called for system role.

    ``completions_to_canonical_request`` filters system messages out
    before reaching ``_message_from_request``; the guard exists as
    defense-in-depth for any future direct caller.
    """
    from lilbee.server.chat_completions_api.models import CompletionsMessage
    from lilbee.server.chat_completions_api.translate import _message_from_request

    system_msg = CompletionsMessage(role="system", content="be terse")
    with pytest.raises(ValueError, match="system messages"):
        _message_from_request(system_msg)
