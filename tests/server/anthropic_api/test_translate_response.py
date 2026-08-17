"""Response translation: canonical -> Anthropic message."""

from __future__ import annotations

from typing import ClassVar

from lilbee.core.config.enums import ReasoningMode
from lilbee.server.anthropic_api.translate import canonical_to_messages_response
from lilbee.server.chat_dispatch.canonical import (
    CanonicalResponse,
    CanonicalUsage,
    StopReason,
    TextBlock,
    ToolUseBlock,
)


def _response(content, stop_reason=StopReason.END_TURN) -> CanonicalResponse:
    return CanonicalResponse(
        id="x",
        model="m",
        content=content,
        stop_reason=stop_reason,
        usage=CanonicalUsage(input_tokens=10, output_tokens=5),
    )


def test_plain_text_response():
    body = canonical_to_messages_response(_response([TextBlock(text="hello")]), response_id="msg_1")
    assert body.id == "msg_1"
    assert body.role == "assistant"
    assert body.content == [{"type": "text", "text": "hello"}]
    assert body.stop_reason == "end_turn"
    assert body.usage.input_tokens == 10
    assert body.usage.output_tokens == 5
    # SDK clients expect the field present-but-null
    assert body.model_dump()["stop_sequence"] is None


def test_inline_think_becomes_thinking_block():
    body = canonical_to_messages_response(
        _response([TextBlock(text="<think>plan</think>answer")]), response_id="msg_1"
    )
    assert body.content == [
        {"type": "thinking", "thinking": "plan"},
        {"type": "text", "text": "answer"},
    ]


def test_tool_use_response_maps_blocks_and_stop_reason():
    body = canonical_to_messages_response(
        _response(
            [TextBlock(text="calling"), ToolUseBlock(id="t1", name="ls", input={"p": "."})],
            stop_reason=StopReason.TOOL_USE,
        ),
        response_id="msg_1",
    )
    assert body.stop_reason == "tool_use"
    assert body.content == [
        {"type": "text", "text": "calling"},
        {"type": "tool_use", "id": "t1", "name": "ls", "input": {"p": "."}},
    ]


def test_empty_content_still_emits_text_block():
    body = canonical_to_messages_response(_response([]), response_id="msg_1")
    assert body.content == [{"type": "text", "text": ""}]


class TestReasoningModeOnResponse:
    """How each mode presents a thinking model's ``<think>`` text."""

    _THINKING: ClassVar = [TextBlock(text="<think>plan</think>answer")]

    def test_separate_keeps_the_thinking_block(self):
        body = canonical_to_messages_response(
            _response(self._THINKING), response_id="msg_1", mode=ReasoningMode.SEPARATE
        )
        assert body.content == [
            {"type": "thinking", "thinking": "plan"},
            {"type": "text", "text": "answer"},
        ]

    def test_inline_folds_the_thinking_into_the_text_block(self):
        body = canonical_to_messages_response(
            _response(self._THINKING), response_id="msg_1", mode=ReasoningMode.INLINE
        )
        assert body.content == [{"type": "text", "text": "plan\n\nanswer"}]

    def test_off_drops_the_thinking_entirely(self):
        """A template that ignores the request still thinks; the answer stays clean."""
        body = canonical_to_messages_response(
            _response(self._THINKING), response_id="msg_1", mode=ReasoningMode.OFF
        )
        assert body.content == [{"type": "text", "text": "answer"}]

    def test_off_with_only_thinking_still_emits_a_text_block(self):
        body = canonical_to_messages_response(
            _response([TextBlock(text="<think>plan</think>")]),
            response_id="msg_1",
            mode=ReasoningMode.OFF,
        )
        assert body.content == [{"type": "text", "text": ""}]
