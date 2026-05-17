"""Tests for the protocol-neutral chat-dispatch types."""

from __future__ import annotations

import pytest

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
    ToolResultBlock,
    ToolUseBlock,
    ToolUseDelta,
)


def test_canonical_request_minimal_defaults() -> None:
    req = CanonicalChatRequest(
        model="m",
        messages=[CanonicalMessage(role="user", content=[TextBlock(text="hi")])],
    )
    assert req.tools is None
    assert req.tool_choice is None
    assert req.system is None
    assert req.stream is False
    assert req.temperature is None
    assert req.top_p is None
    assert req.top_k is None
    assert req.max_tokens is None
    assert req.stop is None


def test_canonical_request_full_payload() -> None:
    tool = CanonicalTool(name="search", description="Search docs", input_schema={"type": "object"})
    choice = CanonicalToolChoice(mode="tool", tool_name="search")
    req = CanonicalChatRequest(
        model="m",
        messages=[CanonicalMessage(role="user", content=[TextBlock(text="hi")])],
        system="be terse",
        tools=[tool],
        tool_choice=choice,
        temperature=0.1,
        top_p=0.9,
        top_k=40,
        max_tokens=128,
        stop=["</s>"],
        stream=True,
    )
    assert req.system == "be terse"
    assert req.tools == [tool]
    assert req.tool_choice == choice
    assert req.stream is True


def test_canonical_message_from_string_normalizes_text_block() -> None:
    msg = CanonicalMessage.from_string(role="user", text="hello")
    assert msg.role == "user"
    assert msg.content == [TextBlock(text="hello")]


def test_canonical_response_text_only() -> None:
    resp = CanonicalResponse(
        id="msg_1",
        model="m",
        content=[TextBlock(text="ok")],
        stop_reason=StopReason.END_TURN,
        usage=CanonicalUsage(input_tokens=3, output_tokens=1),
    )
    assert resp.stop_reason == StopReason.END_TURN
    assert resp.usage.input_tokens == 3
    assert resp.usage.output_tokens == 1


def test_canonical_response_with_tool_use_block() -> None:
    resp = CanonicalResponse(
        id="msg_2",
        model="m",
        content=[
            TextBlock(text=""),
            ToolUseBlock(id="t1", name="search", input={"q": "foo"}),
        ],
        stop_reason=StopReason.TOOL_USE,
        usage=CanonicalUsage(input_tokens=5, output_tokens=2),
    )
    tool_blocks = [b for b in resp.content if isinstance(b, ToolUseBlock)]
    assert len(tool_blocks) == 1
    assert tool_blocks[0].name == "search"
    assert tool_blocks[0].input == {"q": "foo"}


def test_tool_result_block_carries_tool_use_id_and_default_error_flag() -> None:
    block = ToolResultBlock(tool_use_id="t1", content=[TextBlock(text="result")])
    assert block.tool_use_id == "t1"
    assert block.is_error is False
    assert block.content == [TextBlock(text="result")]


def test_tool_result_block_can_flag_error() -> None:
    block = ToolResultBlock(
        tool_use_id="t1",
        content=[TextBlock(text="boom")],
        is_error=True,
    )
    assert block.is_error is True


def test_image_block_carries_media_type_and_data() -> None:
    block = ImageBlock(media_type="image/png", data=b"\x89PNG")
    assert block.media_type == "image/png"
    assert block.data == b"\x89PNG"
    assert block.type == "image"


def test_stop_reason_is_string_enum() -> None:
    # StrEnum members behave as strings; this lets translators emit them
    # directly into protocol envelopes without an explicit ``.value``.
    assert StopReason.END_TURN == "end_turn"
    assert StopReason.MAX_TOKENS == "max_tokens"
    assert StopReason.STOP_SEQUENCE == "stop_sequence"
    assert StopReason.TOOL_USE == "tool_use"
    assert StopReason.ERROR == "error"


def test_canonical_tool_choice_modes() -> None:
    for mode in ("auto", "any", "none", "tool"):
        choice = CanonicalToolChoice(mode=mode)  # type: ignore[arg-type]
        assert choice.mode == mode
    forced = CanonicalToolChoice(mode="tool", tool_name="search")
    assert forced.tool_name == "search"


@pytest.mark.parametrize(
    "event",
    [
        MessageStart(id="m1", model="m"),
        ContentBlockStart(index=0, block=TextBlock(text="")),
        ContentBlockDelta(index=0, delta=TextDelta(text="hi")),
        ContentBlockDelta(index=0, delta=ToolUseDelta(partial_json='{"q"')),
        ContentBlockStop(index=0),
        MessageDelta(stop_reason=StopReason.END_TURN),
        MessageDelta(usage=CanonicalUsage(input_tokens=1, output_tokens=2)),
        MessageStop(),
    ],
)
def test_canonical_stream_event_variants_construct(event: CanonicalStreamEvent) -> None:
    # Each variant must be reachable through the discriminated union so
    # the dispatch and translation layers can match on it directly.
    assert event is not None


def test_canonical_message_holds_tool_result_blocks() -> None:
    # ``role="tool"`` messages carry tool_result blocks; this is the shape
    # the dispatch layer expects when translating multi-turn tool flows.
    result = ToolResultBlock(tool_use_id="t1", content=[TextBlock(text="42")])
    msg = CanonicalMessage(role="tool", content=[result])
    assert msg.role == "tool"
    assert msg.content == [result]
