"""Stream translation: canonical events -> Anthropic SSE event pairs."""

from __future__ import annotations

import json

import pytest

from lilbee.core.config.enums import ReasoningMode
from lilbee.server.anthropic_api.translate import canonical_stream_to_anthropic_events
from lilbee.server.chat_dispatch.canonical import (
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
    ToolUseBlock,
    ToolUseDelta,
)


async def _drain(events, mode=ReasoningMode.SEPARATE):
    async def _aiter():
        for event in events:
            yield event

    return [
        pair
        async for pair in canonical_stream_to_anthropic_events(
            _aiter(), model="m", response_id="msg_1", mode=mode
        )
    ]


def _types(pairs):
    return [t for t, _ in pairs]


@pytest.mark.asyncio
async def test_plain_text_stream_event_sequence():
    pairs = await _drain(
        [
            MessageStart(id="x", model="m"),
            ContentBlockStart(index=0, block=TextBlock(text="")),
            ContentBlockDelta(index=0, delta=TextDelta(text="hel")),
            ContentBlockDelta(index=0, delta=TextDelta(text="lo")),
            ContentBlockStop(index=0),
            MessageDelta(
                stop_reason=StopReason.END_TURN,
                usage=CanonicalUsage(input_tokens=3, output_tokens=2),
            ),
            MessageStop(),
        ]
    )
    assert _types(pairs) == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    start = pairs[1][1]
    assert start["index"] == 0
    assert start["content_block"] == {"type": "text", "text": ""}
    assert pairs[2][1]["delta"] == {"type": "text_delta", "text": "hel"}
    delta = pairs[5][1]
    assert delta["delta"]["stop_reason"] == "end_turn"
    assert delta["usage"] == {"input_tokens": 3, "output_tokens": 2}
    assert pairs[0][1]["message"]["id"] == "msg_1"


@pytest.mark.asyncio
async def test_think_tokens_split_into_thinking_then_text_blocks():
    pairs = await _drain(
        [
            ContentBlockStart(index=0, block=TextBlock(text="")),
            ContentBlockDelta(index=0, delta=TextDelta(text="<think>plan")),
            ContentBlockDelta(index=0, delta=TextDelta(text="</think>answer")),
            ContentBlockStop(index=0),
            MessageDelta(stop_reason=StopReason.END_TURN),
            MessageStop(),
        ]
    )
    kinds = [
        (t, p.get("content_block", {}).get("type"), p.get("index"))
        for t, p in pairs
        if t == "content_block_start"
    ]
    assert kinds == [
        ("content_block_start", "thinking", 0),
        ("content_block_start", "text", 1),
    ]
    deltas = [p["delta"] for t, p in pairs if t == "content_block_delta"]
    assert {"type": "thinking_delta", "thinking": "plan"} in deltas
    assert {"type": "text_delta", "text": "answer"} in deltas
    stops = [p["index"] for t, p in pairs if t == "content_block_stop"]
    assert stops == [0, 1]


@pytest.mark.asyncio
async def test_incremental_tool_call_forwards_argument_deltas():
    """Mirrors dispatch's real shape: input {} at start, arguments as deltas."""
    pairs = await _drain(
        [
            ContentBlockStart(index=0, block=TextBlock(text="")),
            ContentBlockDelta(index=0, delta=TextDelta(text="calling")),
            ContentBlockStart(index=1, block=ToolUseBlock(id="t1", name="ls", input={})),
            ContentBlockDelta(index=1, delta=ToolUseDelta(partial_json='{"p":')),
            ContentBlockDelta(index=1, delta=ToolUseDelta(partial_json=' "."}')),
            ContentBlockStop(index=1),
            MessageDelta(stop_reason=StopReason.TOOL_USE),
            MessageStop(),
        ]
    )
    tool_starts = [
        p
        for t, p in pairs
        if t == "content_block_start" and p["content_block"]["type"] == "tool_use"
    ]
    assert len(tool_starts) == 1
    tool_index = tool_starts[0]["index"]
    assert tool_starts[0]["content_block"] == {
        "type": "tool_use",
        "id": "t1",
        "name": "ls",
        "input": {},
    }
    json_deltas = [
        p
        for t, p in pairs
        if t == "content_block_delta" and p["delta"]["type"] == "input_json_delta"
    ]
    assert [d["delta"]["partial_json"] for d in json_deltas] == ['{"p":', ' "."}']
    assert all(d["index"] == tool_index for d in json_deltas)
    # The open text block closes before the tool block opens, and the tool
    # block closes on the canonical stop.
    stop_indexes = [p["index"] for t, p in pairs if t == "content_block_stop"]
    assert stop_indexes == [0, tool_index]
    assert pairs[-2][0] == "message_delta"
    assert pairs[-2][1]["delta"]["stop_reason"] == "tool_use"
    assert pairs[-1][0] == "message_stop"


@pytest.mark.asyncio
async def test_whole_tool_call_on_start_block_still_carries_arguments():
    """A provider announcing the full call up front still yields arguments."""
    pairs = await _drain(
        [
            ContentBlockStart(index=0, block=ToolUseBlock(id="t1", name="ls", input={"p": "."})),
            ContentBlockStop(index=0),
            MessageDelta(stop_reason=StopReason.TOOL_USE),
            MessageStop(),
        ]
    )
    json_deltas = [
        p
        for t, p in pairs
        if t == "content_block_delta" and p["delta"]["type"] == "input_json_delta"
    ]
    assert len(json_deltas) == 1
    assert json.loads(json_deltas[0]["delta"]["partial_json"]) == {"p": "."}


@pytest.mark.asyncio
async def test_orphan_tool_delta_is_dropped():
    """An argument delta with no open tool block must not crash the stream."""
    pairs = await _drain(
        [
            ContentBlockDelta(index=0, delta=ToolUseDelta(partial_json='{"p": 1}')),
            MessageDelta(stop_reason=StopReason.END_TURN),
            MessageStop(),
        ]
    )
    assert _types(pairs) == ["message_start", "message_delta", "message_stop"]


def test_mapper_skips_empty_tokens():
    """A token with no content must not open a block or emit a delta."""
    from lilbee.retrieval.reasoning import StreamToken
    from lilbee.server.anthropic_api.translate import _AnthropicStreamMapper

    mapper = _AnthropicStreamMapper()
    assert mapper._text_events([StreamToken(content="", is_reasoning=False)]) == []


def _thinking_stream():
    return [
        ContentBlockStart(index=0, block=TextBlock(text="")),
        ContentBlockDelta(index=0, delta=TextDelta(text="<think>pl")),
        ContentBlockDelta(index=0, delta=TextDelta(text="an</think>")),
        ContentBlockDelta(index=0, delta=TextDelta(text="answer")),
        ContentBlockStop(index=0),
        MessageDelta(stop_reason=StopReason.END_TURN),
        MessageStop(),
    ]


def _deltas(pairs):
    return [payload["delta"] for kind, payload in pairs if kind == "content_block_delta"]


def _block_kinds(pairs):
    return [
        payload["content_block"]["type"] for kind, payload in pairs if kind == "content_block_start"
    ]


class TestReasoningModeOnStream:
    """Each mode re-blocks the streamed ``<think>`` text its own way."""

    @pytest.mark.asyncio
    async def test_separate_streams_a_thinking_block(self):
        pairs = await _drain(_thinking_stream(), mode=ReasoningMode.SEPARATE)
        assert _block_kinds(pairs) == ["thinking", "text"]
        assert _deltas(pairs) == [
            {"type": "thinking_delta", "thinking": "pl"},
            {"type": "thinking_delta", "thinking": "an"},
            {"type": "text_delta", "text": "answer"},
        ]

    @pytest.mark.asyncio
    async def test_inline_streams_the_thinking_as_text(self):
        pairs = await _drain(_thinking_stream(), mode=ReasoningMode.INLINE)
        assert _block_kinds(pairs) == ["text"]
        assert _deltas(pairs) == [
            {"type": "text_delta", "text": "pl"},
            {"type": "text_delta", "text": "an"},
            {"type": "text_delta", "text": "answer"},
        ]

    @pytest.mark.asyncio
    async def test_off_emits_no_thinking_delta(self):
        pairs = await _drain(_thinking_stream(), mode=ReasoningMode.OFF)
        assert _block_kinds(pairs) == ["text"]
        assert _deltas(pairs) == [{"type": "text_delta", "text": "answer"}]

    @pytest.mark.asyncio
    async def test_off_with_only_thinking_opens_no_block(self):
        pairs = await _drain(
            [
                ContentBlockStart(index=0, block=TextBlock(text="")),
                ContentBlockDelta(index=0, delta=TextDelta(text="<think>plan</think>")),
                ContentBlockStop(index=0),
                MessageDelta(stop_reason=StopReason.END_TURN),
                MessageStop(),
            ],
            mode=ReasoningMode.OFF,
        )
        assert _types(pairs) == ["message_start", "message_delta", "message_stop"]


def _text_stream(chunks):
    return [
        ContentBlockStart(index=0, block=TextBlock(text="")),
        *(ContentBlockDelta(index=0, delta=TextDelta(text=chunk)) for chunk in chunks),
        ContentBlockStop(index=0),
        MessageDelta(stop_reason=StopReason.END_TURN),
        MessageStop(),
    ]


class TestPseudoThinkingStripOnStream:
    """OFF drops a reply-initial pseudo-thinking block, even split across deltas."""

    @pytest.mark.asyncio
    async def test_off_drops_an_unclosed_tag_split_across_deltas(self):
        pairs = await _drain(
            _text_stream(["<anthropic_", "thinking> plan the", " whole reply"]),
            mode=ReasoningMode.OFF,
        )
        assert _types(pairs) == ["message_start", "message_delta", "message_stop"]

    @pytest.mark.asyncio
    async def test_off_drops_a_closed_pseudo_block_and_streams_the_answer(self):
        pairs = await _drain(
            _text_stream(["<anti_codeblock>plan</anti_", "codeblock>ans", "wer"]),
            mode=ReasoningMode.OFF,
        )
        assert _block_kinds(pairs) == ["text"]
        assert _deltas(pairs) == [
            {"type": "text_delta", "text": "ans"},
            {"type": "text_delta", "text": "wer"},
        ]

    @pytest.mark.asyncio
    async def test_off_leaves_mid_reply_markup_untouched(self):
        pairs = await _drain(
            _text_stream(["answer <anti_codeblock>x", "</anti_codeblock>"]),
            mode=ReasoningMode.OFF,
        )
        assert _block_kinds(pairs) == ["text"]
        joined = "".join(delta["text"] for delta in _deltas(pairs))
        assert joined == "answer <anti_codeblock>x</anti_codeblock>"

    @pytest.mark.asyncio
    async def test_separate_streams_pseudo_tags_verbatim(self):
        pairs = await _drain(
            _text_stream(["<anthropic_thinking> plan"]), mode=ReasoningMode.SEPARATE
        )
        assert _block_kinds(pairs) == ["text"]
        assert _deltas(pairs) == [{"type": "text_delta", "text": "<anthropic_thinking> plan"}]
