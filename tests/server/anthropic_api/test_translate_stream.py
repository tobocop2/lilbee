"""Stream translation: canonical events -> Anthropic SSE event pairs."""

from __future__ import annotations

import json

import pytest

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


async def _drain(events):
    async def _aiter():
        for event in events:
            yield event

    return [
        pair
        async for pair in canonical_stream_to_anthropic_events(
            _aiter(), model="m", response_id="msg_1"
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
