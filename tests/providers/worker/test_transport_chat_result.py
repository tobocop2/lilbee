"""ChatRequest tool fields and ChatResult / ToolCall / ToolCallDelta shape."""

from __future__ import annotations

from lilbee.providers.worker.transport import (
    ChatRequest,
    ChatResult,
    FinishReason,
    ToolCall,
    ToolCallDelta,
)


def test_chat_request_carries_tools() -> None:
    req = ChatRequest(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "x", "parameters": {}}}],
        tool_choice="auto",
    )
    assert req.tools is not None
    assert req.tool_choice == "auto"


def test_chat_request_tools_default_none() -> None:
    req = ChatRequest(messages=[{"role": "user", "content": "hi"}])
    assert req.tools is None
    assert req.tool_choice is None


def test_chat_request_accepts_content_blocks() -> None:
    """``messages`` is widened to ``list[dict[str, Any]]`` for content-blocks."""
    blocks = [{"type": "text", "text": "hi"}]
    req = ChatRequest(messages=[{"role": "user", "content": blocks}])
    assert req.messages[0]["content"] == blocks


def test_chat_result_text_only() -> None:
    res = ChatResult(text="hello", tool_calls=(), finish_reason=FinishReason.STOP)
    assert res.text == "hello"
    assert res.tool_calls == ()
    assert res.finish_reason == FinishReason.STOP


def test_chat_result_with_tool_calls() -> None:
    tc = ToolCall(id="call_1", name="search", arguments='{"q":"foo"}')
    res = ChatResult(text="", tool_calls=(tc,), finish_reason=FinishReason.TOOL_CALLS)
    assert res.tool_calls[0].name == "search"
    assert res.tool_calls[0].arguments == '{"q":"foo"}'
    assert res.finish_reason == FinishReason.TOOL_CALLS


def test_finish_reason_string_values() -> None:
    """``FinishReason`` is a StrEnum mirroring OpenAI's vocabulary."""
    assert FinishReason.STOP == "stop"
    assert FinishReason.LENGTH == "length"
    assert FinishReason.TOOL_CALLS == "tool_calls"
    assert FinishReason.CONTENT_FILTER == "content_filter"


def test_tool_call_delta_partial_shape() -> None:
    delta = ToolCallDelta(index=0, id="c1", name="f", arguments_delta='{"q":')
    assert delta.index == 0
    assert delta.id == "c1"
    assert delta.name == "f"
    assert delta.arguments_delta == '{"q":'


def test_tool_call_delta_all_optional() -> None:
    delta = ToolCallDelta(index=2, id=None, name=None, arguments_delta=None)
    assert delta.index == 2
    assert delta.id is None
