"""Tests for ``parse_response`` over each shipped schema."""

from __future__ import annotations

import json

from lilbee.providers.worker.response_parser import SCHEMAS, ModelFamily, parse_response


def test_qwen3_extracts_single_tool_call_with_json_args() -> None:
    """Qwen3 emits ``<tool_call>{"name":..., "arguments":...}</tool_call>``."""
    text = '<tool_call>\n{"name": "search", "arguments": {"q": "foo"}}\n</tool_call>'
    parsed = parse_response(text, SCHEMAS[ModelFamily.QWEN3])
    assert len(parsed.tool_calls) == 1
    call = parsed.tool_calls[0]
    assert call.name == "search"
    assert json.loads(call.arguments) == {"q": "foo"}


def test_qwen3_keeps_leading_content_before_tool_call() -> None:
    """Content before ``<tool_call>`` is preserved as the cleaned content."""
    text = 'Let me search.\n<tool_call>{"name": "f", "arguments": {}}</tool_call>'
    parsed = parse_response(text, SCHEMAS[ModelFamily.QWEN3])
    assert parsed.content.strip() == "Let me search."
    assert len(parsed.tool_calls) == 1


def test_qwen3_extracts_multiple_parallel_tool_calls() -> None:
    """Two ``<tool_call>...</tool_call>`` blocks become two structured calls."""
    text = (
        '<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>\n'
        '<tool_call>{"name": "b", "arguments": {"y": 2}}</tool_call>'
    )
    parsed = parse_response(text, SCHEMAS[ModelFamily.QWEN3])
    assert [c.name for c in parsed.tool_calls] == ["a", "b"]


def test_qwen3_handles_thinking_block() -> None:
    """A leading ``<think>...</think>`` block is captured as thinking, not content."""
    text = (
        "<think>I should search.</think>"
        '<tool_call>{"name": "search", "arguments": {"q": "x"}}</tool_call>'
    )
    parsed = parse_response(text, SCHEMAS[ModelFamily.QWEN3])
    assert len(parsed.tool_calls) == 1


def test_qwen3_text_only_response_returns_no_tool_calls() -> None:
    """When the model never emits a ``<tool_call>`` block, tool_calls is empty."""
    parsed = parse_response("just chatting today", SCHEMAS[ModelFamily.QWEN3])
    assert parsed.tool_calls == ()
    assert "just chatting today" in parsed.content


def test_qwen3_coder_extracts_function_and_parameters() -> None:
    """Qwen3-Coder uses XML-style ``<function=>`` and ``<parameter=>`` markers."""
    text = (
        "<tool_call>\n<function=search>\n"
        "<parameter=q>\nfoo\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    parsed = parse_response(text, SCHEMAS[ModelFamily.QWEN3_CODER])
    assert len(parsed.tool_calls) == 1
    call = parsed.tool_calls[0]
    assert call.name == "search"
    assert json.loads(call.arguments) == {"q": "foo"}


def test_mistral_extracts_tool_call_array() -> None:
    """Mistral emits ``[TOOL_CALLS] [{"name":..., "arguments":...}]``."""
    text = 'Here is the call: [TOOL_CALLS] [{"name": "search", "arguments": {"q": "foo"}}]'
    parsed = parse_response(text, SCHEMAS[ModelFamily.MISTRAL])
    assert len(parsed.tool_calls) == 1
    call = parsed.tool_calls[0]
    assert call.name == "search"
    assert json.loads(call.arguments) == {"q": "foo"}


def test_mistral_keeps_prefix_content() -> None:
    """Content before ``[TOOL_CALLS]`` is preserved."""
    text = 'Reasoning prose.\n[TOOL_CALLS] [{"name": "f", "arguments": {}}]'
    parsed = parse_response(text, SCHEMAS[ModelFamily.MISTRAL])
    assert "Reasoning prose." in parsed.content
    assert len(parsed.tool_calls) == 1


def test_malformed_json_inside_tool_call_falls_back_to_text() -> None:
    """A ``<tool_call>`` block whose body is not JSON returns no tool calls.

    The parse layer catches the schema's TypeError/ValueError and returns the
    raw text as content rather than raising out of the worker.
    """
    text = "<tool_call>NOT JSON{</tool_call>"
    parsed = parse_response(text, SCHEMAS[ModelFamily.QWEN3])
    assert parsed.tool_calls == ()


def test_empty_input_returns_empty_response() -> None:
    """Parsing an empty string never raises."""
    parsed = parse_response("", SCHEMAS[ModelFamily.QWEN3])
    assert parsed.content == ""
    assert parsed.tool_calls == ()
