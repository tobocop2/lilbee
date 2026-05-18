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


def test_qwen3_strips_thinking_block_from_content() -> None:
    """``<think>...</think>`` is removed from content; tool calls still extract.

    ParsedResponse drops thinking entirely for Stage 1 (no ``thinking`` field
    on the dataclass). Verifying here that the schema's ``x-regex-substitutions``
    removes the block from the emitted content rather than letting it leak.
    """
    text = (
        "<think>I should search.</think>"
        '<tool_call>{"name": "search", "arguments": {"q": "x"}}</tool_call>'
    )
    parsed = parse_response(text, SCHEMAS[ModelFamily.QWEN3])
    assert len(parsed.tool_calls) == 1
    assert "I should search" not in parsed.content


def test_qwen3_preserves_content_after_tool_call() -> None:
    """Text after ``</tool_call>`` is preserved, not silently dropped."""
    text = 'Looking it up. <tool_call>{"name": "f", "arguments": {}}</tool_call> Done.'
    parsed = parse_response(text, SCHEMAS[ModelFamily.QWEN3])
    assert len(parsed.tool_calls) == 1
    assert "Looking it up." in parsed.content
    assert "Done." in parsed.content


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


def test_recursive_parse_value_error_falls_back_to_raw_text() -> None:
    """A ``<tool_call>`` block with invalid JSON triggers ``recursive_parse``'s
    JSON parser path and raises ``ValueError``; the parser catches it and
    returns the raw text as content rather than propagating.
    """
    # `{not_json}` matches the outer regex (braces present) but breaks at the
    # `x-parser: "json"` step inside, raising ValueError from recursive_parse.
    text = "<tool_call>{not_json}</tool_call>"
    parsed = parse_response(text, SCHEMAS[ModelFamily.QWEN3])
    assert parsed.content == text
    assert parsed.tool_calls == ()


def test_root_regex_miss_returns_raw_text() -> None:
    """A schema whose root regex doesn't match the input returns the raw text.

    ``recursive_parse`` yields ``None`` when the root extractor fails; the
    parse layer detects that and falls back to the original text instead of
    raising.
    """
    schema = {
        "type": "object",
        "x-regex": r"NEVER_MATCHES_(\w+)",
        "properties": {"content": {"type": "string"}, "tool_calls": {"type": "array"}},
    }
    parsed = parse_response("just regular text", schema)
    assert parsed.content == "just regular text"
    assert parsed.tool_calls == ()


def test_gemma4_extracts_wrapped_function_call() -> None:
    """Gemma 4 schema wraps each call in ``{type:"function", function:{...}}``.

    Exercises the function-wrapped branch of ``_coerce_one`` where the entry
    has a nested ``function`` dict, distinct from Mistral/Qwen3 schemas that
    emit flat ``{name, arguments}`` dicts.
    """
    text = '<|tool_call>call:weather{"city":"Tokyo"}<tool_call|>'
    parsed = parse_response(text, SCHEMAS[ModelFamily.GEMMA4])
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "weather"


def test_non_dict_entry_in_tool_calls_is_dropped() -> None:
    """``_tool_calls_from_parsed`` skips list items that are not dicts."""
    from lilbee.providers.worker.response_parser.parse import _tool_calls_from_parsed

    out = _tool_calls_from_parsed(["not-a-dict", {"name": "x", "arguments": {}}, 42])
    assert [call.name for call in out] == ["x"]


def test_non_list_tool_calls_returns_empty() -> None:
    """``_tool_calls_from_parsed`` returns ``()`` when the raw value isn't a list."""
    from lilbee.providers.worker.response_parser.parse import _tool_calls_from_parsed

    assert _tool_calls_from_parsed(None) == ()
    assert _tool_calls_from_parsed("not a list") == ()
    assert _tool_calls_from_parsed({"name": "x"}) == ()


def test_entry_with_empty_name_is_dropped() -> None:
    """``_coerce_one`` returns ``None`` when the parsed entry has no name."""
    from lilbee.providers.worker.response_parser.parse import _coerce_one

    assert _coerce_one({"name": "", "arguments": {}}) is None
    assert _coerce_one({"name": 42, "arguments": {}}) is None
    assert _coerce_one({"arguments": {}}) is None


def test_arguments_none_renders_as_empty_object() -> None:
    """A missing ``arguments`` field on a parsed call serialises to ``"{}"``."""
    from lilbee.providers.worker.response_parser.parse import _coerce_one

    call = _coerce_one({"name": "f", "arguments": None})
    assert call is not None
    assert call.arguments == "{}"


def test_arguments_already_a_string_pass_through() -> None:
    """When a schema produces ``arguments`` as a string, the parser preserves it."""
    from lilbee.providers.worker.response_parser.parse import _coerce_one

    call = _coerce_one({"name": "f", "arguments": '{"q":"x"}'})
    assert call is not None
    assert call.arguments == '{"q":"x"}'


def test_non_json_serialisable_arguments_render_empty() -> None:
    """Arguments that ``json.dumps`` rejects fall back to ``"{}"`` rather than raising."""
    from lilbee.providers.worker.response_parser.parse import _coerce_one

    class _NotSerialisable:
        pass

    call = _coerce_one({"name": "f", "arguments": {"obj": _NotSerialisable()}})
    assert call is not None
    assert call.arguments == "{}"


def test_coerce_one_unwraps_function_nested_shape() -> None:
    """``_coerce_one`` reads ``name``/``arguments`` from a nested ``function`` dict.

    HF's gpt-oss schema and other OpenAI-shape schemas emit
    ``{"type": "function", "function": {"name": ..., "arguments": ...}}``
    per call. The parser unwraps the inner dict so the schema author can
    pick whichever shape fits the model's output most directly.
    """
    from lilbee.providers.worker.response_parser.parse import _coerce_one

    call = _coerce_one(
        {"type": "function", "function": {"name": "lookup", "arguments": {"q": "foo"}}}
    )
    assert call is not None
    assert call.name == "lookup"
    assert json.loads(call.arguments) == {"q": "foo"}


def test_missing_transformers_utility_degrades_gracefully(monkeypatch) -> None:
    """A transformers release without ``chat_parsing_utils`` returns the raw text.

    The lazy import is wrapped so a chat without tool extraction still runs
    when the upstream utility is missing.
    """
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "transformers.utils.chat_parsing_utils":
            raise ImportError("simulated missing utility")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    parsed = parse_response(
        '<tool_call>{"name": "x", "arguments": {}}</tool_call>',
        SCHEMAS[ModelFamily.QWEN3],
    )
    assert parsed.tool_calls == ()
    assert "tool_call" in parsed.content
