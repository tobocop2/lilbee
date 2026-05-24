"""Tests for ``parse_response`` over each shipped schema."""

from __future__ import annotations

import json

from lilbee.providers.families import registry
from lilbee.providers.families.profile import OutputFormat
from lilbee.providers.worker.response_parser import (
    ParsedResponse,
    TemplateFamily,
    get_schemas,
    parse_response,
)


def _parse(text: str, family: TemplateFamily) -> ParsedResponse:
    """Run ``parse_response`` keyed on the family's profile-declared output format."""
    profile = registry().by_family(family)
    output_format = profile.output_format if profile is not None else OutputFormat.NATIVE
    return parse_response(text, get_schemas()[family], output_format=output_format)


def test_qwen3_extracts_single_tool_call_with_json_args() -> None:
    """Qwen3 emits ``<tool_call>{"name":..., "arguments":...}</tool_call>``."""
    text = '<tool_call>\n{"name": "search", "arguments": {"q": "foo"}}\n</tool_call>'
    parsed = _parse(text, TemplateFamily.QWEN3)
    assert len(parsed.tool_calls) == 1
    call = parsed.tool_calls[0]
    assert call.name == "search"
    assert json.loads(call.arguments) == {"q": "foo"}


def test_qwen3_keeps_leading_content_before_tool_call() -> None:
    """Content before ``<tool_call>`` is preserved as the cleaned content."""
    text = 'Let me search.\n<tool_call>{"name": "f", "arguments": {}}</tool_call>'
    parsed = _parse(text, TemplateFamily.QWEN3)
    assert parsed.content.strip() == "Let me search."
    assert len(parsed.tool_calls) == 1


def test_qwen3_extracts_multiple_parallel_tool_calls() -> None:
    """Two ``<tool_call>...</tool_call>`` blocks become two structured calls."""
    text = (
        '<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>\n'
        '<tool_call>{"name": "b", "arguments": {"y": 2}}</tool_call>'
    )
    parsed = _parse(text, TemplateFamily.QWEN3)
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
    parsed = _parse(text, TemplateFamily.QWEN3)
    assert len(parsed.tool_calls) == 1
    assert "I should search" not in parsed.content


def test_qwen3_preserves_content_after_tool_call() -> None:
    """Text after ``</tool_call>`` is preserved, not silently dropped."""
    text = 'Looking it up. <tool_call>{"name": "f", "arguments": {}}</tool_call> Done.'
    parsed = _parse(text, TemplateFamily.QWEN3)
    assert len(parsed.tool_calls) == 1
    assert "Looking it up." in parsed.content
    assert "Done." in parsed.content


def test_qwen3_text_only_response_returns_no_tool_calls() -> None:
    """When the model never emits a ``<tool_call>`` block, tool_calls is empty."""
    parsed = _parse("just chatting today", TemplateFamily.QWEN3)
    assert parsed.tool_calls == ()
    assert "just chatting today" in parsed.content


def test_qwen3_coder_extracts_function_and_parameters() -> None:
    """Qwen3-Coder uses XML-style ``<function=>`` and ``<parameter=>`` markers."""
    text = (
        "<tool_call>\n<function=search>\n"
        "<parameter=q>\nfoo\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    parsed = _parse(text, TemplateFamily.QWEN3_CODER)
    assert len(parsed.tool_calls) == 1
    call = parsed.tool_calls[0]
    assert call.name == "search"
    assert json.loads(call.arguments) == {"q": "foo"}


def test_mistral_extracts_tool_call_array() -> None:
    """Mistral emits ``[TOOL_CALLS] [{"name":..., "arguments":...}]``."""
    text = 'Here is the call: [TOOL_CALLS] [{"name": "search", "arguments": {"q": "foo"}}]'
    parsed = _parse(text, TemplateFamily.MISTRAL)
    assert len(parsed.tool_calls) == 1
    call = parsed.tool_calls[0]
    assert call.name == "search"
    assert json.loads(call.arguments) == {"q": "foo"}


def test_mistral_keeps_prefix_content() -> None:
    """Content before ``[TOOL_CALLS]`` is preserved."""
    text = 'Reasoning prose.\n[TOOL_CALLS] [{"name": "f", "arguments": {}}]'
    parsed = _parse(text, TemplateFamily.MISTRAL)
    assert "Reasoning prose." in parsed.content
    assert len(parsed.tool_calls) == 1


def test_cohere_extracts_tool_calls_from_action_block() -> None:
    """Cohere/Command R wraps tool calls in ``<|START_ACTION|>...<|END_ACTION|>``."""
    text = (
        "<|START_RESPONSE|>I'll search.<|END_RESPONSE|>"
        '<|START_ACTION|>[{"tool_name": "search", "parameters": {"q": "x"}}]<|END_ACTION|>'
    )
    parsed = _parse(text, TemplateFamily.COHERE)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "search"
    assert json.loads(parsed.tool_calls[0].arguments) == {"q": "x"}


def test_cohere_extracts_bare_tool_call_array_without_action_wrapper() -> None:
    """Command-R7B (GGUF) emits the tool array as bare JSON, no ``<|START_ACTION|>``."""
    text = (
        "I will use the lilbee_search tool to find the chat worker file.\n"
        '[\n  {"tool_call_id": "0", "tool_name": "lilbee_search", '
        '"parameters": {"query": "chat worker file", "top_k": 5}}\n]'
    )
    parsed = _parse(text, TemplateFamily.COHERE)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "lilbee_search"
    assert json.loads(parsed.tool_calls[0].arguments) == {"query": "chat worker file", "top_k": 5}


def test_functionary_v3_extracts_call_after_recipient_marker() -> None:
    """Functionary v3 emits ``>>>name\\n{json}`` after an optional ``>>>all`` block."""
    text = '>>>all\nLet me look that up.\n>>>lilbee_search\n{"query": "chat worker"}'
    parsed = _parse(text, TemplateFamily.FUNCTIONARY_V3)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "lilbee_search"
    assert json.loads(parsed.tool_calls[0].arguments) == {"query": "chat worker"}
    assert "Let me look that up." in parsed.content


def test_functionary_v3_bare_call_without_all_block() -> None:
    """A direct ``>>>name\\n{json}`` with no ``>>>all`` preamble still parses."""
    text = '>>>get_weather\n{"city": "Paris"}'
    parsed = _parse(text, TemplateFamily.FUNCTIONARY_V3)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "get_weather"
    assert json.loads(parsed.tool_calls[0].arguments) == {"city": "Paris"}


def test_ernie_extracts_tool_calls_between_tool_call_tags() -> None:
    """ERNIE 4.x emits ``<tool_call>{json}</tool_call>`` with content in ``<response>``."""
    text = (
        "<response>\nLet me search.\n</response>"
        '<tool_call>{"name": "search", "arguments": {"q": "x"}}</tool_call>'
    )
    parsed = _parse(text, TemplateFamily.ERNIE)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "search"


def test_gpt_oss_extracts_tool_calls_from_commentary_channel() -> None:
    """GPT-OSS wraps tool calls in ``<|channel|>commentary to=functions.<name>``."""
    text = (
        "<|channel|>final<|message|>I'll search.<|end|>"
        '<|channel|>commentary to=functions.search<|message|>{"q": "x"}<|call|>'
    )
    parsed = _parse(text, TemplateFamily.GPT_OSS)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "search"
    assert json.loads(parsed.tool_calls[0].arguments) == {"q": "x"}


def test_smollm_extracts_tool_call() -> None:
    """SmolLM3 wraps tool calls in ``<tool_call>{json}</tool_call>``."""
    text = '<tool_call>{"name": "search", "arguments": {"q": "x"}}</tool_call>'
    parsed = _parse(text, TemplateFamily.SMOLLM)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "search"
    assert json.loads(parsed.tool_calls[0].arguments) == {"q": "x"}


def test_hermes_extracts_tool_call_with_scratch_pad() -> None:
    """Hermes uses ``<scratch_pad>`` for reasoning instead of ``<think>``."""
    text = (
        "<scratch_pad>Let me think.</scratch_pad>"
        '<tool_call>{"name": "search", "arguments": {"q": "x"}}</tool_call>'
    )
    parsed = _parse(text, TemplateFamily.HERMES)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "search"
    assert json.loads(parsed.tool_calls[0].arguments) == {"q": "x"}


def test_deepseek_v31_extracts_tool_call_with_separator() -> None:
    """DeepSeek V3.1 uses fullwidth-pipe separator markers around name + JSON."""
    text = (
        "<｜tool▁calls▁begin｜>"
        '<｜tool▁call▁begin｜>search<｜tool▁sep｜>{"q": "x"}<｜tool▁call▁end｜>'
        "<｜tool▁calls▁end｜>"
    )
    parsed = _parse(text, TemplateFamily.DEEPSEEK_V31)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "search"
    assert json.loads(parsed.tool_calls[0].arguments) == {"q": "x"}


def test_granite_extracts_top_level_tool_call_array() -> None:
    """IBM Granite emits a JSON array after ``<|tool_call|>`` or ``<tool_call>``."""
    text = '<|tool_call|>[{"name": "search", "arguments": {"q": "x"}}]'
    parsed = _parse(text, TemplateFamily.GRANITE)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "search"


def test_granite_extracts_bare_json_tool_call_via_dual_output_format() -> None:
    """OpenAI-style 'tools' parameter elicits bare JSON from Granite; DUAL fallback catches it."""

    text = '{"name": "search", "arguments": {"q": "x"}}'
    parsed = parse_response(
        text,
        get_schemas()[TemplateFamily.GRANITE],
        output_format=OutputFormat.DUAL,
    )
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "search"
    assert parsed.content == ""


def test_phi4mini_extracts_functools_array() -> None:
    """Phi-4 wraps the tool-call array in ``functools[...]``."""
    text = 'Some reasoning. functools[{"name": "search", "arguments": {"q": "x"}}]'
    parsed = _parse(text, TemplateFamily.PHI4MINI)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "search"


def test_functionary_v3_extracts_recipient_routed_call() -> None:
    """Functionary v3 routes tool calls via ``>>>name`` lines."""
    text = '>>>all\nAnswer text\n>>>search\n{"q": "x"}'
    parsed = _parse(text, TemplateFamily.FUNCTIONARY_V3)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "search"


def test_llama3_extracts_python_tagged_tool_call() -> None:
    """Llama 3.x emits ``<|python_tag|>{json}`` for tool calls."""
    text = '<|python_tag|>{"name": "search", "arguments": {"q": "x"}}'
    parsed = _parse(text, TemplateFamily.LLAMA3)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "search"


def test_llama3_extracts_bare_json_tool_call_via_dual_output_format() -> None:
    """Llama-3 prompted via OpenAI 'tools' parameter emits bare JSON; DUAL fallback catches it."""

    text = '{"name": "search", "arguments": {"q": "x"}}'
    parsed = parse_response(
        text,
        get_schemas()[TemplateFamily.LLAMA3],
        output_format=OutputFormat.DUAL,
    )
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "search"


def test_llama3_bare_json_leaves_no_content_leak() -> None:
    """Bare-JSON tool call must not also surface in content; matches the python_tag arm."""

    text = '{"name": "search", "arguments": {"q": "x"}}'
    parsed = parse_response(
        text,
        get_schemas()[TemplateFamily.LLAMA3],
        output_format=OutputFormat.DUAL,
    )
    assert parsed.content == ""


def test_glm46_extracts_xml_arg_key_value_call() -> None:
    """GLM 4.5/4.6 wraps calls in ``<tool_call>NAME\\n<arg_key>K</arg_key>...</tool_call>``."""
    text = (
        "<tool_call>get_weather\n"
        "<arg_key>city</arg_key>\n"
        '<arg_value>"Berlin"</arg_value>\n'
        "</tool_call>"
    )
    parsed = _parse(text, TemplateFamily.GLM46)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "get_weather"
    assert json.loads(parsed.tool_calls[0].arguments) == {"city": "Berlin"}


def test_glm47_extracts_single_line_xml_call() -> None:
    """GLM 4.7 emits the same wire format on a single line; whitespace is the only difference."""
    text = (
        '<tool_call>get_weather<arg_key>city</arg_key><arg_value>"Berlin"</arg_value></tool_call>'
    )
    parsed = _parse(text, TemplateFamily.GLM47)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "get_weather"


def test_kimi_k2_extracts_tool_call_with_functions_prefix() -> None:
    """Kimi K2 emits ``functions.NAME:IDX`` followed by JSON args."""
    text = (
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>"
        '{"city": "Berlin"}'
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )
    parsed = _parse(text, TemplateFamily.KIMI_K2)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "get_weather"
    assert json.loads(parsed.tool_calls[0].arguments) == {"city": "Berlin"}


def test_internlm2_extracts_action_plugin_call() -> None:
    """InternLM2 wraps the JSON call in ``<|action_start|><|plugin|>`` markers."""
    text = (
        "I'll check.<|action_start|><|plugin|>"
        '{"name": "get_weather", "arguments": {"city": "Berlin"}}'
        "<|action_end|>"
    )
    parsed = _parse(text, TemplateFamily.INTERNLM2)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "get_weather"
    assert json.loads(parsed.tool_calls[0].arguments) == {"city": "Berlin"}


def test_olmo3_extracts_pythonic_function_call() -> None:
    """OLMo 3 emits pythonic ``name(key=value)`` inside ``<function_calls>...</function_calls>``."""
    text = '<function_calls>\nget_weather(city="Berlin")\n</function_calls>'
    parsed = _parse(text, TemplateFamily.OLMO3)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "get_weather"


def test_lfm2_extracts_pythonic_list_call() -> None:
    """LFM2 emits ``<|tool_call_start|>[name(key=value)]<|tool_call_end|>``."""
    text = '<|tool_call_start|>[get_weather(city="Berlin")]<|tool_call_end|>'
    parsed = _parse(text, TemplateFamily.LFM2)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "get_weather"


def test_malformed_json_inside_tool_call_falls_back_to_text() -> None:
    """A ``<tool_call>`` block whose body is not JSON returns no tool calls.

    The parse layer catches the schema's TypeError/ValueError and returns the
    raw text as content rather than raising out of the worker.
    """
    text = "<tool_call>NOT JSON{</tool_call>"
    parsed = _parse(text, TemplateFamily.QWEN3)
    assert parsed.tool_calls == ()


def test_empty_input_returns_empty_response() -> None:
    """Parsing an empty string never raises."""
    parsed = _parse("", TemplateFamily.QWEN3)
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
    parsed = _parse(text, TemplateFamily.QWEN3)
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
    parsed = parse_response("just regular text", schema, output_format=OutputFormat.NATIVE)
    assert parsed.content == "just regular text"
    assert parsed.tool_calls == ()


def test_gemma4_extracts_tool_code_python_call() -> None:
    """Gemma emits ``tool_code\\nname(key="val")`` Python-call blocks.

    The GGUF (e.g. gemma-4-E2B) renders tool calls as a ``tool_code`` block
    with a Python-style invocation, not the JSON envelope older Gemma schemas
    assumed; the schema parses the call name + key=value args.
    """
    text = 'lilbee_search\n<eos>tool_code\nweather(city="Tokyo")\n<eos><eos>'
    parsed = _parse(text, TemplateFamily.GEMMA4)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "weather"
    assert json.loads(parsed.tool_calls[0].arguments) == {"city": "Tokyo"}


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
        get_schemas()[TemplateFamily.QWEN3],
        output_format=OutputFormat.NATIVE,
    )
    assert parsed.tool_calls == ()
    assert "tool_call" in parsed.content


def test_qwen3_dual_bare_json_falls_back_when_template_marker_missing() -> None:
    """Qwen3 DUAL: OpenAI-style clients drop ``<tool_call>`` markers; bare-JSON catches it."""
    text = '{"name": "search", "arguments": {"q": "x"}}'
    parsed = parse_response(
        text,
        get_schemas()[TemplateFamily.QWEN3],
        output_format=OutputFormat.DUAL,
    )
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "search"


def test_chatml_tool_call_output_format_extracts_each_wrapper() -> None:
    """``OutputFormat.CHATML_TOOL_CALL`` extracts every ``<tool_call>{json}</tool_call>``."""
    text = (
        "Some prose. "
        '<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>'
        '<tool_call>{"name": "b", "arguments": {"y": 2}}</tool_call>'
    )
    parsed = parse_response(
        text,
        get_schemas()[TemplateFamily.HERMES],
        output_format=OutputFormat.CHATML_TOOL_CALL,
    )
    assert [c.name for c in parsed.tool_calls] == ["a", "b"]
    assert parsed.content.strip() == "Some prose."


def test_chatml_tool_call_output_format_drops_invalid_json_calls() -> None:
    """A ``<tool_call>`` wrapper whose body isn't valid JSON produces zero calls."""
    text = "<tool_call>not-json</tool_call>"
    parsed = parse_response(
        text,
        get_schemas()[TemplateFamily.HERMES],
        output_format=OutputFormat.CHATML_TOOL_CALL,
    )
    assert parsed.tool_calls == ()


def test_harmony_output_format_extracts_commentary_channel_calls() -> None:
    """``OutputFormat.HARMONY`` extracts ``<|channel|>commentary to=...<|call|>`` blocks."""
    text = (
        "<|channel|>final<|message|>Sure.<|end|>"
        '<|channel|>commentary to=functions.search<|message|>{"q": "x"}<|call|>'
    )
    parsed = parse_response(
        text,
        get_schemas()[TemplateFamily.GPT_OSS],
        output_format=OutputFormat.HARMONY,
    )
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "search"
    assert json.loads(parsed.tool_calls[0].arguments) == {"q": "x"}


def test_bare_json_scanner_handles_nested_objects() -> None:
    """The bare-JSON extractor must walk nested objects via ``json.JSONDecoder.raw_decode``."""
    text = '{"name": "search", "arguments": {"filter": {"deep": {"nested": {"v": 1}}}}}'
    parsed = parse_response(
        text,
        get_schemas()[TemplateFamily.LLAMA3],
        output_format=OutputFormat.BARE_JSON,
    )
    assert len(parsed.tool_calls) == 1
    call = parsed.tool_calls[0]
    assert call.name == "search"
    assert json.loads(call.arguments) == {"filter": {"deep": {"nested": {"v": 1}}}}


def test_bare_json_extractor_skips_object_with_non_string_name() -> None:
    """A ``{"name": 42, ...}`` block isn't a tool call -- must be skipped silently."""
    from lilbee.providers.worker.response_parser.format_fallbacks import bare_json_tool_calls

    assert bare_json_tool_calls('{"name": 42, "arguments": {}}') == ()
    assert bare_json_tool_calls('{"name": "", "arguments": {}}') == ()


def test_bare_json_extractor_accepts_parameters_key_alias() -> None:
    """Mistral-Nemo emits ``parameters`` instead of ``arguments``; treat them the same."""
    from lilbee.providers.worker.response_parser.format_fallbacks import bare_json_tool_calls

    calls = bare_json_tool_calls('{"name": "lilbee_search", "parameters": {"query": "x"}}')
    assert len(calls) == 1
    assert calls[0].name == "lilbee_search"
    assert json.loads(calls[0].arguments) == {"query": "x"}


def test_mistral_extracts_bare_json_array_without_tool_calls_marker() -> None:
    """Mistral-Nemo emits a bare ``[{"name":...,"parameters":...}]`` array, no [TOOL_CALLS]."""
    text = '[{"name": "lilbee_search", "parameters": {"query": "chat worker"}}]'
    parsed = parse_response(
        text,
        get_schemas()[TemplateFamily.MISTRAL],
        output_format=OutputFormat.DUAL,
    )
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "lilbee_search"
    assert json.loads(parsed.tool_calls[0].arguments) == {"query": "chat worker"}


def test_bare_json_split_content_returns_full_text_when_no_match() -> None:
    """``split_content_at_bare_json`` returns *text* unchanged when nothing matches."""
    from lilbee.providers.worker.response_parser.format_fallbacks import split_content_at_bare_json

    assert split_content_at_bare_json("no JSON objects here") == "no JSON objects here"


def test_arguments_to_string_handles_string_passthrough_and_serialiser_failure() -> None:
    """``_arguments_to_string`` covers None, str, and non-JSON-serialisable inputs."""
    from lilbee.providers.worker.response_parser.format_fallbacks import _arguments_to_string

    assert _arguments_to_string(None) == "{}"
    assert _arguments_to_string('{"x":1}') == '{"x":1}'

    class _NotSerialisable:
        pass

    assert _arguments_to_string({"obj": _NotSerialisable()}) == "{}"


def test_chatml_tool_call_extractor_rejects_invalid_json_body() -> None:
    """A body that matches ``\\{.*?\\}`` shape but isn't valid JSON returns no call."""
    schema = get_schemas()[TemplateFamily.HERMES]
    parsed = parse_response(
        "<tool_call>{not valid json}</tool_call>",
        schema,
        output_format=OutputFormat.CHATML_TOOL_CALL,
    )
    assert parsed.tool_calls == ()


def test_chatml_tool_call_extractor_rejects_object_without_name() -> None:
    """``CHATML_TOOL_CALL`` skips a wrapper whose JSON object has no ``name``."""
    schema = get_schemas()[TemplateFamily.HERMES]
    parsed = parse_response(
        '<tool_call>{"arguments": {}}</tool_call>',
        schema,
        output_format=OutputFormat.CHATML_TOOL_CALL,
    )
    assert parsed.tool_calls == ()


def test_harmony_extractor_passes_arguments_through_when_present() -> None:
    """The HARMONY extractor returns the args body verbatim as JSON."""
    schema = get_schemas()[TemplateFamily.GPT_OSS]
    text = '<|channel|>commentary to=functions.s<|message|>{"q":"x"}<|call|>'
    parsed = parse_response(text, schema, output_format=OutputFormat.HARMONY)
    assert parsed.tool_calls[0].arguments == '{"q":"x"}'


def test_chatml_extractor_skips_bad_wrapper_keeps_following_one() -> None:
    """A bad first ``<tool_call>`` wrapper must not block extraction of the next valid one."""
    schema = get_schemas()[TemplateFamily.HERMES]
    text = (
        "<tool_call>not-json</tool_call>"
        '<tool_call>{"name": "ok", "arguments": {"a": 1}}</tool_call>'
    )
    parsed = parse_response(text, schema, output_format=OutputFormat.CHATML_TOOL_CALL)
    assert [c.name for c in parsed.tool_calls] == ["ok"]
