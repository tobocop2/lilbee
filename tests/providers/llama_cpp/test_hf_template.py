"""Tests for the HF-template rendering helpers in the llama.cpp provider."""

from __future__ import annotations

import json

from lilbee.providers.llama_cpp.provider import _tool_args_as_json_strings


def test_tool_args_dict_serialised_to_json_string() -> None:
    """A dict ``arguments`` (lilbee's normalised form) becomes a JSON string.

    Functionary's HF template concatenates ``arguments`` as text, so a dict
    would raise ``TypeError: can only concatenate str (not "dict")``.
    """
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "1", "function": {"name": "search", "arguments": {"q": "x"}}}],
        }
    ]
    out = _tool_args_as_json_strings(messages)
    args = out[0]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, str)
    assert json.loads(args) == {"q": "x"}


def test_tool_args_string_left_unchanged() -> None:
    """An ``arguments`` already in string form is passed through verbatim."""
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "1", "function": {"name": "s", "arguments": '{"q": "x"}'}}],
        }
    ]
    out = _tool_args_as_json_strings(messages)
    assert out[0]["tool_calls"][0]["function"]["arguments"] == '{"q": "x"}'


def test_plain_messages_untouched() -> None:
    """Messages without tool calls are returned as-is."""
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    assert _tool_args_as_json_strings(messages) == messages


def test_hf_template_tool_arg_shape_per_family() -> None:
    """GLM renders args as a dict; functionary/cohere concatenate them as strings.

    GLM-4.5's template iterates ``arguments.items()`` and breaks on a string
    ('str object has no attribute items'), so it must keep lilbee's normalised
    dict. Functionary and Command-R concatenate the args, so they re-stringify.
    Flipping these flips a working family back to a 500.
    """
    from lilbee.providers.families.cohere import PROFILE as COHERE
    from lilbee.providers.families.functionary_v3 import PROFILE as FUNCTIONARY
    from lilbee.providers.families.glm46 import PROFILE as GLM46

    assert GLM46.render_with_hf_template is True
    assert GLM46.stringify_hf_tool_args is False
    assert FUNCTIONARY.stringify_hf_tool_args is True
    assert COHERE.stringify_hf_tool_args is True
