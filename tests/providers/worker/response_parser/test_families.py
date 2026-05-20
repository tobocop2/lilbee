"""Tests for chat-template family detection."""

from __future__ import annotations

import pytest

from lilbee.providers.worker.response_parser.families import TemplateFamily, detect_family


@pytest.mark.parametrize(
    "template, expected",
    [
        (
            "system\n{% for tool in tools %}{% endfor %}<tool_call>{...}</tool_call>",
            TemplateFamily.QWEN3,
        ),
        (
            "system\n<tool_call><function=name><parameter=key>v</parameter></function></tool_call>",
            TemplateFamily.QWEN3_CODER,
        ),
        (
            "...output your tool call as [TOOL_CALLS] [{...}]",
            TemplateFamily.MISTRAL,
        ),
        (
            '...<|"|>some Gemma 4 quoted value<|"|>...',
            TemplateFamily.GEMMA4,
        ),
        (
            "<|START_ACTION|>[{...}]<|END_ACTION|>",
            TemplateFamily.COHERE,
        ),
        (
            "<|channel|>final<|message|>...<|call|>",
            TemplateFamily.GPT_OSS,
        ),
        (
            "<|begin_of_sentence|>system<|end_of_sentence|>",
            TemplateFamily.ERNIE,
        ),
        (
            "You are a function calling AI model with <tool_call> and </tool_call>",
            TemplateFamily.HERMES,
        ),
        (
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>name<｜tool▁sep｜>{}<｜tool▁call▁end｜>",
            TemplateFamily.DEEPSEEK_V31,
        ),
        (
            "<|start_of_role|>system<|end_of_role|>...<|tool_call|>[{...}]",
            TemplateFamily.GRANITE,
        ),
        (
            "<|tool|>schema<|/tool|>...<|end|>",
            TemplateFamily.PHI4MINI,
        ),
        (
            ">>>all\\ncontent\\n>>>name\\n{}",
            TemplateFamily.FUNCTIONARY_V3,
        ),
        (
            "<|python_tag|>{...}",
            TemplateFamily.LLAMA3,
        ),
        (
            "<tool_call>get_x\n<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>",
            TemplateFamily.GLM46,
        ),
        (
            "Use <tool_call>{function-name}<arg_key>{arg-key}</arg_key>"
            "<arg_value>{arg-value}</arg_value></tool_call>",
            TemplateFamily.GLM47,
        ),
        (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>x:0<|tool_call_argument_begin|>{}<|tool_call_end|>",
            TemplateFamily.KIMI_K2,
        ),
        (
            "I'll check.<function_calls>\nget_weather(city='NYC')\n</function_calls>",
            TemplateFamily.OLMO3,
        ),
        (
            "<|tool_list_start|>tool definitions here<|tool_list_end|>",
            TemplateFamily.LFM2,
        ),
        ("", TemplateFamily.UNKNOWN),
        ("no markers here", TemplateFamily.UNKNOWN),
    ],
)
def test_detect_family_classifies(template: str, expected: TemplateFamily) -> None:
    """Each known family is recognised by its distinctive markers."""
    assert detect_family(template) == expected


def test_detect_family_uses_architecture_fallback_for_smollm() -> None:
    """SmolLM3 shares ``<tool_call>`` with Qwen3; architecture metadata
    disambiguates so the SmolLM-specific schema is selected.
    """
    template = "{% for m in messages %}<|im_start|>{{ m.role }}<tool_call>x</tool_call>{% endfor %}"
    assert detect_family(template, architecture="smollm3") == TemplateFamily.SMOLLM
    assert detect_family(template, architecture=None) == TemplateFamily.QWEN3
    assert detect_family(template, architecture="qwen3") == TemplateFamily.QWEN3


def test_detect_family_uses_architecture_fallback_for_internlm2() -> None:
    """InternLM2's chat template is minimal; action-block markers only appear
    in model output. Architecture metadata is the only reliable signal.
    """
    minimal_template = "{% for m in messages %}<|im_start|>{{ m.role }}<|im_end|>{% endfor %}"
    assert detect_family(minimal_template, architecture="internlm2") == TemplateFamily.INTERNLM2
    assert detect_family(minimal_template, architecture="internlm") == TemplateFamily.INTERNLM2
    assert detect_family(minimal_template, architecture=None) == TemplateFamily.UNKNOWN


def test_detect_family_falls_back_to_architecture_with_no_template() -> None:
    """Empty chat template + known architecture still classifies."""
    assert detect_family(None, architecture="internlm2") == TemplateFamily.INTERNLM2
    assert detect_family("", architecture="internlm2") == TemplateFamily.INTERNLM2
    assert detect_family("", architecture="unknown_arch") == TemplateFamily.UNKNOWN


def test_detect_family_handles_none() -> None:
    """A missing chat_template metadata key returns UNKNOWN, not an error."""
    assert detect_family(None) == TemplateFamily.UNKNOWN


def test_qwen3_coder_wins_over_qwen3() -> None:
    """Qwen3-Coder templates also include ``<tool_call>``; the more specific
    marker pair (``<function=`` + ``<parameter=``) must classify first.
    """
    template = "<tool_call><function=foo><parameter=bar>x</parameter></function></tool_call>"
    assert detect_family(template) == TemplateFamily.QWEN3_CODER
