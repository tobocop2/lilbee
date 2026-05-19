"""Tests for chat-template family detection."""

from __future__ import annotations

import pytest

from lilbee.providers.worker.response_parser.families import ModelFamily, detect_family


@pytest.mark.parametrize(
    "template, expected",
    [
        (
            "system\n{% for tool in tools %}{% endfor %}<tool_call>{...}</tool_call>",
            ModelFamily.QWEN3,
        ),
        (
            "system\n<tool_call><function=name><parameter=key>v</parameter></function></tool_call>",
            ModelFamily.QWEN3_CODER,
        ),
        (
            "...output your tool call as [TOOL_CALLS] [{...}]",
            ModelFamily.MISTRAL,
        ),
        (
            '...<|"|>some Gemma 4 quoted value<|"|>...',
            ModelFamily.GEMMA4,
        ),
        (
            "<|START_ACTION|>[{...}]<|END_ACTION|>",
            ModelFamily.COHERE,
        ),
        (
            "<|channel|>final<|message|>...<|call|>",
            ModelFamily.GPT_OSS,
        ),
        (
            "<|begin_of_sentence|>system<|end_of_sentence|>",
            ModelFamily.ERNIE,
        ),
        (
            "You are a function calling AI model with <tool_call> and </tool_call>",
            ModelFamily.HERMES,
        ),
        (
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>name<｜tool▁sep｜>{}<｜tool▁call▁end｜>",
            ModelFamily.DEEPSEEK_V31,
        ),
        (
            "<|start_of_role|>system<|end_of_role|>...<|tool_call|>[{...}]",
            ModelFamily.GRANITE,
        ),
        (
            "<|tool|>schema<|/tool|>...<|end|>",
            ModelFamily.PHI4MINI,
        ),
        (
            ">>>all\\ncontent\\n>>>name\\n{}",
            ModelFamily.FUNCTIONARY_V3,
        ),
        (
            "<|python_tag|>{...}",
            ModelFamily.LLAMA3,
        ),
        (
            "<tool_call>get_x\n<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>",
            ModelFamily.GLM46,
        ),
        (
            "Use <tool_call>{function-name}<arg_key>{arg-key}</arg_key>"
            "<arg_value>{arg-value}</arg_value></tool_call>",
            ModelFamily.GLM47,
        ),
        (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>x:0<|tool_call_argument_begin|>{}<|tool_call_end|>",
            ModelFamily.KIMI_K2,
        ),
        (
            "I'll check.<function_calls>\nget_weather(city='NYC')\n</function_calls>",
            ModelFamily.OLMO3,
        ),
        (
            "<|tool_list_start|>tool definitions here<|tool_list_end|>",
            ModelFamily.LFM2,
        ),
        ("", ModelFamily.UNKNOWN),
        ("no markers here", ModelFamily.UNKNOWN),
    ],
)
def test_detect_family_classifies(template: str, expected: ModelFamily) -> None:
    """Each known family is recognised by its distinctive markers."""
    assert detect_family(template) == expected


def test_detect_family_uses_architecture_fallback_for_smollm() -> None:
    """SmolLM3 shares ``<tool_call>`` with Qwen3; architecture metadata
    disambiguates so the SmolLM-specific schema is selected.
    """
    template = "{% for m in messages %}<|im_start|>{{ m.role }}<tool_call>x</tool_call>{% endfor %}"
    assert detect_family(template, architecture="smollm3") == ModelFamily.SMOLLM
    assert detect_family(template, architecture=None) == ModelFamily.QWEN3
    assert detect_family(template, architecture="qwen3") == ModelFamily.QWEN3


def test_detect_family_uses_architecture_fallback_for_internlm2() -> None:
    """InternLM2's chat template is minimal; action-block markers only appear
    in model output. Architecture metadata is the only reliable signal.
    """
    minimal_template = "{% for m in messages %}<|im_start|>{{ m.role }}<|im_end|>{% endfor %}"
    assert detect_family(minimal_template, architecture="internlm2") == ModelFamily.INTERNLM2
    assert detect_family(minimal_template, architecture="internlm") == ModelFamily.INTERNLM2
    assert detect_family(minimal_template, architecture=None) == ModelFamily.UNKNOWN


def test_detect_family_falls_back_to_architecture_with_no_template() -> None:
    """Empty chat template + known architecture still classifies."""
    assert detect_family(None, architecture="internlm2") == ModelFamily.INTERNLM2
    assert detect_family("", architecture="internlm2") == ModelFamily.INTERNLM2
    assert detect_family("", architecture="unknown_arch") == ModelFamily.UNKNOWN


def test_detect_family_handles_none() -> None:
    """A missing chat_template metadata key returns UNKNOWN, not an error."""
    assert detect_family(None) == ModelFamily.UNKNOWN


def test_qwen3_coder_wins_over_qwen3() -> None:
    """Qwen3-Coder templates also include ``<tool_call>``; the more specific
    marker pair (``<function=`` + ``<parameter=``) must classify first.
    """
    template = "<tool_call><function=foo><parameter=bar>x</parameter></function></tool_call>"
    assert detect_family(template) == ModelFamily.QWEN3_CODER
