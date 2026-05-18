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
        ("", ModelFamily.UNKNOWN),
        ("no markers here", ModelFamily.UNKNOWN),
    ],
)
def test_detect_family_classifies(template: str, expected: ModelFamily) -> None:
    """Each known family is recognised by its distinctive markers."""
    assert detect_family(template) == expected


def test_detect_family_handles_none() -> None:
    """A missing chat_template metadata key returns UNKNOWN, not an error."""
    assert detect_family(None) == ModelFamily.UNKNOWN


def test_qwen3_coder_wins_over_qwen3() -> None:
    """Qwen3-Coder templates also include ``<tool_call>``; the more specific
    marker pair (``<function=`` + ``<parameter=``) must classify first.
    """
    template = "<tool_call><function=foo><parameter=bar>x</parameter></function></tool_call>"
    assert detect_family(template) == ModelFamily.QWEN3_CODER
