"""Wire-model parsing for the Anthropic Messages surface."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from lilbee.server.anthropic_api.models import (
    AnthropicMessage,
    MessagesRequest,
    TextBlockParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
    UnknownBlockParam,
)

_CLAUDE_CODE_STYLE_BODY = {
    "model": "Qwen3-8B",
    "max_tokens": 8192,
    "system": [
        {"type": "text", "text": "You are Claude Code.", "cache_control": {"type": "ephemeral"}}
    ],
    "messages": [
        {"role": "user", "content": "list the files"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "I should call ls", "signature": "abc"},
                {"type": "text", "text": "Listing files."},
                {"type": "tool_use", "id": "toolu_1", "name": "ls", "input": {"path": "."}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "a.py\nb.py"},
            ],
        },
    ],
    "tools": [
        {
            "name": "ls",
            "description": "List files",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
        }
    ],
    "thinking": {"type": "adaptive"},
    "metadata": {"user_id": "u1"},
    "stream": True,
}


def test_realistic_claude_code_body_parses():
    request = MessagesRequest.model_validate(_CLAUDE_CODE_STYLE_BODY)
    assert request.model == "Qwen3-8B"
    assert request.stream is True
    assert request.system is not None and request.system[0].text == "You are Claude Code."
    blocks = request.messages[1].content
    assert isinstance(blocks[0], UnknownBlockParam)  # thinking tolerated, not rejected
    assert isinstance(blocks[1], TextBlockParam)
    assert isinstance(blocks[2], ToolUseBlockParam)
    result = request.messages[2].content[0]
    assert isinstance(result, ToolResultBlockParam)
    assert result.tool_use_id == "toolu_1"


def test_missing_max_tokens_fails_validation():
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    with pytest.raises(ValidationError):
        MessagesRequest.model_validate(body)


def test_unknown_top_level_fields_are_tolerated():
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "output_config": {"effort": "high"},
            "betas": ["compact-2026-01-12"],
        }
    )
    assert request.max_tokens == 16


def test_tool_result_list_content_parses_text_blocks():
    message = AnthropicMessage.model_validate(
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [{"type": "text", "text": "ok"}],
                    "is_error": True,
                }
            ],
        }
    )
    block = message.content[0]
    assert isinstance(block, ToolResultBlockParam)
    assert block.is_error is True
    assert isinstance(block.content[0], TextBlockParam)


class TestThinkingBudgetFloor:
    """``budget_tokens`` carries Anthropic's documented 1024-token minimum."""

    def _body(self, **thinking) -> dict:
        return {
            "model": "m",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "enabled", **thinking},
        }

    def test_budget_at_the_floor_parses(self):
        request = MessagesRequest.model_validate(self._body(budget_tokens=1024))
        assert request.thinking is not None
        assert request.thinking.budget_tokens == 1024

    def test_zero_budget_is_rejected(self):
        """0 would resolve to 0 chars, which the cap reads as unlimited."""
        with pytest.raises(ValidationError):
            MessagesRequest.model_validate(self._body(budget_tokens=0))

    def test_budget_below_the_floor_is_rejected(self):
        with pytest.raises(ValidationError):
            MessagesRequest.model_validate(self._body(budget_tokens=1023))

    def test_negative_budget_is_rejected(self):
        with pytest.raises(ValidationError):
            MessagesRequest.model_validate(self._body(budget_tokens=-1))

    def test_a_disabled_request_is_never_rejected_for_its_budget(self):
        """The strictest possible body must not 400; a disabled budget is ignored."""
        body = {
            "model": "m",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "disabled", "budget_tokens": 0},
        }
        request = MessagesRequest.model_validate(body)
        assert request.thinking is not None
        assert request.thinking.budget_tokens is None

    def test_absent_budget_is_allowed(self):
        request = MessagesRequest.model_validate(self._body())
        assert request.thinking is not None
        assert request.thinking.budget_tokens is None
