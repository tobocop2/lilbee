"""Request translation: Anthropic Messages -> canonical."""

from __future__ import annotations

import pytest

from lilbee.core.config.enums import ReasoningMode
from lilbee.server.anthropic_api.models import AnthropicThinking, MessagesRequest
from lilbee.server.anthropic_api.translate import (
    messages_to_canonical_request,
    resolve_reasoning_mode,
)
from lilbee.server.chat_dispatch.canonical import (
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)


def _request(**overrides) -> MessagesRequest:
    body = {
        "model": "m",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "hi"}],
    }
    body.update(overrides)
    return MessagesRequest.model_validate(body)


def test_string_system_and_sampling_map_through():
    req = messages_to_canonical_request(
        _request(
            system="be brief",
            temperature=0.2,
            top_p=0.9,
            top_k=40,
            stop_sequences=["END"],
            stream=True,
        )
    )
    assert req.system == "be brief"
    assert req.max_tokens == 128
    assert req.temperature == 0.2
    assert req.top_p == 0.9
    assert req.top_k == 40
    assert req.stop == ["END"]
    assert req.stream is True


def test_block_system_concatenates():
    req = messages_to_canonical_request(
        _request(system=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
    )
    assert req.system == "a\n\nb"


def test_tool_result_becomes_tool_message_before_user_text():
    req = messages_to_canonical_request(
        _request(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "here you go"},
                        {"type": "tool_result", "tool_use_id": "t1", "content": "42"},
                    ],
                },
            ]
        )
    )
    assert [m.role for m in req.messages] == ["tool", "user"]
    result = req.messages[0].content[0]
    assert isinstance(result, ToolResultBlock)
    assert result.tool_use_id == "t1"
    assert result.content == [TextBlock(text="42")]
    assert req.messages[1].content == [TextBlock(text="here you go")]


def test_assistant_tool_use_and_thinking_blocks():
    req = messages_to_canonical_request(
        _request(
            messages=[
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "hmm"},
                        {"type": "text", "text": "calling"},
                        {"type": "tool_use", "id": "t1", "name": "ls", "input": {"p": "."}},
                    ],
                },
            ]
        )
    )
    assistant = req.messages[1]
    # thinking dropped; text + tool_use survive
    assert assistant.content == [
        TextBlock(text="calling"),
        ToolUseBlock(id="t1", name="ls", input={"p": "."}),
    ]


def test_tools_and_tool_choice_map():
    req = messages_to_canonical_request(
        _request(
            tools=[{"name": "ls", "description": "list", "input_schema": {"type": "object"}}],
            tool_choice={"type": "tool", "name": "ls"},
        )
    )
    assert req.tools is not None
    assert req.tools[0].name == "ls"
    assert req.tool_choice is not None
    assert req.tool_choice.mode == "tool"
    assert req.tool_choice.tool_name == "ls"


@pytest.mark.parametrize(("wire", "mode"), [("auto", "auto"), ("any", "any"), ("none", "none")])
def test_tool_choice_modes(wire: str, mode: str):
    req = messages_to_canonical_request(_request(tool_choice={"type": wire}))
    assert req.tool_choice is not None
    assert req.tool_choice.mode == mode


def test_tool_choice_tool_without_name_raises():
    with pytest.raises(ValueError, match="requires a name"):
        messages_to_canonical_request(_request(tool_choice={"type": "tool"}))


def test_image_block_raises_value_error():
    with pytest.raises(ValueError, match="Image content"):
        messages_to_canonical_request(
            _request(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "data": "x"}},
                        ],
                    }
                ]
            )
        )


def test_empty_assistant_message_is_dropped():
    req = messages_to_canonical_request(
        _request(
            messages=[
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": [{"type": "thinking", "thinking": "only"}]},
                {"role": "user", "content": "and?"},
            ]
        )
    )
    assert [m.role for m in req.messages] == ["user", "user"]


def test_empty_string_content_message_is_dropped():
    req = messages_to_canonical_request(
        _request(
            messages=[
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": ""},
                {"role": "user", "content": "and?"},
            ]
        )
    )
    assert [m.role for m in req.messages] == ["user", "user"]


def test_tool_result_without_content_yields_empty_result():
    req = messages_to_canonical_request(
        _request(
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "t1"}],
                }
            ]
        )
    )
    result = req.messages[0].content[0]
    assert isinstance(result, ToolResultBlock)
    assert result.content == []


def test_tool_result_list_content_keeps_text_and_drops_unknown():
    req = messages_to_canonical_request(
        _request(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [
                                {"type": "text", "text": "ok"},
                                {"type": "mystery", "payload": 1},
                            ],
                        }
                    ],
                }
            ]
        )
    )
    result = req.messages[0].content[0]
    assert isinstance(result, ToolResultBlock)
    assert result.content == [TextBlock(text="ok")]


def test_tool_result_image_content_raises():
    with pytest.raises(ValueError, match="Image content"):
        messages_to_canonical_request(
            _request(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t1",
                                "content": [{"type": "image", "source": {"data": "x"}}],
                            }
                        ],
                    }
                ]
            )
        )


def test_mid_conversation_system_message_becomes_system_reminder_user_turn():
    """Claude Code sends role:"system" entries mid-conversation; a 400 there
    breaks every session after the first such turn."""
    req = messages_to_canonical_request(
        _request(
            messages=[
                {"role": "user", "content": "go"},
                {"role": "system", "content": "Terse mode enabled."},
            ]
        )
    )
    assert [m.role for m in req.messages] == ["user", "user"]
    reminder = req.messages[1].content[0]
    assert isinstance(reminder, TextBlock)
    assert reminder.text == "<system-reminder>\nTerse mode enabled.\n</system-reminder>"


def test_system_message_with_text_blocks_concatenates():
    req = messages_to_canonical_request(
        _request(
            messages=[
                {"role": "user", "content": "go"},
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "a"},
                        {"type": "text", "text": "b"},
                    ],
                },
            ]
        )
    )
    reminder = req.messages[1].content[0]
    assert isinstance(reminder, TextBlock)
    assert "ab" in reminder.text


def test_empty_system_message_is_dropped():
    req = messages_to_canonical_request(
        _request(
            messages=[
                {"role": "user", "content": "go"},
                {"role": "system", "content": ""},
            ]
        )
    )
    assert [m.role for m in req.messages] == ["user"]


class TestThinkingParameterOnRequest:
    """``thinking`` picks the reasoning mode; the setting supplies the default."""

    def test_disabled_request_asks_the_template_not_to_think(self):
        req = messages_to_canonical_request(_request(), mode=ReasoningMode.OFF)
        assert req.think is False

    def test_separate_leaves_the_template_default(self):
        req = messages_to_canonical_request(_request(), mode=ReasoningMode.SEPARATE)
        assert req.think is None

    def test_inline_leaves_the_template_default(self):
        req = messages_to_canonical_request(_request(), mode=ReasoningMode.INLINE)
        assert req.think is None

    def test_thinking_disabled_maps_to_off(self):
        assert (
            resolve_reasoning_mode(
                AnthropicThinking(type="disabled"), default=ReasoningMode.SEPARATE
            )
            is ReasoningMode.OFF
        )

    def test_absent_thinking_keeps_the_configured_default(self):
        assert resolve_reasoning_mode(None, default=ReasoningMode.INLINE) is ReasoningMode.INLINE

    def test_thinking_enabled_keeps_the_configured_presentation(self):
        mode = resolve_reasoning_mode(
            AnthropicThinking(type="enabled", budget_tokens=1024), default=ReasoningMode.INLINE
        )
        assert mode is ReasoningMode.INLINE

    def test_thinking_enabled_overrides_an_off_default(self):
        """A request that asks for thinking must get it, whatever the setting says."""
        mode = resolve_reasoning_mode(AnthropicThinking(type="enabled"), default=ReasoningMode.OFF)
        assert mode is ReasoningMode.SEPARATE

    def test_budget_tokens_parses_and_is_not_an_error(self):
        request = _request(thinking={"type": "enabled", "budget_tokens": 4096})
        assert request.thinking is not None
        assert request.thinking.budget_tokens == 4096

    def test_unknown_thinking_shape_falls_back_to_the_setting(self):
        """A shape this surface does not know must not 400 the agent."""
        assert _request(thinking={"type": "adaptive"}).thinking is None
        assert _request(thinking="on").thinking is None
