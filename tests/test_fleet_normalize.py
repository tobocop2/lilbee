"""Tests for reshaping an OpenAI tool conversation into strict alternation."""

from __future__ import annotations

from itertools import pairwise
from typing import Any

from lilbee.providers.fleet.normalize import to_alternating

_SYSTEM = "system"
_USER = "user"
_ASSISTANT = "assistant"


def _roles_after_system(messages: list[dict[str, Any]]) -> list[str]:
    return [m["role"] for m in messages if m["role"] != _SYSTEM]


def _assert_strict_alternation(messages: list[dict[str, Any]]) -> None:
    """Output has no tool role, no empty turns, and alternates after the system block."""
    roles = [m["role"] for m in messages]
    assert "tool" not in roles
    assert all(m["content"] for m in messages)
    convo = _roles_after_system(messages)
    assert all(role in (_USER, _ASSISTANT) for role in convo)
    for earlier, later in pairwise(convo):
        assert earlier != later


def test_tool_role_becomes_labelled_user_turn() -> None:
    out = to_alternating(
        [
            {"role": _USER, "content": "search"},
            {"role": _ASSISTANT, "content": "looking"},
            {"role": "tool", "content": "found it", "tool_call_id": "abc"},
            {"role": _ASSISTANT, "content": "done"},
        ]
    )
    _assert_strict_alternation(out)
    assert _roles_after_system(out) == [_USER, _ASSISTANT, _USER, _ASSISTANT]
    assert out[2]["content"] == "Tool result: found it"


def test_assistant_tool_calls_render_as_text_note() -> None:
    out = to_alternating(
        [
            {"role": _USER, "content": "go"},
            {
                "role": _ASSISTANT,
                "content": "",
                "tool_calls": [
                    {"function": {"name": "grep", "arguments": '{"q":"x"}'}},
                ],
            },
            {"role": "tool", "content": "match", "tool_call_id": "id"},
            {"role": _ASSISTANT, "content": "answer"},
        ]
    )
    _assert_strict_alternation(out)
    # The empty assistant content is replaced by a note for the tool call, so the
    # turn is non-empty and the templates accept it.
    assert "Calling tool grep" in out[1]["content"]
    assert '{"q":"x"}' in out[1]["content"]


def test_consecutive_assistant_turns_merge() -> None:
    out = to_alternating(
        [
            {"role": _USER, "content": "q"},
            {"role": _ASSISTANT, "content": "a1"},
            {"role": _ASSISTANT, "content": "a2"},
        ]
    )
    _assert_strict_alternation(out)
    assert _roles_after_system(out) == [_USER, _ASSISTANT]
    assert out[-1]["content"] == "a1\na2"


def test_leading_system_messages_kept_verbatim() -> None:
    out = to_alternating(
        [
            {"role": _SYSTEM, "content": "sys A"},
            {"role": _SYSTEM, "content": "sys B"},
            {"role": _USER, "content": "hi"},
        ]
    )
    assert out[0] == {"role": _SYSTEM, "content": "sys A"}
    assert out[1] == {"role": _SYSTEM, "content": "sys B"}
    _assert_strict_alternation(out)


def test_full_opencode_tool_loop_alternates() -> None:
    # The exact shape the opencode tool loop emits: system, user, then repeated
    # assistant(tool_calls) / tool(result) pairs ending in a plain assistant.
    out = to_alternating(
        [
            {"role": _SYSTEM, "content": "you are a coding agent"},
            {"role": _USER, "content": "find foo"},
            {
                "role": _ASSISTANT,
                "content": "",
                "tool_calls": [{"function": {"name": "grep", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "a.py:1", "tool_call_id": "c1"},
            {
                "role": _ASSISTANT,
                "content": "",
                "tool_calls": [{"function": {"name": "read", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "def foo", "tool_call_id": "c2"},
            {"role": _ASSISTANT, "content": "It is in a.py."},
        ]
    )
    _assert_strict_alternation(out)
    # System stays first; the rest alternates user/assistant with no empty turns.
    assert out[0]["role"] == _SYSTEM
    assert _roles_after_system(out)[0] == _USER


def test_late_system_message_folds_into_user_turn() -> None:
    out = to_alternating(
        [
            {"role": _USER, "content": "q"},
            {"role": _SYSTEM, "content": "mid-conversation system"},
        ]
    )
    _assert_strict_alternation(out)
    # The late system message must not re-open a system block mid-conversation.
    assert out[0]["role"] == _USER
    assert "mid-conversation system" in out[0]["content"]


def test_multipart_content_flattens_to_text() -> None:
    out = to_alternating(
        [
            {
                "role": _USER,
                "content": [
                    {"type": "text", "text": "part one"},
                    {"type": "text", "text": "part two"},
                ],
            },
        ]
    )
    assert out == [{"role": _USER, "content": "part one\npart two"}]


def test_empty_turns_are_dropped() -> None:
    out = to_alternating(
        [
            {"role": _USER, "content": "q"},
            {"role": _ASSISTANT, "content": ""},  # no content, no tool_calls -> dropped
            {"role": _USER, "content": "still here"},
        ]
    )
    _assert_strict_alternation(out)
    # Both user turns survive and merge (the empty assistant between them vanished).
    assert _roles_after_system(out) == [_USER]
    assert out[0]["content"] == "q\nstill here"


def test_malformed_tool_call_entries_skipped() -> None:
    out = to_alternating(
        [
            {"role": _USER, "content": "q"},
            {
                "role": _ASSISTANT,
                "content": "thinking",
                "tool_calls": [
                    "not-a-dict",
                    {"function": {"arguments": "{}"}},  # no name
                    {"function": {"name": "ok", "arguments": "{}"}},
                ],
            },
        ]
    )
    _assert_strict_alternation(out)
    assert "Calling tool ok" in out[-1]["content"]
    assert out[-1]["content"].startswith("thinking")


def test_none_content_assistant_keeps_only_tool_note() -> None:
    # An assistant turn with content=None (OpenAI's shape for a pure tool call)
    # still produces a non-empty turn from its tool-call note.
    out = to_alternating(
        [
            {"role": _USER, "content": "q"},
            {
                "role": _ASSISTANT,
                "content": None,
                "tool_calls": [{"function": {"name": "f", "arguments": "{}"}}],
            },
        ]
    )
    _assert_strict_alternation(out)
    assert out[-1]["content"].startswith("Calling tool f")


def test_non_text_multipart_parts_are_dropped() -> None:
    # Image parts carry no text and must not leak into the flattened content.
    out = to_alternating(
        [
            {
                "role": _USER,
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                    {"type": "text", "text": "describe"},
                ],
            },
        ]
    )
    assert out == [{"role": _USER, "content": "describe"}]


def test_non_string_non_list_content_stringified() -> None:
    # A non-standard scalar content (number) is coerced to text rather than dropped.
    out = to_alternating([{"role": _USER, "content": 42}])
    assert out == [{"role": _USER, "content": "42"}]


def test_non_string_tool_arguments_serialized() -> None:
    out = to_alternating(
        [
            {"role": _USER, "content": "q"},
            {
                "role": _ASSISTANT,
                "content": "",
                "tool_calls": [{"function": {"name": "f", "arguments": {"k": "v"}}}],
            },
        ]
    )
    assert '{"k": "v"}' in out[-1]["content"]
