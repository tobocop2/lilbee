"""Tests for fleet chat-context windowing."""

from __future__ import annotations

import json

from lilbee.providers.fleet.windowing import (
    _message_tokens,
    _tools_tokens,
    estimate_tokens,
    window_messages,
)

_SYS = {"role": "system", "content": "you are a helpful assistant"}
_USER = {"role": "user", "content": "final question"}


def _pair(i: int, arg_len: int = 200, result_len: int = 400) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"c{i}",
                    "type": "function",
                    "function": {"name": "search", "arguments": json.dumps({"q": "x" * arg_len})},
                }
            ],
        },
        {"role": "tool", "tool_call_id": f"c{i}", "content": "y" * result_len},
    ]


def test_estimate_tokens_rounds_up() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 2  # ceil(4/3)


def test_message_tokens_counts_content_and_tool_calls() -> None:
    plain = _message_tokens({"role": "user", "content": "abcdef"})
    with_calls = _message_tokens(_pair(0)[0])
    assert plain > 0
    assert with_calls > plain  # tool-call JSON adds tokens


def test_message_tokens_handles_non_string_content() -> None:
    # vision multipart content (a list) is counted via its JSON, not skipped
    tokens = _message_tokens({"role": "user", "content": [{"type": "text", "text": "hello"}]})
    assert tokens > 0


def test_tools_tokens_zero_when_absent() -> None:
    assert _tools_tokens(None) == 0
    assert _tools_tokens([]) == 0


def test_tools_tokens_counts_schema() -> None:
    tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
    assert _tools_tokens(tools) > 0


def test_window_keeps_everything_when_it_fits() -> None:
    msgs = [_SYS, _USER]
    result = window_messages(msgs, None, budget=10_000)
    assert result.fits is True
    assert result.dropped == 0
    assert result.messages == msgs


def test_window_drops_oldest_turns_keeps_system_and_final() -> None:
    msgs = [_SYS]
    for i in range(20):
        msgs.extend(_pair(i))
    msgs.append(_USER)

    result = window_messages(msgs, None, budget=300)
    assert result.fits is True
    assert result.dropped > 0
    assert result.messages[0] == _SYS  # system always survives
    assert result.messages[-1] == _USER  # current turn always survives
    assert result.prompt_tokens <= 300


def test_window_never_starts_with_orphan_tool_result() -> None:
    # A budget that lands mid-pair must not keep a tool result whose call was dropped.
    msgs = [_SYS]
    for i in range(10):
        msgs.extend(_pair(i))
    msgs.append(_USER)

    result = window_messages(msgs, None, budget=400)
    non_system = [m for m in result.messages if m["role"] != "system"]
    assert non_system[0]["role"] != "tool"


def test_window_trims_orphan_tool_when_only_result_fits() -> None:
    # Budget fits the small tool result but not its huge originating call, so the
    # kept suffix would start with an orphan ``tool`` message and must be trimmed.
    msgs = [
        _SYS,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c0", "type": "function", "function": {"name": "s", "arguments": "z" * 3000}}
            ],
        },
        {"role": "tool", "tool_call_id": "c0", "content": "ok"},
        _USER,
    ]
    result = window_messages(msgs, None, budget=40)
    roles = [m["role"] for m in result.messages]
    assert "tool" not in roles
    assert roles[-1] == "user"


def test_window_overflow_when_system_and_final_exceed_budget() -> None:
    msgs = [_SYS, {"role": "user", "content": "x" * 9000}]
    result = window_messages(msgs, None, budget=50)
    assert result.fits is False
    # best-effort: still returns system + the final message for the error path
    assert result.messages[0] == _SYS
    assert result.messages[-1]["content"].startswith("x")


def test_window_system_only_respects_budget() -> None:
    assert window_messages([_SYS], None, budget=10_000).fits is True
    assert window_messages([_SYS], None, budget=1).fits is False


def test_window_budget_accounts_for_tools() -> None:
    tools = [
        {
            "type": "function",
            "function": {"name": "search", "parameters": {"x": "y" * 3000}},
        }
    ]
    # tools alone blow the budget => even the lone user turn cannot fit
    result = window_messages([_USER], tools, budget=50)
    assert result.fits is False
