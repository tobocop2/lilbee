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


def _chars_over_two(messages: list[dict]) -> int:
    """A stand-in engine count: a fixed 500-token tool preamble plus half a token per char."""
    return 500 + sum(len(m.get("content") or "") for m in messages) // 2


def _long_convo(turns: int) -> list[dict]:
    msgs: list[dict] = [_SYS]
    for _ in range(turns):
        msgs.append({"role": "user", "content": "x" * 280})
        msgs.append({"role": "assistant", "content": "y" * 280})
    msgs.append({"role": "user", "content": "q"})
    return msgs


# The estimate of _long_convo(4) is 842 tokens, just under a budget of 850.
_NEAR_BUDGET = 850


def test_window_measure_drops_turns_the_estimate_keeps() -> None:
    msgs = _long_convo(4)
    assert window_messages(msgs, None, budget=_NEAR_BUDGET).dropped == 0
    measured = window_messages(msgs, None, budget=_NEAR_BUDGET, measure=_chars_over_two)
    assert measured.dropped > 0
    assert measured.fits is True
    assert measured.prompt_tokens == _chars_over_two(measured.messages)
    assert measured.prompt_tokens <= _NEAR_BUDGET
    assert measured.messages[0] == _SYS
    assert measured.messages[-1]["content"] == "q"


def test_window_measure_measures_again_after_each_drop() -> None:
    calls: list[int] = []

    def measure(messages: list[dict]) -> int:
        calls.append(len(messages))
        return 830 + 5 * len(messages)

    result = window_messages(_long_convo(4), None, budget=_NEAR_BUDGET, measure=measure)
    assert result.fits is True
    assert calls == [10, 9, 8, 7, 6, 5, 4]
    assert len(calls) <= len(_long_convo(4))


def test_window_measure_is_called_once_far_under_the_budget() -> None:
    calls: list[int] = []

    def measure(messages: list[dict]) -> int:
        calls.append(len(messages))
        return 1

    result = window_messages(_long_convo(4), None, budget=5_000, measure=measure)
    assert calls == [10]
    assert result.dropped == 0
    assert result.prompt_tokens == 1


def test_window_measure_catches_a_fixed_preamble_the_estimate_leaves_just_under() -> None:
    def with_preamble(messages: list[dict]) -> int:
        return 300 + sum(_message_tokens(m) for m in messages)

    # The estimate (842) is 158 tokens under the budget (1000), 16 percent: under any
    # ratio gate. The preamble is 300, so the real prompt is over.
    budget = 1_000
    assert window_messages(_long_convo(4), None, budget=budget).dropped == 0
    result = window_messages(_long_convo(4), None, budget=budget, measure=with_preamble)
    assert result.dropped > 0
    assert result.fits is True
    assert result.prompt_tokens == with_preamble(result.messages)
    assert result.prompt_tokens <= budget


def test_window_measure_is_called_once_near_the_budget() -> None:
    calls: list[int] = []

    def measure(messages: list[dict]) -> int:
        calls.append(len(messages))
        return 800

    result = window_messages(_long_convo(4), None, budget=_NEAR_BUDGET, measure=measure)
    assert calls == [10]
    assert result.prompt_tokens == 800
    assert result.dropped == 0


def test_window_measure_fits_a_prompt_that_measures_exactly_the_budget() -> None:
    calls: list[int] = []

    def measure(messages: list[dict]) -> int:
        calls.append(len(messages))
        return _NEAR_BUDGET

    result = window_messages(_long_convo(4), None, budget=_NEAR_BUDGET, measure=measure)
    assert calls == [10]
    assert result.dropped == 0
    assert result.fits is True


def _shrinking_measure(over: int, first_len: int = 10):
    """A measure that is *over* tokens past the budget for the full list and fits for less."""

    def measure(messages: list[dict]) -> int:
        return _NEAR_BUDGET + over if len(messages) == first_len else 800

    return measure


def test_window_measure_drops_a_turn_for_an_overshoot_of_one_token() -> None:
    result = window_messages(
        _long_convo(4), None, budget=_NEAR_BUDGET, measure=_shrinking_measure(1)
    )
    assert result.dropped == 1


def test_window_measure_stops_dropping_when_the_estimate_equals_the_overshoot() -> None:
    first_turn_estimate = _message_tokens(_long_convo(4)[1])
    result = window_messages(
        _long_convo(4),
        None,
        budget=_NEAR_BUDGET,
        measure=_shrinking_measure(first_turn_estimate),
    )
    assert result.dropped == 1


def test_window_measure_adds_up_the_estimates_of_the_dropped_turns() -> None:
    first_turn_estimate = _message_tokens(_long_convo(4)[1])
    result = window_messages(
        _long_convo(4),
        None,
        budget=_NEAR_BUDGET,
        measure=_shrinking_measure(first_turn_estimate + 1),
    )
    assert result.dropped == 2


def test_window_system_only_fits_a_budget_equal_to_its_estimate() -> None:
    assert window_messages([_SYS], None, budget=_message_tokens(_SYS)).fits is True


def test_window_keeps_a_turn_that_fills_the_budget_exactly() -> None:
    older = {"role": "user", "content": "x" * 30}
    budget = _message_tokens(_SYS) + _message_tokens(older) + _message_tokens(_USER)
    result = window_messages([_SYS, older, _USER], None, budget=budget)
    assert result.dropped == 0
    assert result.fits is True


def test_window_measure_empty_result_reports_the_estimate_of_the_system_messages() -> None:
    msgs = [_SYS, *_pair(0), *_pair(1)]
    result = window_messages(msgs, None, budget=270, measure=lambda m: 2_000)
    assert result.fits is False
    assert result.messages == [_SYS]
    assert result.prompt_tokens == _message_tokens(_SYS)


def test_window_measure_unfixable_keeps_the_last_turn_and_reports_the_count() -> None:
    result = window_messages(_long_convo(2), None, budget=20, measure=_chars_over_two)
    assert result.fits is False
    assert result.messages[-1]["content"] == "q"
    assert result.prompt_tokens == _chars_over_two(result.messages)


def test_window_measure_not_called_without_a_conversation() -> None:
    def measure(messages: list[dict]) -> int:
        raise AssertionError("no conversation to measure")

    assert window_messages([_SYS], None, budget=10_000, measure=measure).fits is True


def test_window_measure_drops_a_last_turn_left_as_an_orphan_tool_result() -> None:
    msgs = [_SYS, *_pair(0), *_pair(1)]
    result = window_messages(msgs, None, budget=270, measure=lambda m: 2_000)
    assert result.fits is False
    assert result.messages == [_SYS]
