"""Tests for chat-message context-window truncation."""

from __future__ import annotations

import json

from lilbee.providers.worker.windowing import (
    WindowingOutcome,
    count_message_tokens,
    count_tools_overhead,
    window_messages_to_budget,
)


def _bytes_tokenizer(data: bytes) -> list[int]:
    """Stand-in tokenizer: one token per byte. Predictable for assertions."""
    return list(data)


def _sys() -> dict:
    return {"role": "system", "content": "sys"}


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant_text(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _assistant_tool_call(call_id: str, name: str, args: dict) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ],
    }


def _tool_result(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


class TestCountToolsOverhead:
    def test_returns_zero_when_no_tools(self) -> None:
        """No tools means no overhead, regardless of chat template."""
        assert count_tools_overhead(None, _bytes_tokenizer) == 0
        assert count_tools_overhead([], _bytes_tokenizer) == 0

    def test_exact_formula_pins_multiplier_and_preamble(self) -> None:
        """The estimate is ``int(raw * 1.5) + 256``. An accidental tweak to
        either constant (or the formula shape) breaks this assertion, forcing
        the change to be deliberate rather than a silent regression.
        """
        import json

        from lilbee.providers.worker.windowing import (
            _TOOLS_TEMPLATE_OVERHEAD_MULTIPLIER,
            _TOOLS_TEMPLATE_PREAMBLE_TOKENS,
        )

        tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
        raw = len(_bytes_tokenizer(json.dumps(tools).encode("utf-8")))
        expected = int(raw * _TOOLS_TEMPLATE_OVERHEAD_MULTIPLIER) + _TOOLS_TEMPLATE_PREAMBLE_TOKENS
        assert count_tools_overhead(tools, _bytes_tokenizer) == expected


class TestCountMessageTokens:
    def test_counts_content_and_overhead(self) -> None:
        """A plain text message counts its bytes plus per-message overhead."""
        msg = {"role": "user", "content": "hi"}
        assert count_message_tokens(msg, _bytes_tokenizer) == 2 + 4

    def test_counts_assistant_tool_calls(self) -> None:
        """Assistant messages with tool_calls add name + arguments bytes."""
        msg = _assistant_tool_call("c1", "search", {"q": "x"})
        # "search" = 6 bytes, '{"q": "x"}' = 10 bytes, overhead = 4
        assert count_message_tokens(msg, _bytes_tokenizer) == 6 + 10 + 4

    def test_skips_non_dict_tool_call_entries(self) -> None:
        """Malformed tool_calls list items (non-dict) don't break counting."""
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                "not a dict",
                {"id": "c1", "function": {"name": "x", "arguments": "{}"}},
            ],
        }
        assert count_message_tokens(msg, _bytes_tokenizer) == 1 + 2 + 4  # "x" + "{}" + overhead

    def test_counts_dict_shaped_arguments(self) -> None:
        """``arguments`` already parsed as a dict serialises to JSON for the count."""
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": {"q": "x"}}}],
        }
        # "f" (1) + json.dumps({"q":"x"}) (10) + overhead (4) = 15
        assert count_message_tokens(msg, _bytes_tokenizer) == 1 + 10 + 4


class TestWindowMessagesToBudget:
    def test_returns_unchanged_when_under_budget(self) -> None:
        """No truncation happens when the prompt already fits."""
        msgs = [_sys(), _user("hello")]
        outcome = window_messages_to_budget(msgs, budget=1000, tokenize=_bytes_tokenizer)
        assert outcome.messages == msgs
        assert outcome.dropped == 0

    def test_drops_oldest_tool_pair_first(self) -> None:
        """Tool call + result pair gets dropped before user/assistant exchanges."""
        msgs = [
            _sys(),
            _user("first search"),
            _assistant_tool_call("c1", "search", {"q": "long" * 100}),
            _tool_result("c1", "result" * 200),
            _user("now answer"),
            _assistant_text("ok"),
            _user("follow up"),
        ]
        total = sum(count_message_tokens(m, _bytes_tokenizer) for m in msgs)
        # Budget slightly under total but enough to fit everything except the
        # tool pair.
        pair_tokens = count_message_tokens(msgs[2], _bytes_tokenizer) + count_message_tokens(
            msgs[3], _bytes_tokenizer
        )
        outcome = window_messages_to_budget(
            msgs, budget=total - pair_tokens, tokenize=_bytes_tokenizer
        )
        assert outcome.messages is not None
        assert outcome.dropped == 2
        # System and the in-flight (final user) message must survive.
        assert outcome.messages[0]["role"] == "system"
        assert outcome.messages[-1]["content"] == "follow up"
        # Tool call + result are gone.
        roles = [m["role"] for m in outcome.messages]
        assert "tool" not in roles
        assert all("tool_calls" not in m for m in outcome.messages if m["role"] == "assistant")

    def test_drops_oldest_user_assistant_when_no_tool_pairs(self) -> None:
        """Without tool pairs, drop the OLDEST user/assistant exchange first."""
        msgs = [
            _sys(),
            _user("old question"),
            _assistant_text("old answer"),
            _user("new question"),
        ]
        total = sum(count_message_tokens(m, _bytes_tokenizer) for m in msgs)
        smallest_droppable = count_message_tokens(msgs[1], _bytes_tokenizer)
        outcome = window_messages_to_budget(
            msgs, budget=total - smallest_droppable, tokenize=_bytes_tokenizer
        )
        assert outcome.messages is not None
        assert outcome.dropped >= 1
        assert outcome.messages[0]["role"] == "system"
        assert outcome.messages[-1]["content"] == "new question"
        # The OLDEST exchange is the one that's gone.
        remaining_contents = [m["content"] for m in outcome.messages]
        assert "old question" not in remaining_contents

    def test_overflow_when_only_keep_set_exceeds_budget(self) -> None:
        """If system + last user message alone exceeds budget, overflow."""
        msgs = [_sys(), _user("x" * 500)]
        outcome = window_messages_to_budget(msgs, budget=10, tokenize=_bytes_tokenizer)
        assert outcome.messages is None
        assert outcome.requested > outcome.available
        assert outcome.available == 10

    def test_dropped_pair_matches_call_ids(self) -> None:
        """Tool messages are paired by tool_call_id; unrelated tool entries stay."""
        msgs = [
            _sys(),
            _user("first"),
            _assistant_tool_call("c1", "search", {"q": "a"}),
            _tool_result("c1", "result_a" * 100),
            _assistant_tool_call("c2", "search", {"q": "b"}),
            _tool_result("c2", "result_b"),
            _user("now answer"),
        ]
        # Force one pair drop.
        pair_tokens = count_message_tokens(msgs[2], _bytes_tokenizer) + count_message_tokens(
            msgs[3], _bytes_tokenizer
        )
        total = sum(count_message_tokens(m, _bytes_tokenizer) for m in msgs)
        outcome = window_messages_to_budget(
            msgs, budget=total - pair_tokens, tokenize=_bytes_tokenizer
        )
        assert outcome.messages is not None
        # First pair (c1) gone; second pair (c2) preserved.
        remaining_tool_ids = [
            (call.get("id"))
            for m in outcome.messages
            if m.get("role") == "assistant"
            for call in m.get("tool_calls") or []
        ]
        assert "c1" not in remaining_tool_ids
        assert "c2" in remaining_tool_ids

    def test_handles_messages_with_no_user_message(self) -> None:
        """Without a user message, ``_last_role_index`` returns -1 and the
        windowing loop has no droppable indices; an overfull prompt then
        surfaces as overflow rather than crashing. Verifies the -1 branch
        of ``_last_role_index``.
        """
        msgs = [_sys(), _assistant_text("filler" * 200)]
        outcome = window_messages_to_budget(msgs, budget=30, tokenize=_bytes_tokenizer)
        assert outcome.messages is None
        assert outcome.requested > outcome.available

    def test_under_budget_with_no_user_message_passes_through(self) -> None:
        """Without a user message and under budget, returns the list unchanged."""
        msgs = [_sys(), _assistant_text("brief")]
        outcome = window_messages_to_budget(msgs, budget=1000, tokenize=_bytes_tokenizer)
        assert outcome.messages == msgs
        assert outcome.dropped == 0


class TestWindowingOutcome:
    def test_fit_factory(self) -> None:
        out = WindowingOutcome.fit([_user("x")], dropped=0, available=100)
        assert out.messages is not None
        assert out.dropped == 0

    def test_overflow_factory(self) -> None:
        out = WindowingOutcome.overflow(requested=200, available=100)
        assert out.messages is None
        assert out.requested == 200
        assert out.available == 100
