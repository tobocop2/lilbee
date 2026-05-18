"""Tests for the token-budget conversation history windower."""

from __future__ import annotations

from lilbee.retrieval.query.history_window import (
    estimate_tokens,
    windowed_history,
)


def _msg(role: str, chars: int, marker: str = "") -> dict[str, str]:
    body = marker + ("x" * (chars - len(marker)))
    return {"role": role, "content": body}


class TestEstimateTokens:
    def test_at_least_one_token(self) -> None:
        assert estimate_tokens({"role": "user", "content": ""}) == 1

    def test_chars_over_four_rule(self) -> None:
        assert estimate_tokens({"role": "user", "content": "x" * 40}) == 10


class TestWindowedHistory:
    def test_returns_all_when_under_budget(self) -> None:
        msgs = [_msg("user", 100), _msg("assistant", 100)]
        out = windowed_history(msgs, max_tokens=10_000)
        assert out == msgs

    def test_drops_oldest_pairs_when_over_budget(self) -> None:
        # 6 pairs at ~256 tokens each = ~3072 tokens total; budget 1000.
        msgs = []
        for i in range(6):
            msgs.append(_msg("user", 1024, marker=f"u{i}-"))
            msgs.append(_msg("assistant", 1024, marker=f"a{i}-"))
        out = windowed_history(msgs, max_tokens=1000)
        assert len(out) < len(msgs)
        # Newest pair survives.
        assert out[-2]["content"].startswith("u5-")
        assert out[-1]["content"].startswith("a5-")
        # Window starts at a user message (no orphaned assistant).
        assert out[0]["role"] == "user"

    def test_empty_input_returns_empty(self) -> None:
        assert windowed_history([], max_tokens=1000) == []

    def test_zero_or_negative_budget_returns_input_unchanged(self) -> None:
        msgs = [_msg("user", 100), _msg("assistant", 100)]
        assert windowed_history(msgs, max_tokens=0) == msgs

    def test_keeps_last_message_even_if_oversized(self) -> None:
        """A single huge message is kept; the caller decides how to handle it."""
        msgs = [_msg("user", 1_000_000)]
        out = windowed_history(msgs, max_tokens=10)
        assert out == msgs

    def test_realigns_when_history_does_not_start_at_user(self) -> None:
        """Legacy state with a leading assistant message is realigned to user-start."""
        msgs = [
            _msg("assistant", 4096, marker="orphan-"),
            _msg("user", 4096, marker="u0-"),
            _msg("assistant", 4096, marker="a0-"),
            _msg("user", 4096, marker="u1-"),
            _msg("assistant", 4096, marker="a1-"),
        ]
        out = windowed_history(msgs, max_tokens=2000)
        assert out[0]["role"] == "user"
