"""History compaction: what gets folded away, and what the prompt carries."""

from __future__ import annotations

from lilbee.retrieval.query.compaction import (
    COMPACT_KEEP_RECENT,
    COMPACT_MAX_TOKENS,
    SUMMARY_REQUEST,
    batch_overflow,
    foldable,
    overflow,
    prompt_history,
    summary_cap,
    summary_messages,
    summary_word_budget,
)
from lilbee.retrieval.query.history_window import estimate_tokens


def _msgs(n: int, size: int = 400) -> list[dict[str, str]]:
    """n alternating turns, each roughly size/4 tokens."""
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i} " + "x" * size}
        for i in range(n)
    ]


def test_no_summary_yields_no_synthetic_messages() -> None:
    assert summary_messages("") == []
    assert summary_messages("   ") == []


def test_summary_rides_as_an_alternating_user_assistant_pair() -> None:
    """A second system message would be dropped by most chat templates."""
    pair = summary_messages("we discussed torque")
    assert [m["role"] for m in pair] == ["user", "assistant"]
    assert pair[0]["content"] == SUMMARY_REQUEST
    assert pair[1]["content"] == "we discussed torque"


def test_overflow_is_empty_when_everything_fits() -> None:
    assert overflow(_msgs(4), max_tokens=10_000) == []


def test_overflow_returns_the_oldest_turns_that_do_not_fit() -> None:
    history = _msgs(10)
    dropped = overflow(history, max_tokens=300)
    assert dropped, "expected the oldest turns to overflow a tight budget"
    assert dropped == history[: len(dropped)]
    assert len(dropped) % 2 == 0, "turns drop in pairs, never orphaning a reply"


def test_prompt_history_puts_the_summary_first_and_keeps_recent_turns() -> None:
    history = _msgs(10)
    out = prompt_history(history, "earlier: torque specs", max_tokens=600)
    assert out[0]["content"] == SUMMARY_REQUEST
    assert out[1]["content"] == "earlier: torque specs"
    assert out[-1] == history[-1], "the newest turn must always survive"


def test_prompt_history_without_a_summary_is_just_the_window() -> None:
    history = _msgs(10)
    out = prompt_history(history, "", max_tokens=600)
    assert all(m["content"] != SUMMARY_REQUEST for m in out)
    assert out[-1] == history[-1]


def test_summary_cap_scales_down_with_a_small_window() -> None:
    """A flat cap would spend a third of a 2048-target model's history budget."""
    assert summary_cap(32768) == COMPACT_MAX_TOKENS
    assert summary_cap(2048) < COMPACT_MAX_TOKENS
    assert summary_cap(256) >= 64, "still large enough to hold a useful note"


def test_the_word_budget_always_fits_the_token_cap() -> None:
    """The prompt's word ask and num_predict must agree at every window size:
    asking for more words than the cap holds guarantees a clipped final
    sentence."""
    for ctx in (256, 2048, 8192, 32768, 131072):
        words = summary_word_budget(ctx)
        assert words <= summary_cap(ctx), f"ctx={ctx}: asked {words} words > cap"
        assert words >= 48, f"ctx={ctx}: {words} words is too small for a useful note"


def test_batches_each_fit_the_current_model_window() -> None:
    """Switching a long conversation onto a small model must not build one huge prompt.

    A 32k-model conversation compacted for a 2k model drops its whole backlog at
    once; summarizing that in a single call would overflow the small model and the
    fallback would lose every dropped turn.
    """
    dropped = _msgs(200)  # ~20k tokens, i.e. a large conversation
    ctx = 2048
    batches = batch_overflow(dropped, "", ctx_target=ctx)
    assert len(batches) > 1, "a 20k backlog must not be summarized in one 2k call"
    # `estimate < ctx` is the invariant that shipped a live failure: a batch
    # estimated at 1728 tokens reached a 2048-token server as ~2666 real tokens
    # (terse text tokenizes denser than chars/4, and the server adds a chat
    # template), the call was rejected, and every turn stranded. The real
    # invariant: even at the worst-case estimate gap, prompt plus reply fits.
    worst_case_ratio = 1.6
    for batch in batches:
        cost = sum(estimate_tokens(m) for m in batch)
        real_worst = int(cost * worst_case_ratio) + 320 + 64  # + reply cap + wrapper
        assert real_worst <= ctx, (
            f"batch estimated at {cost} could reach the server as {real_worst} "
            f"tokens and overflow the {ctx} window it is shrinking history for"
        )
    # every dropped turn is accounted for, none skipped
    assert sum(len(b) for b in batches) == len(dropped)


def test_batches_leave_room_for_the_previous_summary() -> None:
    """The previous notes ride in every batch prompt, so they must be budgeted."""
    dropped = _msgs(60)
    lean = batch_overflow(dropped, "", ctx_target=4096)
    fat = batch_overflow(dropped, "s" * 6000, ctx_target=4096)
    assert len(fat) >= len(lean), "a long previous summary must shrink the batches"


def test_a_single_oversized_turn_is_clipped_rather_than_sent_to_fail() -> None:
    """One turn bigger than the window would fail every call and lose the lot."""
    huge = [{"role": "user", "content": "y" * 200_000}]
    batches = batch_overflow(huge, "", ctx_target=2048)
    assert len(batches) == 1
    assert estimate_tokens(batches[0][0]) < 2048
    assert batches[0][0]["content"].endswith("[…clipped]")


def test_no_overflow_yields_no_batches() -> None:
    assert batch_overflow([], "", ctx_target=2048) == []


def test_an_oversized_turn_flushes_the_batch_being_built() -> None:
    """A giant turn must not be merged into the batch already in progress."""
    dropped = [
        {"role": "user", "content": "small one"},
        {"role": "assistant", "content": "small two"},
        {"role": "user", "content": "y" * 200_000},  # bigger than any batch
        {"role": "assistant", "content": "small three"},
    ]
    batches = batch_overflow(dropped, "", ctx_target=2048)
    # the two small turns batch together, the giant one stands alone (clipped)
    assert [len(b) for b in batches] == [2, 1, 1]
    assert batches[0][0]["content"] == "small one"
    assert batches[1][0]["content"].endswith("[…clipped]")
    assert batches[2][0]["content"] == "small three"


def test_a_summary_that_fits_buys_its_room_from_recent_turns() -> None:
    """Carrying the summary must not push the prompt past the limit it enforces."""
    history = _msgs(10)  # ~100 tokens each, ~1000 total
    # Sized so the summary (~312 with its request line) forces turns out, while
    # the newest pair still fits the remainder: the buy-back case, not the drop.
    budget = 1000
    with_summary = prompt_history(history, "s" * 1200, max_tokens=budget)
    without = prompt_history(history, "", max_tokens=budget)
    assert with_summary[0]["content"] == SUMMARY_REQUEST, "the summary rode along"
    assert len(with_summary) < len(without), "and it was paid for out of the recent turns"
    assert sum(estimate_tokens(m) for m in with_summary) <= budget


def test_a_summary_that_cannot_fit_is_dropped_not_stacked() -> None:
    """When the newest turns alone bust the budget, the notes lose, not the question.

    windowed_history keeps the newest pair whatever it costs, so stacking a
    summary on top of an already-oversized window would push the prompt further
    past the budget than carrying no summary at all -- and an overflowed prompt
    is an engine failure rather than a worse answer.
    """
    history = _msgs(10)
    budget = 600  # the newest pair alone is over this
    with_summary = prompt_history(history, "s" * 2000, max_tokens=budget)
    without = prompt_history(history, "", max_tokens=budget)
    assert all(m["content"] != SUMMARY_REQUEST for m in with_summary), "the summary was dropped"
    assert with_summary == without, "and the prompt is exactly what it would be without one"


class TestFoldableAlignment:
    """The fold boundary must leave the kept window opening on a user turn."""

    def test_odd_history_keeps_alternation_intact(self):
        """An interrupted turn persists an unpaired user message, so a history
        can be odd-length. Cutting a fixed count then leaves the kept window
        starting on an assistant reply, and the assembled prompt runs
        user, assistant, assistant -- the shape strict chat templates reject."""
        history = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
        ]
        dropped = foldable(history)
        kept = history[len(dropped) :]
        assert kept, "folding must not consume the whole history"
        assert kept[0]["role"] == "user"

    def test_even_history_is_unchanged(self):
        history = _msgs(10)
        dropped = foldable(history)
        assert len(dropped) == len(history) - COMPACT_KEEP_RECENT
        assert history[len(dropped)]["role"] == "user"

    def test_short_history_folds_nothing(self):
        assert foldable(_msgs(COMPACT_KEEP_RECENT)) == []

    def test_a_tail_with_no_user_turn_keeps_the_plain_boundary(self):
        """Nothing to align to: scanning past the end and folding everything
        away would throw the whole conversation out to fix its shape."""
        history = [
            {"role": "assistant", "content": f"a{i}"} for i in range(COMPACT_KEEP_RECENT + 4)
        ]
        dropped = foldable(history)
        assert len(dropped) == len(history) - COMPACT_KEEP_RECENT
