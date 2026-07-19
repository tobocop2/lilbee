"""Searcher.summarize_history: the model call compaction folds turns through."""

from __future__ import annotations

from unittest.mock import MagicMock

from lilbee.core.config import cfg
from lilbee.retrieval.query import Searcher
from lilbee.retrieval.query.compaction import MAX_COMPACT_CALLS, plan_compaction, summary_cap


def _searcher(provider: MagicMock) -> Searcher:
    return Searcher(cfg, provider, MagicMock(), MagicMock(), MagicMock(), MagicMock())


def _provider(text: str = "notes") -> MagicMock:
    provider = MagicMock()
    provider.chat.return_value = MagicMock(text=text)
    return provider


def _msgs(n: int, size: int = 400) -> list[dict[str, str]]:
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i} " + "x" * size}
        for i in range(n)
    ]


def _thinking_provider() -> MagicMock:
    """A model that reasons, reproducing what a real Qwen3 returns.

    Measured against Qwen3-4B on real hardware: given a budget and thinking left
    on, it spends the whole budget reasoning, llama.cpp force-closes the block at
    the limit, and the reply is one complete ``<think>...</think>`` and nothing
    else -- which ``strip_reasoning`` deletes in full, leaving "". Every other
    provider double in this file returns clean prose, which no reasoning model
    does; that is why this shipped green.
    """
    provider = MagicMock()

    def chat(_messages, *, stream=False, options=None):
        thinking_off = (options or {}).get("think") is False
        text = "notes" if thinking_off else "<think>Okay, let me work through this. That</think>"
        return MagicMock(text=text)

    provider.chat.side_effect = chat
    return provider


def test_the_summarize_call_turns_thinking_off() -> None:
    """The contract: internal utility calls do not get to reason on the budget."""
    provider = _provider("notes")
    _searcher(provider).summarize_history(_msgs(6), "")
    options = provider.chat.call_args.kwargs["options"]
    assert options["think"] is False
    assert options["temperature"] == 0


def test_a_reasoning_model_does_not_strand_every_turn() -> None:
    """The pod failure, pinned: 60 turns dropped, nothing condensed, summary empty.

    With thinking left on this returned CompactionResult(summary="", condensed=0,
    stranded=<all>) -- the turns were dropped from context for a summary that was
    never written, and the only symptom was a long pause.
    """
    result = _searcher(_thinking_provider()).summarize_history(_msgs(6), "")
    assert result.summary, "a reasoning model must still produce a summary"
    assert result.condensed > 0
    assert result.stranded == 0, "turns must not be dropped for a summary that was never written"


def test_a_non_native_reasoning_model_recovers_the_reasoning() -> None:
    """think=False reaches only llama-server; other providers strip it.

    Over Ollama, LM Studio, or a cloud API a reasoning model can spend the whole
    budget inside <think> even with think=False requested, because that provider
    ignored the flag. The reasoning is a summary of the turns, so it must be
    recovered rather than the turns stranded for an empty answer.
    """
    provider = MagicMock()
    # An answerless reply, as a provider that could not disable thinking returns.
    only_reasoning = "<think>They discussed head bolt torque: 30, 60, then 90 deg.</think>"
    provider.chat.return_value = MagicMock(text=only_reasoning)
    result = _searcher(provider).summarize_history(_msgs(6), "")
    assert "90 deg" in result.summary, "the reasoning content is the recovered summary"
    assert result.condensed > 0
    assert result.stranded == 0


def test_an_overflowing_batch_splits_instead_of_stranding() -> None:
    """The live failure, pinned: batch sizing works from a token estimate, the
    estimate ran under the server's real count, and the summarize call itself
    overflowed the window -- stranding every turn it was preserving. Overflow
    must halve and retry, not fail."""
    from lilbee.providers.base import ProviderError, ProviderErrorKind

    provider = MagicMock()
    calls: list[int] = []

    def chat(messages, *, stream=False, options=None):
        # Refuse anything above a tiny window, as the real preflight does.
        content = messages[0]["content"]
        calls.append(len(content))
        if len(content) > 2000:
            raise ProviderError(
                "too big", provider="fleet", kind=ProviderErrorKind.CONTEXT_OVERFLOW
            )
        return MagicMock(text="notes for a slice")

    provider.chat.side_effect = chat
    result = _searcher(provider).summarize_history(_msgs(8, size=600), "")
    assert result.stranded == 0, "overflow must split, not strand"
    assert result.condensed == 8
    assert result.summary
    assert len(calls) > 1, "the oversized call must have been retried in halves"


def test_a_single_message_overflow_still_fails_safe() -> None:
    """One message too big for the window cannot be split further; it strands
    with the warning rather than recursing forever."""
    from lilbee.providers.base import ProviderError, ProviderErrorKind

    provider = MagicMock()
    provider.chat.side_effect = ProviderError(
        "too big", provider="fleet", kind=ProviderErrorKind.CONTEXT_OVERFLOW
    )
    result = _searcher(provider).summarize_history(_msgs(1, size=99999), "")
    assert result.condensed == 0
    assert result.stranded == 1


def test_summarizes_a_small_backlog_in_one_call() -> None:
    provider = _provider("they compared torque specs")
    cfg.chat_n_ctx_target = 8192
    result = _searcher(provider).summarize_history(_msgs(4))
    assert result.summary == "they compared torque specs"
    assert result.condensed == 4
    assert result.stranded == 0
    assert provider.chat.call_count == 1


def test_folds_a_large_backlog_batch_by_batch() -> None:
    """A 32k conversation compacted for a 2k model must not be one giant call."""
    provider = _provider()
    cfg.chat_n_ctx_target = 2048
    _searcher(provider).summarize_history(_msgs(200))
    assert provider.chat.call_count > 1, "the backlog must be folded in batches"
    for call in provider.chat.call_args_list:
        prompt = call.args[0][0]["content"]
        # each prompt must fit the window it is summarizing for
        assert len(prompt) // 4 < cfg.chat_n_ctx_target


def test_each_batch_is_summarized_once_not_through_the_running_summary() -> None:
    """No batch may be handed the running notes: that is a summary of a summary.

    Folding notes through each batch makes the earliest turns a summary nested as
    deep as there are batches (~16 at a 2k window), which a small model degrades
    into drift. Each batch is summarized once and the notes are merged instead.
    """
    provider = MagicMock()
    provider.chat.side_effect = [MagicMock(text=f"notes {i}") for i in range(50)]
    cfg.chat_n_ctx_target = 2048
    result = _searcher(provider).summarize_history(_msgs(200))
    for call in provider.chat.call_args_list:
        prompt = call.args[0][0]["content"]
        assert "Earlier notes:" not in prompt, "a batch was fed the running summary"
        assert "notes 0" not in prompt, "batch output leaked into another batch's prompt"
    assert result.summary


def test_a_huge_backlog_is_bounded_and_reports_what_it_dropped() -> None:
    """A 100k conversation on a 2k model must not cost 100 calls and a stall.

    It cannot be remembered at any price, so condense the recent slice and say
    how much was dropped rather than stalling minutes to produce mush.
    """
    provider = _provider()
    cfg.chat_n_ctx_target = 2048
    huge = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": "word " * 400}
        for i in range(200)
    ]
    result = _searcher(provider).summarize_history(huge)
    assert provider.chat.call_count <= MAX_COMPACT_CALLS + 1, "one merge pass at most on top"
    assert result.stranded > 0, "the turns beyond the budget are reported, not hidden"
    assert result.condensed > 0
    assert result.condensed + result.stranded == len(huge)


def test_merged_notes_over_the_cap_get_one_compression_pass() -> None:
    """Several batches of notes can outgrow the cap; compress once, at the end.

    Compressing per batch would be the summary-of-a-summary this design avoids,
    and leaving them uncompressed would spend the window the cap protects.
    """
    long_note = "note " * 300  # ~375 tokens, well over summary_cap(2048)
    provider = _provider(long_note)
    cfg.chat_n_ctx_target = 2048
    plan = plan_compaction(_msgs(60), "", ctx_target=2048)
    result = _searcher(provider).summarize_history(_msgs(60))
    assert provider.chat.call_count == len(plan.batches) + 1, "one merge pass on top of the batches"
    assert result.summary


def test_caps_the_reply_length_to_the_window() -> None:
    provider = _provider()
    cfg.chat_n_ctx_target = 2048
    _searcher(provider).summarize_history(_msgs(4))
    assert provider.chat.call_args.kwargs["options"]["num_predict"] == summary_cap(2048)


def test_a_provider_failure_keeps_the_previous_notes() -> None:
    """Losing the summary would drop every turn it stood for."""
    provider = MagicMock()
    provider.chat.side_effect = RuntimeError("engine died")
    cfg.chat_n_ctx_target = 8192
    result = _searcher(provider).summarize_history(_msgs(4), "earlier: the oil spec")
    assert result.summary == "earlier: the oil spec"


def test_turns_a_failed_batch_reports_as_stranded_not_condensed() -> None:
    """A batch the model could not condense is gone; saying otherwise misleads.

    condensed counts what actually reached the notes, not what was planned: the
    marker built from it tells the user what the model still knows.
    """
    provider = MagicMock()
    provider.chat.side_effect = RuntimeError("engine died mid-compaction")
    cfg.chat_n_ctx_target = 8192
    result = _searcher(provider).summarize_history(_msgs(8))
    assert result.condensed == 0, "nothing was condensed, so claim nothing"
    assert result.stranded == 8, "those turns are gone and must be reported"


def test_a_partial_failure_counts_only_the_batches_that_landed() -> None:
    """One batch failing must not invalidate the notes the others produced."""
    provider = MagicMock()
    # first batch summarizes, second fails, third summarizes
    provider.chat.side_effect = [
        MagicMock(text="notes A"),
        RuntimeError("hiccup"),
        MagicMock(text="notes C"),
        MagicMock(text="notes D"),
    ]
    cfg.chat_n_ctx_target = 2048
    result = _searcher(provider).summarize_history(_msgs(40))
    assert result.stranded > 0, "the failed batch's turns are reported as lost"
    assert result.condensed > 0, "the batches that landed are still counted"
    assert "notes A" in result.summary


def test_an_empty_reply_keeps_the_previous_notes() -> None:
    provider = _provider("   ")
    cfg.chat_n_ctx_target = 8192
    result = _searcher(provider).summarize_history(_msgs(4), "earlier: the oil spec")
    assert result.summary == "earlier: the oil spec"


def test_nothing_to_summarize_makes_no_call() -> None:
    provider = _provider()
    cfg.chat_n_ctx_target = 8192
    assert _searcher(provider).summarize_history([], "keep me").summary == "keep me"
    provider.chat.assert_not_called()


def test_on_batch_hears_each_batch_before_its_model_call() -> None:
    """Progress runs 1..total in order, so a client can tick a live indicator."""
    heard: list[tuple[int, int]] = []
    _searcher(_provider("notes")).summarize_history(
        _msgs(6), "", on_batch=lambda batch, total: heard.append((batch, total))
    )
    assert heard, "at least one batch must be reported"
    total = heard[0][1]
    assert heard == [(index + 1, total) for index in range(len(heard))]
