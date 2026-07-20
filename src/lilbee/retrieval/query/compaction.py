"""Condense the turns that no longer fit the prompt into a rolling summary.

Windowing alone drops the oldest turns outright, so a resumed conversation the
user can still scroll is one the model cannot see: it answers as if the earlier
turns never happened. Compaction keeps a summary of what was dropped and carries
it at the head of the prompt, so old context degrades to a gist instead of
vanishing. The transcript on disk and on screen is never touched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from lilbee.retrieval.query.history_window import (
    chars_for_tokens,
    estimate_text_tokens,
    estimate_tokens,
    windowed_history,
)

if TYPE_CHECKING:
    from lilbee.retrieval.query.searcher import ChatMessage

# Kept simple: 0.6B models botch elaborate structured-summary instructions.
# {words} is filled from summary_word_budget so the ask agrees with the cap.
COMPACT_PROMPT = (
    "Condense the conversation below into brief factual notes that let an "
    "assistant carry it on. Keep names, numbers, decisions, and anything left "
    "unresolved; drop pleasantries. Under {words} words. Return ONLY the notes.\n\n"
    "{previous}Conversation:\n{transcript}"
)

# Ceiling on a summary's tokens; ctx/8 governs below it. A tighter ceiling
# flattens long conversations to a few hundred words even on a 32k window.
COMPACT_MAX_TOKENS = 1024


def summary_word_budget(ctx_target: int) -> int:
    """The word count the prompt asks for, derived from the token cap.

    Roughly three words per four tokens, so the instruction and ``num_predict``
    agree: asking for more words than the cap can hold guarantees a truncated
    final sentence, and asking for far fewer wastes the window.
    """
    return summary_cap(ctx_target) * 3 // 4


# A summary must never eat the window it exists to protect: at a 2048 target the
# flat 320-token cap would be a third of the history budget. Scale it down with
# the model, with a floor that still fits a useful note.
_SUMMARY_CTX_FRACTION = 8
_SUMMARY_MIN_TOKENS = 64

# Rough cost of the instruction wrapper around a batch transcript.
_PROMPT_OVERHEAD_TOKENS = 64
# Fraction of the window a batch may claim: chars/4 under-counts terse text by
# up to ~1.8x (measured), and an overflowing batch strands its turns.
_ESTIMATE_SAFETY_FRACTION = 0.6
# Never build a batch smaller than this, however tight the window.
_MIN_BATCH_TOKENS = 128

# Most model calls one compaction may spend. Switching a 100k-token conversation
# onto a 2k model produces ~100 batches; folding them all would stall the user's
# next turn for minutes to produce a 256-token note, i.e. ~99% of the content is
# discarded either way. Condense the most recent slice well and say plainly that
# the rest was dropped, rather than stalling to produce mush.
MAX_COMPACT_CALLS = 4

# Compaction fires at a fraction of the budget rather than at the limit, and
# clears down to a summary plus the newest exchanges.
#
# Firing AT the limit and folding only the overflow leaves the history still at
# the limit, so the next turn overflows again and every later turn pays a model
# call. Triggering early and clearing deep buys many turns of headroom for the
# same one call: driving these functions over a 40-turn conversation of ~350
# token turns costs 2 calls at the 8192 context floor and 18 at 2048 (0 at 32k,
# which never fills). The fire-at-the-limit shape this replaces modelled at ~34
# and ~70 for the same runs -- modelled, not measured, since that code is gone.
# test_a_long_chat_does_not_compact_on_every_turn pins the property.
#
# This is the shape Anthropic's own compaction uses (a token threshold, then the
# conversation replaced by a summary). The difference here is that the newest
# exchanges stay verbatim: a chat has to answer the question just asked, not a
# paraphrase of it.
COMPACT_TRIGGER_FRACTION = 0.8
# Messages (not exchanges) kept verbatim when compaction clears the history.
COMPACT_KEEP_RECENT = 4

# Fraction of ``chat_n_ctx_target`` a conversation may spend on its history;
# the rest is for the system prompt, RAG context, question, and reasoning.
HISTORY_TOKEN_BUDGET_FRACTION = 0.5


def history_budget(ctx_target: int) -> int:
    """Token budget for everything a conversation carries into the prompt."""
    return int(ctx_target * HISTORY_TOKEN_BUDGET_FRACTION)


@dataclass(frozen=True)
class CompactionResult:
    """Notes produced by one compaction, and what they do and do not cover."""

    summary: str
    condensed: int
    """Messages folded into the notes (not exchanges: a user+assistant pair is two)."""
    stranded: int
    """Messages dropped with no notes. Non-zero means the conversation lost detail
    outright, which the UI must say plainly rather than let the model appear to
    have forgotten for no reason."""


@dataclass(frozen=True)
class CompactionPlan:
    """What one compaction will attempt, and what it gives up on before it starts."""

    batches: list[list[ChatMessage]]
    stranded: int
    """Messages dropped with no notes because the backlog exceeded MAX_COMPACT_CALLS.

    Deliberately no ``condensed`` counterpart: a plan cannot know what will be
    condensed, only what it will try. Whether a batch lands depends on the model
    answering, so only the caller that watched it can count. A planned-not-actual
    count is what made the UI claim turns were summarized when they were lost.
    """


# The summary rides in as a user/assistant pair rather than a second system
# message: the prompt is assembled as [system] + history + [user], and most chat
# templates accept only the leading system message, silently dropping or
# rejecting a later one. A pair keeps user/assistant alternation intact for
# every template, and windowed_history drops in pairs for the same reason.
SUMMARY_REQUEST = "Before we go on, remind me what we have covered so far."


def summary_messages(summary: str) -> list[ChatMessage]:
    """The synthetic pair that carries *summary* into the prompt, or nothing."""
    if not summary.strip():
        return []
    return [
        {"role": "user", "content": SUMMARY_REQUEST},
        {"role": "assistant", "content": summary},
    ]


def prompt_history(
    history: list[ChatMessage], summary: str, *, max_tokens: int
) -> list[ChatMessage]:
    """Assemble the history a prompt should carry: the summary, then recent turns.

    The summary is charged against the same budget and reserved first, so adding
    it can never push the prompt over the limit it exists to respect.

    When even that does not fit, the summary is dropped rather than stacked on
    top. windowed_history deliberately keeps the newest pair whatever it costs
    (an empty prompt is useless), so adding notes to an already-oversized window
    would push the prompt further past the budget than carrying no summary at
    all -- and overflow is an engine failure, not a worse answer. Faced with a
    turn too big to share, the live question beats notes about old ones.
    """
    pair = summary_messages(summary)
    reserved = sum(estimate_tokens(m) for m in pair)
    recent = windowed_history(history, max_tokens=max(1, max_tokens - reserved))
    if reserved + sum(estimate_tokens(m) for m in recent) > max_tokens:
        return windowed_history(history, max_tokens=max_tokens)
    return pair + recent


def overflow(history: list[ChatMessage], *, max_tokens: int) -> list[ChatMessage]:
    """The oldest turns that do not fit *max_tokens*, i.e. what compaction folds away."""
    kept = windowed_history(history, max_tokens=max_tokens)
    dropped = len(history) - len(kept)
    return history[:dropped] if dropped > 0 else []


def compaction_due(history: list[ChatMessage], summary: str, *, max_tokens: int) -> bool:
    """Whether the conversation has filled enough of its budget to compact.

    Deliberately below the limit: waiting until the prompt no longer fits means
    compacting on every subsequent turn, because folding just the overflow leaves
    it full again. See COMPACT_TRIGGER_FRACTION.
    """
    used = sum(estimate_tokens(m) for m in history)
    used += sum(estimate_tokens(m) for m in summary_messages(summary))
    return used > max_tokens * COMPACT_TRIGGER_FRACTION


def foldable(history: list[ChatMessage]) -> list[ChatMessage]:
    """Everything compaction folds into notes: all but the newest exchanges.

    Clearing this much is what buys headroom. The tail stays verbatim because a
    chat has to answer the question just asked; a summary of it is not the same
    thing, which is where a plain "summarize everything" would go wrong.

    No ``keep`` parameter: COMPACT_KEEP_RECENT is the policy, and a knob nobody
    turns is just a second place for it to disagree with itself.
    """
    keep = COMPACT_KEEP_RECENT
    return history[:-keep] if len(history) > keep else []


def summary_cap(ctx_target: int) -> int:
    """How many tokens a summary may spend, scaled to the model it is written for."""
    return max(_SUMMARY_MIN_TOKENS, min(COMPACT_MAX_TOKENS, ctx_target // _SUMMARY_CTX_FRACTION))


def batch_overflow(
    dropped: list[ChatMessage], previous_summary: str, *, ctx_target: int
) -> list[list[ChatMessage]]:
    """Split *dropped* into batches whose summarize prompt each fit *ctx_target*.

    Compaction usually nibbles a pair at a time, but switching models does not:
    dropping from a 32k model to a 2k one turns tens of thousands of tokens into
    overflow at once. Summarizing that in a single call would overflow the very
    model being compacted for, and the failure path would keep the old summary,
    losing every one of those turns. Folding batch by batch keeps each call
    inside the current window, so a switch costs several small calls instead of
    one impossible one.

    A lone turn larger than a whole batch is truncated rather than sent to
    certain failure: half its text summarized beats the entire turn dropped.
    """
    # Proportional, not a fixed pad: the estimate error and the server-side
    # chat-template cost both grow with content (a raw-room batch measured
    # ~2666 real tokens against a 2048 window and stranded every turn).
    room = max(
        _MIN_BATCH_TOKENS,
        int(
            (
                ctx_target
                - summary_cap(ctx_target)
                - estimate_text_tokens(previous_summary)
                - _PROMPT_OVERHEAD_TOKENS
            )
            * _ESTIMATE_SAFETY_FRACTION
        ),
    )
    batches: list[list[ChatMessage]] = []
    current: list[ChatMessage] = []
    current_tokens = 0
    for message in dropped:
        cost = estimate_tokens(message)
        if cost > room:
            if current:
                batches.append(current)
                current, current_tokens = [], 0
            batches.append([_truncated(message, room)])
            continue
        if current and current_tokens + cost > room:
            batches.append(current)
            current, current_tokens = [], 0
        current.append(message)
        current_tokens += cost
    if current:
        batches.append(current)
    return batches


def plan_compaction(
    dropped: list[ChatMessage], previous_summary: str, *, ctx_target: int
) -> CompactionPlan:
    """Decide what one compaction condenses, keeping its cost bounded.

    Beyond MAX_COMPACT_CALLS batches the oldest turns are stranded rather than
    folded: a 2k-context model cannot carry a 100k conversation whatever we
    spend, so buying a marginally better note with minutes of stall is a bad
    trade. The most recent slice is the part still worth remembering, and the
    caller reports the stranded count instead of pretending it survived.
    """
    batches = batch_overflow(dropped, previous_summary, ctx_target=ctx_target)
    if len(batches) <= MAX_COMPACT_CALLS:
        return CompactionPlan(batches=batches, stranded=0)
    return CompactionPlan(
        batches=batches[-MAX_COMPACT_CALLS:],
        stranded=sum(len(b) for b in batches[:-MAX_COMPACT_CALLS]),
    )


def merge_notes(previous_summary: str, notes: list[str]) -> str:
    """Join carried-forward notes with fresh per-batch notes, oldest first."""
    parts = [part.strip() for part in [previous_summary, *notes] if part.strip()]
    return "\n".join(parts)


def _truncated(message: ChatMessage, max_tokens: int) -> ChatMessage:
    """A copy of *message* clipped to roughly *max_tokens*, marked as clipped."""
    keep = chars_for_tokens(max_tokens)
    return {"role": message["role"], "content": message["content"][:keep] + " […clipped]"}
