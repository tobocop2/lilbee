"""Fit a chat message list to a served context window by dropping oldest turns."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from lilbee.providers.base import CHAT_MESSAGE_OVERHEAD_TOKENS

_SYSTEM_ROLE = "system"
_TOOL_ROLE = "tool"
# Conservative chars-per-token: below the ~4 English average so the estimate
# over-counts tokens and the window errs toward dropping more, never overflowing.
_CHARS_PER_TOKEN = 3
# Per-message token overhead for the role markers and chat-template wrappers the
# server adds around each message. Owned by base; retrieval budgets the same number.


def estimate_tokens(text: str) -> int:
    """Conservative token estimate for a text fragment."""
    return math.ceil(len(text) / _CHARS_PER_TOKEN)


def _message_tokens(message: dict[str, Any]) -> int:
    """Estimated tokens a wire message contributes (content + tool-call JSON + overhead)."""
    total = CHAT_MESSAGE_OVERHEAD_TOKENS
    content = message.get("content")
    if isinstance(content, str):
        total += estimate_tokens(content)
    elif content:
        total += estimate_tokens(json.dumps(content))
    tool_calls = message.get("tool_calls")
    if tool_calls:
        total += estimate_tokens(json.dumps(tool_calls))
    return total


def _tools_tokens(tools: list[dict[str, Any]] | None) -> int:
    """Estimated tokens the tool schemas contribute to the rendered prompt."""
    if not tools:
        return 0
    return estimate_tokens(json.dumps(tools))


def _without_orphan_tools(kept: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """*kept* without a leading ``tool`` result whose originating call was dropped."""
    while kept and kept[0].get("role") == _TOOL_ROLE:
        kept = kept[1:]
    return kept


def _shrink_to_measured(
    system: list[dict[str, Any]],
    kept: list[dict[str, Any]],
    budget: int,
    measure: Callable[[list[dict[str, Any]]], int],
) -> tuple[list[dict[str, Any]], int]:
    """Drop the oldest kept turns until the measured prompt fits; return them and the count.

    Each round drops enough turns, by estimate, to cover the measured overshoot
    (at least one), then measures again. The last turn is never dropped.
    """
    while True:
        measured = measure(system + kept)
        if measured <= budget or len(kept) == 1:
            return kept, measured
        overshoot = measured - budget
        dropped = 0
        cut = 0
        while cut < len(kept) - 1 and dropped < overshoot:
            dropped += _message_tokens(kept[cut])
            cut += 1
        kept = _without_orphan_tools(kept[cut:])
        if not kept:
            return kept, measured


@dataclass(frozen=True)
class WindowResult:
    """Outcome of fitting messages to a budget."""

    messages: list[dict[str, Any]]  # system + kept suffix (best-effort even on overflow)
    fits: bool
    prompt_tokens: int  # tokens of ``messages`` plus the tools passed in (measured when asked)
    dropped: int  # number of conversation messages dropped


def window_messages(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    budget: int,
    measure: Callable[[list[dict[str, Any]]], int] | None = None,
) -> WindowResult:
    """Drop oldest conversation turns until the prompt fits ``budget``.

    System messages and the most recent turn are always kept; tool-call/result
    pairs drop together (a kept suffix never starts with an orphan ``tool``
    message whose originating call was dropped). ``fits`` is False when even the
    system messages, tools, and the final message exceed the budget; the caller
    turns that into a context-overflow error.

    The estimate reads the wire JSON, so it misses the chat template's own
    tool preamble. When *measure* is given it returns the engine's own token
    count for a message list (the caller renders the tools). The kept suffix
    shrinks until that count fits ``budget``. The preamble is a fixed size,
    not a share of the estimate, so no estimate is trusted: *measure* runs
    at least once whenever a conversation remains.
    """
    system = [m for m in messages if m.get("role") == _SYSTEM_ROLE]
    convo = [m for m in messages if m.get("role") != _SYSTEM_ROLE]
    fixed = sum(_message_tokens(m) for m in system) + _tools_tokens(tools)

    if not convo:
        return WindowResult(list(system), fixed <= budget, fixed, 0)

    # Keep conversation messages newest-first while they fit; the most recent is
    # always kept (an empty guard) so the current turn survives.
    kept_rev: list[dict[str, Any]] = []
    used = fixed
    for msg in reversed(convo):
        cost = _message_tokens(msg)
        if kept_rev and used + cost > budget:
            break
        kept_rev.append(msg)
        used += cost
    kept = list(reversed(kept_rev))

    kept = _without_orphan_tools(kept)

    prompt_tokens = fixed + sum(_message_tokens(m) for m in kept)
    if measure is not None and kept:
        kept, prompt_tokens = _shrink_to_measured(system, kept, budget, measure)
        if not kept:
            prompt_tokens = fixed
    fits = prompt_tokens <= budget and bool(kept)
    return WindowResult(system + kept, fits, prompt_tokens, len(convo) - len(kept))
