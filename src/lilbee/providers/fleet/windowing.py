"""Fit a chat message list to a served context window by dropping oldest turns."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

_SYSTEM_ROLE = "system"
_TOOL_ROLE = "tool"
# Conservative chars-per-token: below the ~4 English average so the estimate
# over-counts tokens and the window errs toward dropping more, never overflowing.
_CHARS_PER_TOKEN = 3
# Per-message token overhead for the role markers and chat-template wrappers the
# server adds around each message.
_PER_MESSAGE_OVERHEAD = 8


def estimate_tokens(text: str) -> int:
    """Conservative token estimate for a text fragment."""
    return math.ceil(len(text) / _CHARS_PER_TOKEN)


def _message_tokens(message: dict[str, Any]) -> int:
    """Estimated tokens a wire message contributes (content + tool-call JSON + overhead)."""
    total = _PER_MESSAGE_OVERHEAD
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


@dataclass(frozen=True)
class WindowResult:
    """Outcome of fitting messages to a budget."""

    messages: list[dict[str, Any]]  # system + kept suffix (best-effort even on overflow)
    fits: bool
    prompt_tokens: int  # estimated tokens of ``messages`` plus the tools passed in
    dropped: int  # number of conversation messages dropped


def window_messages(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    budget: int,
) -> WindowResult:
    """Drop oldest conversation turns until the estimated prompt fits ``budget``.

    System messages and the most recent turn are always kept; tool-call/result
    pairs drop together (a kept suffix never starts with an orphan ``tool``
    message whose originating call was dropped). ``fits`` is False when even the
    system messages, tools, and the final message exceed the budget; the caller
    turns that into a context-overflow error.
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

    # A kept suffix must not begin with an orphan tool result (its call dropped).
    while kept and kept[0].get("role") == _TOOL_ROLE:
        kept = kept[1:]

    prompt_tokens = fixed + sum(_message_tokens(m) for m in kept)
    fits = prompt_tokens <= budget and bool(kept)
    return WindowResult(system + kept, fits, prompt_tokens, len(convo) - len(kept))
