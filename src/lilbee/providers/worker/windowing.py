"""Trim a chat message list to fit the loaded model's context window."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Roles used by /v1/chat/completions wire format. Constants prevent typos
# from silently breaking pair-matching.
_SYSTEM = "system"
_USER = "user"
_ASSISTANT = "assistant"
_TOOL = "tool"

# Conservative per-message overhead the chat template adds around every
# message (role markers, separators). Slightly over-estimating is fine: it
# only widens the safety margin against tokenizer drift between count-time
# and inference-time.
_PER_MESSAGE_OVERHEAD = 4

Tokenizer = Callable[[bytes], list[int]]


@dataclass(frozen=True)
class WindowingOutcome:
    """Result of running :func:`window_messages_to_budget`.

    ``messages`` is the trimmed list ready to forward to inference;
    ``dropped`` is how many messages were removed. When the prompt cannot
    be reduced below ``budget`` (system + last user message alone exceeds
    it), ``messages`` is ``None`` and ``requested`` carries the smallest
    achievable token count for the un-droppable subset.
    """

    messages: list[dict[str, Any]] | None
    dropped: int
    requested: int
    available: int

    @classmethod
    def fit(
        cls, messages: list[dict[str, Any]], *, dropped: int, available: int
    ) -> WindowingOutcome:
        """Outcome for a successfully trimmed (or already-fitting) message list."""
        return cls(messages=messages, dropped=dropped, requested=0, available=available)

    @classmethod
    def overflow(cls, *, requested: int, available: int) -> WindowingOutcome:
        """Outcome when no further messages can be dropped and the budget is still exceeded."""
        return cls(messages=None, dropped=0, requested=requested, available=available)


def count_message_tokens(message: dict[str, Any], tokenize: Tokenizer) -> int:
    """Estimate the token cost of one OpenAI-wire-format chat message.

    Sums the tokenised content, any tool-call JSON the assistant emitted,
    and a fixed per-message overhead for the chat template's role markers
    and separators. Counts on the bytes-encoded payload because llama-cpp's
    ``tokenize`` only accepts bytes.
    """
    total = _PER_MESSAGE_OVERHEAD
    content = message.get("content")
    if isinstance(content, str) and content:
        total += len(tokenize(content.encode("utf-8")))
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if isinstance(function, dict):
                name = function.get("name")
                if isinstance(name, str):
                    total += len(tokenize(name.encode("utf-8")))
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    total += len(tokenize(arguments.encode("utf-8")))
                elif arguments is not None:
                    total += len(tokenize(json.dumps(arguments).encode("utf-8")))
    return total


def window_messages_to_budget(
    messages: list[dict[str, Any]],
    *,
    budget: int,
    tokenize: Tokenizer,
) -> WindowingOutcome:
    """Drop oldest tool-call/tool-result pairs and old turns to fit *budget*.

    Drop order:
      1. Oldest assistant tool-call message + every ``tool`` message
         keyed to one of its ``tool_calls[*].id`` values.
      2. Oldest non-system user/assistant exchange that is not part of
         the in-flight turn (the trailing user message and any trailing
         assistant message after it).
    Returns :class:`WindowingOutcome.overflow` if the un-droppable subset
    (system + trailing user message) alone exceeds ``budget``.
    """
    counts = [count_message_tokens(m, tokenize) for m in messages]
    total = sum(counts)
    if total <= budget:
        return WindowingOutcome.fit(messages, dropped=0, available=budget)

    keep: list[bool] = [True] * len(messages)
    in_flight_user_idx = _last_role_index(messages, _USER)
    droppable_idx = _droppable_indices(messages, in_flight_user_idx=in_flight_user_idx)

    for idx in droppable_idx:
        if total <= budget:
            break
        if not keep[idx]:
            continue
        keep[idx] = False
        total -= counts[idx]

    if total > budget:
        return WindowingOutcome.overflow(requested=total, available=budget)

    trimmed = [m for i, m in enumerate(messages) if keep[i]]
    dropped = len(messages) - len(trimmed)
    return WindowingOutcome.fit(trimmed, dropped=dropped, available=budget)


def _last_role_index(messages: list[dict[str, Any]], role: str) -> int:
    """Return the index of the last message whose role matches; -1 if absent."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == role:
            return i
    return -1


def _droppable_indices(
    messages: list[dict[str, Any]],
    *,
    in_flight_user_idx: int,
) -> list[int]:
    """Return drop priorities ordered oldest-first.

    Tool-call/result pairs first (cheaper to lose: the tool already ran;
    its result is what dominates token counts in agent loops), then non-
    system user/assistant exchanges that are not the in-flight turn.
    """
    tool_pair_indices: list[int] = []
    pair_call_ids: set[str] = set()

    for i, msg in enumerate(messages):
        if i >= in_flight_user_idx >= 0:
            break
        role = msg.get("role")
        if role == _ASSISTANT:
            tool_call_ids = _collect_tool_call_ids(msg)
            if tool_call_ids:
                tool_pair_indices.append(i)
                pair_call_ids.update(tool_call_ids)
        elif role == _TOOL and msg.get("tool_call_id") in pair_call_ids:
            tool_pair_indices.append(i)

    exchange_indices: list[int] = [
        i
        for i, msg in enumerate(messages)
        if 0 <= i < in_flight_user_idx
        and msg.get("role") in (_USER, _ASSISTANT)
        and i not in tool_pair_indices
    ]

    return tool_pair_indices + exchange_indices


def _collect_tool_call_ids(message: dict[str, Any]) -> set[str]:
    """Return the set of tool_call ids attached to an assistant message."""
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return set()
    out: set[str] = set()
    for call in tool_calls:
        if isinstance(call, dict):
            call_id = call.get("id")
            if isinstance(call_id, str) and call_id:
                out.add(call_id)
    return out
