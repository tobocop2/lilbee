"""Trim a chat message list to fit the loaded model's context window."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ChatRole(StrEnum):
    """Closed set of chat-message roles in the OpenAI wire format."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


# Per-message overhead from chat-template role markers + separators.
_PER_MESSAGE_OVERHEAD = 4

Tokenizer = Callable[[bytes], list[int]]


@dataclass(frozen=True)
class WindowingOutcome:
    """Result of running :func:`window_messages_to_budget`."""

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
    """Estimate the token cost of one OpenAI-wire-format chat message."""
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


_TOOLS_TEMPLATE_OVERHEAD_MULTIPLIER = 1.5
"""Fudge factor for chat-template Jinja inflation of the tools schema.

The model's chat template wraps the bare JSON tools list in family-specific
markup (``<tools>...</tools>``, ``# Tools\n\n...``, etc.) and may inline tool
descriptions / schema fields multiple times. Across the families lilbee
supports, the rendered prompt is typically 1.3-1.7x the raw JSON dump. We use
1.5 as a conservative middle and add a fixed preamble allowance so models
with small ``n_ctx`` raise a clean 400 before the worker reaches the
llama-cpp ``ValueError`` for over-budget prompts.
"""

_TOOLS_TEMPLATE_PREAMBLE_TOKENS = 256
"""Fixed allowance for the chat-template's tools preamble (introductory text
like 'You may call one or more functions to assist with the user query.')."""


def count_tools_overhead(tools: list[dict[str, Any]] | None, tokenize: Tokenizer) -> int:
    """Estimate the tokens llama-cpp will inject for the ``tools`` schema.

    The estimate accounts for chat-template inflation: the raw JSON is
    multiplied by a fixed factor and a small preamble allowance is added.
    The pre-flight budget computed off this estimate is intentionally
    conservative; a precise figure would require rendering the chat template
    which is not exposed by llama-cpp-python.
    """
    if not tools:
        return 0
    raw_tokens = len(tokenize(json.dumps(tools).encode("utf-8")))
    return int(raw_tokens * _TOOLS_TEMPLATE_OVERHEAD_MULTIPLIER) + _TOOLS_TEMPLATE_PREAMBLE_TOKENS


def window_messages_to_budget(
    messages: list[dict[str, Any]],
    *,
    budget: int,
    tokenize: Tokenizer,
) -> WindowingOutcome:
    """Drop oldest tool-call/tool-result pairs and old turns to fit *budget*."""
    counts = [count_message_tokens(m, tokenize) for m in messages]
    total = sum(counts)
    if total <= budget:
        return WindowingOutcome.fit(messages, dropped=0, available=budget)

    keep: list[bool] = [True] * len(messages)
    in_flight_user_idx = _last_role_index(messages, ChatRole.USER)
    droppable_idx = _droppable_indices(messages, in_flight_user_idx=in_flight_user_idx)

    for idx in droppable_idx:
        if total <= budget:
            break
        keep[idx] = False
        total -= counts[idx]

    if total > budget:
        return WindowingOutcome.overflow(requested=total, available=budget)

    trimmed = [m for i, m in enumerate(messages) if keep[i]]
    dropped = len(messages) - len(trimmed)
    return WindowingOutcome.fit(trimmed, dropped=dropped, available=budget)


def _last_role_index(messages: list[dict[str, Any]], role: ChatRole) -> int:
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
    """Drop oldest tool-call/result pairs first, then oldest user/assistant exchanges."""
    tool_pair_indices: list[int] = []
    pair_call_ids: set[str] = set()

    for i, msg in enumerate(messages):
        if i >= in_flight_user_idx >= 0:
            break
        role = msg.get("role")
        if role == ChatRole.ASSISTANT:
            tool_call_ids = _collect_tool_call_ids(msg)
            if tool_call_ids:
                tool_pair_indices.append(i)
                pair_call_ids.update(tool_call_ids)
        elif role == ChatRole.TOOL and msg.get("tool_call_id") in pair_call_ids:
            tool_pair_indices.append(i)

    exchange_indices: list[int] = [
        i
        for i, msg in enumerate(messages)
        if 0 <= i < in_flight_user_idx
        and msg.get("role") in (ChatRole.USER, ChatRole.ASSISTANT)
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
