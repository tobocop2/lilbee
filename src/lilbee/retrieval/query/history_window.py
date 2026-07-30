"""Token-budget history windowing for chat conversations."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from lilbee.data.extract.chunk import CHARS_PER_TOKEN

if TYPE_CHECKING:
    from lilbee.retrieval.query.searcher import ChatMessage


def estimate_text_tokens(text: str) -> int:
    """Cheap char/4 token estimate for a string."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def chars_for_tokens(tokens: int) -> int:
    """Rough char budget for a token budget: the inverse of estimate_text_tokens.

    The chars-per-token ratio itself is owned by :mod:`lilbee.data.extract.chunk`, the
    one place it is defined; this is the token->char direction of it.
    """
    return tokens * CHARS_PER_TOKEN


def estimate_tokens(message: ChatMessage) -> int:
    """Cheap char/4 token estimate for one message."""
    return estimate_text_tokens(message["content"])


def windowed_history(
    messages: list[ChatMessage],
    *,
    max_tokens: int,
    estimator: Callable[[ChatMessage], int] = estimate_tokens,
) -> list[ChatMessage]:
    """Return the suffix of *messages* whose token cost fits in *max_tokens*.

    Drops messages from the front in pairs so the window starts at a user
    message; never strands an orphan assistant reply with no preceding user
    turn for the model to anchor to. The newest pair is always kept even
    if it exceeds the budget on its own (caller decides what to do then).

    A non-positive *max_tokens* disables windowing and returns everything,
    rather than windowing hardest. No production caller can reach it today
    (the context target has a floor), but a caller deriving a budget that
    goes non-positive gets the full history, not an empty one.
    """
    if max_tokens <= 0 or not messages:
        return list(messages)
    sizes = [estimator(m) for m in messages]
    total = sum(sizes)
    if total <= max_tokens:
        return list(messages)
    start = 0
    # ``len(messages) - 2`` keeps the newest user/assistant pair even when it
    # exceeds the budget on its own. The caller decides what to do if the
    # final pair is over-sized (typically: send it anyway and let the chat
    # server error if it must, rather than send nothing at all).
    while start < len(messages) - 2 and total > max_tokens:
        # Drop the front pair (user + assistant). If the front isn't a user
        # message (malformed input), drop one to realign.
        drop = 2 if messages[start]["role"] == "user" else 1
        for i in range(start, min(start + drop, len(messages))):
            total -= sizes[i]
        start += drop
    return list(messages[start:])
