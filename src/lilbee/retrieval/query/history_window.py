"""Token-budget history windowing for chat conversations."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lilbee.retrieval.query.searcher import ChatMessage

# Conservative char->token estimator. Matches OpenAI's "4 chars ~= 1 token"
# rule of thumb for English; under-counts non-ASCII slightly but the
# budget already leaves headroom for that.
_CHARS_PER_TOKEN = 4


def estimate_text_tokens(text: str) -> int:
    """Cheap char/4 token estimate for a string."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


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
    # final pair is over-sized (typically: send it anyway and let llama-cpp
    # error if it must, rather than send nothing at all).
    while start < len(messages) - 2 and total > max_tokens:
        # Drop the front pair (user + assistant). If the front isn't a user
        # message (malformed input), drop one to realign.
        drop = 2 if messages[start]["role"] == "user" else 1
        for i in range(start, min(start + drop, len(messages))):
            total -= sizes[i]
        start += drop
    return list(messages[start:])
