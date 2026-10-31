"""Re-inlines server-extracted reasoning as ``<think>`` text, for every provider."""

from __future__ import annotations

from lilbee.providers.base import THINK_CLOSE_TAG, THINK_OPEN_TAG


class ThinkInliner:
    """Re-inlines server-extracted reasoning deltas as inline ``<think>`` text.

    The server parses each model's reasoning format natively (``--reasoning-format``)
    and streams it as ``reasoning_content``; lilbee's pipeline speaks inline
    ``<think>`` text, so the chat boundary opens the tag on the first reasoning
    delta and closes it when the answer starts (or at end of stream). Disabled
    (the non-chat roles), reasoning is dropped and content passes through, matching
    the server-extracted default those roles already ran with.
    """

    def __init__(self, *, enabled: bool) -> None:
        self._enabled = enabled
        self._in_think = False

    def feed(self, reasoning: str, content: str) -> str:
        if not self._enabled:
            return content
        parts: list[str] = []
        if reasoning:
            if not self._in_think:
                self._in_think = True
                parts.append(THINK_OPEN_TAG)
            parts.append(reasoning)
        if content:
            if self._in_think:
                self._in_think = False
                parts.append(THINK_CLOSE_TAG)
            parts.append(content)
        return "".join(parts)

    def finish(self) -> str:
        """Close an unterminated think block at end of stream."""
        if self._in_think:
            self._in_think = False
            return THINK_CLOSE_TAG
        return ""


def inline_reasoning(reasoning: str, content: str) -> str:
    """A whole message's text with its extracted *reasoning* re-inlined before *content*."""
    if reasoning:
        return f"{THINK_OPEN_TAG}{reasoning}{THINK_CLOSE_TAG}{content}"
    return content
