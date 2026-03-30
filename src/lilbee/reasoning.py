"""Reasoning token filter — detects <think>...</think> tags in streaming output.

Reasoning models (Qwen3, DeepSeek-R1) wrap their thinking process in
``<think>...</think>`` tags. This module provides a stateful filter that
classifies tokens as reasoning or response content.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

_OPEN_TAG = "<think>"
_CLOSE_TAG = "</think>"


@dataclass
class StreamToken:
    """A classified token from the stream."""

    content: str
    is_reasoning: bool


class _TagParser:
    """Stateful parser that tracks whether we're inside a thinking block."""

    def __init__(self, *, show: bool) -> None:
        self.show = show
        self.buf = ""
        self.in_thinking = False

    def feed(self, token: str) -> list[StreamToken]:
        """Feed a token and return any complete StreamTokens."""
        self.buf += token
        result: list[StreamToken] = []
        while self.buf:
            emitted = self._process_thinking() if self.in_thinking else self._process_normal()
            if emitted is None:
                break  # waiting for more data (partial tag)
            if emitted.content:
                result.append(emitted)
        return result

    def flush(self) -> StreamToken | None:
        """Flush remaining buffer at end of stream."""
        if not self.buf:
            return None
        if self.in_thinking:
            return StreamToken(content=self.buf, is_reasoning=True) if self.show else None
        return StreamToken(content=self.buf, is_reasoning=False)

    def _process_thinking(self) -> StreamToken | None:
        """Process buffer while inside a <think> block. Returns None if waiting."""
        close_idx = self.buf.find(_CLOSE_TAG)
        if close_idx == -1:
            if _could_be_partial(_CLOSE_TAG, self.buf):
                return None  # wait for more
            content = self.buf
            self.buf = ""
            return (
                StreamToken(content=content, is_reasoning=True)
                if self.show
                else StreamToken(content="", is_reasoning=True)
            )

        thinking_content = self.buf[:close_idx]
        self.buf = self.buf[close_idx + len(_CLOSE_TAG) :]
        self.in_thinking = False
        if thinking_content and self.show:
            return StreamToken(content=thinking_content, is_reasoning=True)
        return StreamToken(content="", is_reasoning=True)

    def _process_normal(self) -> StreamToken | None:
        """Process buffer while outside thinking blocks. Returns None if waiting."""
        open_idx = self.buf.find(_OPEN_TAG)
        if open_idx == -1:
            if _could_be_partial(_OPEN_TAG, self.buf):
                return None  # wait for more
            content = self.buf
            self.buf = ""
            return StreamToken(content=content, is_reasoning=False)

        before = self.buf[:open_idx]
        self.buf = self.buf[open_idx + len(_OPEN_TAG) :]
        self.in_thinking = True
        return StreamToken(content=before, is_reasoning=False)


_MAX_REASONING_CHARS = 16_000  # ~4K tokens — safety limit for runaway reasoning


def filter_reasoning(tokens: Iterator[str], *, show: bool) -> Iterator[StreamToken]:
    """Filter ``<think>...</think>`` tags from a token stream.

    When *show* is True, yields thinking content as ``StreamToken(is_reasoning=True)``.
    When *show* is False, strips thinking content entirely.
    Tokens outside thinking blocks are always yielded as ``is_reasoning=False``.

    Reasoning is capped at ``_MAX_REASONING_CHARS`` to prevent runaway
    thinking loops (common with Qwen3 and similar models).
    """
    parser = _TagParser(show=show)
    reasoning_chars = 0
    for token in tokens:
        for st in parser.feed(token):
            if st.content:
                if st.is_reasoning:
                    reasoning_chars += len(st.content)
                    if reasoning_chars > _MAX_REASONING_CHARS:
                        parser.in_thinking = False
                        parser.buf = ""
                        yield StreamToken(content="\n[reasoning truncated]", is_reasoning=True)
                        break
                yield st
        else:
            continue
        break  # reasoning limit hit — exit outer loop too
    # Drain remaining non-reasoning tokens after truncation
    for token in tokens:
        for st in parser.feed(token):
            if st.content and not st.is_reasoning:
                yield st
    final = parser.flush()
    if final and final.content:
        yield final


def _could_be_partial(tag: str, buf: str) -> bool:
    """Check if the end of buf could be the start of the given tag."""
    return any(buf.endswith(tag[:length]) for length in range(1, len(tag)))
