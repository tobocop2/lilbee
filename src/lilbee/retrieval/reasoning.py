"""Reasoning token filter: detects <think>...</think> tags in streaming output.

Reasoning models (Qwen3, DeepSeek-R1) wrap their thinking process in
``<think>...</think>`` tags. This module provides a stateful filter that
classifies tokens as reasoning or response content and notifies the
caller when reasoning exceeds a caller-supplied cap.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from lilbee.providers.base import ClosableIterator

_OPEN_TAG = "<think>"
_CLOSE_TAG = "</think>"
_THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>\s*|<think>[\s\S]*$")
_PROGRESS_TICK_CHARS = 256
"""Coarseness of the progress callback: fire when reasoning grows by at least this many chars."""


@dataclass
class StreamToken:
    """A classified token from the stream."""

    content: str
    is_reasoning: bool


@dataclass
class _TagParser:
    """Stateful parser that tracks whether we're inside a thinking block."""

    show: bool
    buf: str = ""
    in_thinking: bool = False
    reasoning_chars: int = 0
    reasoning_text: list[str] = field(default_factory=list)

    def captured_reasoning(self) -> str:
        """Concatenated reasoning text seen so far, regardless of ``show``."""
        return "".join(self.reasoning_text)

    def feed(self, token: str) -> list[StreamToken]:
        """Feed a token and return any complete StreamTokens."""
        self.buf += token
        result: list[StreamToken] = []
        while self.buf:
            emitted = self._process_thinking() if self.in_thinking else self._process_normal()
            if emitted is None:
                break
            if emitted.content:
                result.append(emitted)
        return result

    def flush(self) -> StreamToken | None:
        """Flush remaining buffer at end of stream."""
        if not self.buf:
            return None
        if self.in_thinking:
            self._absorb_reasoning(self.buf)
            return StreamToken(content=self.buf, is_reasoning=True) if self.show else None
        return StreamToken(content=self.buf, is_reasoning=False)

    def _absorb_reasoning(self, content: str) -> None:
        self.reasoning_chars += len(content)
        self.reasoning_text.append(content)

    def _process_thinking(self) -> StreamToken | None:
        close_idx = self.buf.find(_CLOSE_TAG)
        if close_idx == -1:
            if _could_be_partial(_CLOSE_TAG, self.buf):
                return None
            content = self.buf
            self._absorb_reasoning(content)
            self.buf = ""
            return (
                StreamToken(content=content, is_reasoning=True)
                if self.show
                else StreamToken(content="", is_reasoning=True)
            )
        thinking_content = self.buf[:close_idx]
        self._absorb_reasoning(thinking_content)
        self.buf = self.buf[close_idx + len(_CLOSE_TAG) :]
        self.in_thinking = False
        if thinking_content and self.show:
            return StreamToken(content=thinking_content, is_reasoning=True)
        return StreamToken(content="", is_reasoning=True)

    def _process_normal(self) -> StreamToken | None:
        open_idx = self.buf.find(_OPEN_TAG)
        if open_idx == -1:
            if _could_be_partial(_OPEN_TAG, self.buf):
                return None
            content = self.buf
            self.buf = ""
            return StreamToken(content=content, is_reasoning=False)
        before = self.buf[:open_idx]
        self.buf = self.buf[open_idx + len(_OPEN_TAG) :]
        self.in_thinking = True
        return StreamToken(content=before, is_reasoning=False)


def filter_reasoning(
    tokens: Iterator[str],
    *,
    show: bool,
    cap_chars: int,
    on_cap: Callable[[str], None] | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> Iterator[StreamToken]:
    """Filter ``<think>...</think>`` tags and signal when reasoning exceeds the cap.

    *cap_chars* bounds reasoning content. When exceeded, ``on_cap`` is
    called with the captured reasoning text, the upstream iterator is
    closed, and iteration stops; the caller decides what to do next
    (stop the response, re-issue the chat with a continuation prompt,
    etc.). *on_progress* is fired with the running reasoning-chars count
    each time it grows by at least 256 characters.

    A non-positive *cap_chars* disables the cap.
    """
    parser = _TagParser(show=show)
    last_progress_tick = 0
    try:
        for token in tokens:
            for st in parser.feed(token):
                if st.content:
                    yield st
            if (
                on_progress is not None
                and parser.reasoning_chars >= last_progress_tick + _PROGRESS_TICK_CHARS
            ):
                last_progress_tick = parser.reasoning_chars
                on_progress(parser.reasoning_chars)
            if cap_chars > 0 and parser.reasoning_chars > cap_chars:
                if on_cap is not None:
                    on_cap(parser.captured_reasoning())
                return
        final = parser.flush()
        if final and final.content:
            yield final
        if on_progress is not None and parser.reasoning_chars > last_progress_tick:
            on_progress(parser.reasoning_chars)
    finally:
        _close_iterator(tokens)


def _close_iterator(tokens: Iterator[str]) -> None:
    """Close *tokens* if it satisfies the ClosableIterator protocol."""
    if isinstance(tokens, ClosableIterator):
        with contextlib.suppress(Exception):
            tokens.close()


def strip_reasoning(text: str) -> str:
    """Remove ``<think>...</think>`` blocks from a complete (non-streaming) string."""
    return _THINK_BLOCK_RE.sub("", text)


def _could_be_partial(tag: str, buf: str) -> bool:
    """Check if the end of buf could be the start of the given tag."""
    return any(buf.endswith(tag[:length]) for length in range(1, len(tag)))
