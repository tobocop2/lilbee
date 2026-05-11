"""Reasoning token filter and cap-aware chat orchestrator.

Reasoning models (Qwen3, DeepSeek-R1) wrap their thinking process in
``<think>...</think>`` tags. This module provides:

- ``filter_reasoning``: a stateful streaming filter that classifies
  tokens as reasoning vs response and signals when reasoning exceeds a
  caller-supplied cap.
- ``stream_chat_with_cap``: the high-level orchestrator. Wraps a
  provider call with the filter; when the cap fires, re-issues the
  chat with a "stop thinking, answer directly" nudge. All chat surfaces
  (HTTP/SSE, CLI, TUI) consume this so cap behavior is uniform.
- ``effective_reasoning_cap``: resolves the cap from the global config
  with per-model ``ModelDefaults`` overrides.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from lilbee.core.config import cfg
from lilbee.providers.base import ClosableIterator

if TYPE_CHECKING:
    from lilbee.providers.base import LLMProvider

_OPEN_TAG = "<think>"
_CLOSE_TAG = "</think>"
_THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>\s*|<think>[\s\S]*$")
_PROGRESS_TICK_CHARS = 256
"""Coarseness of the progress callback: fire when reasoning grows by at least this many chars."""

CAP_CONTINUATION_PROMPT = (
    "Stop thinking now. Give your final answer directly, without any further <think> blocks."
)
"""The user-message nudge appended on the continuation call after the cap fires."""

CAP_NOTICE_TEMPLATE = "\n[reasoning capped at {chars} chars, asking for a direct answer]\n"
"""User-visible marker emitted between the truncated reasoning and the continuation answer."""


@dataclass
class StreamToken:
    """A classified token from the stream."""

    content: str
    is_reasoning: bool


@dataclass
class CapNotice:
    """Emitted once when the reasoning cap fires, before the continuation stream."""

    cap_chars: int


@dataclass
class _TagParser:
    """Stateful parser that tracks whether we're inside a thinking block."""

    show: bool
    buf: str = ""
    in_thinking: bool = False
    reasoning_chars: int = 0

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
            self.reasoning_chars += len(self.buf)
            return StreamToken(content=self.buf, is_reasoning=True) if self.show else None
        return StreamToken(content=self.buf, is_reasoning=False)

    def _process_thinking(self) -> StreamToken | None:
        close_idx = self.buf.find(_CLOSE_TAG)
        if close_idx == -1:
            if _could_be_partial(_CLOSE_TAG, self.buf):
                return None
            content = self.buf
            self.reasoning_chars += len(content)
            self.buf = ""
            return (
                StreamToken(content=content, is_reasoning=True)
                if self.show
                else StreamToken(content="", is_reasoning=True)
            )
        thinking_content = self.buf[:close_idx]
        self.reasoning_chars += len(thinking_content)
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
    on_cap: Callable[[], None] | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> Iterator[StreamToken]:
    """Classify ``<think>...</think>`` tokens and stop when reasoning exceeds the cap.

    *cap_chars* bounds reasoning content. When exceeded, ``on_cap`` is
    fired (no payload), the upstream iterator is closed, and iteration
    stops. The caller decides what to do next via the higher-level
    ``stream_chat_with_cap`` orchestrator. *on_progress* is fired with
    the running reasoning-chars count each time it grows by at least 256
    characters. A non-positive *cap_chars* disables the cap.
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
                    on_cap()
                return
        final = parser.flush()
        if final and final.content:
            yield final
        if on_progress is not None and parser.reasoning_chars > last_progress_tick:
            on_progress(parser.reasoning_chars)
    finally:
        _close_iterator(tokens)


def effective_reasoning_cap() -> int:
    """Return the active reasoning cap; 0 means unlimited.

    A per-model ``ModelDefaults.max_reasoning_chars`` value (including
    ``0`` for "this model is allowed to think forever") beats the global
    ``cfg.max_reasoning_chars`` setting. Only ``None`` falls through to
    the global, so a per-model 0 means the user explicitly opted that
    model out of the cap.
    """
    defaults = cfg.model_defaults
    override = defaults.max_reasoning_chars if defaults is not None else None
    return override if isinstance(override, int) and override >= 0 else cfg.max_reasoning_chars


def stream_chat_with_cap(
    provider: LLMProvider,
    messages: list[dict[str, Any]],
    *,
    options: dict[str, Any] | None,
    model: str,
    show_reasoning: bool,
    cap_chars: int,
) -> Generator[StreamToken | CapNotice, None, None]:
    """Stream chat tokens; on cap-fire, re-issue with a stop-thinking nudge.

    Yields ``StreamToken`` events for both reasoning and response tokens
    in the first pass. If reasoning exceeds *cap_chars*, the upstream
    iterator is closed, a single ``CapNotice`` is yielded, and the
    continuation stream starts (same messages plus a user message asking
    the model to answer directly). Continuation tokens stream as
    ``StreamToken(is_reasoning=False)``.
    """
    cap_fired = False

    def _on_cap() -> None:
        nonlocal cap_fired
        cap_fired = True

    first_stream = provider.chat(messages, stream=True, options=options or None, model=model)
    yield from filter_reasoning(
        first_stream,
        show=show_reasoning,
        cap_chars=cap_chars,
        on_cap=_on_cap,
    )
    if not cap_fired:
        return
    yield CapNotice(cap_chars=cap_chars)
    nudged = [*messages, {"role": "user", "content": CAP_CONTINUATION_PROMPT}]
    second_stream = provider.chat(nudged, stream=True, options=options or None, model=model)
    try:
        for chunk in second_stream:
            if chunk:
                yield StreamToken(content=chunk, is_reasoning=False)
    finally:
        _close_iterator(second_stream)


def cap_events_as_stream_tokens(
    events: Iterator[StreamToken | CapNotice],
) -> Iterator[StreamToken]:
    """Render ``CapNotice`` events as user-visible reasoning ``StreamToken``s.

    Library and CLI surfaces speak ``StreamToken`` only. This helper lets
    them consume the orchestrator's union output without a per-call
    isinstance dance for the cap notice.
    """
    for event in events:
        if isinstance(event, CapNotice):
            yield StreamToken(
                content=CAP_NOTICE_TEMPLATE.format(chars=event.cap_chars),
                is_reasoning=True,
            )
        elif event.content:
            yield event


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
