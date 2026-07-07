"""Reasoning token filter and cap-aware chat orchestrator.

Reasoning models (Qwen3, DeepSeek-R1) wrap their thinking process in
``<think>...</think>`` tags. This module provides:

- ``filter_reasoning``: a stateful streaming filter that classifies
  tokens as reasoning vs response and signals when reasoning exceeds a
  caller-supplied cap.
- ``stream_chat_with_cap``: the high-level orchestrator. Wraps a
  provider call with the filter; when the cap fires, re-issues the
  chat with a "stop thinking, answer directly" nudge. The ask/search
  streaming path and CLI/TUI consume it directly; the canonical
  chat-dispatch path mirrors the same filter + cap-nudge behavior over
  its own async driver.
- ``effective_reasoning_cap``: resolves the cap from the global config
  with per-model ``ModelDefaults`` overrides.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
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

# OpenAI harmony control tokens (gpt-oss). The stream leaks them as text because
# the chat server runs with --reasoning-format none (keeps <think> inline); lilbee
# classifies the channels itself, the same way it parses <think>.
_HARMONY_OPEN = "<|"
_HARMONY_CLOSE = "|>"
_HARMONY_CHANNEL = "<|channel|>"
_HARMONY_MESSAGE = "<|message|>"
_HARMONY_START = "<|start|>"
_HARMONY_END = "<|end|>"
_HARMONY_RETURN = "<|return|>"
_HARMONY_CALL = "<|call|>"
_HARMONY_BODY_ENDERS = frozenset({_HARMONY_START, _HARMONY_END, _HARMONY_RETURN, _HARMONY_CALL})
_FINAL_CHANNEL = "final"

CAP_CONTINUATION_PROMPT = (
    "Stop thinking now. Give your final answer directly, without any further <think> blocks."
)
"""The user-message nudge appended on the continuation call after the cap fires."""

CAP_NOTICE_TEMPLATE = "\n[reasoning capped at {chars} chars, asking for a direct answer]\n"
"""User-visible marker emitted between the truncated reasoning and the continuation answer."""

REASONING_EXHAUSTED_NOTICE = (
    "The model spent its whole response budget on reasoning and produced no final "
    "answer. Try a shorter question, raise the generation token limit, or lower the "
    "reasoning effort."
)
"""Returned in place of an empty answer when reasoning consumed the entire generation.

Lets a caller tell "the model thought itself to death" apart from a genuine empty
response, which an empty string alone cannot."""


class _HarmonySection(StrEnum):
    """Which part of a harmony message the parser is currently reading."""

    HEADER = "header"  # role name after <|start|>; not emitted
    CHANNEL = "channel"  # channel name after <|channel|>; captured, not emitted
    BODY = "body"  # message text after <|message|>; classified by channel


class _Route(StrEnum):
    """Which stream dialect the dispatcher committed to."""

    HARMONY = "harmony"  # gpt-oss channels
    TAG = "tag"  # plain text or <think> reasoning


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
class TagParser:
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


@dataclass
class HarmonyParser:
    """Stateful parser for gpt-oss harmony channels.

    Emits the ``final`` channel as answer text and every other channel (analysis,
    commentary) as reasoning, stripping all ``<|...|>`` control tokens. Mirrors
    ``TagParser``'s ``feed`` / ``flush`` / ``reasoning_chars`` contract so the same
    orchestrator and cap logic drive either dialect.
    """

    show: bool
    buf: str = ""
    section: _HarmonySection = _HarmonySection.HEADER
    channel: str = ""
    channel_buf: str = ""
    reasoning_chars: int = 0

    def feed(self, token: str) -> list[StreamToken]:
        """Feed a token and return any complete StreamTokens."""
        self.buf += token
        result: list[StreamToken] = []
        while self.buf:
            emitted = self._step()
            if emitted is None:
                break
            if emitted.content:
                result.append(emitted)
        return result

    def flush(self) -> StreamToken | None:
        """Bodies stream fully during feed; only a trailing partial token is left to drop."""
        self.buf = ""
        return None

    def _step(self) -> StreamToken | None:
        open_idx = self.buf.find(_HARMONY_OPEN)
        if open_idx == -1:
            return self._emit_plain()
        if open_idx > 0:
            text, self.buf = self.buf[:open_idx], self.buf[open_idx:]
            return self._classify(text)
        close_idx = self.buf.find(_HARMONY_CLOSE)
        if close_idx == -1:
            return None  # incomplete control token; wait for more
        marker, self.buf = (
            self.buf[: close_idx + len(_HARMONY_CLOSE)],
            self.buf[close_idx + len(_HARMONY_CLOSE) :],
        )
        self._handle_marker(marker)
        return StreamToken(content="", is_reasoning=False)

    def _emit_plain(self) -> StreamToken | None:
        """Consume buffered text with no control token, holding back a lone trailing '<'."""
        if self.buf == _HARMONY_OPEN[0]:
            return None  # could be the start of the next control token
        if self.buf.endswith(_HARMONY_OPEN[0]):
            text, self.buf = self.buf[:-1], _HARMONY_OPEN[0]
            return self._classify(text)
        text, self.buf = self.buf, ""
        return self._classify(text)

    def _classify(self, text: str) -> StreamToken:
        """Route body text by channel; capture channel names; drop header text."""
        if self.section is _HarmonySection.CHANNEL:
            self.channel_buf += text
            return StreamToken(content="", is_reasoning=False)
        if self.section is not _HarmonySection.BODY:
            return StreamToken(content="", is_reasoning=False)
        if self.channel == _FINAL_CHANNEL:
            return StreamToken(content=text, is_reasoning=False)
        self.reasoning_chars += len(text)
        return StreamToken(content=text if self.show else "", is_reasoning=True)

    def _handle_marker(self, marker: str) -> None:
        if marker == _HARMONY_CHANNEL:
            self.section = _HarmonySection.CHANNEL
            self.channel_buf = ""
        elif marker == _HARMONY_MESSAGE:
            if self.section is _HarmonySection.CHANNEL:
                self.channel = self.channel_buf.strip()
            self.section = _HarmonySection.BODY
        elif marker in _HARMONY_BODY_ENDERS:
            self.section = _HarmonySection.HEADER


@dataclass
class ReasoningParser:
    """Routes a chat stream to harmony or ``<think>`` handling by its opening tokens.

    A harmony stream (gpt-oss) opens with a ``<|`` control token; everything else,
    plain text and ``<think>`` reasoning, is handled by ``TagParser``. The choice is
    made from the leading bytes and never switches mid-stream.
    """

    show: bool
    reasoning_chars: int = 0
    _inner: TagParser | HarmonyParser | None = field(default=None, repr=False)
    _probe: str = ""

    def feed(self, token: str) -> list[StreamToken]:
        if self._inner is None:
            self._probe += token
            route = _route_reasoning(self._probe)
            if route is None:
                return []  # ambiguous prefix; keep buffering until the dialect is clear
            self._inner = (
                HarmonyParser(show=self.show)
                if route is _Route.HARMONY
                else TagParser(show=self.show)
            )
            token, self._probe = self._probe, ""
        out = self._inner.feed(token)
        self.reasoning_chars = self._inner.reasoning_chars
        return out

    def flush(self) -> StreamToken | None:
        if self._inner is not None:
            tail = self._inner.flush()
            self.reasoning_chars = self._inner.reasoning_chars
            return tail
        if self._probe:
            tail, self._probe = StreamToken(content=self._probe, is_reasoning=False), ""
            return tail
        return None


def _route_reasoning(probe: str) -> _Route | None:
    """Pick the stream dialect from its leading bytes; None while still ambiguous."""
    lead = probe.lstrip()
    if lead.startswith(_HARMONY_OPEN):
        return _Route.HARMONY
    if lead in ("", _HARMONY_OPEN[0]):
        return None  # could still become a "<|" control token
    return _Route.TAG


def filter_reasoning(
    tokens: Iterator[str],
    *,
    show: bool,
    cap_chars: int,
    on_cap: Callable[[], None] | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> Iterator[StreamToken]:
    """Classify reasoning tokens (``<think>`` or harmony) and stop past the cap.

    *cap_chars* bounds reasoning content. When exceeded, ``on_cap`` is
    fired (no payload), the upstream iterator is closed, and iteration
    stops. The caller decides what to do next via the higher-level
    ``stream_chat_with_cap`` orchestrator. *on_progress* is fired with
    the running reasoning-chars count each time it grows by at least 256
    characters. A non-positive *cap_chars* disables the cap.
    """
    parser = ReasoningParser(show=show)
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
        _text_only(first_stream),
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
        for chunk in _text_only(second_stream):
            if chunk:
                yield StreamToken(content=chunk, is_reasoning=False)
    finally:
        _close_iterator(second_stream)


def _text_only(stream: Iterator[Any]) -> Iterator[str]:
    """Filter a chat stream down to its text deltas.

    Tool-call deltas (when ``tools`` is passed) and the trailing token-usage
    frame both ride the same iterator; the RAG / reasoning paths only consume
    text, so any non-str frame is dropped here rather than crashing the chat.
    """
    try:
        for item in stream:
            if isinstance(item, str):
                yield item
    finally:
        # Forward close to the source: when a consumer (filter_reasoning on
        # cap-fire) closes this generator, a plain for-loop would not propagate
        # GeneratorExit to *stream*, leaking its HTTP connection / in_flight slot.
        _close_iterator(stream)


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


def _close_iterator(tokens: Iterator[Any]) -> None:
    """Close *tokens* if it satisfies the ClosableIterator protocol."""
    if isinstance(tokens, ClosableIterator):
        with contextlib.suppress(Exception):
            tokens.close()


def strip_reasoning(text: str) -> str:
    """Reduce a complete (non-streaming) string to its final-answer text.

    Removes ``<think>...</think>`` blocks, and for gpt-oss harmony output keeps
    only the ``final`` channel, stripping the analysis channel and every
    ``<|...|>`` control token.
    """
    if _HARMONY_OPEN in text:
        return _final_channel_text(text)
    return _THINK_BLOCK_RE.sub("", text)


def _final_channel_text(text: str) -> str:
    """Run harmony *text* through the parser and keep only answer (final) tokens."""
    parser = HarmonyParser(show=False)
    return "".join(tok.content for tok in parser.feed(text) if not tok.is_reasoning)


def _could_be_partial(tag: str, buf: str) -> bool:
    """Check if the end of buf could be the start of the given tag."""
    return any(buf.endswith(tag[:length]) for length in range(1, len(tag)))
