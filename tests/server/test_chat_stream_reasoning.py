"""Wire-contract tests for `/api/chat/stream` reasoning split and cap."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from lilbee.core.config import cfg
from lilbee.retrieval.reasoning import CAP_NOTICE_TEMPLATE
from lilbee.server.chat_dispatch.canonical import (
    CanonicalChatRequest,
    CanonicalStreamEvent,
    ContentBlockDelta,
    ContentBlockStart,
    ContentBlockStop,
    MessageStart,
    MessageStop,
    TextBlock,
    TextDelta,
)
from lilbee.server.handlers import rag as _rag


def _events_from(texts: list[str]) -> list[CanonicalStreamEvent]:
    return [
        MessageStart(id="msg_test", model=cfg.chat_model),
        ContentBlockStart(index=0, block=TextBlock(text="")),
        *(ContentBlockDelta(index=0, delta=TextDelta(text=t)) for t in texts),
        ContentBlockStop(index=0),
        MessageStop(),
    ]


def _async_stream(events: list[CanonicalStreamEvent]) -> AsyncIterator[CanonicalStreamEvent]:
    async def _gen():
        for event in events:
            yield event

    return _gen()


def _rag_return() -> tuple[list[Any], list[dict[str, str]]]:
    return (
        [],
        [
            {"role": "system", "content": "ctx"},
            {"role": "user", "content": "q"},
        ],
    )


def _parse(events: list[str]) -> list[tuple[str, str]]:
    """Return (event_name, raw_data_line) for each non-empty SSE frame."""
    out: list[tuple[str, str]] = []
    for frame in events:
        if not frame:
            continue
        name = ""
        data = ""
        for line in frame.split("\n"):
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = line[len("data: ") :]
        if name:
            out.append((name, data))
    return out


class TestChatStreamReasoningSplit:
    """`/api/chat/stream` must split `<think>` blocks into `event: reasoning` frames."""

    async def test_think_block_emits_reasoning_event_and_token_event(
        self, mock_svc, monkeypatch
    ) -> None:
        """Reasoning text rides on `event: reasoning`; answer on `event: token`."""
        mock_svc.searcher.build_rag_context.return_value = _rag_return()
        snapshot_show = cfg.show_reasoning
        cfg.show_reasoning = True

        def _stream(_req: CanonicalChatRequest):
            return _async_stream(_events_from(["<think>foo</think>", "bar"]))

        monkeypatch.setattr(_rag, "dispatch_chat_stream", _stream)
        try:
            frames = [e async for e in _rag.chat_stream("q", [])]
        finally:
            cfg.show_reasoning = snapshot_show

        parsed = _parse(frames)
        reasoning_data = [d for name, d in parsed if name == "reasoning"]
        token_data = [d for name, d in parsed if name == "token"]
        assert any("foo" in d for d in reasoning_data), parsed
        assert any("bar" in d for d in token_data), parsed
        # Raw <think> tags must never leak to clients.
        assert not any("<think>" in d for _, d in parsed)
        assert not any("</think>" in d for _, d in parsed)

    async def test_reasoning_hidden_when_show_reasoning_false(self, mock_svc, monkeypatch) -> None:
        """With show_reasoning off, reasoning tokens are dropped, answer still ships."""
        mock_svc.searcher.build_rag_context.return_value = _rag_return()
        snapshot_show = cfg.show_reasoning
        cfg.show_reasoning = False

        def _stream(_req: CanonicalChatRequest):
            return _async_stream(_events_from(["<think>foo</think>", "bar"]))

        monkeypatch.setattr(_rag, "dispatch_chat_stream", _stream)
        try:
            frames = [e async for e in _rag.chat_stream("q", [])]
        finally:
            cfg.show_reasoning = snapshot_show

        parsed = _parse(frames)
        reasoning_data = [d for name, d in parsed if name == "reasoning"]
        token_data = [d for name, d in parsed if name == "token"]
        assert not any("foo" in d for d in reasoning_data)
        assert any("bar" in d for d in token_data)
        # show_reasoning=False still strips tags entirely; clients never see them.
        assert not any("<think>" in d for _, d in parsed)
        assert not any("foo" in d for _, d in parsed)


class TestChatStreamReasoningCap:
    """The reasoning cap must re-issue the request when reasoning blows past the limit."""

    async def test_cap_fires_emits_notice_then_continuation(self, mock_svc, monkeypatch) -> None:
        """Long `<think>` triggers CapNotice + a second dispatch call for the answer."""
        mock_svc.searcher.build_rag_context.return_value = _rag_return()
        snapshot_cap = cfg.max_reasoning_chars
        snapshot_show = cfg.show_reasoning
        cfg.max_reasoning_chars = 64
        cfg.show_reasoning = True

        long_reasoning = "<think>" + ("x" * 200) + "</think>never reached"
        call_count = {"n": 0}

        def _stream(_req: CanonicalChatRequest):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _async_stream(_events_from([long_reasoning]))
            return _async_stream(_events_from(["final ", "answer."]))

        monkeypatch.setattr(_rag, "dispatch_chat_stream", _stream)
        try:
            frames = [e async for e in _rag.chat_stream("q", [])]
        finally:
            cfg.max_reasoning_chars = snapshot_cap
            cfg.show_reasoning = snapshot_show

        parsed = _parse(frames)
        notice_marker = CAP_NOTICE_TEMPLATE.format(chars=64).strip()
        reasoning_data = [d for name, d in parsed if name == "reasoning"]
        token_data = [d for name, d in parsed if name == "token"]
        assert any(notice_marker in d for d in reasoning_data), parsed
        assert any("final " in d for d in token_data)
        assert any("answer." in d for d in token_data)
        assert call_count["n"] == 2

    async def test_partial_tag_at_eof_flushes_buffered_text(self, mock_svc, monkeypatch) -> None:
        """A stream that ends mid-``<thi`` flushes the buffered text to the client."""
        mock_svc.searcher.build_rag_context.return_value = _rag_return()
        snapshot_show = cfg.show_reasoning
        cfg.show_reasoning = True

        def _stream(_req: CanonicalChatRequest):
            # ``<thi`` is a possible-partial open tag; the parser holds it in
            # its buffer until either the rest of the tag arrives or the
            # stream ends, at which point ``flush()`` returns it as plain text.
            return _async_stream(_events_from(["hello <thi"]))

        monkeypatch.setattr(_rag, "dispatch_chat_stream", _stream)
        try:
            frames = [e async for e in _rag.chat_stream("q", [])]
        finally:
            cfg.show_reasoning = snapshot_show

        parsed = _parse(frames)
        token_data = [d for name, d in parsed if name == "token"]
        assert any("hello <thi" in d for d in token_data), parsed

    async def test_stream_without_aclose_is_handled_silently(self, mock_svc, monkeypatch) -> None:
        """An async iterator that lacks ``aclose`` still survives cap-fire."""
        mock_svc.searcher.build_rag_context.return_value = _rag_return()
        snapshot_cap = cfg.max_reasoning_chars
        snapshot_show = cfg.show_reasoning
        cfg.max_reasoning_chars = 32
        cfg.show_reasoning = True

        class _NoCloseStream:
            def __init__(self, events: list[CanonicalStreamEvent]) -> None:
                self._events = list(events)

            def __aiter__(self) -> _NoCloseStream:
                return self

            async def __anext__(self) -> CanonicalStreamEvent:
                if not self._events:
                    raise StopAsyncIteration
                return self._events.pop(0)

        long_reasoning = "<think>" + ("x" * 200) + "</think>"
        call_count = {"n": 0}

        def _stream(_req: CanonicalChatRequest):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _NoCloseStream(_events_from([long_reasoning]))
            return _async_stream(_events_from(["done"]))

        monkeypatch.setattr(_rag, "dispatch_chat_stream", _stream)
        try:
            frames = [e async for e in _rag.chat_stream("q", [])]
        finally:
            cfg.max_reasoning_chars = snapshot_cap
            cfg.show_reasoning = snapshot_show

        parsed = _parse(frames)
        token_data = [d for name, d in parsed if name == "token"]
        assert any("done" in d for d in token_data)
        assert call_count["n"] == 2

    async def test_cap_disabled_skips_continuation(self, mock_svc, monkeypatch) -> None:
        """Short reasoning under the cap does not re-issue."""
        mock_svc.searcher.build_rag_context.return_value = _rag_return()
        snapshot_cap = cfg.max_reasoning_chars
        snapshot_show = cfg.show_reasoning
        cfg.max_reasoning_chars = 64_000
        cfg.show_reasoning = True

        call_count = {"n": 0}

        def _stream(_req: CanonicalChatRequest):
            call_count["n"] += 1
            return _async_stream(_events_from(["<think>brief</think>", "answer"]))

        monkeypatch.setattr(_rag, "dispatch_chat_stream", _stream)
        try:
            frames = [e async for e in _rag.chat_stream("q", [])]
        finally:
            cfg.max_reasoning_chars = snapshot_cap
            cfg.show_reasoning = snapshot_show

        parsed = _parse(frames)
        notice_marker = CAP_NOTICE_TEMPLATE.format(chars=64_000).strip()
        assert not any(notice_marker in d for _, d in parsed)
        assert call_count["n"] == 1


@pytest.fixture
def mock_svc():
    """Mirror the lightweight services stub used by tests/test_server_handlers.py."""
    from unittest.mock import MagicMock

    from lilbee.app.services import set_services
    from tests.conftest import make_mock_services

    searcher = MagicMock()
    searcher.build_rag_context.return_value = None
    services = make_mock_services(searcher=searcher)
    services.registry.list_installed = MagicMock(return_value=[_installed_manifest(cfg.chat_model)])
    set_services(services)
    yield services
    set_services(None)


def _installed_manifest(ref: str):
    from unittest.mock import MagicMock

    m = MagicMock()
    m.ref = ref
    m.task = "chat"
    return m
