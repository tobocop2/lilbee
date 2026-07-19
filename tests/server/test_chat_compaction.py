"""HTTP chat context management: windowing, compaction, and its SSE events."""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import pytest

from lilbee.core.config import cfg
from lilbee.retrieval.query.compaction import (
    COMPACT_KEEP_RECENT,
    HISTORY_TOKEN_BUDGET_FRACTION,
    CompactionResult,
    history_budget,
    summary_messages,
)
from lilbee.server.chat_dispatch.canonical import (
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
from lilbee.server.models import ChatRequest, CompactionInfo
from lilbee.sessions import SessionNotFoundError
from tests.server.conftest import parse_sse_events


def _msgs(n: int, size: int = 400) -> list[dict[str, str]]:
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": "x" * size} for i in range(n)
    ]


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


def _installed_manifest(ref: str) -> MagicMock:
    m = MagicMock()
    m.ref = ref
    m.task = "chat"
    return m


@pytest.fixture
def mock_svc():
    """Mirror the lightweight services stub used by tests/test_server_handlers.py."""
    from lilbee.app.services import set_services
    from tests.conftest import make_mock_services

    searcher = MagicMock()
    searcher.build_rag_context.return_value = None
    searcher.search_unavailable.return_value = False
    searcher.pre_retrieval_answer.return_value = None
    searcher.direct_messages.return_value = [{"role": "user", "content": "q"}]
    services = dataclasses.replace(make_mock_services(searcher=searcher), session_store=MagicMock())
    services.registry.list_installed = MagicMock(return_value=[_installed_manifest(cfg.chat_model)])
    set_services(services)
    yield services
    set_services(None)


def test_history_budget_is_the_tui_fraction() -> None:
    assert history_budget(2048) == int(2048 * HISTORY_TOKEN_BUDGET_FRACTION)


class TestManageHistory:
    def test_windows_without_compacting_when_the_toggle_is_off(self, mock_svc, monkeypatch) -> None:
        monkeypatch.setattr(cfg, "chat_compaction", False)
        monkeypatch.setattr(cfg, "chat_n_ctx_target", 2048)
        history = _msgs(40)

        managed, info = _rag._manage_history(history, "")

        assert info is None
        mock_svc.searcher.summarize_history.assert_not_called()
        assert len(managed) < len(history)
        assert managed == history[-len(managed) :]

    def test_compacts_when_due_and_returns_the_result(self, mock_svc, monkeypatch) -> None:
        monkeypatch.setattr(cfg, "chat_compaction", True)
        monkeypatch.setattr(cfg, "chat_n_ctx_target", 2048)
        history = _msgs(40)
        dropped = history[:-COMPACT_KEEP_RECENT]
        mock_svc.searcher.summarize_history.return_value = CompactionResult(
            summary="the notes", condensed=len(dropped), stranded=2
        )

        managed, info = _rag._manage_history(history, "old notes")

        mock_svc.searcher.summarize_history.assert_called_once_with(
            dropped, "old notes", on_batch=None
        )
        assert info == CompactionInfo(summary="the notes", condensed=len(dropped), stranded=2)
        assert managed[:2] == summary_messages("the notes")
        assert managed[2:] == history[-COMPACT_KEEP_RECENT:]

    def test_leaves_a_conversation_under_the_trigger_alone(self, mock_svc, monkeypatch) -> None:
        monkeypatch.setattr(cfg, "chat_compaction", True)
        monkeypatch.setattr(cfg, "chat_n_ctx_target", 8192)
        history = _msgs(4)

        managed, info = _rag._manage_history(history, "")

        assert info is None
        mock_svc.searcher.summarize_history.assert_not_called()
        assert managed == history

    def test_windows_when_only_the_tail_fills_the_budget(self, mock_svc, monkeypatch) -> None:
        monkeypatch.setattr(cfg, "chat_compaction", True)
        monkeypatch.setattr(cfg, "chat_n_ctx_target", 2048)
        history = _msgs(COMPACT_KEEP_RECENT, size=4000)

        managed, info = _rag._manage_history(history, "")

        assert info is None
        mock_svc.searcher.summarize_history.assert_not_called()
        assert len(managed) < len(history)


class TestPersistSummary:
    def test_writes_fresh_notes_to_the_session(self, mock_svc, monkeypatch) -> None:
        monkeypatch.setattr(cfg, "sessions_enabled", True)
        info = CompactionInfo(summary="notes", condensed=3, stranded=0)

        _rag._persist_summary("s1", info)

        mock_svc.session_store.set_summary.assert_called_once_with("s1", "notes")

    @pytest.mark.parametrize(
        ("session_id", "info", "sessions_on"),
        [
            pytest.param("s1", None, True, id="no-compaction"),
            pytest.param(
                None, CompactionInfo(summary="n", condensed=1, stranded=0), True, id="no-session"
            ),
            pytest.param(
                "s1", CompactionInfo(summary="", condensed=0, stranded=4), True, id="empty-summary"
            ),
            pytest.param(
                "s1", CompactionInfo(summary="n", condensed=1, stranded=0), False, id="sessions-off"
            ),
        ],
    )
    def test_skips_when_there_is_nothing_or_nowhere_to_write(
        self, mock_svc, monkeypatch, session_id, info, sessions_on
    ) -> None:
        monkeypatch.setattr(cfg, "sessions_enabled", sessions_on)

        _rag._persist_summary(session_id, info)

        mock_svc.session_store.set_summary.assert_not_called()

    def test_tolerates_a_session_deleted_mid_chat(self, mock_svc, monkeypatch) -> None:
        monkeypatch.setattr(cfg, "sessions_enabled", True)
        mock_svc.session_store.set_summary.side_effect = SessionNotFoundError("s1")

        _rag._persist_summary("s1", CompactionInfo(summary="n", condensed=1, stranded=0))


class TestChatStreamCompactionEvents:
    async def _frames(self, history: list[dict[str, str]], summary: str = "") -> list[str]:
        return [
            frame
            async for frame in _rag.chat_stream(
                "q", history=history, top_k=0, summary=summary, session_id="s1"
            )
        ]

    async def test_emits_compacting_then_compaction_before_the_answer(
        self, mock_svc, monkeypatch
    ) -> None:
        monkeypatch.setattr(cfg, "chat_compaction", True)
        monkeypatch.setattr(cfg, "chat_n_ctx_target", 2048)
        monkeypatch.setattr(cfg, "sessions_enabled", True)
        monkeypatch.setattr(
            _rag, "dispatch_chat_stream", lambda req: _async_stream(_events_from(["hi"]))
        )
        mock_svc.searcher.summarize_history.return_value = CompactionResult(
            summary="the notes", condensed=6, stranded=0
        )

        frames = await self._frames(_msgs(40))

        events = parse_sse_events("".join(frames).encode())
        names = [name for name, _ in events]
        assert names.index("compacting") < names.index("compaction") < names.index("token")
        compaction = dict(events)["compaction"]
        assert compaction == {"summary": "the notes", "condensed": 6, "stranded": 0}
        mock_svc.session_store.set_summary.assert_called_once_with("s1", "the notes")

    async def test_relays_per_batch_progress_while_condensing(self, mock_svc, monkeypatch) -> None:
        monkeypatch.setattr(cfg, "chat_compaction", True)
        monkeypatch.setattr(cfg, "chat_n_ctx_target", 2048)
        monkeypatch.setattr(cfg, "sessions_enabled", True)
        monkeypatch.setattr(
            _rag, "dispatch_chat_stream", lambda req: _async_stream(_events_from(["hi"]))
        )

        def _condense(dropped, summary, on_batch=None):
            if on_batch is not None:
                on_batch(1, 2)
                on_batch(2, 2)
            return CompactionResult(summary="the notes", condensed=len(dropped), stranded=0)

        mock_svc.searcher.summarize_history.side_effect = _condense

        frames = await self._frames(_msgs(40))

        events = parse_sse_events("".join(frames).encode())
        compacting = [data for name, data in events if name == "compacting"]
        assert compacting[0] == {}
        assert {"batch": 1, "batches": 2} in compacting
        assert {"batch": 2, "batches": 2} in compacting
        names = [name for name, _ in events]
        assert names.index("compaction") < names.index("token")

    async def test_stays_silent_when_compaction_is_off(self, mock_svc, monkeypatch) -> None:
        monkeypatch.setattr(cfg, "chat_compaction", False)
        monkeypatch.setattr(cfg, "chat_n_ctx_target", 2048)
        monkeypatch.setattr(
            _rag, "dispatch_chat_stream", lambda req: _async_stream(_events_from(["hi"]))
        )

        frames = await self._frames(_msgs(40))

        names = [name for name, _ in parse_sse_events("".join(frames).encode())]
        assert "compacting" not in names
        assert "compaction" not in names
        mock_svc.searcher.summarize_history.assert_not_called()


class TestChatResponseCompaction:
    async def test_non_stream_chat_carries_and_persists_the_result(
        self, mock_svc, monkeypatch
    ) -> None:
        monkeypatch.setattr(cfg, "chat_compaction", True)
        monkeypatch.setattr(cfg, "chat_n_ctx_target", 2048)
        monkeypatch.setattr(cfg, "sessions_enabled", True)
        response = MagicMock()
        response.content = [TextBlock(text="an answer")]
        monkeypatch.setattr(_rag, "dispatch_chat", lambda req: response)
        mock_svc.searcher.summarize_history.return_value = CompactionResult(
            summary="the notes", condensed=6, stranded=1
        )

        result = await _rag.chat("q", history=_msgs(40), top_k=0, summary="", session_id="s1")

        assert result.compaction == CompactionInfo(summary="the notes", condensed=6, stranded=1)
        mock_svc.session_store.set_summary.assert_called_once_with("s1", "the notes")

    async def test_non_stream_chat_reports_no_compaction_when_none_ran(
        self, mock_svc, monkeypatch
    ) -> None:
        monkeypatch.setattr(cfg, "chat_compaction", False)
        response = MagicMock()
        response.content = [TextBlock(text="an answer")]
        monkeypatch.setattr(_rag, "dispatch_chat", lambda req: response)

        result = await _rag.chat("q", history=_msgs(4), top_k=0)

        assert result.compaction is None


def test_chat_request_defaults_carry_no_conversation_state() -> None:
    req = ChatRequest(question="q")
    assert req.summary == ""
    assert req.session_id is None
