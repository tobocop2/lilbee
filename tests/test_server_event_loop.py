"""Store, session and model-file calls in async handlers run off the event loop."""

from __future__ import annotations

import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lilbee.app import services as svc_mod
from lilbee.app.models import RemoveResult
from lilbee.core.config import cfg
from lilbee.data.ingest import SyncResult
from lilbee.server import handlers
from lilbee.server.chat_dispatch.dispatch import ModelNotFoundError
from lilbee.server.handlers import documents as document_handlers
from lilbee.server.handlers import ingest as ingest_handlers
from lilbee.server.handlers import models as model_handlers
from lilbee.server.handlers import rag as rag_handlers
from lilbee.server.handlers import sessions as session_handlers
from lilbee.server.handlers.sse import SseStream
from lilbee.server.models import (
    CompactionInfo,
    SessionCreateRequest,
    SessionMessageCreateRequest,
    SessionSummaryRequest,
)
from lilbee.sessions import MessageRole, SessionMessage, SessionOrigin
from tests.conftest import make_mock_services


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.documents_dir = tmp_path / "documents"
    yield
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture
def services():
    services = make_mock_services()
    svc_mod.set_services(services)
    yield services
    svc_mod.set_services(None)


def _recording(threads: list[int], wrapped=None, returns=None):
    """A stand-in that notes the thread it ran on, then calls through or returns."""

    def record(*args, **kwargs):
        threads.append(threading.get_ident())
        return wrapped(*args, **kwargs) if wrapped is not None else returns

    return record


def _assert_off_the_loop(threads: list[int], calls: int) -> None:
    assert len(threads) == calls
    assert threading.get_ident() not in threads


class TestStoreCallsStayOffTheLoop:
    async def test_list_documents_reads_both_counts_off_the_loop(self, services):
        threads: list[int] = []
        services.store.get_sources.side_effect = _recording(threads, returns=[])
        services.store.count_sources.side_effect = _recording(threads, returns=0)
        await document_handlers.list_documents()
        _assert_off_the_loop(threads, 2)

    async def test_status_gathers_off_the_loop(self, services):
        threads: list[int] = []
        wrapped = services.store.source_summary
        services.store.source_summary = MagicMock(side_effect=_recording(threads, wrapped))
        await handlers.status()
        _assert_off_the_loop(threads, 1)

    async def test_health_reads_store_state_off_the_loop(self, services):
        threads: list[int] = []
        services.store.health_warnings.side_effect = _recording(threads, returns=[])
        services.store.upgrade_progress.side_effect = _recording(threads, returns=None)
        await handlers.health()
        _assert_off_the_loop(threads, 2)

    async def test_models_delete_removes_files_off_the_loop(self, services):
        threads: list[int] = []
        result = RemoveResult(model="m", deleted=True, freed_gb=1.0)
        with patch(
            "lilbee.app.models.remove_model_data", side_effect=_recording(threads, returns=result)
        ):
            await model_handlers.models_delete("m", source="native")
        _assert_off_the_loop(threads, 1)

    async def test_an_upload_looks_up_same_content_off_the_loop(self, services):
        threads: list[int] = []
        services.store.get_sources.side_effect = _recording(threads, returns=[])
        sse = SseStream()
        with patch("lilbee.data.ingest.sync", new=AsyncMock(return_value=SyncResult())):
            await ingest_handlers._run_upload([("note.md", b"# note\n")], sse)
        _assert_off_the_loop(threads, 1)


class TestChatSummaryPersistStaysOffTheLoop:
    """A fresh compaction summary is saved to the session store on a worker thread."""

    _INFO = CompactionInfo(summary="notes", condensed=2, stranded=0)

    async def test_chat_saves_the_summary_off_the_loop(self, services):
        threads: list[int] = []
        with (
            patch.object(rag_handlers, "_manage_history", return_value=([], self._INFO)),
            patch.object(rag_handlers, "_persist_summary", side_effect=_recording(threads)),
            # No chat model is installed here, so the turn ends after the summary is saved.
            pytest.raises(ModelNotFoundError, match="is not installed"),
        ):
            await rag_handlers.chat("q", [], top_k=0, session_id="s1")
        _assert_off_the_loop(threads, 1)

    async def test_the_stream_saves_the_summary_off_the_loop(self, services):
        threads: list[int] = []
        with (
            patch.object(rag_handlers, "_compaction_pending", return_value=True),
            patch.object(rag_handlers, "_manage_history", return_value=([], self._INFO)),
            patch.object(rag_handlers, "_persist_summary", side_effect=_recording(threads)),
        ):
            frames = [f async for f in rag_handlers._context_management_frames([], "", "s1")]
        assert any(isinstance(f, str) and "event: compaction" in f for f in frames)
        _assert_off_the_loop(threads, 1)


class TestSessionCallsStayOffTheLoop:
    """Each handler's store calls run on a worker thread; one session exists to act on."""

    @pytest.fixture
    def store(self, services):
        return services.session_store

    @pytest.fixture
    def session_id(self, store):
        sid = store.create("model", "all", SessionOrigin.HTTP)
        store.add_message(sid, SessionMessage(role=MessageRole.USER, content="hello"))
        return sid

    @staticmethod
    def _spy(store, threads: list[int]) -> None:
        for name in (
            "list",
            "get",
            "create",
            "add_message",
            "fork",
            "transfer",
            "set_summary",
            "set_title",
            "delete",
        ):
            setattr(store, name, _recording(threads, getattr(store, name)))

    async def test_list(self, store, session_id):
        threads: list[int] = []
        self._spy(store, threads)
        await session_handlers.list_sessions()
        _assert_off_the_loop(threads, 1)

    async def test_get(self, store, session_id):
        threads: list[int] = []
        self._spy(store, threads)
        await session_handlers.get_session(session_id)
        _assert_off_the_loop(threads, 1)

    async def test_markdown(self, store, session_id):
        threads: list[int] = []
        self._spy(store, threads)
        await session_handlers.get_session_markdown(session_id)
        _assert_off_the_loop(threads, 1)

    async def test_create(self, store):
        threads: list[int] = []
        self._spy(store, threads)
        await session_handlers.create_session(SessionCreateRequest(model_ref="model", scope="all"))
        _assert_off_the_loop(threads, 2)

    async def test_add_message(self, store, session_id):
        threads: list[int] = []
        self._spy(store, threads)
        data = SessionMessageCreateRequest(role=MessageRole.USER, content="more", sources=[])
        await session_handlers.add_session_message(session_id, data)
        _assert_off_the_loop(threads, 2)

    async def test_fork(self, store, session_id):
        threads: list[int] = []
        self._spy(store, threads)
        await session_handlers.fork_session(session_id, None)
        # A fork copies messages through the store's own get and create, so the
        # spy sees more than the two calls the handler makes.
        _assert_off_the_loop(threads, 4)

    async def test_claim(self, store, session_id):
        threads: list[int] = []
        self._spy(store, threads)
        await session_handlers.claim_session(session_id)
        _assert_off_the_loop(threads, 2)

    async def test_set_summary(self, store, session_id):
        threads: list[int] = []
        self._spy(store, threads)
        data = SessionSummaryRequest(summary="notes")
        await session_handlers.set_session_summary(session_id, data)
        _assert_off_the_loop(threads, 2)

    async def test_rename(self, store, session_id):
        threads: list[int] = []
        self._spy(store, threads)
        await session_handlers.rename_session(session_id, "a title")
        _assert_off_the_loop(threads, 1)

    async def test_delete(self, store, session_id):
        threads: list[int] = []
        self._spy(store, threads)
        await session_handlers.delete_session(session_id)
        _assert_off_the_loop(threads, 1)
