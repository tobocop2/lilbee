"""Store, session and model-file calls in async handlers run off the event loop."""

from __future__ import annotations

import asyncio
import hashlib
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lilbee.app import services as svc_mod
from lilbee.app.models import RemoveResult
from lilbee.catalog import CatalogResult
from lilbee.catalog.types import ModelSource
from lilbee.core.config import cfg
from lilbee.core.web_names import WEB_PREFIX
from lilbee.data.ingest import SyncResult
from lilbee.data.store import SearchChunk
from lilbee.retrieval.query.searcher import (
    GROUNDED_REFUSAL,
    SEARCH_NEEDS_EMBEDDER,
    AskResult,
    RagContext,
)
from lilbee.runtime.progress.types import LibraryUpgradeEvent, UpgradePhase
from lilbee.server import handlers
from lilbee.server import wiki as server_wiki
from lilbee.server.chat_dispatch.canonical import (
    CanonicalResponse,
    CanonicalUsage,
    StopReason,
    TextBlock,
)
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
from lilbee.sessions.store import TitleSource
from lilbee.wiki import lint as lint_mod
from lilbee.wiki.shared import WikiSubdir
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


class TestFileWalksStayOffTheLoop:
    """Directory walks and file writes in async handlers run on a worker thread."""

    @staticmethod
    def _assert_all_off_the_loop(threads: list[int]) -> None:
        assert threads
        assert threading.get_ident() not in threads

    async def test_models_catalog_lists_installed_models_off_the_loop(self, services):
        threads: list[int] = []
        services.registry.list_installed.side_effect = _recording(threads, returns=[])
        with (
            patch.object(model_handlers, "get_families", return_value=[]),
            patch.object(
                model_handlers,
                "get_catalog",
                return_value=CatalogResult(total=0, limit=20, offset=0, models=[]),
            ),
        ):
            await model_handlers.models_catalog(task="chat", source="native")
        _assert_off_the_loop(threads, 1)

    async def test_an_upload_creates_and_writes_files_off_the_loop(self, services):
        mkdir_threads: list[int] = []
        write_threads: list[int] = []
        services.store.get_sources.return_value = []
        with (
            patch.object(Path, "mkdir", _recording(mkdir_threads, Path.mkdir)),
            patch.object(Path, "write_bytes", _recording(write_threads, Path.write_bytes)),
            patch("lilbee.data.ingest.sync", new=AsyncMock(return_value=SyncResult())),
        ):
            await ingest_handlers._run_upload([("sub/note.md", b"# note\n")], SseStream())
        self._assert_all_off_the_loop(mkdir_threads)
        self._assert_all_off_the_loop(write_threads)
        assert (cfg.documents_dir / "sub" / "note.md").read_bytes() == b"# note\n"

    async def test_an_upload_creates_a_missing_documents_directory(self, services):
        cfg.documents_dir = cfg.data_root / "missing" / "documents"
        services.store.get_sources.return_value = []
        with patch("lilbee.data.ingest.sync", new=AsyncMock(return_value=SyncResult())):
            await ingest_handlers._run_upload([("note.md", b"# note\n")], SseStream())
        assert (cfg.documents_dir / "note.md").read_bytes() == b"# note\n"

    async def test_wiki_status_walks_the_tree_off_the_loop(self, services):
        threads: list[int] = []
        cfg.wiki = True
        wiki_root = cfg.data_root / cfg.wiki_dir
        for subdir, names in (
            (WikiSubdir.SUMMARIES, ["a.md"]),
            (WikiSubdir.DRAFTS, ["b.md", "nested/c.md"]),
            (WikiSubdir.CONCEPTS, ["d.md"]),
        ):
            for name in names:
                page = wiki_root / subdir / name
                page.parent.mkdir(parents=True, exist_ok=True)
                page.write_text("# page\n", encoding="utf-8")
        report = MagicMock(error_count=0, warning_count=0)
        with (
            patch.object(Path, "rglob", _recording(threads, Path.rglob)),
            patch.object(lint_mod, "lint_all", return_value=report),
        ):
            result = await server_wiki.wiki_status_route.fn()
        self._assert_all_off_the_loop(threads)
        assert (result.summaries, result.drafts, result.pages) == (1, 2, 2)


class TestOffLoopCallsKeepTheirArguments:
    """Each call moved onto a worker thread still hands the real store its arguments."""

    async def test_health_carries_the_upgrade_progress(self, services):
        event = LibraryUpgradeEvent(
            phase=UpgradePhase.TABLE, table="chunks", tables_done=1, tables_total=3
        )
        services.store.health_warnings.return_value = []
        services.store.upgrade_progress.return_value = event
        response = await handlers.health()
        assert response.upgrade == event

    async def test_models_delete_hands_the_model_and_source_to_the_manager(self, services):
        services.model_manager.remove.return_value = True
        await model_handlers.models_delete("org/repo/m.gguf", source="ollama")
        services.model_manager.remove.assert_called_once_with(
            "org/repo/m.gguf", source=ModelSource.OLLAMA
        )

    async def test_an_upload_keeps_a_file_whose_name_the_index_already_holds(self, services):
        content = b"# note\n"
        cfg.documents_dir.mkdir(parents=True)
        (cfg.documents_dir / "note.md").write_bytes(b"# stale\n")
        services.store.get_sources.return_value = [
            {"filename": "note.md", "file_hash": hashlib.sha256(content).hexdigest()}
        ]
        with patch("lilbee.data.ingest.sync", new=AsyncMock(return_value=SyncResult())):
            await ingest_handlers._run_upload([("note.md", content)], SseStream())
        # The same name is not a move: the upload overwrites the stale bytes.
        assert (cfg.documents_dir / "note.md").read_bytes() == content


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

    async def test_chat_hands_its_session_id_to_the_store(self, services):
        services.session_store.set_summary = MagicMock()
        with (
            patch.object(rag_handlers, "_manage_history", return_value=([], self._INFO)),
            pytest.raises(ModelNotFoundError, match="is not installed"),
        ):
            await rag_handlers.chat("q", [], top_k=0, session_id="s1")
        services.session_store.set_summary.assert_called_once_with("s1", "notes")

    async def test_the_stream_hands_its_session_id_to_the_store(self, services):
        services.session_store.set_summary = MagicMock()
        with (
            patch.object(rag_handlers, "_compaction_pending", return_value=True),
            patch.object(rag_handlers, "_manage_history", return_value=([], self._INFO)),
        ):
            [f async for f in rag_handlers._context_management_frames([], "", "s1")]
        services.session_store.set_summary.assert_called_once_with("s1", "notes")

    async def test_the_stream_without_a_fresh_summary_saves_and_announces_nothing(self, services):
        persist = MagicMock()
        with (
            patch.object(rag_handlers, "_compaction_pending", return_value=True),
            patch.object(rag_handlers, "_manage_history", return_value=(["kept"], None)),
            patch.object(rag_handlers, "_persist_summary", persist),
        ):
            frames = [f async for f in rag_handlers._context_management_frames([], "", "s1")]
        assert frames[-1] == (["kept"], None)
        assert not any(isinstance(f, str) and "event: compaction" in f for f in frames)
        persist.assert_not_called()


_ANSWER = CanonicalResponse(
    id="msg_test",
    model="m",
    content=[TextBlock(text="ok")],
    stop_reason=StopReason.END_TURN,
    usage=CanonicalUsage(input_tokens=0, output_tokens=0),
)


class TestChatStreamWarmsBeforeRetrieval:
    """The chat stream yields the warming notice from its own worker call, before retrieval."""

    async def test_the_warming_probe_runs_off_the_loop(self, services):
        threads: list[int] = []
        with (
            patch.object(
                rag_handlers, "_chat_warming_events", side_effect=_recording(threads, returns=[])
            ),
            patch.object(services.searcher, "search_unavailable", return_value=True),
        ):
            [f async for f in rag_handlers.chat_stream("q", [])]
        _assert_off_the_loop(threads, 1)

    async def test_the_warming_notice_precedes_retrieval(self, services):
        release = threading.Event()
        retrieval_waited: list[bool] = []

        def retrieve(question: str) -> str:
            # The notice must reach the client while retrieval is still to run.
            retrieval_waited.append(release.wait(timeout=5))
            return "canned"

        notice = "event: warming\n\n"
        searcher = services.searcher
        with (
            patch.object(rag_handlers, "_chat_warming_events", return_value=[notice]),
            patch.object(searcher, "search_unavailable", return_value=False),
            patch.object(searcher, "skip_retrieval", return_value=False),
            patch.object(searcher, "pre_retrieval_answer", side_effect=retrieve),
        ):
            frames = []
            async for frame in rag_handlers.chat_stream("q", []):
                frames.append(frame)
                if frame == notice:
                    release.set()
        assert retrieval_waited == [True]
        answer_at = next(i for i, frame in enumerate(frames) if "canned" in frame)
        assert frames.count(notice) == 1
        assert frames.index(notice) < answer_at

    async def test_the_warming_notice_precedes_the_embedder_refusal(self, services):
        notice = "event: warming\n\n"
        with (
            patch.object(rag_handlers, "_chat_warming_events", return_value=[notice]),
            patch.object(services.searcher, "search_unavailable", return_value=True),
        ):
            frames = [f async for f in rag_handlers.chat_stream("q", [])]
        refusal_at = next(i for i, frame in enumerate(frames) if SEARCH_NEEDS_EMBEDDER in frame)
        assert frames.index(notice) < refusal_at

    async def test_a_client_that_leaves_during_warming_never_starts_retrieval(self, services):
        notice = "event: warming\n\n"
        searcher = services.searcher
        with (
            patch.object(rag_handlers, "_chat_warming_events", return_value=[notice]),
            patch.object(searcher, "search_unavailable", return_value=False),
            patch.object(searcher, "skip_retrieval", return_value=False),
            patch.object(searcher, "pre_retrieval_answer", return_value="canned") as retrieve,
        ):
            stream = rag_handlers.chat_stream("q", [])
            async with asyncio.timeout(5):
                first = await anext(stream)
                await stream.aclose()
        assert first == notice
        retrieve.assert_not_called()


class TestChatRetrievalStaysOffTheLoop:
    """The one-shot chat route reads the library and memory on a worker thread."""

    async def test_an_ungrounded_turn_builds_its_messages_off_the_loop(self, services):
        threads: list[int] = []
        searcher = services.searcher
        with (
            patch.object(
                searcher,
                "direct_messages",
                side_effect=_recording(threads, returns=[{"role": "user", "content": "q"}]),
            ),
            pytest.raises(ModelNotFoundError, match="is not installed"),
        ):
            await rag_handlers.chat("q", [], top_k=0)
        _assert_off_the_loop(threads, 1)

    async def test_a_canned_answer_is_looked_up_off_the_loop(self, services):
        threads: list[int] = []
        searcher = services.searcher
        with (
            patch.object(searcher, "search_unavailable", return_value=False),
            patch.object(searcher, "skip_retrieval", return_value=False),
            patch.object(
                searcher, "pre_retrieval_answer", side_effect=_recording(threads, returns="none")
            ),
        ):
            result = await rag_handlers.chat("how many documents?", [])
        assert result.answer == "none"
        _assert_off_the_loop(threads, 1)

    async def test_a_grounded_turn_retrieves_off_the_loop(self, services):
        threads: list[int] = []
        searcher = services.searcher
        refused = RagContext([], [], refused=True)
        with (
            patch.object(searcher, "search_unavailable", return_value=False),
            patch.object(searcher, "skip_retrieval", return_value=False),
            patch.object(searcher, "pre_retrieval_answer", return_value=None),
            patch.object(
                searcher, "build_rag_context", side_effect=_recording(threads, returns=refused)
            ),
        ):
            result = await rag_handlers.chat("q", [])
        assert result.answer == GROUNDED_REFUSAL
        _assert_off_the_loop(threads, 1)

    async def test_the_embedder_probe_runs_off_the_loop(self, services):
        threads: list[int] = []
        with patch.object(
            services.searcher,
            "search_unavailable",
            side_effect=_recording(threads, returns=True),
        ):
            result = await rag_handlers.chat("q", [])
        assert result.answer == SEARCH_NEEDS_EMBEDDER
        _assert_off_the_loop(threads, 1)

    async def test_a_crawled_page_source_is_titled_off_the_loop(self, services):
        threads: list[int] = []
        page = SearchChunk(
            source=f"{WEB_PREFIX}page.md",
            content_type="text",
            chunk="text",
            distance=0.1,
            page_start=1,
            page_end=1,
            line_start=0,
            line_end=0,
            chunk_index=0,
            vector=[0.1],
        )
        grounded = RagContext([page], [{"role": "user", "content": "q"}])
        searcher = services.searcher
        services.store.web_titles.side_effect = _recording(threads, returns={})
        with (
            patch.object(searcher, "search_unavailable", return_value=False),
            patch.object(searcher, "skip_retrieval", return_value=False),
            patch.object(searcher, "pre_retrieval_answer", return_value=None),
            patch.object(searcher, "build_rag_context", return_value=grounded),
            patch.object(rag_handlers, "dispatch_chat", return_value=_ANSWER),
            patch.object(rag_handlers, "_store_extracted_memories", AsyncMock(return_value=[])),
        ):
            result = await rag_handlers.chat("q", [])
        assert [source.source for source in result.sources] == [page.source]
        # Titles are read for the sources, the cited subset and the dropped list.
        _assert_off_the_loop(threads, 3)


def _web_chunk() -> SearchChunk:
    return SearchChunk(
        source=f"{WEB_PREFIX}page.md",
        content_type="text",
        chunk="text",
        distance=0.1,
        page_start=1,
        page_end=1,
        line_start=0,
        line_end=0,
        chunk_index=0,
        vector=[0.1],
    )


class TestAskStaysOffTheLoop:
    """The ask routes read the store and probe the engine on a worker thread."""

    async def test_the_one_shot_response_is_titled_off_the_loop(self, services):
        threads: list[int] = []
        page = _web_chunk()
        services.store.web_titles.side_effect = _recording(threads, returns={})
        with (
            patch.object(
                services.searcher, "ask_raw", return_value=AskResult(answer="a", sources=[page])
            ),
            patch.object(services.searcher, "search_unavailable", return_value=True),
        ):
            result = await rag_handlers.ask("q")
        assert [source.source for source in result.sources] == [page.source]
        # Titles are read for the sources, the cited subset and the dropped list.
        _assert_off_the_loop(threads, 3)

    async def test_the_one_shot_embedder_probe_runs_off_the_loop(self, services):
        threads: list[int] = []
        with (
            patch.object(
                services.searcher, "ask_raw", return_value=AskResult(answer="a", sources=[])
            ),
            patch.object(
                services.searcher,
                "search_unavailable",
                side_effect=_recording(threads, returns=True),
            ),
        ):
            await rag_handlers.ask("q")
        _assert_off_the_loop(threads, 1)

    async def test_the_stream_probes_the_engine_and_embedder_off_the_loop(self, services):
        warming: list[int] = []
        unavailable: list[int] = []
        with (
            patch.object(
                rag_handlers, "_chat_warming_events", side_effect=_recording(warming, returns=[])
            ),
            patch.object(
                services.searcher,
                "search_unavailable",
                side_effect=_recording(unavailable, returns=True),
            ),
        ):
            [f async for f in rag_handlers.ask_stream("q")]
        _assert_off_the_loop(warming, 1)
        _assert_off_the_loop(unavailable, 1)

    async def test_the_stream_decides_to_skip_retrieval_off_the_loop(self, services):
        threads: list[int] = []
        searcher = services.searcher
        with (
            patch.object(searcher, "search_unavailable", return_value=False),
            patch.object(searcher, "skip_retrieval", side_effect=_recording(threads, returns=True)),
            patch.object(rag_handlers, "_resolve_stream_context", return_value=([], None, [])),
        ):
            [f async for f in rag_handlers.ask_stream("q")]
        _assert_off_the_loop(threads, 1)

    async def test_the_sources_frame_is_titled_off_the_loop(self, services):
        threads: list[int] = []
        page = _web_chunk()
        services.store.web_titles.side_effect = _recording(threads, returns={})
        with patch.object(rag_handlers, "_store_extracted_memories", AsyncMock(return_value=[])):
            frames = [f async for f in rag_handlers._emit_sources_and_memories("q", ["a"], [page])]
        assert any(page.source in frame for frame in frames)
        _assert_off_the_loop(threads, 1)


class TestSyncPlanningStaysOffTheLoop:
    """The sync behind an upload walks the disk and reads the index on a worker thread."""

    async def test_the_walk_and_the_index_reads_run_off_the_loop(self, services):
        from lilbee.data.ingest import pipeline
        from lilbee.data.ingest.discovery import CorpusScan

        ran: dict[str, list[int]] = {
            name: []
            for name in (
                "plan_fanout",
                "discover_corpus",
                "_sources_in_scope",
                "_forget_refused",
                "_move_pool",
                "_missing_from_index",
                "_describe_failures",
            )
        }
        services.store.embedding_mismatch.return_value = None
        services.store.held_records.return_value = {}
        services.store.get_sources.return_value = []

        def stub(name: str, returns):
            return patch.object(pipeline, name, side_effect=_recording(ran[name], returns=returns))

        with (
            stub("plan_fanout", []),
            stub("discover_corpus", CorpusScan({}, {})),
            stub("_sources_in_scope", []),
            stub("_forget_refused", []),
            stub("_move_pool", pipeline._MovePool([], {})),
            stub("_missing_from_index", []),
            stub("_describe_failures", []),
        ):
            await pipeline.sync(quiet=True)
        for name, threads in ran.items():
            assert threads, name
            assert threading.get_ident() not in threads, name

    async def test_the_embedder_probe_runs_off_the_loop_when_a_file_is_planned(self, services):
        from lilbee.data.ingest import pipeline
        from lilbee.data.ingest.discovery import CorpusScan

        async def one_batch(*_args):
            yield [MagicMock()]

        probed: list[int] = []
        services.store.embedding_mismatch.return_value = None
        services.store.held_records.return_value = {}
        services.store.get_sources.return_value = []

        with (
            patch.object(pipeline, "plan_fanout", return_value=[]),
            patch.object(pipeline, "discover_corpus", return_value=CorpusScan({}, {})),
            patch.object(pipeline, "_forget_refused", return_value=[]),
            patch.object(pipeline, "_move_pool", return_value=pipeline._MovePool([], {})),
            patch.object(pipeline, "_plan_batches", one_batch),
            patch.object(pipeline, "_require_embedding_model", _recording(probed)),
            patch.object(pipeline, "ingest_stream", new_callable=AsyncMock) as ingest,
            patch.object(pipeline, "_run_post_ingest_passes", new_callable=AsyncMock),
            patch.object(pipeline, "_persist_failures"),
            patch.object(pipeline, "_missing_from_index", return_value=[]),
            patch.object(pipeline, "_describe_failures", return_value=[]),
        ):
            await pipeline.sync(quiet=True)
        ingest.assert_awaited_once()
        _assert_off_the_loop(probed, 1)

    async def test_the_documents_dir_and_the_ignore_file_are_made_and_read_off_the_loop(
        self, services
    ):
        from lilbee.data.ingest import pipeline
        from lilbee.data.ingest.discovery import CorpusScan

        made: list[int] = []
        read: list[int] = []
        make_dir = Path.mkdir
        services.store.embedding_mismatch.return_value = None
        services.store.held_records.return_value = {}
        services.store.get_sources.return_value = []

        def record_mkdir(path, *args, **kwargs):
            if path == cfg.documents_dir:
                made.append(threading.get_ident())
            return make_dir(path, *args, **kwargs)

        with (
            patch.object(Path, "mkdir", record_mkdir),
            patch.object(
                pipeline.IgnoreRules,
                "for_corpus",
                _recording(read, wrapped=pipeline.IgnoreRules.for_corpus),
            ),
            patch.object(pipeline, "plan_fanout", return_value=[]),
            patch.object(pipeline, "discover_corpus", return_value=CorpusScan({}, {})),
            patch.object(pipeline, "_forget_refused", return_value=[]),
            patch.object(pipeline, "_move_pool", return_value=pipeline._MovePool([], {})),
            patch.object(pipeline, "_missing_from_index", return_value=[]),
            patch.object(pipeline, "_describe_failures", return_value=[]),
        ):
            await pipeline.sync(quiet=True)
        _assert_off_the_loop(made, 1)
        _assert_off_the_loop(read, 1)


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

    async def test_create_records_the_http_origin(self, store):
        created = await session_handlers.create_session(
            SessionCreateRequest(model_ref="model", scope="all")
        )
        assert store.get(created.meta.id).meta.origin == SessionOrigin.HTTP

    async def test_claim_hands_the_session_to_the_http_origin(self, store, session_id):
        store.transfer = MagicMock(wraps=store.transfer)
        await session_handlers.claim_session(session_id)
        store.transfer.assert_called_once_with(session_id, SessionOrigin.HTTP)

    async def test_rename_marks_the_title_as_custom(self, store, session_id):
        store.set_title = MagicMock(wraps=store.set_title)
        await session_handlers.rename_session(session_id, "a title")
        store.set_title.assert_called_once_with(session_id, "a title", TitleSource.CUSTOM)

    async def test_delete(self, store, session_id):
        threads: list[int] = []
        self._spy(store, threads)
        await session_handlers.delete_session(session_id)
        _assert_off_the_loop(threads, 1)
