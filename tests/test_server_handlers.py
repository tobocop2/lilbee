"""Tests for the framework-agnostic server handlers."""

import asyncio
import json
import logging
from unittest.mock import MagicMock, patch

import pytest

import lilbee.app.services as svc_mod
from lilbee.core.config import cfg
from lilbee.data.ingest import SyncResult
from lilbee.data.store import SearchChunk
from lilbee.server import handlers
from lilbee.server.handlers import (
    ingest as _ingest_h,
)
from lilbee.server.handlers import (
    rag as _rag_h,
)
from lilbee.server.handlers import (
    sse as _sse_h,
)

_SAMPLE_CHUNK = SearchChunk(
    source="a.pdf",
    content_type="pdf",
    chunk="text",
    distance=0.1,
    page_start=1,
    page_end=1,
    line_start=0,
    line_end=0,
    chunk_index=0,
    vector=[0.1],
)


def _rag_return(chunks: list[SearchChunk] | None = None):
    """Build a mock build_rag_context return value."""
    results = chunks or [_SAMPLE_CHUNK]
    messages = [{"role": "system", "content": "test"}, {"role": "user", "content": "q"}]
    return results, messages


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    """Redirect config paths for all handler tests."""
    snapshot = cfg.model_copy()
    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir(exist_ok=True)
    cfg.data_dir = tmp_path / "data"
    cfg.data_root = tmp_path
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    yield tmp_path
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture(autouse=True)
def mock_svc():
    """Provide a mock Services container for all handler tests."""
    from tests.conftest import make_mock_services

    searcher = MagicMock()
    searcher.search.return_value = []
    searcher.ask_raw.return_value = MagicMock(answer="", sources=[])
    searcher.build_rag_context.return_value = None
    services = make_mock_services(searcher=searcher)
    svc_mod.set_services(services)
    yield services
    svc_mod.set_services(None)


class TestSseEvent:
    def test_format_string_data(self):
        result = handlers.sse_event("token", {"token": "hi"})
        assert result == 'event: token\ndata: {"token": "hi"}\n\n'

    def test_format_empty_dict(self):
        result = handlers.sse_event("done", {})
        assert result == "event: done\ndata: {}\n\n"

    def test_format_list_data(self):
        result = handlers.sse_event("sources", [{"a": 1}])
        assert result == 'event: sources\ndata: [{"a": 1}]\n\n'


class TestHealth:
    async def test_returns_status_and_version(self):
        with patch("lilbee.server.handlers.get_version", return_value="1.2.3"):
            result = await handlers.health()
        assert result.status == "ok"
        assert result.version == "1.2.3"


class TestStatus:
    async def test_returns_config_and_sources(self):
        from lilbee.app.status import StatusConfig, StatusResult

        mock_status = StatusResult(
            config=StatusConfig(
                documents_dir="docs",
                data_dir="data",
                chat_model="test:latest",
                embedding_model="embed:latest",
            ),
            sources=[],
            total_chunks=0,
        )
        with patch("lilbee.server.handlers.gather_status", return_value=mock_status):
            result = await handlers.status()
        assert result.sources == []
        assert result.total_chunks == 0

    async def test_exposes_all_four_model_roles(self):
        """/api/status config payload surfaces vision and reranker slots."""
        cfg.vision_model = ""
        cfg.reranker_model = ""
        result = await handlers.status()
        assert hasattr(result.config, "vision_model")
        assert hasattr(result.config, "reranker_model")
        assert result.config.vision_model == ""
        assert result.config.reranker_model == ""


class TestSearch:
    async def test_returns_grouped_results(self, mock_svc):
        chunk = SearchChunk(
            source="doc.pdf",
            content_type="pdf",
            chunk="hello",
            distance=0.2,
            page_start=1,
            page_end=1,
            line_start=0,
            line_end=0,
            chunk_index=0,
            vector=[0.1],
        )
        mock_svc.searcher.search.return_value = [chunk]
        result = await handlers.search("test", top_k=3)
        assert len(result) == 1
        assert result[0].source == "doc.pdf"
        mock_svc.searcher.search.assert_called_once_with("test", top_k=3, chunk_type=None)

    async def test_empty_results(self, mock_svc):
        mock_svc.searcher.search.return_value = []
        result = await handlers.search("nothing")
        assert result == []

    async def test_empty_query_raises(self):
        with pytest.raises(ValueError, match="query must not be empty"):
            await handlers.search("")

    async def test_whitespace_query_raises(self):
        with pytest.raises(ValueError, match="query must not be empty"):
            await handlers.search("   ")

    async def test_chunk_type_passed_to_searcher(self, mock_svc):
        mock_svc.searcher.search.return_value = []
        await handlers.search("test", chunk_type="wiki")
        mock_svc.searcher.search.assert_called_once_with("test", top_k=5, chunk_type="wiki")

    async def test_chunk_type_defaults_to_none(self, mock_svc):
        mock_svc.searcher.search.return_value = []
        await handlers.search("test")
        mock_svc.searcher.search.assert_called_once_with("test", top_k=5, chunk_type=None)


class TestAsk:
    async def test_returns_answer_and_sources(self, mock_svc):
        from lilbee.retrieval.query import AskResult

        mock_svc.searcher.ask_raw.return_value = AskResult(
            answer="42",
            sources=[
                SearchChunk(
                    source="doc.pdf",
                    content_type="pdf",
                    page_start=1,
                    page_end=1,
                    line_start=0,
                    line_end=0,
                    chunk="c",
                    chunk_index=0,
                    distance=0.1,
                    vector=[0.1],
                )
            ],
        )
        result = await handlers.ask("what?")
        assert result.answer == "42"
        assert len(result.sources) == 1
        assert result.sources[0].distance == 0.1

    async def test_no_sources(self, mock_svc):
        from lilbee.retrieval.query import AskResult

        mock_svc.searcher.ask_raw.return_value = AskResult(answer="No docs found.", sources=[])
        result = await handlers.ask("what?")
        assert result.answer == "No docs found."
        assert result.sources == []

    async def test_empty_question_raises(self):
        with pytest.raises(ValueError, match="question must not be empty"):
            await handlers.ask("")

    async def test_whitespace_question_raises(self):
        with pytest.raises(ValueError, match="question must not be empty"):
            await handlers.ask("   ")

    async def test_forwards_chunk_type_to_searcher(self, mock_svc):
        from lilbee.retrieval.query import AskResult

        mock_svc.searcher.ask_raw.return_value = AskResult(answer="a", sources=[])
        await handlers.ask("q", chunk_type="wiki")
        assert mock_svc.searcher.ask_raw.call_args.kwargs.get("chunk_type") == "wiki"


class TestAskStream:
    async def test_no_results_yields_error(self, mock_svc):
        mock_svc.searcher.build_rag_context.return_value = None
        events = [e async for e in handlers.ask_stream("test")]
        non_empty = [e for e in events if e]
        assert len(non_empty) == 1
        parsed = json.loads(non_empty[0].split("data: ")[1].strip())
        assert "No relevant documents found" in parsed["message"]

    async def test_yields_token_sources_done(self, mock_svc):
        mock_svc.searcher.build_rag_context.return_value = _rag_return()
        mock_svc.provider.chat.return_value = iter(["answer"])
        events = [e async for e in handlers.ask_stream("question")]

        non_empty = [e for e in events if e]
        event_types = [e.split("\n")[0].replace("event: ", "") for e in non_empty]
        assert "token" in event_types
        assert "sources" in event_types
        assert "done" in event_types

    async def test_forwards_chunk_type_to_build_rag_context(self, mock_svc):
        mock_svc.searcher.build_rag_context.return_value = None
        async for _ in handlers.ask_stream("q", chunk_type="wiki"):
            pass
        assert mock_svc.searcher.build_rag_context.call_args.kwargs.get("chunk_type") == "wiki"

    async def test_provider_error_yields_error_event(self, mock_svc):
        mock_svc.searcher.build_rag_context.return_value = _rag_return()
        mock_svc.provider.chat.side_effect = RuntimeError("model missing")
        events = [e async for e in handlers.ask_stream("question")]

        non_empty = [e for e in events if e]
        error_events = [e for e in non_empty if e.startswith("event: error")]
        assert len(error_events) == 1
        assert "Internal error" in error_events[0]
        parsed = json.loads(error_events[0].split("data: ")[1].strip())
        assert "code" not in parsed

    async def test_oom_load_error_yields_structured_code(self, mock_svc):
        mock_svc.searcher.build_rag_context.return_value = _rag_return()
        mock_svc.provider.chat.side_effect = ValueError(
            "Failed to load Qwen3-4B (2.4 GB) with n_ctx=4096. "
            "Host has 1.2 GB free RAM. Try a smaller model."
        )
        events = [e async for e in handlers.ask_stream("question")]

        non_empty = [e for e in events if e]
        error_events = [e for e in non_empty if e.startswith("event: error")]
        assert len(error_events) == 1
        parsed = json.loads(error_events[0].split("data: ")[1].strip())
        assert parsed["code"] == "model_too_large"
        assert parsed["message"] == "Model too large for available RAM"
        assert "Failed to load Qwen3-4B" in parsed["detail"]

    async def test_cancel_sets_cancel_event(self, mock_svc):
        """Closing the generator mid-stream signals the thread to stop."""
        import threading

        barrier = threading.Event()

        def blocking_chat(*args, **kwargs):
            yield "first"
            barrier.wait(timeout=2)
            yield "second"

        mock_svc.searcher.build_rag_context.return_value = _rag_return()
        mock_svc.provider.chat.side_effect = blocking_chat
        gen = handlers.ask_stream("question")
        events = []
        async for event in gen:
            events.append(event)
            if event and "first" in event:
                await gen.aclose()
                barrier.set()
                break

        non_empty = [e for e in events if e]
        assert any("first" in e for e in non_empty)
        assert not any("second" in e for e in non_empty)

    async def test_cancel_logs_message(self, mock_svc, caplog):
        """Closing the generator mid-stream logs a cancellation message."""
        import threading

        barrier = threading.Event()

        def blocking_chat(*args, **kwargs):
            yield "first"
            barrier.wait(timeout=2)
            yield "second"

        mock_svc.searcher.build_rag_context.return_value = _rag_return()
        mock_svc.provider.chat.side_effect = blocking_chat
        caplog.set_level(logging.INFO, logger="lilbee.server.handlers")
        gen = handlers.ask_stream("question")
        async for event in gen:
            if event and "first" in event:
                await gen.aclose()
                barrier.set()
                break
        # Give async generator cleanup a tick to fire the log
        await asyncio.sleep(0.05)
        assert any("cancelled by client" in r.message for r in caplog.records)

    async def test_skips_empty_tokens(self, mock_svc):
        """Empty strings from provider are not emitted."""
        mock_svc.searcher.build_rag_context.return_value = _rag_return()
        mock_svc.provider.chat.return_value = iter(["", "answer"])
        events = [e async for e in handlers.ask_stream("question")]

        non_empty = [e for e in events if e]
        token_events = [e for e in non_empty if e.startswith("event: token")]
        assert len(token_events) == 1
        assert "answer" in token_events[0]


class TestChat:
    async def test_passes_history(self, mock_svc):
        from lilbee.retrieval.query import AskResult

        mock_svc.searcher.ask_raw.return_value = AskResult(answer="ok", sources=[])
        history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        result = await handlers.chat("follow up", history)
        assert result.answer == "ok"
        mock_svc.searcher.ask_raw.assert_called_once_with(
            "follow up", top_k=0, history=history, options=None, chunk_type=None
        )

    async def test_forwards_chunk_type(self, mock_svc):
        from lilbee.retrieval.query import AskResult

        mock_svc.searcher.ask_raw.return_value = AskResult(answer="ok", sources=[])
        await handlers.chat("q", [], chunk_type="raw")
        assert mock_svc.searcher.ask_raw.call_args.kwargs.get("chunk_type") == "raw"


class TestChatStream:
    async def test_no_results_yields_error(self, mock_svc):
        mock_svc.searcher.build_rag_context.return_value = None
        events = [e async for e in handlers.chat_stream("test", [])]
        non_empty = [e for e in events if e]
        assert any("error" in e for e in non_empty)

    async def test_yields_events_with_history(self, mock_svc):
        mock_svc.searcher.build_rag_context.return_value = _rag_return()
        mock_svc.provider.chat.return_value = iter(["reply"])
        history = [{"role": "user", "content": "hi"}]
        events = [e async for e in handlers.chat_stream("follow up", history)]

        non_empty = [e for e in events if e]
        event_types = [e.split("\n")[0].replace("event: ", "") for e in non_empty]
        assert "token" in event_types
        assert "done" in event_types

    async def test_forwards_chunk_type_to_build_rag_context(self, mock_svc):
        mock_svc.searcher.build_rag_context.return_value = None
        async for _ in handlers.chat_stream("q", [], chunk_type="raw"):
            pass
        assert mock_svc.searcher.build_rag_context.call_args.kwargs.get("chunk_type") == "raw"

    async def test_provider_error_yields_error_event(self, mock_svc):
        mock_svc.searcher.build_rag_context.return_value = _rag_return()
        mock_svc.provider.chat.side_effect = ConnectionError("provider down")
        events = [e async for e in handlers.chat_stream("q", [])]

        non_empty = [e for e in events if e]
        error_events = [e for e in non_empty if e.startswith("event: error")]
        assert len(error_events) == 1
        assert "Internal error" in error_events[0]

    async def test_cancel_sets_cancel_event(self, mock_svc):
        """Closing the chat_stream generator mid-stream signals the thread to stop."""
        import threading

        barrier = threading.Event()

        def blocking_chat(*args, **kwargs):
            yield "first"
            barrier.wait(timeout=2)
            yield "second"

        mock_svc.searcher.build_rag_context.return_value = _rag_return()
        mock_svc.provider.chat.side_effect = blocking_chat
        gen = handlers.chat_stream("question", [])
        events = []
        async for event in gen:
            events.append(event)
            if event and "first" in event:
                await gen.aclose()
                barrier.set()
                break

        non_empty = [e for e in events if e]
        assert any("first" in e for e in non_empty)
        assert not any("second" in e for e in non_empty)

    async def test_cancel_logs_message(self, mock_svc, caplog):
        """Closing the chat_stream generator mid-stream logs a cancellation message."""
        import threading

        barrier = threading.Event()

        def blocking_chat(*args, **kwargs):
            yield "first"
            barrier.wait(timeout=2)
            yield "second"

        mock_svc.searcher.build_rag_context.return_value = _rag_return()
        mock_svc.provider.chat.side_effect = blocking_chat
        caplog.set_level(logging.INFO, logger="lilbee.server.handlers")
        gen = handlers.chat_stream("question", [])
        async for event in gen:
            if event and "first" in event:
                await gen.aclose()
                barrier.set()
                break
        await asyncio.sleep(0.05)
        assert any("cancelled by client" in r.message for r in caplog.records)

    async def test_skips_empty_tokens(self, mock_svc):
        """Empty strings from provider are not emitted."""
        mock_svc.searcher.build_rag_context.return_value = _rag_return()
        mock_svc.provider.chat.return_value = iter(["", "reply"])
        events = [e async for e in handlers.chat_stream("question", [])]

        non_empty = [e for e in events if e]
        token_events = [e for e in non_empty if e.startswith("event: token")]
        assert len(token_events) == 1
        assert "reply" in token_events[0]


class TestSyncStream:
    async def test_yields_progress_and_done(self):
        sync_result = SyncResult(added=["a.txt"], unchanged=0)

        async def fake_sync(force_rebuild=False, quiet=False, *, on_progress=None, cancel=None):
            if on_progress:
                from lilbee.runtime.progress import FileDoneEvent, SyncDoneEvent

                on_progress("file_done", FileDoneEvent(file="a.txt", status="ok", chunks=3))
                on_progress("done", SyncDoneEvent(added=1, updated=0, removed=0, failed=0))
            return sync_result

        with patch("lilbee.data.ingest.sync", side_effect=fake_sync):
            events = [e async for e in handlers.sync_stream()]

        non_empty = [e for e in events if e]
        # Last event should be "done" (from sync_stream itself, after sync finishes)
        done_events = [e for e in non_empty if e.startswith("event: done")]
        assert len(done_events) >= 1
        last_done = done_events[-1]
        done_data = json.loads(last_done.split("data: ")[1].strip())
        assert "a.txt" in done_data["added"]

    async def test_yields_progress_events(self):
        sync_result = SyncResult(added=["b.txt"])

        async def fake_sync(force_rebuild=False, quiet=False, *, on_progress=None, cancel=None):
            if on_progress:
                from lilbee.runtime.progress import FileDoneEvent, FileStartEvent, SyncDoneEvent

                on_progress(
                    "file_start",
                    FileStartEvent(file="b.txt", total_files=1, current_file=1),
                )
                on_progress("file_done", FileDoneEvent(file="b.txt", status="ok", chunks=2))
                on_progress("done", SyncDoneEvent(added=1, updated=0, removed=0, failed=0))
            return sync_result

        with patch("lilbee.data.ingest.sync", side_effect=fake_sync):
            events = [e async for e in handlers.sync_stream()]

        non_empty = [e for e in events if e]
        file_start_events = [e for e in non_empty if e.startswith("event: file_start")]
        assert len(file_start_events) >= 1
        data = json.loads(file_start_events[0].split("data: ")[1].strip())
        assert data["file"] == "b.txt"

    async def test_timeout_continue_when_no_progress(self):
        """Sync with no progress events exercises the TimeoutError continue branch."""
        sync_result = SyncResult()

        async def slow_sync(force_rebuild=False, quiet=False, *, on_progress=None, cancel=None):
            await asyncio.sleep(0.2)  # force at least one timeout iteration
            return sync_result

        with patch("lilbee.data.ingest.sync", side_effect=slow_sync):
            events = [e async for e in handlers.sync_stream()]

        done_events = [e for e in events if e.startswith("event: done")]
        assert len(done_events) == 1

    async def test_cancel_sets_cancel_event(self, caplog):
        """Closing the sync_stream generator signals cancel to the sync task."""
        import threading

        barrier = threading.Event()
        captured_cancel: list[threading.Event] = []

        async def blocking_sync(force_rebuild=False, quiet=False, *, on_progress=None, cancel=None):
            captured_cancel.append(cancel)
            if on_progress:
                from lilbee.runtime.progress import FileStartEvent

                on_progress(
                    "file_start", FileStartEvent(file="a.txt", total_files=1, current_file=1)
                )
            barrier.wait(timeout=2)
            return SyncResult()

        caplog.set_level(logging.INFO, logger="lilbee.server.handlers")
        with patch("lilbee.data.ingest.sync", side_effect=blocking_sync):
            gen = handlers.sync_stream()
            async for event in gen:
                if event and "file_start" in event:
                    await gen.aclose()
                    barrier.set()
                    break
            await asyncio.sleep(0.05)

        assert captured_cancel and captured_cancel[0].is_set()
        assert any("Sync stream cancelled by client" in r.message for r in caplog.records)


class TestAddFiles:
    async def test_stream_yields_events(self, isolated_env):
        """add_files_stream yields SSE events and a done event."""
        test_file = isolated_env / "documents" / "test.txt"
        test_file.write_text("test content")

        async def fake_sync(**kwargs):
            return SyncResult()

        with patch("lilbee.data.ingest.sync", side_effect=fake_sync):
            events = []
            async for event in handlers.add_files_stream({"paths": [str(test_file)]}):
                events.append(event)
            assert any("done" in e for e in events)

    async def test_stream_emits_sentinel_on_sync_failure(self, isolated_env):
        """Sync failure emits one error frame and closes the stream without a done frame."""
        test_file = isolated_env / "documents" / "test.txt"
        test_file.write_text("test content")

        async def failing_sync(**kwargs):
            raise RuntimeError("sync blew up")

        with patch("lilbee.data.ingest.sync", side_effect=failing_sync):
            events = []
            async for event in handlers.add_files_stream({"paths": [str(test_file)]}):
                events.append(event)

        error_events = [e for e in events if e.startswith("event: error")]
        done_events = [e for e in events if e.startswith("event: done")]
        assert len(error_events) == 1
        assert "sync blew up" in error_events[0]
        # No done event when the task failed.
        assert done_events == []


class TestSyncStreamDoneDelivery:
    """Regression tests for the SSE drain race (bb-7enj)."""

    async def test_done_event_delivered_on_fast_completion(self):
        """A fast-completing sync still delivers both done frames in order."""
        sync_result = SyncResult(added=["fast.txt"])

        async def instant_sync(force_rebuild=False, quiet=False, *, on_progress=None, cancel=None):
            if on_progress:
                from lilbee.runtime.progress import SyncDoneEvent

                on_progress("done", SyncDoneEvent(added=1, updated=0, removed=0, failed=0))
            return sync_result

        with patch("lilbee.data.ingest.sync", side_effect=instant_sync):
            events = [e async for e in handlers.sync_stream()]

        # The sync-emitted done (SyncDoneEvent counts) must be delivered, and
        # the handler-emitted done (SyncResult lists) must follow it.
        done_events = [e for e in events if e.startswith("event: done")]
        assert len(done_events) == 2, f"expected 2 done events, got {done_events}"

        counts_done = json.loads(done_events[0].split("data: ")[1].strip())
        lists_done = json.loads(done_events[1].split("data: ")[1].strip())
        assert counts_done == {"added": 1, "updated": 0, "removed": 0, "failed": 0, "skipped": 0}
        assert lists_done["added"] == ["fast.txt"]

    async def test_done_event_delivered_on_noop_sync(self):
        """A sync with zero file changes still emits the done frame."""
        sync_result = SyncResult()  # empty: no added/updated/removed

        async def noop_sync(force_rebuild=False, quiet=False, *, on_progress=None, cancel=None):
            if on_progress:
                from lilbee.runtime.progress import SyncDoneEvent

                on_progress("done", SyncDoneEvent(added=0, updated=0, removed=0, failed=0))
            return sync_result

        with patch("lilbee.data.ingest.sync", side_effect=noop_sync):
            events = [e async for e in handlers.sync_stream()]

        done_events = [e for e in events if e.startswith("event: done")]
        assert len(done_events) == 2
        counts = json.loads(done_events[0].split("data: ")[1].strip())
        assert counts == {"added": 0, "updated": 0, "removed": 0, "failed": 0, "skipped": 0}

    async def test_drain_exits_cleanly_when_producer_raises(self):
        """A raising producer still enqueues the sentinel via finally so drain closes."""
        from lilbee.server.handlers import SseStream

        sse = SseStream()

        async def raising_producer():
            try:
                raise RuntimeError("boom")
            finally:
                sse.queue.put_nowait(None)

        task = asyncio.create_task(raising_producer())
        events = [e async for e in sse.drain(task, "raising")]
        assert events == []
        assert task.done()
        assert isinstance(task.exception(), RuntimeError)

    async def test_stream_reports_error_when_sync_raises(self):
        """If the sync coroutine raises, sync_stream emits an error SSE frame."""

        async def failing_sync(force_rebuild=False, quiet=False, *, on_progress=None, cancel=None):
            raise RuntimeError("boom")

        with patch("lilbee.data.ingest.sync", side_effect=failing_sync):
            events = [e async for e in handlers.sync_stream()]

        error_events = [e for e in events if e.startswith("event: error")]
        done_events = [e for e in events if e.startswith("event: done")]
        assert len(error_events) == 1
        assert "boom" in error_events[0]
        # No done frame should be emitted when sync failed.
        assert done_events == []

    async def test_force_rebuild_flag_reaches_sync(self):
        """sync_stream(force_rebuild=True) plumbs the flag through to ingest.sync."""
        sync_result = SyncResult(added=[])
        observed: dict[str, bool] = {}

        async def fake_sync(force_rebuild=False, quiet=False, *, on_progress=None, cancel=None):
            observed["force_rebuild"] = force_rebuild
            return sync_result

        with patch("lilbee.data.ingest.sync", side_effect=fake_sync):
            _ = [e async for e in handlers.sync_stream(force_rebuild=True)]

        assert observed["force_rebuild"] is True

    async def test_force_rebuild_defaults_to_false(self):
        """sync_stream() with no arguments leaves the existing incremental sync semantics."""
        sync_result = SyncResult(added=[])
        observed: dict[str, bool] = {}

        async def fake_sync(force_rebuild=False, quiet=False, *, on_progress=None, cancel=None):
            observed["force_rebuild"] = force_rebuild
            return sync_result

        with patch("lilbee.data.ingest.sync", side_effect=fake_sync):
            _ = [e async for e in handlers.sync_stream()]

        assert observed["force_rebuild"] is False


class TestDrainFallback:
    async def test_drain_exits_when_task_done_without_sentinel(self):
        """drain() exits cleanly when a task finishes with no sentinel."""
        from lilbee.server.handlers import SseStream

        sse = SseStream()

        async def quiet_producer():
            # Never emits anything; simulates a crashed or misbehaving producer.
            return None

        task = asyncio.create_task(quiet_producer())
        events = [e async for e in sse.drain(task, "no-sentinel test")]
        assert events == []
        assert task.done()


class TestDrainHeartbeat:
    """Root cause for obsidian-lilbee-v8y: a long-running producer (vision OCR
    on an image-heavy PDF) keeps the queue empty for longer than the plugin's
    120s STREAM_IDLE_TIMEOUT_MS, so drain() must emit heartbeat events while
    the producer is alive but silent."""

    async def test_heartbeat_emitted_while_queue_idle(self):
        """drain() emits heartbeat events while the producer is silent."""
        from lilbee.server.handlers import SseStream

        cfg.sse_heartbeat_interval = 0.05
        sse = SseStream()

        async def slow_producer():
            await asyncio.sleep(0.25)
            sse.queue.put_nowait("event: progress\ndata: {}\n\n")
            sse.queue.put_nowait(None)

        task = asyncio.create_task(slow_producer())
        events = [e async for e in sse.drain(task, "slow")]

        heartbeats = [e for e in events if e.startswith("event: heartbeat")]
        progress = [e for e in events if e.startswith("event: progress")]
        assert heartbeats, "expected at least one heartbeat while producer was idle"
        assert progress == ["event: progress\ndata: {}\n\n"]
        # Heartbeat payload must be JSON with a monotonic timestamp.
        ts_payload = json.loads(heartbeats[0].split("data: ")[1].strip())
        assert isinstance(ts_payload.get("ts"), int | float)

    async def test_heartbeat_disabled_when_interval_is_zero(self):
        """``cfg.sse_heartbeat_interval == 0`` opts out of heartbeats."""
        from lilbee.server.handlers import SseStream

        cfg.sse_heartbeat_interval = 0.0
        sse = SseStream()

        async def slow_producer():
            await asyncio.sleep(0.25)
            sse.queue.put_nowait(None)

        task = asyncio.create_task(slow_producer())
        events = [e async for e in sse.drain(task, "disabled")]
        assert not any(e.startswith("event: heartbeat") for e in events)

    async def test_heartbeat_skipped_when_producer_emits_faster_than_interval(self):
        """A chatty producer never triggers a heartbeat."""
        from lilbee.server.handlers import SseStream

        cfg.sse_heartbeat_interval = 5.0
        sse = SseStream()

        async def chatty_producer():
            for i in range(3):
                sse.queue.put_nowait(f'event: progress\ndata: {{"i": {i}}}\n\n')
                await asyncio.sleep(0.01)
            sse.queue.put_nowait(None)

        task = asyncio.create_task(chatty_producer())
        events = [e async for e in sse.drain(task, "chatty")]

        heartbeats = [e for e in events if e.startswith("event: heartbeat")]
        assert heartbeats == []
        progress = [e for e in events if e.startswith("event: progress")]
        assert len(progress) == 3


class TestListModels:
    @patch("lilbee.server.handlers.models.get_services")
    async def test_returns_catalogs(self, mock_get_mm):
        installed = "Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"
        mock_get_mm.return_value.model_manager.list_installed.return_value = [installed]
        result = await handlers.list_models()

        assert result.chat.active == cfg.chat_model
        assert isinstance(result.chat.catalog, list)
        assert len(result.chat.catalog) > 0
        assert installed in result.chat.installed

    @patch("lilbee.server.handlers.models.get_services")
    async def test_installed_flag_in_catalog(self, mock_get_mm):
        installed = "Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"
        mock_get_mm.return_value.model_manager.list_installed.return_value = [installed]
        result = await handlers.list_models()

        catalog = result.chat.catalog
        qwen_entry = next(m for m in catalog if "Qwen3 0.6B" in m.name)
        assert qwen_entry.installed is True

        mistral_entry = next(m for m in catalog if "Mistral" in m.name)
        assert mistral_entry.installed is False

    @patch("lilbee.server.handlers.models.get_services")
    async def test_reranker_installed_detected_via_registry(self, mock_get_mm):
        """Rerankers install as GGUFs like chat/embedding; the ref match marks them installed."""
        from lilbee.catalog import FEATURED_RERANK

        bge = FEATURED_RERANK[0]
        mock_get_mm.return_value.model_manager.list_installed.return_value = [bge.ref]
        result = await handlers.list_models()
        bge_entry = next(m for m in result.reranker.catalog if bge.display_name in m.name)
        assert bge_entry.installed is True

    @patch("lilbee.server.handlers.models.get_services")
    async def test_embedding_installed_detected_via_registry(self, mock_get_mm):
        """Embedding installs surface via the same registry path as chat/reranker."""
        from lilbee.catalog import FEATURED_EMBEDDING

        entry = FEATURED_EMBEDDING[0]
        mock_get_mm.return_value.model_manager.list_installed.return_value = [entry.ref]
        result = await handlers.list_models()
        embed_entry = next(m for m in result.embedding.catalog if entry.display_name in m.name)
        assert embed_entry.installed is True

    @patch("lilbee.server.handlers.models.get_services")
    async def test_returns_vision_and_reranker_sections(self, mock_get_mm):
        """list_models exposes chat, embedding, vision, and reranker catalogs."""
        mock_get_mm.return_value.model_manager.list_installed.return_value = []
        cfg.vision_model = ""
        cfg.reranker_model = ""
        result = await handlers.list_models()

        assert result.vision.active == ""
        assert result.reranker.active == ""
        assert len(result.vision.catalog) > 0
        assert len(result.reranker.catalog) > 0
        assert result.embedding.active == cfg.embedding_model
        assert len(result.embedding.catalog) > 0


class TestSetChatModel:
    async def test_updates_config_and_persists(self, tmp_path, mock_svc):
        mock_svc.provider.list_models.return_value = [_CHAT_REF]
        result = await handlers.set_chat_model(_CHAT_REF)
        assert result.model == _CHAT_REF
        assert cfg.chat_model == _CHAT_REF

    async def test_rejects_unavailable_model(self, tmp_path, mock_svc):
        mock_svc.provider.list_models.return_value = [_CHAT_REF]
        with pytest.raises(ValueError, match="not available"):
            await handlers.set_chat_model("org/Nonexistent-GGUF/none.gguf")

    async def test_accepts_ollama_prefixed_ref(self, tmp_path, mock_svc):
        """``ollama/<name>:tag`` is accepted verbatim when the backend reports it."""
        mock_svc.provider.list_models.return_value = ["ollama/qwen3:0.6b"]
        result = await handlers.set_chat_model("ollama/qwen3:0.6b")
        assert result.model == "ollama/qwen3:0.6b"
        assert cfg.chat_model == "ollama/qwen3:0.6b"

    async def test_resolves_bare_repo_to_installed_quant(self, tmp_path, mock_svc):
        """Bare ``hf_repo`` resolves to whichever quant of that repo is installed."""
        mock_svc.provider.list_models.return_value = [_CHAT_REF]
        result = await handlers.set_chat_model("Qwen/Qwen3-0.6B-GGUF")
        assert result.model == _CHAT_REF
        assert cfg.chat_model == _CHAT_REF

    async def test_rejects_out_of_catalog_model(self, tmp_path, mock_svc):
        """Out-of-catalog installed models cannot be assigned to any role."""
        custom = "org/Custom-GGUF/custom.gguf"
        mock_svc.provider.list_models.return_value = [custom]
        with pytest.raises(ValueError, match="featured catalog"):
            await handlers.set_chat_model(custom)

    async def test_rejects_vision_model_on_chat_endpoint(self, tmp_path, mock_svc):
        """Setting a vision-task model on the chat endpoint returns 422 semantics."""
        mock_svc.provider.list_models.return_value = [_VISION_REF]
        with pytest.raises(ValueError, match="not chat"):
            await handlers.set_chat_model(_VISION_REF)

    async def test_rejects_unparseable_ref(self, tmp_path, mock_svc):
        """An unparseable bare ``name:tag`` errors with the standard not-available message."""
        mock_svc.provider.list_models.return_value = [_CHAT_REF]
        with pytest.raises(ValueError, match="not available"):
            await handlers.set_chat_model("qwen3:0.6b")

    async def test_provider_prefixed_bare_form_resolves_via_parse(self, tmp_path, mock_svc):
        """An ``ollama/<name>`` resolves to bare ``<name>`` when that's what list_models returns."""
        mock_svc.provider.list_models.return_value = ["qwen3:0.6b"]
        with pytest.raises(ValueError, match="featured catalog"):
            # parse path: "ollama/qwen3:0.6b" -> name="qwen3:0.6b" which is in
            # available; the downstream catalog check rejects it for chat role.
            await handlers.set_chat_model("ollama/qwen3:0.6b")


class TestModelsCatalog:
    @staticmethod
    def _manifest(hf_repo: str, gguf_filename: str, task: str = "chat"):
        from lilbee.modelhub.registry import ModelManifest

        return ModelManifest(
            hf_repo=hf_repo,
            gguf_filename=gguf_filename,
            size_bytes=1,
            task=task,
            downloaded_at="2026-01-01T00:00:00+00:00",
        )

    @patch("lilbee.server.handlers.models.get_catalog")
    async def test_returns_catalog_response(self, mock_get_catalog, mock_svc):
        from lilbee.catalog import CatalogModel, CatalogResult

        mock_get_catalog.return_value = CatalogResult(
            total=1,
            limit=20,
            offset=0,
            models=[
                CatalogModel(
                    hf_repo="Qwen/Qwen3-8B-GGUF",
                    gguf_filename="*Q4_K_M.gguf",
                    size_gb=5.0,
                    min_ram_gb=8.0,
                    description="Medium model",
                    featured=True,
                    downloads=1000,
                    task="chat",
                )
            ],
        )
        mock_svc.registry.list_installed.return_value = [
            self._manifest("Qwen/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf")
        ]
        result = await handlers.models_catalog()

        assert result.total == 1
        assert result.has_more is False
        assert len(result.models) == 1
        m = result.models[0]
        assert m.hf_repo == "Qwen/Qwen3-8B-GGUF"
        assert m.task == "chat"
        assert m.featured is True
        assert m.downloads == 1000
        assert m.param_count == "8B"
        assert m.installed is True
        assert m.source == "native"

    @patch("lilbee.server.handlers.models.get_catalog")
    async def test_filters_passed_to_catalog(self, mock_get_catalog, mock_svc):
        from lilbee.catalog import CatalogResult

        mock_get_catalog.return_value = CatalogResult(total=0, limit=10, offset=5, models=[])
        mock_svc.registry.list_installed.return_value = []
        await handlers.models_catalog(
            task="chat",
            search="qwen",
            size="small",
            installed=True,
            featured=True,
            sort="downloads",
            limit=10,
            offset=5,
        )
        mock_get_catalog.assert_called_once_with(
            task="chat",
            search="qwen",
            size="small",
            installed=True,
            featured=True,
            sort="downloads",
            limit=10,
            offset=5,
        )

    @patch("lilbee.server.handlers.models.get_catalog")
    async def test_installed_flag(self, mock_get_catalog, mock_svc):
        from lilbee.catalog import CatalogModel, CatalogResult

        mock_get_catalog.return_value = CatalogResult(
            total=1,
            limit=20,
            offset=0,
            models=[
                CatalogModel(
                    hf_repo="Qwen/Qwen3-8B-GGUF",
                    gguf_filename="*Q4_K_M.gguf",
                    size_gb=5.0,
                    min_ram_gb=8.0,
                    description="test",
                    featured=True,
                    downloads=0,
                    task="chat",
                )
            ],
        )
        mock_svc.registry.list_installed.return_value = [
            self._manifest("Qwen/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf")
        ]
        result = await handlers.models_catalog()
        assert result.models[0].installed is True

    @patch("lilbee.server.handlers.models.get_catalog")
    async def test_has_more_propagated(self, mock_get_catalog, mock_svc):
        from lilbee.catalog import CatalogResult

        mock_get_catalog.return_value = CatalogResult(
            total=0, limit=20, offset=0, models=[], has_more=True
        )
        mock_svc.registry.list_installed.return_value = []
        result = await handlers.models_catalog()
        assert result.has_more is True

    @patch("lilbee.server.handlers.models.get_catalog")
    async def test_installed_reflects_registry_not_routing_provider(
        self, mock_get_catalog, mock_svc
    ):
        """installed flag must reflect on-disk GGUFs only, not litellm routability.

        The routing provider's ``list_models()`` can include models that are
        routable via litellm proxies but have no GGUF on disk. The catalog's
        ``installed`` field should only be True when a ModelManifest exists
        in the registry for that ref.
        """
        from lilbee.catalog import CatalogModel, CatalogResult
        from lilbee.modelhub.registry import ModelManifest

        mock_get_catalog.return_value = CatalogResult(
            total=2,
            limit=20,
            offset=0,
            models=[
                CatalogModel(
                    hf_repo="Qwen/Qwen3-8B-GGUF",
                    gguf_filename="*Q4_K_M.gguf",
                    size_gb=5.0,
                    min_ram_gb=8.0,
                    description="on disk",
                    featured=True,
                    downloads=0,
                    task="chat",
                ),
                CatalogModel(
                    hf_repo="TheBloke/Mistral-7B-GGUF",
                    gguf_filename="*Q4_K_M.gguf",
                    size_gb=4.0,
                    min_ram_gb=8.0,
                    description="routable only",
                    featured=False,
                    downloads=0,
                    task="chat",
                ),
            ],
        )
        # Registry has only qwen3 on disk.
        mock_svc.registry.list_installed.return_value = [
            ModelManifest(
                hf_repo="Qwen/Qwen3-8B-GGUF",
                gguf_filename="Qwen3-8B-Q4_K_M.gguf",
                size_bytes=1,
                task="chat",
                downloaded_at="2026-01-01T00:00:00+00:00",
            )
        ]
        # Routing provider also claims mistral is routable (litellm proxy),
        # but its GGUF is not on disk.
        mock_svc.provider.list_models.return_value = [
            "Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf",
            "ollama/mistral:7b",
        ]

        result = await handlers.models_catalog()

        by_repo = {m.hf_repo: m for m in result.models}
        assert by_repo["Qwen/Qwen3-8B-GGUF"].installed is True
        assert by_repo["TheBloke/Mistral-7B-GGUF"].installed is False

    @patch("lilbee.server.handlers.models.available_memory_for_fit", return_value=16 * 1024**3)
    @patch("lilbee.server.handlers.models.get_catalog")
    async def test_fit_set_for_native_installed_row(self, mock_get_catalog, _mock_mem, mock_svc):
        """A native row with a known footprint gets a server-computed fit level."""
        from lilbee.catalog import CatalogModel, CatalogResult
        from lilbee.runtime.hardware import FitLevel

        mock_get_catalog.return_value = CatalogResult(
            total=1,
            limit=20,
            offset=0,
            models=[
                CatalogModel(
                    hf_repo="Qwen/Qwen3-8B-GGUF",
                    gguf_filename="*Q4_K_M.gguf",
                    size_gb=5.0,
                    min_ram_gb=8.0,
                    description="",
                    featured=True,
                    downloads=0,
                    task="chat",
                )
            ],
        )
        mock_svc.registry.list_installed.return_value = [
            self._manifest("Qwen/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf")
        ]
        result = await handlers.models_catalog()
        assert result.models[0].fit is FitLevel.FITS

    @patch("lilbee.server.handlers.models.available_memory_for_fit", return_value=4 * 1024**3)
    @patch("lilbee.server.handlers.models.get_catalog")
    async def test_fit_set_for_native_not_installed_row(
        self, mock_get_catalog, _mock_mem, mock_svc
    ):
        """A native, not-yet-installed row still carries fit so users see it before pulling."""
        from lilbee.catalog import CatalogModel, CatalogResult
        from lilbee.runtime.hardware import FitLevel

        mock_get_catalog.return_value = CatalogResult(
            total=1,
            limit=20,
            offset=0,
            models=[
                CatalogModel(
                    hf_repo="Qwen/Qwen3-8B-GGUF",
                    gguf_filename="*Q4_K_M.gguf",
                    size_gb=5.0,
                    min_ram_gb=8.0,
                    description="",
                    featured=True,
                    downloads=0,
                    task="chat",
                )
            ],
        )
        mock_svc.registry.list_installed.return_value = []
        result = await handlers.models_catalog()
        assert result.models[0].installed is False
        assert result.models[0].fit is FitLevel.WONT_RUN

    @patch("lilbee.server.handlers.models.available_memory_for_fit", return_value=None)
    @patch("lilbee.server.handlers.models.get_catalog")
    async def test_fit_none_when_memory_probe_unavailable(
        self, mock_get_catalog, _mock_mem, mock_svc
    ):
        """When the host probe fails, every row degrades to fit=None instead of guessing."""
        from lilbee.catalog import CatalogModel, CatalogResult

        mock_get_catalog.return_value = CatalogResult(
            total=1,
            limit=20,
            offset=0,
            models=[
                CatalogModel(
                    hf_repo="Qwen/Qwen3-8B-GGUF",
                    gguf_filename="*Q4_K_M.gguf",
                    size_gb=5.0,
                    min_ram_gb=8.0,
                    description="",
                    featured=True,
                    downloads=0,
                    task="chat",
                )
            ],
        )
        mock_svc.registry.list_installed.return_value = []
        result = await handlers.models_catalog()
        assert result.models[0].fit is None

    @patch("lilbee.server.handlers.models.get_catalog")
    async def test_fit_none_when_size_unknown(self, mock_get_catalog, mock_svc):
        """Rows without a resolved footprint cannot be assessed against host memory."""
        from lilbee.catalog import CatalogModel, CatalogResult

        mock_get_catalog.return_value = CatalogResult(
            total=1,
            limit=20,
            offset=0,
            models=[
                CatalogModel(
                    hf_repo="adhoc/repo",
                    gguf_filename="*.gguf",
                    size_gb=0.0,
                    min_ram_gb=2.0,
                    description="",
                    featured=False,
                    downloads=0,
                    task="chat",
                )
            ],
        )
        mock_svc.registry.list_installed.return_value = []
        result = await handlers.models_catalog()
        assert result.models[0].fit is None

    @patch("lilbee.server.handlers.models.get_catalog")
    async def test_size_variants_attached_for_featured_family(self, mock_get_catalog, mock_svc):
        """A featured row exposes its sibling family variants on the response."""
        from lilbee.catalog import FEATURED_CHAT, CatalogResult

        # Pick the first featured chat repo; its family should yield at least
        # one SizeVariantInfo regardless of how many siblings it has.
        anchor = FEATURED_CHAT[0]
        mock_get_catalog.return_value = CatalogResult(total=1, limit=20, offset=0, models=[anchor])
        mock_svc.registry.list_installed.return_value = []
        result = await handlers.models_catalog()
        variants = result.models[0].size_variants
        assert len(variants) >= 1
        assert all(v.size_gb >= 0 for v in variants)
        assert anchor.hf_repo in {v.ref for v in variants}

    @patch("lilbee.server.handlers.models.get_catalog")
    async def test_size_variants_empty_for_non_family_row(self, mock_get_catalog, mock_svc):
        """An ad-hoc HF row with no family grouping returns an empty variant strip."""
        from lilbee.catalog import CatalogModel, CatalogResult

        mock_get_catalog.return_value = CatalogResult(
            total=1,
            limit=20,
            offset=0,
            models=[
                CatalogModel(
                    hf_repo="adhoc/some-repo",
                    gguf_filename="*.gguf",
                    size_gb=2.0,
                    min_ram_gb=4.0,
                    description="",
                    featured=False,
                    downloads=0,
                    task="chat",
                )
            ],
        )
        mock_svc.registry.list_installed.return_value = []
        result = await handlers.models_catalog()
        assert result.models[0].size_variants == []

    @patch("lilbee.server.handlers.models.get_catalog")
    async def test_fit_none_for_non_native_source(self, mock_get_catalog, mock_svc):
        """Non-native (off-host) rows leave fit unset since the footprint isn't local.

        ``enrich_catalog`` currently always emits ``source="native"``, but the
        handler must still return ``fit=None`` for any row whose source is not
        native so a future router that mixes cloud rows in this response keeps
        the contract (cloud rows have no on-host footprint to assess).
        """
        from dataclasses import replace

        from lilbee.catalog import CatalogModel, CatalogResult, enrich_catalog
        from lilbee.catalog.types import ModelSource

        mock_get_catalog.return_value = CatalogResult(
            total=1,
            limit=20,
            offset=0,
            models=[
                CatalogModel(
                    hf_repo="cloud/example",
                    gguf_filename="",
                    size_gb=0.0,
                    min_ram_gb=0.0,
                    description="cloud chat model",
                    featured=False,
                    downloads=0,
                    task="chat",
                )
            ],
        )
        mock_svc.registry.list_installed.return_value = []

        def _enrich_as_remote(result, installed_refs):
            enriched = enrich_catalog(result, installed_refs)
            return [replace(e, source=ModelSource.REMOTE.value) for e in enriched]

        with patch(
            "lilbee.server.handlers.models.enrich_catalog",
            side_effect=_enrich_as_remote,
        ):
            result = await handlers.models_catalog()
        assert result.models[0].source == "remote"
        assert result.models[0].fit is None


class TestModelsInstalled:
    async def test_returns_installed_models(self):
        mock_manager = MagicMock()
        mock_manager.list_installed.return_value = ["ollama/qwen3:8b", "ollama/mistral:7b"]
        from lilbee.catalog.types import ModelSource

        mock_manager.get_source.return_value = ModelSource.REMOTE
        with patch(
            "lilbee.server.handlers.models.get_services",
            return_value=MagicMock(model_manager=mock_manager),
        ):
            result = await handlers.models_installed()
        assert len(result.models) == 2
        assert result.models[0].source == "remote"

    async def test_unknown_source_defaults_to_litellm(self):
        mock_manager = MagicMock()
        mock_manager.list_installed.return_value = ["unknown"]
        mock_manager.get_source.return_value = None
        with patch(
            "lilbee.server.handlers.models.get_services",
            return_value=MagicMock(model_manager=mock_manager),
        ):
            result = await handlers.models_installed()
        assert result.models[0].source == "remote"


class TestModelsPull:
    async def test_yields_progress_events_native(self):
        mock_manager = MagicMock()

        def fake_pull(model, source, *, on_progress=None, on_bytes=None):
            if on_bytes:
                on_bytes(500, 1000)
                on_bytes(1000, 1000)
            return

        mock_manager.pull.side_effect = fake_pull
        with patch(
            "lilbee.server.handlers.models.get_services",
            return_value=MagicMock(model_manager=mock_manager),
        ):
            events = [e async for e in handlers.models_pull("test", source="native")]
        non_empty = [e for e in events if e]
        assert any('"current": 500' in e for e in non_empty)
        assert any('"total": 1000' in e for e in non_empty)

    async def test_yields_progress_events_litellm(self):
        """Litellm pulls use on_progress (dict), not on_bytes (int, int)."""
        mock_manager = MagicMock()

        def fake_pull(model, source, *, on_progress=None, on_bytes=None):
            if on_progress:
                on_progress({"status": "downloading"})
                on_progress({"status": "success"})
            return

        mock_manager.pull.side_effect = fake_pull
        with patch(
            "lilbee.server.handlers.models.get_services",
            return_value=MagicMock(model_manager=mock_manager),
        ):
            events = [e async for e in handlers.models_pull("test", source="remote")]
        non_empty = [e for e in events if e]
        assert any("downloading" in e for e in non_empty)
        assert any("success" in e for e in non_empty)

    async def test_error_yields_error_event(self):
        mock_manager = MagicMock()
        mock_manager.pull.side_effect = RuntimeError("fail")
        with patch(
            "lilbee.server.handlers.models.get_services",
            return_value=MagicMock(model_manager=mock_manager),
        ):
            events = [e async for e in handlers.models_pull("bad", source="native")]
        non_empty = [e for e in events if e]
        assert any("error" in e and "fail" in e for e in non_empty)

    async def test_cancel_stops_pull(self, caplog):
        """Closing the pull generator mid-stream sets cancel and logs."""
        import threading

        barrier = threading.Event()
        mock_manager = MagicMock()

        def blocking_pull(model, source, *, on_progress=None, on_bytes=None):
            if on_bytes:
                on_bytes(100, 1000)
            barrier.wait(timeout=2)
            if on_bytes:
                on_bytes(1000, 1000)

        mock_manager.pull.side_effect = blocking_pull
        caplog.set_level(logging.INFO, logger="lilbee.server.handlers")
        with patch(
            "lilbee.server.handlers.models.get_services",
            return_value=MagicMock(model_manager=mock_manager),
        ):
            gen = handlers.models_pull("test", source="native")
            async for event in gen:
                if event and "current" in event:
                    await gen.aclose()
                    barrier.set()
                    break
            await asyncio.sleep(0.05)

        assert any("Model pull stream cancelled by client" in r.message for r in caplog.records)


class TestModelsDelete:
    async def test_returns_deleted_true(self):
        mock_manager = MagicMock()
        mock_manager.remove.return_value = True
        with patch(
            "lilbee.server.handlers.models.get_services",
            return_value=MagicMock(model_manager=mock_manager),
        ):
            result = await handlers.models_delete("test", source="remote")
        assert result.deleted is True
        assert result.model == "test"

    async def test_returns_deleted_false(self):
        mock_manager = MagicMock()
        mock_manager.remove.return_value = False
        with patch(
            "lilbee.server.handlers.models.get_services",
            return_value=MagicMock(model_manager=mock_manager),
        ):
            result = await handlers.models_delete("missing", source="native")
        assert result.deleted is False
        assert result.freed_gb == 0.0


class TestModelsShow:
    async def test_returns_params(self, mock_svc):
        mock_svc.provider.show_model.return_value = {"parameters": "temp 0.7"}
        result = await handlers.models_show("ollama/qwen3:8b")
        assert result.model_dump() == {"parameters": "temp 0.7"}

    async def test_returns_empty_when_none(self, mock_svc):
        mock_svc.provider.show_model.return_value = None
        result = await handlers.models_show("unknown")
        assert result.model_dump() == {}


class TestDeleteDocuments:
    async def test_removes_known_documents(self, mock_svc):
        from lilbee.data.store import RemoveResult

        mock_svc.store.remove_documents.return_value = RemoveResult(removed=["a.md"], not_found=[])
        result = await handlers.delete_documents(["a.md"])
        assert result.removed == ["a.md"]
        assert result.not_found == []

    async def test_not_found(self, mock_svc):
        from lilbee.data.store import RemoveResult

        mock_svc.store.remove_documents.return_value = RemoveResult(
            removed=[], not_found=["missing.md"]
        )
        result = await handlers.delete_documents(["missing.md"])
        assert result.removed == []
        assert result.not_found == ["missing.md"]

    async def test_delete_files_removes_from_disk(self, mock_svc, tmp_path):
        from lilbee.data.store import RemoveResult

        cfg.documents_dir = tmp_path
        f = tmp_path / "a.md"
        f.write_text("content")

        def fake_remove(names, *, delete_files=False):
            if delete_files and f.exists():
                f.unlink()
            return RemoveResult(removed=["a.md"], not_found=[])

        mock_svc.store.remove_documents.side_effect = fake_remove
        result = await handlers.delete_documents(["a.md"], delete_files=True)
        assert result.removed == ["a.md"]
        assert not f.exists()


class TestUpdateConfig:
    async def test_update_config_valid(self, tmp_path):
        result = await handlers.update_config({"temperature": 0.7})
        assert result.updated == ["temperature"]
        assert result.reindex_required is False
        assert cfg.temperature == 0.7

    async def test_update_config_reindex(self, tmp_path):
        result = await handlers.update_config({"chunk_size": 1024})
        assert result.reindex_required is True
        assert cfg.chunk_size == 1024

    async def test_update_config_null_reset(self, tmp_path):
        cfg.temperature = 0.5
        result = await handlers.update_config({"temperature": None})
        assert result.updated == ["temperature"]
        assert cfg.temperature is None
        # Verify delete_value was called (file should not contain temperature)
        from lilbee.core import settings as s

        stored = s.load(cfg.data_root)
        assert "temperature" not in stored

    async def test_update_config_unknown_field(self):
        with pytest.raises(ValueError, match="Unknown or read-only config field"):
            await handlers.update_config({"bogus_field": 123})

    async def test_non_nullable_field_rejects_null(self):
        with pytest.raises(ValueError, match="does not accept null"):
            await handlers.update_config({"rag_system_prompt": None})

    async def test_empty_dict_returns_no_updates(self):
        result = await handlers.update_config({})
        assert result.updated == []
        assert result.reindex_required is False

    async def test_invalid_type_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            await handlers.update_config({"chunk_size": "not_a_number"})

    async def test_chunk_size_below_minimum_rejected(self):
        with pytest.raises(ValueError, match="chunk_size must be >= 64"):
            await handlers.update_config({"chunk_size": 5})

    async def test_chunk_size_at_minimum_accepted(self, tmp_path):
        result = await handlers.update_config({"chunk_size": 64})
        assert result.updated == ["chunk_size"]
        assert cfg.chunk_size == 64

    async def test_llm_api_key_write_only(self, tmp_path):
        """llm_api_key can be written via PATCH but is excluded from GET."""
        result = await handlers.update_config({"llm_api_key": "sk-test123"})
        assert result.updated == ["llm_api_key"]
        assert cfg.llm_api_key == "sk-test123"
        # Verify it's excluded from GET /api/config
        config = await handlers.get_config()
        assert "llm_api_key" not in config.model_dump()

    async def test_wiki_toggle_via_patch(self, tmp_path):
        """The wiki feature flag must round-trip through PATCH /api/config.

        settings_map exposes it under /settings and docs advertise
        LILBEE_WIKI as a supported override; the HTTP surface has to
        accept it too or the three entry points drift out of parity.
        """
        cfg.wiki = True
        result = await handlers.update_config({"wiki": False})
        assert result.updated == ["wiki"]
        assert cfg.wiki is False
        # And back on.
        result = await handlers.update_config({"wiki": True})
        assert result.updated == ["wiki"]
        assert cfg.wiki is True

    async def test_multi_field_bad_second_no_partial_apply(self):
        """If second field is invalid, first field should NOT be applied."""
        original_temp = cfg.temperature
        with pytest.raises(ValueError, match="does not accept null"):
            await handlers.update_config({"temperature": 0.9, "rag_system_prompt": None})
        # temperature should be unchanged: validation happens before apply
        assert cfg.temperature == original_temp

    async def test_multi_field_success(self, tmp_path):
        """Multiple valid fields are applied and persisted in one call."""
        result = await handlers.update_config({"temperature": 0.7, "top_k": 5})
        assert set(result.updated) == {"temperature", "top_k"}
        assert result.reindex_required is False
        assert cfg.temperature == 0.7
        assert cfg.top_k == 5
        # Verify both persisted
        from lilbee.core import settings as s

        stored = s.load(cfg.data_root)
        assert stored["temperature"] == "0.7"
        assert stored["top_k"] == "5"

    async def test_api_key_update_injects_provider_keys(self, tmp_path):
        with patch("lilbee.server.handlers.config.inject_provider_keys") as mock_inject:
            result = await handlers.update_config({"openai_api_key": "sk-new"})
        assert result.updated == ["openai_api_key"]
        mock_inject.assert_called_once()

    async def test_non_key_update_skips_injection(self, tmp_path):
        with patch("lilbee.server.handlers.config.inject_provider_keys") as mock_inject:
            await handlers.update_config({"temperature": 0.5})
        mock_inject.assert_not_called()

    async def test_update_config_rejects_chat_model(self):
        """Role fields only move via PUT /api/models/<role>; PATCH must 422."""
        with pytest.raises(ValueError, match="read-only"):
            await handlers.update_config({"chat_model": "Qwen/Qwen3-0.6B-GGUF"})

    async def test_update_config_rejects_embedding_model(self):
        with pytest.raises(ValueError, match="read-only"):
            await handlers.update_config({"embedding_model": "nomic-ai/nomic-embed-text-v1.5-GGUF"})

    async def test_update_config_rejects_vision_model(self):
        with pytest.raises(ValueError, match="read-only"):
            await handlers.update_config({"vision_model": "lightonai/LightOnOCR-2.1B-GGUF"})

    async def test_update_config_rejects_reranker_model(self):
        with pytest.raises(ValueError, match="read-only"):
            await handlers.update_config(
                {"reranker_model": "ggml-org/bge-reranker-v2-m3-Q8_0-GGUF"}
            )


_EMBED_REF = "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf"
_CHAT_REF = "Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"


class TestSetEmbeddingModel:
    @patch("lilbee.server.handlers.models.get_services")
    async def test_updates_config_and_persists(self, mock_svc, tmp_path):
        mock_svc.return_value.provider.list_models.return_value = [_EMBED_REF]
        result = await handlers.set_embedding_model(_EMBED_REF)
        assert result.model == _EMBED_REF
        assert cfg.embedding_model == _EMBED_REF
        from lilbee.core import settings as s

        stored = s.load(cfg.data_root)
        assert stored["embedding_model"] == _EMBED_REF

    @patch("lilbee.server.handlers.models.get_services")
    async def test_empty_string_rejected(self, mock_svc):
        mock_svc.return_value.provider.list_models.return_value = []
        with pytest.raises(ValueError, match="not available"):
            await handlers.set_embedding_model("")

    @patch("lilbee.server.handlers.models.get_services")
    async def test_embedding_model_bare_repo_resolves_to_installed_quant(self, mock_svc, tmp_path):
        """Setting an embedding by bare ``hf_repo`` resolves to whichever
        quant of that repo is currently installed.
        """
        mock_svc.return_value.provider.list_models.return_value = [_EMBED_REF]
        repo = "nomic-ai/nomic-embed-text-v1.5-GGUF"
        result = await handlers.set_embedding_model(repo)
        assert result.model == _EMBED_REF
        assert cfg.embedding_model == _EMBED_REF

    @patch("lilbee.server.handlers.models.get_services")
    async def test_rejects_unavailable_embedding_model(self, mock_svc):
        mock_svc.return_value.provider.list_models.return_value = [_EMBED_REF]
        with pytest.raises(ValueError, match="not available"):
            await handlers.set_embedding_model("org/Bogus-GGUF/bogus.gguf")

    @patch("lilbee.server.handlers.models.get_services")
    async def test_rejects_out_of_catalog_model(self, mock_svc):
        """Non-catalog installed model cannot be assigned to the embedding role."""
        custom = "org/Custom-Embed-GGUF/custom.gguf"
        mock_svc.return_value.provider.list_models.return_value = [custom]
        with pytest.raises(ValueError, match="featured catalog"):
            await handlers.set_embedding_model(custom)

    @patch("lilbee.server.handlers.models.get_services")
    async def test_rejects_chat_model_on_embedding_endpoint(self, mock_svc):
        """Setting a chat-task model as embedding returns a task-mismatch error."""
        mock_svc.return_value.provider.list_models.return_value = [_CHAT_REF]
        with pytest.raises(ValueError, match="not embedding"):
            await handlers.set_embedding_model(_CHAT_REF)

    @patch("lilbee.server.handlers.models.get_services")
    async def test_returns_reindex_required_when_persisted_meta_differs(self, mock_svc):
        """Switching to a model different from the one that built the store flags rebuild."""
        mock_svc.return_value.provider.list_models.return_value = [_EMBED_REF]
        mock_svc.return_value.store.get_meta.return_value = {
            "embedding_model": "previous-model:v1",
            "embedding_dim": 768,
            "schema_version": 1,
            "updated_at": "2026-04-25T00:00:00+00:00",
        }
        result = await handlers.set_embedding_model(_EMBED_REF)
        assert result.model == _EMBED_REF
        assert result.reindex_required is True

    @patch("lilbee.server.handlers.models.get_services")
    async def test_returns_no_reindex_required_on_empty_store(self, mock_svc):
        """A store with no _meta row (fresh install) does not need a rebuild."""
        mock_svc.return_value.provider.list_models.return_value = [_EMBED_REF]
        mock_svc.return_value.store.get_meta.return_value = None
        result = await handlers.set_embedding_model(_EMBED_REF)
        assert result.reindex_required is False

    @patch("lilbee.server.handlers.models.get_services")
    async def test_returns_no_reindex_required_when_same_model(self, mock_svc):
        """Re-setting the same model that already built the store does not need a rebuild."""
        mock_svc.return_value.provider.list_models.return_value = [_EMBED_REF]
        mock_svc.return_value.store.get_meta.return_value = {
            "embedding_model": _EMBED_REF,
            "embedding_dim": 768,
            "schema_version": 1,
            "updated_at": "2026-04-25T00:00:00+00:00",
        }
        result = await handlers.set_embedding_model(_EMBED_REF)
        assert result.reindex_required is False

    @patch("lilbee.server.handlers.models.get_services")
    async def test_pins_legacy_meta_before_cfg_mutation(self, mock_svc):
        """A pre-upgrade store with chunks but no _meta is pinned under the OLD cfg.

        Asserts call ordering: initialize_meta_if_legacy must run BEFORE cfg is
        mutated. A regression that swaps the two would let lazy-init adopt the
        NEW cfg as the legacy identity, hiding the drift the caller just introduced.
        """
        mock_svc.return_value.provider.list_models.return_value = [_EMBED_REF]
        store_mock = mock_svc.return_value.store
        store_mock.get_meta.return_value = {
            "embedding_model": "previous-model:v1",
            "embedding_dim": 768,
            "schema_version": 1,
            "updated_at": "2026-04-25T00:00:00+00:00",
        }
        cfg_at_pin: dict[str, str] = {}

        def record_cfg() -> None:
            cfg_at_pin["embedding_model"] = cfg.embedding_model

        store_mock.initialize_meta_if_legacy.side_effect = record_cfg
        original_model = cfg.embedding_model
        await handlers.set_embedding_model(_EMBED_REF)
        store_mock.initialize_meta_if_legacy.assert_called_once()
        assert cfg_at_pin["embedding_model"] == original_model


class TestGetConfig:
    async def test_returns_all_config_keys(self):
        result = await handlers.get_config()
        dumped = result.model_dump()
        assert "chat_model" in dumped
        assert "rag_system_prompt" in dumped
        assert "remote_base_url" in dumped
        assert "diversity_max_per_source" in dumped
        assert "mmr_lambda" in dumped
        assert "query_expansion_count" in dumped
        assert "adaptive_threshold_step" in dumped
        assert "temperature" in dumped
        assert "max_context_sources" in dumped
        assert "hyde" in dumped
        assert "hyde_weight" in dumped
        assert "temporal_filtering" in dumped
        assert "concept_graph" in dumped
        assert "concept_boost_weight" in dumped
        assert "concept_max_per_chunk" in dumped
        assert "expansion_guardrails" in dumped
        assert "expansion_similarity_threshold" in dumped
        assert "llm_api_key" not in dumped


def _fake_source_store(mock_svc, all_sources: list[dict]) -> None:
    """Wire mock_svc.store.get_sources + count_sources to simulate DB-side
    pagination: filter by ``search`` against filename, then slice by
    offset/limit. Mirrors the real LanceDB behavior so list_documents
    tests don't have to special-case the mock call shape."""

    def _filter(search):
        if not search:
            return list(all_sources)
        search_lower = search.lower()
        return [s for s in all_sources if search_lower in s["filename"].lower()]

    def _get(*, search=None, limit=None, offset=0):
        matches = _filter(search)
        if limit is None:
            return matches[offset:]
        return matches[offset : offset + limit]

    def _count(*, search=None):
        return len(_filter(search))

    mock_svc.store.get_sources.side_effect = _get
    mock_svc.store.count_sources.side_effect = _count


class TestListDocuments:
    async def test_returns_documents(self, mock_svc):
        _fake_source_store(
            mock_svc,
            [{"filename": "a.md", "chunk_count": 5, "ingested_at": "2026-01-01"}],
        )
        result = await handlers.list_documents()
        assert result.total == 1
        assert result.documents[0].filename == "a.md"
        assert result.documents[0].chunk_count == 5
        assert result.has_more is False

    async def test_empty(self, mock_svc):
        _fake_source_store(mock_svc, [])
        result = await handlers.list_documents()
        assert result.total == 0
        assert result.documents == []
        assert result.has_more is False

    async def test_pagination(self, mock_svc):
        _fake_source_store(
            mock_svc,
            [{"filename": f"doc{i}.md", "chunk_count": i} for i in range(10)],
        )
        result = await handlers.list_documents(limit=3, offset=2)
        assert result.total == 10
        assert len(result.documents) == 3
        assert result.documents[0].filename == "doc2.md"
        assert result.limit == 3
        assert result.offset == 2
        # offset + returned (2 + 3) == 5 < total 10 → more pages remain
        assert result.has_more is True

    async def test_pagination_final_page_clears_has_more(self, mock_svc):
        _fake_source_store(
            mock_svc,
            [{"filename": f"doc{i}.md", "chunk_count": i} for i in range(10)],
        )
        result = await handlers.list_documents(limit=5, offset=5)
        assert result.total == 10
        assert result.has_more is False

    async def test_empty_page_with_positive_total_is_not_has_more(self, mock_svc):
        """Race guard: empty page alongside a non-zero total must not loop.

        Happens when a concurrent writer mutates the SOURCES table
        between count_sources() and get_sources(). Without the
        ``len(page) > 0`` guard, a naive client reading has_more to
        decide whether to keep fetching would spin past the end.
        """
        mock_svc.store.get_sources.return_value = []
        mock_svc.store.count_sources.return_value = 42
        result = await handlers.list_documents(limit=10, offset=0)
        assert result.total == 42
        assert result.documents == []
        assert result.has_more is False

    async def test_search_filter(self, mock_svc):
        _fake_source_store(
            mock_svc,
            [
                {"filename": "readme.md", "chunk_count": 3},
                {"filename": "setup.py", "chunk_count": 1},
                {"filename": "readme_dev.md", "chunk_count": 2},
            ],
        )
        result = await handlers.list_documents(search="readme")
        assert result.total == 2
        assert all("readme" in d.filename for d in result.documents)
        assert result.has_more is False


class TestGetConfigReranker:
    async def test_exposes_reranker_fields(self):
        """Reranker is a first-class public role; its fields always surface."""
        result = await handlers.get_config()
        dumped = result.model_dump()
        assert "reranker_model" in dumped
        assert "rerank_candidates" in dumped


class TestGetConfigDefaults:
    async def test_model_role_defaults_surface(self):
        """Model-role fields are non-writable but appear in defaults.

        UI reset-to-default affordances hit ``PUT /api/models/<role>`` to
        reset each slot. The defaults endpoint must surface the canonical
        blank/default values so clients don't hardcode them locally.
        """
        result = await handlers.get_config_defaults()
        dumped = result.model_dump()
        assert dumped["chat_model"] == _CHAT_REF
        assert dumped["embedding_model"] == _EMBED_REF
        assert dumped["vision_model"] == ""
        assert dumped["reranker_model"] == ""

    async def test_writable_defaults_surface(self):
        """Defaults include writable public fields (chunk_size, temperature, ...)."""
        result = await handlers.get_config_defaults()
        dumped = result.model_dump()
        assert "chunk_size" in dumped
        assert "temperature" in dumped


_VISION_REF = "noctrex/LightOnOCR-2-1B-GGUF/lightonocr-Q4_K_M.gguf"


class TestSetVisionModel:
    @patch("lilbee.server.handlers.models.get_services")
    async def test_sets_vision_model(self, mock_svc, tmp_path):
        mock_svc.return_value.provider.list_models.return_value = [_VISION_REF]
        result = await handlers.set_vision_model(_VISION_REF)
        assert result.model == _VISION_REF
        assert cfg.vision_model == _VISION_REF

    @patch("lilbee.server.handlers.models.settings.set_value")
    @patch("lilbee.server.handlers.models.get_services")
    async def test_empty_string_unsets(self, mock_svc, mock_set_value):
        cfg.vision_model = _VISION_REF
        result = await handlers.set_vision_model("")
        assert result.model == ""
        assert cfg.vision_model == ""
        mock_set_value.assert_called_once_with(cfg.data_root, "vision_model", "")

    @patch("lilbee.server.handlers.models.get_services")
    async def test_whitespace_string_unsets(self, mock_svc):
        """Whitespace-only strings must not bypass the empty-check guard."""
        cfg.vision_model = _VISION_REF
        result = await handlers.set_vision_model("   ")
        assert result.model == ""
        assert cfg.vision_model == ""

    @patch("lilbee.server.handlers.models.get_services")
    async def test_rejects_chat_model(self, mock_svc):
        from lilbee.catalog.types import ModelTask
        from lilbee.core.config.validators import TaskMismatchError

        mock_svc.return_value.provider.list_models.return_value = [_CHAT_REF]
        with pytest.raises(TaskMismatchError) as exc:
            await handlers.set_vision_model(_CHAT_REF)
        assert exc.value.entry_task == ModelTask.CHAT
        assert exc.value.expected_task == ModelTask.VISION

    @patch("lilbee.server.handlers.models.get_services")
    async def test_rejects_out_of_catalog(self, mock_svc):
        custom = "org/Custom-Vision-GGUF/custom.gguf"
        mock_svc.return_value.provider.list_models.return_value = [custom]
        with pytest.raises(ValueError, match="featured catalog"):
            await handlers.set_vision_model(custom)


_RERANK_REF = "gpustack/bge-reranker-v2-m3-GGUF/bge-reranker-Q4_K_M.gguf"


class TestSetRerankerModel:
    @patch("lilbee.server.handlers.models.get_services")
    async def test_sets_reranker_model(self, mock_svc, tmp_path):
        mock_svc.return_value.provider.list_models.return_value = [_RERANK_REF]
        result = await handlers.set_reranker_model(_RERANK_REF)
        assert result.model == _RERANK_REF
        assert cfg.reranker_model == _RERANK_REF

    @patch("lilbee.server.handlers.models.settings.set_value")
    @patch("lilbee.server.handlers.models.get_services")
    async def test_empty_string_unsets(self, mock_svc, mock_set_value):
        cfg.reranker_model = _RERANK_REF
        result = await handlers.set_reranker_model("")
        assert result.model == ""
        assert cfg.reranker_model == ""
        mock_set_value.assert_called_once_with(cfg.data_root, "reranker_model", "")

    @patch("lilbee.server.handlers.models.get_services")
    async def test_whitespace_string_unsets(self, mock_svc):
        """Whitespace-only strings must not bypass the empty-check guard."""
        cfg.reranker_model = _RERANK_REF
        result = await handlers.set_reranker_model("   ")
        assert result.model == ""
        assert cfg.reranker_model == ""

    @patch("lilbee.server.handlers.models.get_services")
    async def test_resolves_bare_repo_to_installed_quant(self, mock_svc, tmp_path):
        """PUT with bare ``hf_repo`` resolves to whichever quant is installed.

        The handler matches against installed refs of the form
        ``hf_repo/filename`` and returns the canonical full ref.
        """
        mock_svc.return_value.provider.list_models.return_value = [_RERANK_REF]
        result = await handlers.set_reranker_model("gpustack/bge-reranker-v2-m3-GGUF")
        assert result.model == _RERANK_REF
        assert cfg.reranker_model == _RERANK_REF

    @patch("lilbee.server.handlers.models.get_services")
    async def test_rejects_chat_model(self, mock_svc):
        from lilbee.catalog.types import ModelTask
        from lilbee.core.config.validators import TaskMismatchError

        mock_svc.return_value.provider.list_models.return_value = [_CHAT_REF]
        with pytest.raises(TaskMismatchError) as exc:
            await handlers.set_reranker_model(_CHAT_REF)
        assert exc.value.entry_task == ModelTask.CHAT
        assert exc.value.expected_task == ModelTask.RERANK

    @patch("lilbee.server.handlers.models.get_services")
    async def test_rejects_vision_model(self, mock_svc):
        from lilbee.catalog.types import ModelTask
        from lilbee.core.config.validators import TaskMismatchError

        mock_svc.return_value.provider.list_models.return_value = [_VISION_REF]
        with pytest.raises(TaskMismatchError) as exc:
            await handlers.set_reranker_model(_VISION_REF)
        assert exc.value.entry_task == ModelTask.VISION
        assert exc.value.expected_task == ModelTask.RERANK

    @patch("lilbee.server.handlers.models.get_services")
    async def test_rejects_embedding_model_in_vision_slot(self, mock_svc):
        """Embedding model in the reranker slot carries the structured task pair."""
        from lilbee.catalog.types import ModelTask
        from lilbee.core.config.validators import TaskMismatchError

        mock_svc.return_value.provider.list_models.return_value = [_EMBED_REF]
        with pytest.raises(TaskMismatchError) as exc:
            await handlers.set_reranker_model(_EMBED_REF)
        assert exc.value.entry_task == ModelTask.EMBEDDING
        assert exc.value.expected_task == ModelTask.RERANK

    @patch("lilbee.server.handlers.models.get_services")
    async def test_rejects_reranker_in_vision_slot(self, mock_svc):
        """Reranker in the vision slot reports the rerank/vision task pair."""
        from lilbee.catalog.types import ModelTask
        from lilbee.core.config.validators import TaskMismatchError

        mock_svc.return_value.provider.list_models.return_value = [_RERANK_REF]
        with pytest.raises(TaskMismatchError) as exc:
            await handlers.set_vision_model(_RERANK_REF)
        assert exc.value.entry_task == ModelTask.RERANK
        assert exc.value.expected_task == ModelTask.VISION


class TestTaskEndpointPath:
    """TASK_ENDPOINT_PATH is keyed by ``ModelTask`` members.

    Consumers that hold a plain ``entry.task`` string coerce via
    ``ModelTask(entry.task)`` at the indexing boundary.
    """

    def test_index_with_enum(self) -> None:
        from lilbee.catalog.types import ModelTask
        from lilbee.server.handlers import TASK_ENDPOINT_PATH

        assert TASK_ENDPOINT_PATH[ModelTask.CHAT] == "chat"
        assert TASK_ENDPOINT_PATH[ModelTask.EMBEDDING] == "embedding"
        assert TASK_ENDPOINT_PATH[ModelTask.VISION] == "vision"
        assert TASK_ENDPOINT_PATH[ModelTask.RERANK] == "reranker"

    def test_coerces_catalog_task_string(self) -> None:
        """Coercion via ``ModelTask(entry.task)`` resolves to the enum key."""
        from lilbee.catalog.types import ModelTask
        from lilbee.server.handlers import TASK_ENDPOINT_PATH

        assert TASK_ENDPOINT_PATH[ModelTask("chat")] == "chat"
        assert TASK_ENDPOINT_PATH[ModelTask("rerank")] == "reranker"


class TestCrawlStream:
    @pytest.fixture(autouse=True)
    def _skip_chromium_bootstrap(self):
        """Treat Chromium as installed so crawl_stream doesn't subprocess out."""
        with patch("lilbee.crawler.bootstrap.chromium_installed", return_value=True):
            yield

    @patch("lilbee.crawler.crawl_and_save")
    @patch("lilbee.crawler.validate_crawl_url")
    async def test_streams_events_and_done(self, _mock_validate, mock_crawl):
        from pathlib import Path

        async def fake_crawl(url, *, depth, max_pages, on_progress, cancel=None):
            from lilbee.runtime.progress import CrawlDoneEvent, CrawlPageEvent, CrawlStartEvent

            on_progress("crawl_start", CrawlStartEvent(url=url, depth=depth))
            on_progress("crawl_page", CrawlPageEvent(url=url, current=1, total=1))
            on_progress("crawl_done", CrawlDoneEvent(pages_crawled=1, files_written=1))
            return [Path("test.md")]

        mock_crawl.side_effect = fake_crawl
        events = []
        async for event in handlers.crawl_stream("https://example.com", depth=1, max_pages=10):
            events.append(event)
        assert any("crawl_start" in e for e in events)
        assert any("done" in e for e in events)

    @patch("lilbee.crawler.crawl_and_save")
    @patch("lilbee.crawler.validate_crawl_url")
    async def test_streams_error_on_exception(self, _mock_validate, mock_crawl):
        mock_crawl.side_effect = RuntimeError("network fail")
        events = []
        async for event in handlers.crawl_stream("https://example.com"):
            events.append(event)
        assert any("error" in e and "network fail" in e for e in events)

    @patch("lilbee.crawler.crawl_and_save")
    @patch("lilbee.crawler.validate_crawl_url")
    async def test_cancel_stops_crawl(self, _mock_validate, mock_crawl, caplog):
        """Closing the crawl generator mid-stream sets cancel and logs."""
        import threading

        barrier = threading.Event()

        async def blocking_crawl(url, *, depth, max_pages, on_progress, cancel=None):
            from lilbee.runtime.progress import CrawlStartEvent

            on_progress("crawl_start", CrawlStartEvent(url=url, depth=depth))
            barrier.wait(timeout=2)
            return []

        mock_crawl.side_effect = blocking_crawl
        caplog.set_level(logging.INFO, logger="lilbee.server.handlers")
        gen = handlers.crawl_stream("https://example.com", depth=1)
        async for event in gen:
            if event and "crawl_start" in event:
                await gen.aclose()
                barrier.set()
                break
        await asyncio.sleep(0.05)
        assert any("Crawl stream cancelled by client" in r.message for r in caplog.records)


class TestSseHelpers:
    def test_sse_error(self):
        result = handlers.sse_error("oops")
        assert result == 'event: error\ndata: {"message": "oops"}\n\n'

    def test_sse_error_with_code_and_detail(self):
        result = handlers.sse_error(
            "Internal error", code="model_too_large", detail="Failed to load X"
        )
        parsed = json.loads(result.split("data: ")[1].strip())
        assert parsed == {
            "message": "Internal error",
            "code": "model_too_large",
            "detail": "Failed to load X",
        }

    def test_sse_error_omits_unset_fields(self):
        result = handlers.sse_error("oops", code=None, detail=None)
        parsed = json.loads(result.split("data: ")[1].strip())
        assert parsed == {"message": "oops"}

    def test_sse_done(self):
        result = handlers.sse_done({"count": 1})
        assert result == 'event: done\ndata: {"count": 1}\n\n'


class TestClassifyLoadError:
    def test_oom_diagnostic_is_classified_as_model_too_large(self):
        msg = (
            "Failed to load Qwen3-4B (2.4 GB) with n_ctx=4096. "
            "Host has 1.2 GB free RAM. Try a smaller model."
        )
        code, user_message = handlers.classify_load_error(msg)
        assert code == "model_too_large"
        assert user_message == "Model too large for available RAM"

    def test_llama_context_signature_is_classified(self):
        code, _ = handlers.classify_load_error("llama_context: failed to allocate")
        assert code == "model_too_large"

    def test_unknown_message_falls_back_to_internal(self):
        code, user_message = handlers.classify_load_error("Network unreachable")
        assert code is None
        assert user_message == "Internal error"

    def test_classifier_is_case_insensitive(self):
        code, _ = handlers.classify_load_error("FAILED TO LOAD model.gguf")
        assert code == "model_too_large"


class TestResolveGenerationOptions:
    def test_with_options(self):
        result = _sse_h._resolve_generation_options({"temperature": 0.5})
        assert result is not None

    def test_without_options(self):
        result = _sse_h._resolve_generation_options(None)
        assert result is None


class TestListExternalModels:
    """Tests for the external model discovery handler."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        """Reset the external models cache before each test."""
        from lilbee.server.handlers import models as h

        h._external_cache = h._ExternalModelsCache()
        yield
        h._external_cache = h._ExternalModelsCache()

    @patch("lilbee.server.handlers.models.get_services")
    async def test_returns_provider_models(self, mock_svc):
        mock_svc.return_value.provider.list_models.return_value = ["model-a", "model-b"]
        result = await handlers.list_external_models()
        assert result.models == ["model-a", "model-b"]
        assert result.error is None

    @patch("lilbee.server.handlers.models.get_services")
    async def test_error_returns_empty_with_message(self, mock_svc):
        mock_svc.return_value.provider.list_models.side_effect = RuntimeError("connection refused")
        result = await handlers.list_external_models()
        assert result.models == []
        assert result.error is not None

    @patch("lilbee.server.handlers.models.get_services")
    async def test_cache_reuses_result(self, mock_svc):
        mock_svc.return_value.provider.list_models.return_value = ["model-a"]
        result1 = await handlers.list_external_models()
        result2 = await handlers.list_external_models()
        assert result1 == result2
        assert result1.models == ["model-a"]
        mock_svc.return_value.provider.list_models.assert_called_once()

    @patch("lilbee.server.handlers.models.time")
    @patch("lilbee.server.handlers.models.get_services")
    async def test_cache_expires(self, mock_svc, mock_time):
        mock_svc.return_value.provider.list_models.return_value = ["model-a"]
        mock_time.monotonic.return_value = 0.0
        await handlers.list_external_models()

        mock_time.monotonic.return_value = 61.0
        await handlers.list_external_models()

        assert mock_svc.return_value.provider.list_models.call_count == 2

    @patch("lilbee.server.handlers.models.get_services")
    async def test_cache_invalidates_on_config_change(self, mock_svc):
        mock_svc.return_value.provider.list_models.return_value = ["model-a"]
        cfg.remote_base_url = "https://provider-a.example"
        await handlers.list_external_models()

        cfg.remote_base_url = "https://provider-b.example"
        await handlers.list_external_models()

        assert mock_svc.return_value.provider.list_models.call_count == 2


# ---------------------------------------------------------------------------
# Phase 4: SSE cancel checks, model pull progress cancel
# ---------------------------------------------------------------------------


class TestRunLlmStreamCancel:
    def test_cancel_stops_streaming(self):
        """When cancel is set, _run_llm_stream breaks out of the loop."""
        import threading

        cancel = threading.Event()
        cancel.set()  # pre-set cancel

        queue: asyncio.Queue[str | None] = asyncio.Queue()
        error_holder: list[str] = []

        mock_provider = MagicMock()
        # Return some stream tokens that would normally be emitted
        stream_data = iter(["token1", "token2"])
        mock_provider.chat.return_value = stream_data

        with patch("lilbee.server.handlers.rag.get_services") as mock_svc:
            mock_svc.return_value.provider = mock_provider
            _rag_h._run_llm_stream(
                [{"role": "user", "content": "hi"}],
                None,
                queue,
                cancel,
                error_holder,
            )
        # Should have None sentinel
        items = []
        while not queue.empty():
            items.append(queue.get_nowait())
        assert items[-1] is None


class TestReasoningCapHandling:
    """SSE handler forwards events from the shared cap-aware orchestrator."""

    def _drain(self, queue: asyncio.Queue[str | None]) -> list[str]:
        items: list[str] = []
        while not queue.empty():
            value = queue.get_nowait()
            if value is None:
                break
            items.append(value)
        return items

    def test_cap_fire_emits_notice_event(self):
        """When the orchestrator yields a CapNotice, the handler turns it into an SSE event."""
        import threading

        from lilbee.retrieval.reasoning import CAP_NOTICE_TEMPLATE

        notice_marker = CAP_NOTICE_TEMPLATE.format(chars=512).strip()

        snapshot_cap = cfg.max_reasoning_chars
        try:
            cfg.max_reasoning_chars = 512

            mock_provider = MagicMock()
            long_reasoning = "<think>" + ("x " * 400) + "</think>not reached"
            mock_provider.chat.side_effect = [iter([long_reasoning]), iter(["final ", "answer."])]

            queue: asyncio.Queue[str | None] = asyncio.Queue()
            cancel = threading.Event()
            error_holder: list[str] = []

            with patch("lilbee.server.handlers.rag.get_services") as mock_svc:
                mock_svc.return_value.provider = mock_provider
                _rag_h._run_llm_stream(
                    [{"role": "user", "content": "explain quantum tunneling"}],
                    None,
                    queue,
                    cancel,
                    error_holder,
                )

            events = self._drain(queue)
            assert any(notice_marker in e for e in events)
            assert any("final " in e for e in events)
            assert any("answer." in e for e in events)
            assert mock_provider.chat.call_count == 2
        finally:
            cfg.max_reasoning_chars = snapshot_cap

    def test_cap_does_not_fire_under_threshold(self):
        """Short reasoning skips the re-issue path entirely."""
        import threading

        from lilbee.retrieval.reasoning import CAP_NOTICE_TEMPLATE

        snapshot_cap = cfg.max_reasoning_chars
        try:
            cfg.max_reasoning_chars = 64_000

            mock_provider = MagicMock()
            mock_provider.chat.return_value = iter(["<think>brief</think>", "the ", "answer"])

            queue: asyncio.Queue[str | None] = asyncio.Queue()
            cancel = threading.Event()
            error_holder: list[str] = []

            with patch("lilbee.server.handlers.rag.get_services") as mock_svc:
                mock_svc.return_value.provider = mock_provider
                _rag_h._run_llm_stream(
                    [{"role": "user", "content": "hi"}],
                    None,
                    queue,
                    cancel,
                    error_holder,
                )

            assert mock_provider.chat.call_count == 1
            events = self._drain(queue)
            notice_marker = CAP_NOTICE_TEMPLATE.format(chars=64_000).strip()
            assert not any(notice_marker in e for e in events)
        finally:
            cfg.max_reasoning_chars = snapshot_cap

    def test_cap_fire_with_cancel_skips_continuation(self):
        """If the user cancels mid-stream, the continuation re-issue does not run."""
        import threading

        snapshot_cap = cfg.max_reasoning_chars
        try:
            cfg.max_reasoning_chars = 512

            mock_provider = MagicMock()

            def first_pass():
                yield "<think>"
                for _ in range(400):
                    yield "x "
                yield "</think>"

            mock_provider.chat.return_value = first_pass()

            queue: asyncio.Queue[str | None] = asyncio.Queue()
            cancel = threading.Event()
            cancel.set()
            error_holder: list[str] = []

            with patch("lilbee.server.handlers.rag.get_services") as mock_svc:
                mock_svc.return_value.provider = mock_provider
                _rag_h._run_llm_stream(
                    [{"role": "user", "content": "hi"}],
                    None,
                    queue,
                    cancel,
                    error_holder,
                )

            assert mock_provider.chat.call_count == 1
        finally:
            cfg.max_reasoning_chars = snapshot_cap

    def test_cancel_during_continuation_stops_emitting(self):
        """Cancel set mid-second-wave breaks the continuation loop without crashing."""
        import threading

        snapshot_cap = cfg.max_reasoning_chars
        try:
            cfg.max_reasoning_chars = 512
            cancel = threading.Event()
            mock_provider = MagicMock()
            long_reasoning = "<think>" + ("x " * 400) + "</think>"

            def second_pass():
                yield "first chunk "
                cancel.set()
                yield "second chunk should be ignored"

            mock_provider.chat.side_effect = [iter([long_reasoning]), second_pass()]
            queue: asyncio.Queue[str | None] = asyncio.Queue()
            error_holder: list[str] = []

            with patch("lilbee.server.handlers.rag.get_services") as mock_svc:
                mock_svc.return_value.provider = mock_provider
                _rag_h._run_llm_stream(
                    [{"role": "user", "content": "long question"}],
                    None,
                    queue,
                    cancel,
                    error_holder,
                )

            events = self._drain(queue)
            assert any("first chunk" in e for e in events)
            assert not any("second chunk" in e for e in events)
        finally:
            cfg.max_reasoning_chars = snapshot_cap


class TestParseOcrParams:
    def test_ocr_timeout_coerced_to_float(self):
        """_parse_ocr_params coerces ocr_timeout to float."""
        enable_ocr, ocr_timeout = _ingest_h._parse_ocr_params({"ocr_timeout": "60"})
        assert ocr_timeout == 60.0
        assert isinstance(ocr_timeout, float)
        assert enable_ocr is None


class TestAddHandlerCancel:
    async def test_cancel_returns_early(self):
        """When cancel is set before sync, add returns early with copy-only summary."""
        from lilbee.server.handlers import SseStream

        sse = SseStream()
        sse.cancel.set()

        copy_result = MagicMock()
        copy_result.copied = ["test.txt"]
        copy_result.skipped = []

        with patch("lilbee.server.handlers.ingest.copy_files", return_value=copy_result):
            result = await _ingest_h._run_add(
                paths=[],
                force=False,
                enable_ocr=None,
                ocr_timeout=None,
                sse=sse,
            )
        assert result is not None
        assert result.copied == ["test.txt"]


class TestModelPullProgressCancel:
    async def test_cancel_skips_progress(self):
        """When cancel is set, the progress callback returns early."""
        from lilbee.server.handlers import SseStream

        sse = SseStream()
        sse.cancel.set()

        # Simulate the progress callback pattern from models_pull
        def _on_progress(data):
            if sse.cancel.is_set():
                return
            sse.queue.put_nowait("should_not_appear")

        _on_progress({"status": "downloading"})
        assert sse.queue.empty()

    async def test_cancel_during_pull_skips_later_progress(self):
        """When cancel is set before pull starts, all progress calls return early."""
        import threading

        mock_manager = MagicMock()
        progress_called = threading.Event()

        def fake_pull(model, source, *, on_progress=None, on_bytes=None):
            if on_bytes:
                # All progress calls should see cancel already set
                on_bytes(500, 1000)
                progress_called.set()

        mock_manager.pull.side_effect = fake_pull

        # Patch SseStream so cancel is pre-set
        original_init = handlers.SseStream.__init__

        def patched_init(self):
            original_init(self)
            self.cancel.set()  # Pre-set cancel before pull starts

        with (
            patch(
                "lilbee.server.handlers.models.get_services",
                return_value=MagicMock(model_manager=mock_manager),
            ),
            patch.object(handlers.SseStream, "__init__", patched_init),
        ):
            gen = handlers.models_pull("test", source="native")
            events = []
            async for event in gen:
                events.append(event)
            # Wait for the pull thread to complete
            await asyncio.sleep(0.2)

        assert progress_called.is_set()  # Pull did call on_bytes
        assert not any("current" in e for e in events if e)

    async def test_cancel_during_litellm_pull_skips_progress(self):
        """When cancel is set before litellm pull starts, on_progress returns early."""
        import threading

        mock_manager = MagicMock()
        progress_called = threading.Event()

        def fake_pull(model, source, *, on_progress=None, on_bytes=None):
            if on_progress:
                on_progress({"status": "should_be_suppressed"})
                progress_called.set()

        mock_manager.pull.side_effect = fake_pull

        original_init = handlers.SseStream.__init__

        def patched_init(self):
            original_init(self)
            self.cancel.set()

        with (
            patch(
                "lilbee.server.handlers.models.get_services",
                return_value=MagicMock(model_manager=mock_manager),
            ),
            patch.object(handlers.SseStream, "__init__", patched_init),
        ):
            gen = handlers.models_pull("test", source="remote")
            events = []
            async for event in gen:
                events.append(event)
            await asyncio.sleep(0.2)

        assert progress_called.is_set()
        assert not any("should_be_suppressed" in e for e in events if e)


class TestSearchRouteErrors:
    """Cover error handling wrappers in search and ask routes."""

    @staticmethod
    def _auth_headers() -> dict[str, str]:
        from lilbee.server import auth as _auth_mod

        return {"Authorization": f"Bearer {_auth_mod.session_manager.token}"}

    async def test_empty_search_returns_400(self):
        from litestar.testing import AsyncTestClient

        from lilbee.server.app import create_app

        async with AsyncTestClient(create_app()) as client:
            resp = await client.get("/api/search", params={"q": ""}, headers=self._auth_headers())
        assert resp.status_code == 400

    async def test_search_provider_error_returns_503(self):
        from litestar.testing import AsyncTestClient

        from lilbee.server.app import create_app

        async with AsyncTestClient(create_app()) as client:
            with patch("lilbee.server.handlers.rag.get_services") as mock_svc:
                mock_svc.return_value.searcher.search.side_effect = RuntimeError("down")
                resp = await client.get(
                    "/api/search", params={"q": "test"}, headers=self._auth_headers()
                )
        assert resp.status_code == 503

    async def test_empty_ask_returns_400(self):
        from litestar.testing import AsyncTestClient

        from lilbee.server.app import create_app

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post(
                "/api/ask", json={"question": ""}, headers=self._auth_headers()
            )
        assert resp.status_code == 400

    async def test_ask_provider_error_returns_503(self):
        from litestar.testing import AsyncTestClient

        from lilbee.server.app import create_app

        async with AsyncTestClient(create_app()) as client:
            with patch("lilbee.server.handlers.rag.get_services") as mock_svc:
                mock_svc.return_value.searcher.ask_raw.side_effect = RuntimeError("down")
                resp = await client.post(
                    "/api/ask", json={"question": "test"}, headers=self._auth_headers()
                )
        assert resp.status_code == 503


class TestSourceContentRoute:
    """GET /api/source routing: JSON shape, raw streaming, and error paths.

    ``get_source_content`` is covered through the route so we exercise the
    ``isinstance(result, tuple)`` branch in ``source_content_route`` at the
    same time; separate handler-level tests would double the surface
    without adding signal.
    """

    @staticmethod
    def _auth_headers() -> dict[str, str]:
        from lilbee.server import auth as _auth_mod

        return {"Authorization": f"Bearer {_auth_mod.session_manager.token}"}

    async def test_empty_source_returns_400(self, isolated_env):
        from litestar.testing import AsyncTestClient

        from lilbee.server.app import create_app

        async with AsyncTestClient(create_app()) as client:
            resp = await client.get(
                "/api/source", params={"source": "   "}, headers=self._auth_headers()
            )
        assert resp.status_code == 400

    async def test_missing_file_returns_404(self, isolated_env):
        from litestar.testing import AsyncTestClient

        from lilbee.server.app import create_app

        async with AsyncTestClient(create_app()) as client:
            resp = await client.get(
                "/api/source", params={"source": "nope.md"}, headers=self._auth_headers()
            )
        assert resp.status_code == 404

    async def test_path_traversal_returns_400(self, isolated_env):
        """``..`` traversal escapes ``documents_dir`` → ValueError → 400.

        ``validate_path_within`` resolves the path and checks
        ``is_relative_to``; the resulting ``ValueError`` is translated to
        ValidationException by the route wrapper.
        """
        from litestar.testing import AsyncTestClient

        from lilbee.server.app import create_app

        async with AsyncTestClient(create_app()) as client:
            resp = await client.get(
                "/api/source",
                params={"source": "../../etc/passwd"},
                headers=self._auth_headers(),
            )
        assert resp.status_code == 400

    async def test_text_file_returns_markdown_json(self, isolated_env):
        """A markdown source returns JSON with markdown, title, content_type."""
        from litestar.testing import AsyncTestClient

        from lilbee.server.app import create_app

        (cfg.documents_dir / "doc.md").write_text("# Hello\n\nbody\n", encoding="utf-8")
        async with AsyncTestClient(create_app()) as client:
            resp = await client.get(
                "/api/source", params={"source": "doc.md"}, headers=self._auth_headers()
            )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["markdown"].startswith("# Hello")
        assert payload["title"] == "Hello"
        assert payload["content_type"].startswith("text/")

    async def test_binary_file_returns_empty_markdown(self, isolated_env):
        """Non-text types return empty markdown; client should re-request raw=1."""
        from litestar.testing import AsyncTestClient

        from lilbee.server.app import create_app

        (cfg.documents_dir / "doc.pdf").write_bytes(b"%PDF-1.4\nbinarydata")
        async with AsyncTestClient(create_app()) as client:
            resp = await client.get(
                "/api/source", params={"source": "doc.pdf"}, headers=self._auth_headers()
            )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["markdown"] == ""
        assert payload["content_type"] == "application/pdf"
        assert payload["title"] is None

    async def test_unknown_extension_falls_back_to_octet_stream(self, isolated_env):
        """A file without an extension maps to application/octet-stream."""
        from litestar.testing import AsyncTestClient

        from lilbee.server.app import create_app

        (cfg.documents_dir / "mystery").write_bytes(b"\x00\x01\x02")
        async with AsyncTestClient(create_app()) as client:
            resp = await client.get(
                "/api/source", params={"source": "mystery"}, headers=self._auth_headers()
            )
        assert resp.status_code == 200
        assert resp.json()["content_type"] == "application/octet-stream"

    async def test_raw_returns_bytes_with_content_type(self, isolated_env):
        """raw=1 streams the file bytes with a real Content-Type header."""
        from litestar.testing import AsyncTestClient

        from lilbee.server.app import create_app

        payload = b"%PDF-1.4\nbinarydata\n"
        (cfg.documents_dir / "doc.pdf").write_bytes(payload)
        async with AsyncTestClient(create_app()) as client:
            resp = await client.get(
                "/api/source",
                params={"source": "doc.pdf", "raw": True},
                headers=self._auth_headers(),
            )
        assert resp.status_code == 200
        assert resp.content == payload
        assert resp.headers["content-type"].startswith("application/pdf")
        assert resp.headers["x-content-type-options"] == "nosniff"
        assert "content-disposition" not in resp.headers

    async def test_raw_html_source_forces_octet_stream_attachment(self, isolated_env):
        """An attacker-named .html source must not render as text/html.

        The server caps the Content-Type to a known-safe allowlist; .html falls
        outside it, so the response is served as application/octet-stream with
        Content-Disposition: attachment and X-Content-Type-Options: nosniff.
        """
        from litestar.testing import AsyncTestClient

        from lilbee.server.app import create_app

        (cfg.documents_dir / "evil.html").write_bytes(b"<script>alert(1)</script>")
        async with AsyncTestClient(create_app()) as client:
            resp = await client.get(
                "/api/source",
                params={"source": "evil.html", "raw": True},
                headers=self._auth_headers(),
            )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/octet-stream")
        assert resp.headers["x-content-type-options"] == "nosniff"
        assert "attachment" in resp.headers["content-disposition"]
        assert "evil.html" in resp.headers["content-disposition"]

    @pytest.mark.parametrize(
        ("filename", "body"),
        [
            ("evil.js", b"alert(1)"),
            ("evil.css", b"body { background: url(javascript:1); }"),
            ("evil.xhtml", b"<x:y xmlns:x='ns' />"),
        ],
    )
    async def test_raw_script_carrying_text_types_force_attachment(
        self, isolated_env, filename, body
    ):
        """Text-category types that can carry script are denied inline.

        ``.mjs`` is intentionally omitted from this parametrization because
        ``mimetypes.guess_type`` resolves it from the Windows registry and
        returns inconsistent answers across platforms; ``.js`` covers the
        same ``text/javascript`` deny path with deterministic mapping.
        """
        from litestar.testing import AsyncTestClient

        from lilbee.server.app import create_app

        (cfg.documents_dir / filename).write_bytes(body)
        async with AsyncTestClient(create_app()) as client:
            resp = await client.get(
                "/api/source",
                params={"source": filename, "raw": True},
                headers=self._auth_headers(),
            )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/octet-stream")
        assert "attachment" in resp.headers["content-disposition"]

    async def test_raw_svg_source_forces_octet_stream_attachment(self, isolated_env):
        """SVG can carry script; not in the allowlist → forced to octet-stream."""
        from litestar.testing import AsyncTestClient

        from lilbee.server.app import create_app

        (cfg.documents_dir / "evil.svg").write_bytes(
            b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'
        )
        async with AsyncTestClient(create_app()) as client:
            resp = await client.get(
                "/api/source",
                params={"source": "evil.svg", "raw": True},
                headers=self._auth_headers(),
            )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/octet-stream")
        assert "attachment" in resp.headers["content-disposition"]

    async def test_raw_markdown_serves_text_markdown_no_attachment(self, isolated_env):
        """text/markdown is in the allowlist → served inline with nosniff only."""
        from litestar.testing import AsyncTestClient

        from lilbee.server.app import create_app

        (cfg.documents_dir / "note.md").write_text("# hi", encoding="utf-8")
        async with AsyncTestClient(create_app()) as client:
            resp = await client.get(
                "/api/source",
                params={"source": "note.md", "raw": True},
                headers=self._auth_headers(),
            )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/markdown")
        assert resp.headers["x-content-type-options"] == "nosniff"
        assert "content-disposition" not in resp.headers

    async def test_raw_unknown_extension_forces_attachment(self, isolated_env):
        """No extension → mimetypes returns None → octet-stream + attachment."""
        from litestar.testing import AsyncTestClient

        from lilbee.server.app import create_app

        (cfg.documents_dir / "mystery").write_bytes(b"\x00\x01\x02")
        async with AsyncTestClient(create_app()) as client:
            resp = await client.get(
                "/api/source",
                params={"source": "mystery", "raw": True},
                headers=self._auth_headers(),
            )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/octet-stream")
        assert "attachment" in resp.headers["content-disposition"]

    async def test_raw_csv_serves_text_csv_no_attachment(self, isolated_env):
        """text/csv passes through under the category-based policy.

        The previous static allowlist rejected anything not pre-enumerated;
        the category policy admits any ``text/*`` (except ``text/html``) so
        plausible-but-unenumerated types like CSV serve inline.
        """
        from litestar.testing import AsyncTestClient

        from lilbee.server.app import create_app

        (cfg.documents_dir / "data.csv").write_text("a,b,c\n1,2,3\n", encoding="utf-8")
        async with AsyncTestClient(create_app()) as client:
            resp = await client.get(
                "/api/source",
                params={"source": "data.csv", "raw": True},
                headers=self._auth_headers(),
            )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/csv")
        assert resp.headers["x-content-type-options"] == "nosniff"
        assert "content-disposition" not in resp.headers

    async def test_raw_avif_serves_image_avif_no_attachment(self, isolated_env):
        """image/avif passes through under the category-based policy.

        Modern image formats not present in the previous static allowlist
        (e.g. AVIF) are admitted now that ``image/*`` is allowed by category.
        """
        from litestar.testing import AsyncTestClient

        from lilbee.server.app import create_app

        (cfg.documents_dir / "pic.avif").write_bytes(b"\x00\x00\x00 ftypavif")
        async with AsyncTestClient(create_app()) as client:
            resp = await client.get(
                "/api/source",
                params={"source": "pic.avif", "raw": True},
                headers=self._auth_headers(),
            )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("image/avif")
        assert resp.headers["x-content-type-options"] == "nosniff"
        assert "content-disposition" not in resp.headers
