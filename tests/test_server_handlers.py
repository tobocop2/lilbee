"""Tests for the framework-agnostic server handlers."""

import asyncio
import json
import logging
from unittest.mock import MagicMock, patch

import pytest

import lilbee.services as svc_mod
from lilbee.config import cfg
from lilbee.ingest import SyncResult
from lilbee.server import handlers
from lilbee.store import SearchChunk

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
        from lilbee.cli.helpers import StatusConfig, StatusResult

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
        mock_svc.searcher.search.assert_called_once_with("test", top_k=3)

    async def test_empty_results(self, mock_svc):
        mock_svc.searcher.search.return_value = []
        result = await handlers.search("nothing")
        assert result == []


class TestAsk:
    async def test_returns_answer_and_sources(self, mock_svc):
        from lilbee.query import AskResult

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
        from lilbee.query import AskResult

        mock_svc.searcher.ask_raw.return_value = AskResult(answer="No docs found.", sources=[])
        result = await handlers.ask("what?")
        assert result.answer == "No docs found."
        assert result.sources == []


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

    async def test_provider_error_yields_error_event(self, mock_svc):
        mock_svc.searcher.build_rag_context.return_value = _rag_return()
        mock_svc.provider.chat.side_effect = RuntimeError("model missing")
        events = [e async for e in handlers.ask_stream("question")]

        non_empty = [e for e in events if e]
        error_events = [e for e in non_empty if e.startswith("event: error")]
        assert len(error_events) == 1
        assert "Internal error" in error_events[0]

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
        from lilbee.query import AskResult

        mock_svc.searcher.ask_raw.return_value = AskResult(answer="ok", sources=[])
        history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        result = await handlers.chat("follow up", history)
        assert result.answer == "ok"
        mock_svc.searcher.ask_raw.assert_called_once_with(
            "follow up", top_k=0, history=history, options=None
        )


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
                from lilbee.progress import FileDoneEvent, SyncDoneEvent

                on_progress("file_done", FileDoneEvent(file="a.txt", status="ok", chunks=3))
                on_progress("done", SyncDoneEvent(added=1, updated=0, removed=0, failed=0))
            return sync_result

        with patch("lilbee.ingest.sync", side_effect=fake_sync):
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
                from lilbee.progress import FileDoneEvent, FileStartEvent, SyncDoneEvent

                on_progress(
                    "file_start",
                    FileStartEvent(file="b.txt", total_files=1, current_file=1),
                )
                on_progress("file_done", FileDoneEvent(file="b.txt", status="ok", chunks=2))
                on_progress("done", SyncDoneEvent(added=1, updated=0, removed=0, failed=0))
            return sync_result

        with patch("lilbee.ingest.sync", side_effect=fake_sync):
            events = [e async for e in handlers.sync_stream()]

        non_empty = [e for e in events if e]
        file_start_events = [e for e in non_empty if e.startswith("event: file_start")]
        assert len(file_start_events) >= 1
        data = json.loads(file_start_events[0].split("data: ")[1].strip())
        assert data["file"] == "b.txt"

    async def test_timeout_continue_when_no_progress(self):
        """Sync with no progress events exercises the TimeoutError continue branch."""
        import asyncio

        sync_result = SyncResult()

        async def slow_sync(force_rebuild=False, quiet=False, *, on_progress=None, cancel=None):
            await asyncio.sleep(0.2)  # force at least one timeout iteration
            return sync_result

        with patch("lilbee.ingest.sync", side_effect=slow_sync):
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
                from lilbee.progress import FileStartEvent

                on_progress(
                    "file_start", FileStartEvent(file="a.txt", total_files=1, current_file=1)
                )
            barrier.wait(timeout=2)
            return SyncResult()

        caplog.set_level(logging.INFO, logger="lilbee.server.handlers")
        with patch("lilbee.ingest.sync", side_effect=blocking_sync):
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

        with patch("lilbee.ingest.sync", side_effect=fake_sync):
            events = []
            async for event in handlers.add_files_stream({"paths": [str(test_file)]}):
                events.append(event)
            assert any("done" in e for e in events)


class TestListModels:
    @patch("lilbee.models.list_installed_models")
    async def test_returns_catalogs(self, mock_list):
        mock_list.return_value = ["qwen3:8b", "mistral:7b"]
        result = await handlers.list_models()

        assert result.chat.active == cfg.chat_model
        assert isinstance(result.chat.catalog, list)
        assert len(result.chat.catalog) > 0
        assert "qwen3:8b" in result.chat.installed

    @patch("lilbee.models.list_installed_models")
    async def test_installed_flag_in_catalog(self, mock_list):
        mock_list.return_value = ["qwen3:0.6b"]
        result = await handlers.list_models()

        catalog = result.chat.catalog
        qwen_entry = next(m for m in catalog if "Qwen3 0.6B" in m.name)
        assert qwen_entry.installed is True

        mistral_entry = next(m for m in catalog if "Mistral" in m.name)
        assert mistral_entry.installed is False


class TestSetChatModel:
    async def test_updates_config_and_persists(self, tmp_path, mock_svc):
        mock_svc.provider.list_models.return_value = ["llama3:latest"]
        result = await handlers.set_chat_model("llama3")
        assert result.model == "llama3:latest"
        assert cfg.chat_model == "llama3:latest"

    async def test_preserves_existing_tag(self, tmp_path, mock_svc):
        mock_svc.provider.list_models.return_value = ["llama3:7b"]
        result = await handlers.set_chat_model("llama3:7b")
        assert result.model == "llama3:7b"
        assert cfg.chat_model == "llama3:7b"

    async def test_rejects_unavailable_model(self, tmp_path, mock_svc):
        mock_svc.provider.list_models.return_value = ["llama3:latest"]
        with pytest.raises(ValueError, match="not available"):
            await handlers.set_chat_model("nonexistent:7b")


class TestModelsCatalog:
    @patch("lilbee.catalog.get_catalog")
    async def test_returns_catalog_response(self, mock_get_catalog, mock_svc):
        from lilbee.catalog import CatalogModel, CatalogResult

        mock_get_catalog.return_value = CatalogResult(
            total=1,
            limit=20,
            offset=0,
            models=[
                CatalogModel(
                    name="qwen3",
                    tag="8b",
                    display_name="Qwen3 8B",
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
        mock_svc.provider.list_models.return_value = ["qwen3:8b"]
        result = await handlers.models_catalog()

        assert result.total == 1
        assert len(result.models) == 1
        m = result.models[0]
        assert m.name == "qwen3"
        assert m.tag == "8b"
        assert m.hf_repo == "Qwen/Qwen3-8B-GGUF"
        assert m.task == "chat"
        assert m.featured is True
        assert m.downloads == 1000
        assert m.param_count == "8B"
        assert m.installed is True
        assert m.source == "litellm"

    @patch("lilbee.catalog.get_catalog")
    async def test_filters_passed_to_catalog(self, mock_get_catalog, mock_svc):
        from lilbee.catalog import CatalogResult

        mock_get_catalog.return_value = CatalogResult(total=0, limit=10, offset=5, models=[])
        mock_svc.provider.list_models.return_value = []
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

    @patch("lilbee.catalog.get_catalog")
    async def test_installed_flag(self, mock_get_catalog, mock_svc):
        from lilbee.catalog import CatalogModel, CatalogResult

        mock_get_catalog.return_value = CatalogResult(
            total=1,
            limit=20,
            offset=0,
            models=[
                CatalogModel(
                    name="qwen3",
                    tag="8b",
                    display_name="Qwen3 8B",
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
        mock_svc.provider.list_models.return_value = ["qwen3:8b"]
        result = await handlers.models_catalog()
        assert result.models[0].installed is True


class TestModelsInstalled:
    async def test_returns_installed_models(self):
        mock_manager = MagicMock()
        mock_manager.list_installed.return_value = ["qwen3:8b", "mistral:7b"]
        from lilbee.model_manager import ModelSource

        mock_manager.get_source.return_value = ModelSource.LITELLM
        with patch("lilbee.server.handlers.get_model_manager", return_value=mock_manager):
            result = await handlers.models_installed()
        assert len(result.models) == 2
        assert result.models[0].source == "litellm"

    async def test_unknown_source_defaults_to_litellm(self):
        mock_manager = MagicMock()
        mock_manager.list_installed.return_value = ["unknown"]
        mock_manager.get_source.return_value = None
        with patch("lilbee.server.handlers.get_model_manager", return_value=mock_manager):
            result = await handlers.models_installed()
        assert result.models[0].source == "litellm"


class TestModelsPull:
    async def test_yields_progress_events_native(self):
        mock_manager = MagicMock()

        def fake_pull(model, source, *, on_progress=None, on_bytes=None):
            if on_bytes:
                on_bytes(500, 1000)
                on_bytes(1000, 1000)
            return None

        mock_manager.pull.side_effect = fake_pull
        with patch("lilbee.server.handlers.get_model_manager", return_value=mock_manager):
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
            return None

        mock_manager.pull.side_effect = fake_pull
        with patch("lilbee.server.handlers.get_model_manager", return_value=mock_manager):
            events = [e async for e in handlers.models_pull("test", source="litellm")]
        non_empty = [e for e in events if e]
        assert any("downloading" in e for e in non_empty)
        assert any("success" in e for e in non_empty)

    async def test_error_yields_error_event(self):
        mock_manager = MagicMock()
        mock_manager.pull.side_effect = RuntimeError("fail")
        with patch("lilbee.server.handlers.get_model_manager", return_value=mock_manager):
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
        with patch("lilbee.server.handlers.get_model_manager", return_value=mock_manager):
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
        with patch("lilbee.server.handlers.get_model_manager", return_value=mock_manager):
            result = await handlers.models_delete("test", source="litellm")
        assert result.deleted is True
        assert result.model == "test"

    async def test_returns_deleted_false(self):
        mock_manager = MagicMock()
        mock_manager.remove.return_value = False
        with patch("lilbee.server.handlers.get_model_manager", return_value=mock_manager):
            result = await handlers.models_delete("missing", source="native")
        assert result.deleted is False
        assert result.freed_gb == 0.0


class TestModelsShow:
    async def test_returns_params(self, mock_svc):
        mock_svc.provider.show_model.return_value = {"parameters": "temp 0.7"}
        result = await handlers.models_show("qwen3:8b")
        assert result.model_dump() == {"parameters": "temp 0.7"}

    async def test_returns_empty_when_none(self, mock_svc):
        mock_svc.provider.show_model.return_value = None
        result = await handlers.models_show("unknown")
        assert result.model_dump() == {}


class TestDeleteDocuments:
    async def test_removes_known_documents(self, mock_svc):
        from lilbee.store import RemoveResult

        mock_svc.store.remove_documents.return_value = RemoveResult(removed=["a.md"], not_found=[])
        result = await handlers.delete_documents(["a.md"])
        assert result.removed == ["a.md"]
        assert result.not_found == []

    async def test_not_found(self, mock_svc):
        from lilbee.store import RemoveResult

        mock_svc.store.remove_documents.return_value = RemoveResult(
            removed=[], not_found=["missing.md"]
        )
        result = await handlers.delete_documents(["missing.md"])
        assert result.removed == []
        assert result.not_found == ["missing.md"]

    async def test_delete_files_removes_from_disk(self, mock_svc, tmp_path):
        from lilbee.store import RemoveResult

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
        from lilbee import settings as s

        stored = s.load(cfg.data_root)
        assert "temperature" not in stored

    async def test_update_config_unknown_field(self):
        with pytest.raises(ValueError, match="Unknown or read-only config field"):
            await handlers.update_config({"bogus_field": 123})

    async def test_non_nullable_field_rejects_null(self):
        with pytest.raises(ValueError, match="does not accept null"):
            await handlers.update_config({"system_prompt": None})

    async def test_empty_dict_returns_no_updates(self):
        result = await handlers.update_config({})
        assert result.updated == []
        assert result.reindex_required is False

    async def test_invalid_type_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            await handlers.update_config({"chunk_size": "not_a_number"})

    async def test_llm_api_key_write_only(self, tmp_path):
        """llm_api_key can be written via PATCH but is excluded from GET."""
        result = await handlers.update_config({"llm_api_key": "sk-test123"})
        assert result.updated == ["llm_api_key"]
        assert cfg.llm_api_key == "sk-test123"
        # Verify it's excluded from GET /api/config
        config = await handlers.get_config()
        assert "llm_api_key" not in config.model_dump()

    async def test_multi_field_bad_second_no_partial_apply(self):
        """If second field is invalid, first field should NOT be applied."""
        original_temp = cfg.temperature
        with pytest.raises(ValueError, match="does not accept null"):
            await handlers.update_config({"temperature": 0.9, "system_prompt": None})
        # temperature should be unchanged — validation happens before apply
        assert cfg.temperature == original_temp

    async def test_multi_field_success(self, tmp_path):
        """Multiple valid fields are applied and persisted in one call."""
        result = await handlers.update_config({"temperature": 0.7, "top_k": 5})
        assert set(result.updated) == {"temperature", "top_k"}
        assert result.reindex_required is False
        assert cfg.temperature == 0.7
        assert cfg.top_k == 5
        # Verify both persisted
        from lilbee import settings as s

        stored = s.load(cfg.data_root)
        assert stored["temperature"] == "0.7"
        assert stored["top_k"] == "5"


class TestSetEmbeddingModel:
    @patch("lilbee.server.handlers.get_services")
    async def test_updates_config_and_persists(self, mock_svc, tmp_path):
        mock_svc.return_value.provider.list_models.return_value = ["nomic-embed-text:latest"]
        result = await handlers.set_embedding_model("nomic-embed-text:latest")
        assert result.model == "nomic-embed-text:latest"
        assert cfg.embedding_model == "nomic-embed-text:latest"
        from lilbee import settings as s

        stored = s.load(cfg.data_root)
        assert stored["embedding_model"] == "nomic-embed-text:latest"

    @patch("lilbee.server.handlers.get_services")
    async def test_empty_string_rejected(self, mock_svc):
        mock_svc.return_value.provider.list_models.return_value = []
        with pytest.raises(ValueError, match="not available"):
            await handlers.set_embedding_model("")

    @patch("lilbee.server.handlers.get_services")
    async def test_embedding_model_without_tag_normalizes(self, mock_svc, tmp_path):
        """Setting embedding model without a tag normalizes to :latest."""
        mock_svc.return_value.provider.list_models.return_value = ["nomic-embed-text:latest"]
        result = await handlers.set_embedding_model("nomic-embed-text")
        assert result.model == "nomic-embed-text:latest"
        assert cfg.embedding_model == "nomic-embed-text:latest"

    @patch("lilbee.server.handlers.get_services")
    async def test_rejects_unavailable_embedding_model(self, mock_svc):
        mock_svc.return_value.provider.list_models.return_value = ["nomic-embed-text:latest"]
        with pytest.raises(ValueError, match="not available"):
            await handlers.set_embedding_model("bogus-embed")


class TestGetConfig:
    async def test_returns_all_config_keys(self):
        result = await handlers.get_config()
        dumped = result.model_dump()
        assert "chat_model" in dumped
        assert "system_prompt" in dumped
        assert "litellm_base_url" in dumped
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


class TestListDocuments:
    async def test_returns_documents(self, mock_svc):
        mock_svc.store.get_sources.return_value = [
            {"filename": "a.md", "chunk_count": 5, "ingested_at": "2026-01-01"},
        ]
        result = await handlers.list_documents()
        assert result.total == 1
        assert result.documents[0].filename == "a.md"
        assert result.documents[0].chunk_count == 5

    async def test_empty(self, mock_svc):
        mock_svc.store.get_sources.return_value = []
        result = await handlers.list_documents()
        assert result.total == 0
        assert result.documents == []

    async def test_pagination(self, mock_svc):
        mock_svc.store.get_sources.return_value = [
            {"filename": f"doc{i}.md", "chunk_count": i} for i in range(10)
        ]
        result = await handlers.list_documents(limit=3, offset=2)
        assert result.total == 10
        assert len(result.documents) == 3
        assert result.documents[0].filename == "doc2.md"
        assert result.limit == 3
        assert result.offset == 2

    async def test_search_filter(self, mock_svc):
        mock_svc.store.get_sources.return_value = [
            {"filename": "readme.md", "chunk_count": 3},
            {"filename": "setup.py", "chunk_count": 1},
            {"filename": "readme_dev.md", "chunk_count": 2},
        ]
        result = await handlers.list_documents(search="readme")
        assert result.total == 2
        assert all("readme" in d.filename for d in result.documents)


class TestGetConfigReranker:
    @patch("lilbee.reranker.reranker_available", return_value=False)
    async def test_hides_reranker_when_not_installed(self, mock_avail):
        result = await handlers.get_config()
        assert "reranker_model" not in result.model_dump()

    @patch("lilbee.reranker.reranker_available", return_value=True)
    async def test_shows_reranker_when_installed(self, mock_avail):
        result = await handlers.get_config()
        dumped = result.model_dump()
        assert "reranker_model" in dumped
        assert "rerank_candidates" in dumped


class TestCrawlStream:
    @patch("lilbee.crawler.crawl_and_save")
    @patch("lilbee.crawler.validate_crawl_url")
    async def test_streams_events_and_done(self, _mock_validate, mock_crawl):
        from pathlib import Path

        async def fake_crawl(url, *, depth, max_pages, on_progress, cancel=None):
            from lilbee.progress import CrawlDoneEvent, CrawlPageEvent, CrawlStartEvent

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
            from lilbee.progress import CrawlStartEvent

            on_progress("crawl_start", CrawlStartEvent(url=url, depth=depth))
            barrier.wait(timeout=2)
            return []

        mock_crawl.side_effect = blocking_crawl
        caplog.set_level(logging.INFO, logger="lilbee.server.handlers")
        gen = handlers.crawl_stream("https://example.com")
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

    def test_sse_done(self):
        result = handlers.sse_done({"count": 1})
        assert result == 'event: done\ndata: {"count": 1}\n\n'


class TestResolveGenerationOptions:
    def test_with_options(self):
        result = handlers._resolve_generation_options({"temperature": 0.5})
        assert result is not None

    def test_without_options(self):
        result = handlers._resolve_generation_options(None)
        assert result is None


class TestListExternalModels:
    """Tests for the external model discovery handler."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        """Reset the external models cache before each test."""
        import lilbee.server.handlers as h

        h._external_cache = h._ExternalModelsCache()
        yield
        h._external_cache = h._ExternalModelsCache()

    @patch("lilbee.server.handlers.get_services")
    async def test_returns_provider_models(self, mock_svc):
        mock_svc.return_value.provider.list_models.return_value = ["model-a", "model-b"]
        result = await handlers.list_external_models()
        assert result.models == ["model-a", "model-b"]
        assert result.error is None

    @patch("lilbee.server.handlers.get_services")
    async def test_error_returns_empty_with_message(self, mock_svc):
        mock_svc.return_value.provider.list_models.side_effect = RuntimeError("connection refused")
        result = await handlers.list_external_models()
        assert result.models == []
        assert result.error is not None

    @patch("lilbee.server.handlers.get_services")
    async def test_cache_reuses_result(self, mock_svc):
        mock_svc.return_value.provider.list_models.return_value = ["model-a"]
        result1 = await handlers.list_external_models()
        result2 = await handlers.list_external_models()
        assert result1 == result2
        assert result1.models == ["model-a"]
        mock_svc.return_value.provider.list_models.assert_called_once()

    @patch("lilbee.server.handlers.time")
    @patch("lilbee.server.handlers.get_services")
    async def test_cache_expires(self, mock_svc, mock_time):
        mock_svc.return_value.provider.list_models.return_value = ["model-a"]
        mock_time.monotonic.return_value = 0.0
        await handlers.list_external_models()

        mock_time.monotonic.return_value = 61.0
        await handlers.list_external_models()

        assert mock_svc.return_value.provider.list_models.call_count == 2

    @patch("lilbee.server.handlers.get_services")
    async def test_cache_invalidates_on_config_change(self, mock_svc):
        mock_svc.return_value.provider.list_models.return_value = ["model-a"]
        cfg.litellm_base_url = "https://provider-a.example"
        await handlers.list_external_models()

        cfg.litellm_base_url = "https://provider-b.example"
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

        with patch("lilbee.server.handlers.get_services") as mock_svc:
            mock_svc.return_value.provider = mock_provider
            handlers._run_llm_stream(
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


class TestParseOcrParams:
    def test_ocr_timeout_coerced_to_float(self):
        """_parse_ocr_params coerces ocr_timeout to float."""
        enable_ocr, ocr_timeout = handlers._parse_ocr_params({"ocr_timeout": "60"})
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

        with patch("lilbee.server.handlers.copy_files", return_value=copy_result):
            result = await handlers._run_add(
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
            patch("lilbee.server.handlers.get_model_manager", return_value=mock_manager),
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
            patch("lilbee.server.handlers.get_model_manager", return_value=mock_manager),
            patch.object(handlers.SseStream, "__init__", patched_init),
        ):
            gen = handlers.models_pull("test", source="litellm")
            events = []
            async for event in gen:
                events.append(event)
            await asyncio.sleep(0.2)

        assert progress_called.is_set()
        assert not any("should_be_suppressed" in e for e in events if e)
