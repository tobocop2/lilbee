"""Tests for the framework-agnostic server handlers."""

import json
from dataclasses import fields, replace
from unittest.mock import MagicMock, patch

import pytest

from lilbee.config import cfg
from lilbee.ingest import SyncResult
from lilbee.server import handlers


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    """Redirect config paths for all handler tests."""
    snapshot = replace(cfg)
    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir()
    cfg.data_dir = tmp_path / "data"
    cfg.data_root = tmp_path
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    yield tmp_path
    for f in fields(cfg):
        setattr(cfg, f.name, getattr(snapshot, f.name))


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
        with patch("lilbee.cli.helpers.get_version", return_value="1.2.3"):
            result = await handlers.health()
        assert result == {"status": "ok", "version": "1.2.3"}


class TestStatus:
    async def test_returns_config_and_sources(self):
        expected = {"config": {}, "sources": [], "total_chunks": 0}
        with patch("lilbee.cli.helpers.gather_status", return_value=expected):
            result = await handlers.status()
        assert result == expected


class TestSearch:
    @patch("lilbee.query.search_context")
    async def test_returns_grouped_results(self, mock_search):
        mock_search.return_value = [
            {
                "source": "doc.pdf",
                "content_type": "pdf",
                "chunk": "hello",
                "_distance": 0.2,
                "page_start": 1,
                "page_end": 1,
                "line_start": 0,
                "line_end": 0,
            }
        ]
        result = await handlers.search("test", top_k=3)
        assert len(result) == 1
        assert result[0]["source"] == "doc.pdf"
        mock_search.assert_called_once_with("test", top_k=3)

    @patch("lilbee.query.search_context", return_value=[])
    async def test_empty_results(self, _search):
        result = await handlers.search("nothing")
        assert result == []


class TestAsk:
    @patch("lilbee.query.ask_raw")
    async def test_returns_answer_and_sources(self, mock_ask):
        from lilbee.query import AskResult

        mock_ask.return_value = AskResult(
            answer="42",
            sources=[{"source": "doc.pdf", "chunk": "c", "_distance": 0.1, "vector": [0.1]}],
        )
        result = await handlers.ask("what?")
        assert result["answer"] == "42"
        assert len(result["sources"]) == 1
        assert "vector" not in result["sources"][0]
        assert result["sources"][0]["distance"] == 0.1

    @patch("lilbee.query.ask_raw")
    async def test_no_sources(self, mock_ask):
        from lilbee.query import AskResult

        mock_ask.return_value = AskResult(answer="No docs found.", sources=[])
        result = await handlers.ask("what?")
        assert result["answer"] == "No docs found."
        assert result["sources"] == []


class TestAskStream:
    @patch("lilbee.query.search_context", return_value=[])
    async def test_no_results_yields_error(self, _search):
        events = [e async for e in handlers.ask_stream("test")]
        # First event is the empty yield, then the error
        non_empty = [e for e in events if e]
        assert len(non_empty) == 1
        parsed = json.loads(non_empty[0].split("data: ")[1].strip())
        assert "No relevant documents found" in parsed["message"]

    @patch("lilbee.query.search_context")
    async def test_yields_token_sources_done(self, mock_search):
        mock_search.return_value = [
            {
                "source": "a.pdf",
                "content_type": "pdf",
                "chunk": "text",
                "_distance": 0.1,
                "page_start": 1,
                "page_end": 1,
                "line_start": 0,
                "line_end": 0,
            }
        ]
        mock_chunk = MagicMock()
        mock_chunk.message.content = "answer"
        mock_stream = [mock_chunk]

        with patch("lilbee.server.handlers.asyncio.to_thread", return_value=mock_stream):
            events = [e async for e in handlers.ask_stream("question")]

        non_empty = [e for e in events if e]
        event_types = [e.split("\n")[0].replace("event: ", "") for e in non_empty]
        assert "token" in event_types
        assert "sources" in event_types
        assert "done" in event_types

    @patch("lilbee.query.search_context")
    async def test_ollama_error_yields_error_event(self, mock_search):
        mock_search.return_value = [
            {
                "source": "a.pdf",
                "content_type": "pdf",
                "chunk": "text",
                "_distance": 0.1,
                "page_start": 1,
                "page_end": 1,
                "line_start": 0,
                "line_end": 0,
            }
        ]

        with patch(
            "lilbee.server.handlers.asyncio.to_thread", side_effect=RuntimeError("model missing")
        ):
            events = [e async for e in handlers.ask_stream("question")]

        non_empty = [e for e in events if e]
        error_events = [e for e in non_empty if e.startswith("event: error")]
        assert len(error_events) == 1
        assert "model missing" in error_events[0]


class TestChat:
    @patch("lilbee.query.ask_raw")
    async def test_passes_history(self, mock_ask):
        from lilbee.query import AskResult

        mock_ask.return_value = AskResult(answer="ok", sources=[])
        history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        result = await handlers.chat("follow up", history)
        assert result["answer"] == "ok"
        mock_ask.assert_called_once_with("follow up", top_k=0, history=history)


class TestChatStream:
    @patch("lilbee.query.search_context", return_value=[])
    async def test_no_results_yields_error(self, _search):
        events = [e async for e in handlers.chat_stream("test", [])]
        non_empty = [e for e in events if e]
        assert any("error" in e for e in non_empty)

    @patch("lilbee.query.search_context")
    async def test_yields_events_with_history(self, mock_search):
        mock_search.return_value = [
            {
                "source": "a.pdf",
                "content_type": "pdf",
                "chunk": "text",
                "_distance": 0.1,
                "page_start": 1,
                "page_end": 1,
                "line_start": 0,
                "line_end": 0,
            }
        ]
        mock_chunk = MagicMock()
        mock_chunk.message.content = "reply"
        mock_stream = [mock_chunk]

        with patch("lilbee.server.handlers.asyncio.to_thread", return_value=mock_stream):
            history = [{"role": "user", "content": "hi"}]
            events = [e async for e in handlers.chat_stream("follow up", history)]

        non_empty = [e for e in events if e]
        event_types = [e.split("\n")[0].replace("event: ", "") for e in non_empty]
        assert "token" in event_types
        assert "done" in event_types

    @patch("lilbee.query.search_context")
    async def test_ollama_error_yields_error_event(self, mock_search):
        mock_search.return_value = [
            {
                "source": "a.pdf",
                "content_type": "pdf",
                "chunk": "text",
                "_distance": 0.1,
                "page_start": 1,
                "page_end": 1,
                "line_start": 0,
                "line_end": 0,
            }
        ]
        with patch(
            "lilbee.server.handlers.asyncio.to_thread",
            side_effect=ConnectionError("ollama down"),
        ):
            events = [e async for e in handlers.chat_stream("q", [])]

        non_empty = [e for e in events if e]
        error_events = [e for e in non_empty if e.startswith("event: error")]
        assert len(error_events) == 1
        assert "ollama down" in error_events[0]


class TestSyncStream:
    async def test_yields_progress_and_done(self):
        sync_result = SyncResult(added=["a.txt"], unchanged=0)

        async def fake_sync(
            force_rebuild=False, quiet=False, *, force_vision=False, on_progress=None
        ):
            if on_progress:
                on_progress("a.txt", "ingested", 1, 1)
            return sync_result

        with patch("lilbee.ingest.sync", side_effect=fake_sync):
            events = [e async for e in handlers.sync_stream()]

        assert len(events) >= 1
        # Last event should be "done"
        done_events = [e for e in events if e.startswith("event: done")]
        assert len(done_events) == 1
        done_data = json.loads(done_events[0].split("data: ")[1].strip())
        assert "a.txt" in done_data["added"]

    async def test_yields_progress_events(self):
        sync_result = SyncResult(added=["b.txt"])

        async def fake_sync(
            force_rebuild=False, quiet=False, *, force_vision=False, on_progress=None
        ):
            if on_progress:
                on_progress("b.txt", "ingested", 1, 1)
            return sync_result

        with patch("lilbee.ingest.sync", side_effect=fake_sync):
            events = [e async for e in handlers.sync_stream()]

        progress_events = [e for e in events if e.startswith("event: progress")]
        assert len(progress_events) >= 1
        data = json.loads(progress_events[0].split("data: ")[1].strip())
        assert data["file"] == "b.txt"
        assert data["status"] == "ingested"

    async def test_timeout_continue_when_no_progress(self):
        """Sync with no progress events exercises the TimeoutError continue branch."""
        import asyncio

        sync_result = SyncResult()

        async def slow_sync(
            force_rebuild=False, quiet=False, *, force_vision=False, on_progress=None
        ):
            await asyncio.sleep(0.2)  # force at least one timeout iteration
            return sync_result

        with patch("lilbee.ingest.sync", side_effect=slow_sync):
            events = [e async for e in handlers.sync_stream()]

        done_events = [e for e in events if e.startswith("event: done")]
        assert len(done_events) == 1


class TestListModels:
    @patch("lilbee.cli.chat.list_ollama_models")
    async def test_returns_catalogs(self, mock_list):
        mock_list.return_value = ["qwen3:8b", "mistral:7b"]
        result = await handlers.list_models()

        assert result["chat"]["active"] == cfg.chat_model
        assert isinstance(result["chat"]["catalog"], list)
        assert len(result["chat"]["catalog"]) > 0
        assert "qwen3:8b" in result["chat"]["installed"]

        assert isinstance(result["vision"]["catalog"], list)
        assert isinstance(result["vision"]["installed"], list)

    @patch("lilbee.cli.chat.list_ollama_models")
    async def test_installed_flag_in_catalog(self, mock_list):
        mock_list.return_value = ["qwen3:8b"]
        result = await handlers.list_models()

        catalog = result["chat"]["catalog"]
        qwen_entry = next(m for m in catalog if m["name"] == "qwen3:8b")
        assert qwen_entry["installed"] is True

        mistral_entry = next(m for m in catalog if m["name"] == "mistral:7b")
        assert mistral_entry["installed"] is False


class TestPullModel:
    async def test_yields_progress_and_done(self):
        mock_event = MagicMock()
        mock_event.status = "downloading"
        mock_event.total = 1000
        mock_event.completed = 500

        with patch("ollama.pull", return_value=[mock_event]):
            events = [e async for e in handlers.pull_model("test:latest")]

        non_empty = [e for e in events if e]
        progress_events = [e for e in non_empty if e.startswith("event: progress")]
        done_events = [e for e in non_empty if e.startswith("event: done")]
        assert len(progress_events) >= 1
        assert len(done_events) == 1

        data = json.loads(progress_events[0].split("data: ")[1].strip())
        assert data["model"] == "test:latest"
        assert data["completed"] == 500


class TestSetChatModel:
    async def test_updates_config_and_persists(self, tmp_path):
        result = await handlers.set_chat_model("llama3")
        assert result["model"] == "llama3:latest"
        assert cfg.chat_model == "llama3:latest"

    async def test_preserves_existing_tag(self, tmp_path):
        result = await handlers.set_chat_model("llama3:7b")
        assert result["model"] == "llama3:7b"
        assert cfg.chat_model == "llama3:7b"


class TestSetVisionModel:
    async def test_updates_config_and_persists(self, tmp_path):
        result = await handlers.set_vision_model("minicpm-v:latest")
        assert result["model"] == "minicpm-v:latest"
        assert cfg.vision_model == "minicpm-v:latest"

    async def test_empty_string_disables(self, tmp_path):
        cfg.vision_model = "some-model:latest"
        result = await handlers.set_vision_model("")
        assert result["model"] == ""
        assert cfg.vision_model == ""
