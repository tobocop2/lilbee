"""Tests for the Litestar HTTP adapter."""

import asyncio
import json
from unittest import mock
from unittest.mock import AsyncMock

import pytest
from litestar.exceptions import NotAuthorizedException
from litestar.testing import TestClient

from lilbee.catalog.types import ModelTask
from lilbee.core.config import cfg
from lilbee.modelhub.role_validator import TaskMismatchError
from lilbee.runtime.progress import EmbedEvent, EventType
from tests.server.conftest import parse_sse_events


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    """Redirect config paths for all adapter tests."""
    snapshot = cfg.model_copy()
    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir(exist_ok=True)
    cfg.data_dir = tmp_path / "data"
    cfg.data_root = tmp_path
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    yield tmp_path
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture()
def client():
    import lilbee.server.auth as auth_mod
    from lilbee.server.app import create_app

    auth_mod.session_manager.token = None  # disable auth for route-level tests
    app = create_app()
    yield TestClient(app)
    auth_mod.session_manager.token = None


async def mock_async_gen(*events):
    for e in events:
        yield e


async def mock_failing_gen(*events, exc):
    for e in events:
        yield e
    raise exc


class TestHealthRoute:
    @mock.patch(
        "lilbee.server.handlers.health",
        new_callable=AsyncMock,
        return_value={"status": "ok", "version": "1.0.0"},
    )
    def test_returns_json(self, mock_patched, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["version"] == "1.0.0"


class TestStatusRoute:
    @mock.patch(
        "lilbee.server.handlers.status",
        new_callable=AsyncMock,
        return_value={"config": {}, "sources": [], "total_chunks": 0},
    )
    def test_returns_json(self, mock_patched, client):
        resp = client.get("/api/status")
        assert resp.status_code == 200
        assert resp.json()["total_chunks"] == 0


class TestSearchRoute:
    @mock.patch("lilbee.server.handlers.search", new_callable=AsyncMock, return_value=[])
    def test_empty_results(self, mock_search, client):
        resp = client.get("/api/search", params={"q": "hello", "top_k": "3"})
        assert resp.status_code == 200
        assert resp.json() == []
        mock_search.assert_awaited_once_with("hello", top_k=3, chunk_type=None)

    @mock.patch(
        "lilbee.server.handlers.search",
        new_callable=AsyncMock,
        return_value=[{"source": "a.md", "chunks": []}],
    )
    def test_with_results(self, mock_patched, client):
        resp = client.get("/api/search", params={"q": "test"})
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    @mock.patch("lilbee.server.handlers.search", new_callable=AsyncMock, return_value=[])
    def test_default_top_k(self, mock_search, client):
        client.get("/api/search", params={"q": "x"})
        mock_search.assert_awaited_once_with("x", top_k=5, chunk_type=None)

    def test_invalid_chunk_type_rejected_with_400(self, client):
        resp = client.get("/api/search", params={"q": "x", "chunk_type": "bogus"})
        assert resp.status_code == 400


class TestAskRoute:
    @mock.patch(
        "lilbee.server.handlers.ask",
        new_callable=AsyncMock,
        return_value={"answer": "42", "sources": []},
    )
    def test_returns_answer(self, mock_ask, client):
        resp = client.post("/api/ask", json={"question": "meaning?"})
        assert resp.status_code == 201
        assert resp.json()["answer"] == "42"
        mock_ask.assert_awaited_once_with(
            question="meaning?", top_k=0, options=None, chunk_type=None
        )

    @mock.patch(
        "lilbee.server.handlers.ask",
        new_callable=AsyncMock,
        return_value={"answer": "yes", "sources": []},
    )
    def test_forwards_top_k(self, mock_ask, client):
        resp = client.post("/api/ask", json={"question": "q", "top_k": 10})
        assert resp.status_code == 201
        mock_ask.assert_awaited_once_with(question="q", top_k=10, options=None, chunk_type=None)

    @mock.patch(
        "lilbee.server.handlers.ask",
        new_callable=AsyncMock,
        return_value={
            "answer": "ok",
            "sources": [
                {
                    "source": "doc.pdf",
                    "content_type": "pdf",
                    "chunk": "text",
                    "distance": 0.1,
                    "page_start": 1,
                    "page_end": 1,
                    "line_start": 0,
                    "line_end": 0,
                    "chunk_index": 0,
                }
            ],
        },
    )
    def test_returns_sources_as_typed_models(self, mock_patched, client):
        resp = client.post("/api/ask", json={"question": "q"})
        assert resp.status_code == 201
        sources = resp.json()["sources"]
        assert len(sources) == 1
        assert sources[0]["source"] == "doc.pdf"
        assert sources[0]["distance"] == 0.1

    @mock.patch(
        "lilbee.server.handlers.ask",
        new_callable=AsyncMock,
        return_value={"answer": "yes", "sources": []},
    )
    def test_forwards_chunk_type_raw(self, mock_ask, client):
        resp = client.post("/api/ask", json={"question": "q", "chunk_type": "raw"})
        assert resp.status_code == 201
        assert mock_ask.call_args.kwargs.get("chunk_type") == "raw"

    @mock.patch(
        "lilbee.server.handlers.ask",
        new_callable=AsyncMock,
        return_value={"answer": "yes", "sources": []},
    )
    def test_chunk_type_both_normalizes_to_none(self, mock_ask, client):
        resp = client.post("/api/ask", json={"question": "q", "chunk_type": "both"})
        assert resp.status_code == 201
        assert mock_ask.call_args.kwargs.get("chunk_type") is None

    def test_invalid_chunk_type_rejected_with_400(self, client):
        resp = client.post("/api/ask", json={"question": "q", "chunk_type": "bogus"})
        assert resp.status_code == 400

    @mock.patch("lilbee.server.handlers.ask", new_callable=AsyncMock)
    def test_model_not_found_returns_404(self, mock_ask, client):
        from lilbee.server.chat_dispatch.dispatch import ModelNotFoundError

        mock_ask.side_effect = ModelNotFoundError("vendor/missing.gguf")
        resp = client.post("/api/ask", json={"question": "q"})
        assert resp.status_code == 404
        assert "vendor/missing.gguf" in resp.text
        assert "lilbee model pull" in resp.text

    @mock.patch("lilbee.server.handlers.ask", new_callable=AsyncMock)
    def test_model_without_tool_support_returns_400(self, mock_ask, client):
        from lilbee.server.chat_dispatch.dispatch import ModelDoesNotSupportToolsError

        mock_ask.side_effect = ModelDoesNotSupportToolsError("vendor/m.gguf")
        resp = client.post("/api/ask", json={"question": "q"})
        assert resp.status_code == 400
        assert "vendor/m.gguf" in resp.text
        assert "tool-aware" in resp.text

    @mock.patch("lilbee.server.handlers.ask", new_callable=AsyncMock)
    def test_context_window_exceeded_returns_400(self, mock_ask, client):
        from lilbee.providers.base import ProviderError, ProviderErrorKind

        mock_ask.side_effect = ProviderError(
            "Prompt of 9000 tokens exceeds...", kind=ProviderErrorKind.CONTEXT_OVERFLOW
        )
        resp = client.post("/api/ask", json={"question": "q"})
        assert resp.status_code == 400
        assert "9000" in resp.text

    @mock.patch("lilbee.server.handlers.ask", new_callable=AsyncMock)
    def test_not_found_provider_error_returns_404(self, mock_ask, client):
        # A NOT_FOUND ProviderError (e.g. a role model isn't installed) maps to
        # 404, matching the typed ModelNotFoundError path through one classifier.
        from lilbee.providers.base import ProviderError, ProviderErrorKind

        mock_ask.side_effect = ProviderError(
            "embed model missing", kind=ProviderErrorKind.NOT_FOUND
        )
        resp = client.post("/api/ask", json={"question": "q"})
        assert resp.status_code == 404


class TestAskStreamRoute:
    @mock.patch("lilbee.server.handlers.ask_stream")
    def test_returns_sse(self, mock_stream, client):
        mock_stream.return_value = mock_async_gen("event: token\ndata: {}\n\n")
        resp = client.post("/api/ask/stream", json={"question": "hi"})
        assert resp.status_code == 201
        assert "text/event-stream" in resp.headers["content-type"]
        assert b"event: token" in resp.content

    @mock.patch("lilbee.server.handlers.ask_stream")
    def test_forwards_chunk_type(self, mock_stream, client):
        mock_stream.return_value = mock_async_gen("")
        resp = client.post("/api/ask/stream", json={"question": "hi", "chunk_type": "wiki"})
        assert resp.status_code == 201
        assert mock_stream.call_args.kwargs.get("chunk_type") == "wiki"


class TestChatRoute:
    @mock.patch(
        "lilbee.server.handlers.chat",
        new_callable=AsyncMock,
        return_value={"answer": "reply", "sources": []},
    )
    def test_forwards_history(self, mock_chat, client):
        history = [{"role": "user", "content": "hi"}]
        resp = client.post("/api/chat", json={"question": "q", "history": history})
        assert resp.status_code == 201
        mock_chat.assert_awaited_once_with(
            question="q", history=history, top_k=0, options=None, chunk_type=None
        )

    @mock.patch(
        "lilbee.server.handlers.chat",
        new_callable=AsyncMock,
        return_value={"answer": "a", "sources": []},
    )
    def test_default_empty_history(self, mock_chat, client):
        client.post("/api/chat", json={"question": "q"})
        mock_chat.assert_awaited_once_with(
            question="q", history=[], top_k=0, options=None, chunk_type=None
        )

    @mock.patch("lilbee.server.handlers.chat", new_callable=AsyncMock)
    def test_model_not_found_returns_404(self, mock_chat, client):
        """Dispatch's ``ModelNotFoundError`` becomes a 404 with the actionable hint."""
        from lilbee.server.chat_dispatch.dispatch import ModelNotFoundError

        mock_chat.side_effect = ModelNotFoundError("vendor/missing.gguf")
        resp = client.post("/api/chat", json={"question": "q", "history": []})
        assert resp.status_code == 404
        assert "vendor/missing.gguf" in resp.text
        assert "lilbee model pull" in resp.text

    @mock.patch("lilbee.server.handlers.chat", new_callable=AsyncMock)
    def test_model_without_tool_support_returns_400(self, mock_chat, client):
        """``ModelDoesNotSupportToolsError`` becomes a 400 with the actionable hint."""
        from lilbee.server.chat_dispatch.dispatch import ModelDoesNotSupportToolsError

        mock_chat.side_effect = ModelDoesNotSupportToolsError("vendor/m.gguf")
        resp = client.post("/api/chat", json={"question": "q", "history": []})
        assert resp.status_code == 400
        assert "vendor/m.gguf" in resp.text
        assert "tool-aware" in resp.text

    @mock.patch("lilbee.server.handlers.chat", new_callable=AsyncMock)
    def test_context_window_exceeded_returns_400(self, mock_chat, client):
        """A CONTEXT_OVERFLOW ``ProviderError`` from the dispatch becomes a 400."""
        from lilbee.providers.base import ProviderError, ProviderErrorKind

        mock_chat.side_effect = ProviderError(
            "Prompt of 9000 tokens exceeds...", kind=ProviderErrorKind.CONTEXT_OVERFLOW
        )
        resp = client.post("/api/chat", json={"question": "q", "history": []})
        assert resp.status_code == 400
        assert "9000" in resp.text

    @mock.patch("lilbee.server.handlers.chat", new_callable=AsyncMock)
    def test_value_error_becomes_validation_exception(self, mock_chat, client):
        """Plain ValueError surfaces as a 400 / ValidationException, mirroring /api/ask."""
        mock_chat.side_effect = ValueError("question must not be empty")
        resp = client.post("/api/chat", json={"question": "q", "history": []})
        assert resp.status_code == 400
        assert "empty" in resp.text

    @mock.patch("lilbee.server.handlers.chat", new_callable=AsyncMock)
    def test_unexpected_exception_becomes_503(self, mock_chat, client):
        """An uncaught provider failure surfaces as 503 instead of 500, mirroring /api/ask."""
        mock_chat.side_effect = RuntimeError("downstream broke")
        resp = client.post("/api/chat", json={"question": "q", "history": []})
        assert resp.status_code == 503
        assert "downstream broke" in resp.text


class TestChatStreamRoute:
    @mock.patch("lilbee.server.handlers.chat_stream")
    def test_returns_sse(self, mock_stream, client):
        mock_stream.return_value = mock_async_gen("event: done\ndata: {}\n\n")
        resp = client.post(
            "/api/chat/stream",
            json={"question": "hi", "history": []},
        )
        assert resp.status_code == 201
        assert b"event: done" in resp.content

    @mock.patch("lilbee.server.handlers.chat_stream")
    def test_forwards_chunk_type(self, mock_stream, client):
        mock_stream.return_value = mock_async_gen("")
        resp = client.post(
            "/api/chat/stream",
            json={"question": "hi", "history": [], "chunk_type": "raw"},
        )
        assert resp.status_code == 201
        assert mock_stream.call_args.kwargs.get("chunk_type") == "raw"


class TestStreamErrorEvents:
    """A failure inside a streaming handler surfaces as an SSE error event."""

    @mock.patch("lilbee.server.handlers.chat_stream")
    def test_chat_stream_failure_yields_sse_error(self, mock_stream, client):
        mock_stream.return_value = mock_failing_gen(
            exc=ValueError("Embedding dimension mismatch: expected 768, got 384")
        )
        resp = client.post("/api/chat/stream", json={"question": "hi", "history": []})
        assert resp.status_code == 201
        assert b"event: error" in resp.content
        assert b"Embedding dimension mismatch" in resp.content

    @mock.patch("lilbee.server.handlers.chat_stream")
    def test_chat_stream_failure_mid_stream_yields_sse_error(self, mock_stream, client):
        mock_stream.return_value = mock_failing_gen(
            "event: token\ndata: {}\n\n", exc=RuntimeError("worker died")
        )
        resp = client.post("/api/chat/stream", json={"question": "hi", "history": []})
        assert b"event: token" in resp.content
        assert b"event: error" in resp.content
        assert b"worker died" in resp.content

    @mock.patch("lilbee.server.handlers.ask_stream")
    def test_ask_stream_failure_yields_sse_error(self, mock_stream, client):
        mock_stream.return_value = mock_failing_gen(exc=RuntimeError("boom"))
        resp = client.post("/api/ask/stream", json={"question": "hi"})
        assert resp.status_code == 201
        assert b"event: error" in resp.content
        assert b"boom" in resp.content

    @mock.patch("lilbee.server.handlers.chat_stream")
    def test_lock_released_after_stream_failure(self, mock_stream, client):
        mock_stream.return_value = mock_failing_gen(exc=RuntimeError("boom"))
        client.post("/api/chat/stream", json={"question": "hi", "history": []})
        mock_stream.return_value = mock_async_gen("event: done\ndata: {}\n\n")
        resp = client.post("/api/chat/stream", json={"question": "hi", "history": []})
        assert resp.status_code == 201
        assert b"event: done" in resp.content


class TestChatRouteErrorMapping:
    """chat_route maps errors like ask_route: ValueError 400, others 503."""

    @mock.patch("lilbee.server.handlers.chat", new_callable=AsyncMock)
    def test_value_error_maps_to_400(self, mock_chat, client):
        mock_chat.side_effect = ValueError("Embedding dimension mismatch: expected 768, got 384")
        resp = client.post("/api/chat", json={"question": "q", "history": []})
        assert resp.status_code == 400
        assert "Embedding dimension mismatch" in resp.json()["detail"]

    @mock.patch("lilbee.server.handlers.chat", new_callable=AsyncMock)
    def test_unexpected_error_maps_to_503(self, mock_chat, client):
        mock_chat.side_effect = RuntimeError("worker died")
        resp = client.post("/api/chat", json={"question": "q", "history": []})
        assert resp.status_code == 503
        assert "worker died" in resp.json()["detail"]


class TestStreamSingleFlightGate:
    """The chat endpoints admit up to the backend's slot capacity, then 429."""

    @pytest.fixture(autouse=True)
    def _isolate_gate(self):
        """Reset the lru_cache-backed gate so each test starts at zero in-flight."""
        from lilbee.server.chat_dispatch.concurrency import chat_gate

        chat_gate.cache_clear()
        yield
        chat_gate.cache_clear()

    @staticmethod
    def _fill_gate() -> None:
        """Mark every chat slot busy so the next request must wait, then 429."""
        from lilbee.server.chat_dispatch.concurrency import chat_gate

        chat_gate()._in_flight = 999

    @mock.patch("lilbee.server.handlers.ask_stream")
    def test_concurrent_ask_stream_returns_429_after_wait_timeout(self, mock_stream, client):
        """A second /api/ask/stream while every slot is busy waits, then 429s."""
        from lilbee.server.chat_dispatch import concurrency as concurrency_mod

        mock_stream.return_value = mock_async_gen("event: token\ndata: {}\n\n")
        self._fill_gate()
        with mock.patch.object(concurrency_mod, "DEFAULT_BUSY_WAIT_S", 0.05):
            resp = client.post("/api/ask/stream", json={"question": "hi"})
        assert resp.status_code == 429
        assert resp.headers.get("retry-after") == "1"

    @mock.patch("lilbee.server.handlers.chat_stream")
    def test_concurrent_chat_stream_returns_429_after_wait_timeout(self, mock_stream, client):
        """A second /api/chat/stream while every slot is busy waits, then 429s."""
        from lilbee.server.chat_dispatch import concurrency as concurrency_mod

        mock_stream.return_value = mock_async_gen("event: done\ndata: {}\n\n")
        self._fill_gate()
        with mock.patch.object(concurrency_mod, "DEFAULT_BUSY_WAIT_S", 0.05):
            resp = client.post("/api/chat/stream", json={"question": "hi", "history": []})
        assert resp.status_code == 429
        assert resp.headers.get("retry-after") == "1"

    @mock.patch("lilbee.server.handlers.ask", new_callable=AsyncMock)
    def test_concurrent_ask_returns_429_after_wait_timeout(self, mock_ask, client):
        """Non-streaming /api/ask shares the same admission gate."""
        from lilbee.server.chat_dispatch import concurrency as concurrency_mod

        mock_ask.return_value = {"answer": "x", "sources": []}
        self._fill_gate()
        with mock.patch.object(concurrency_mod, "DEFAULT_BUSY_WAIT_S", 0.05):
            resp = client.post("/api/ask", json={"question": "hi"})
        assert resp.status_code == 429
        assert resp.headers.get("retry-after") == "1"

    @mock.patch("lilbee.server.handlers.chat", new_callable=AsyncMock)
    def test_concurrent_chat_returns_429_after_wait_timeout(self, mock_chat, client):
        """Non-streaming /api/chat shares the same admission gate."""
        from lilbee.server.chat_dispatch import concurrency as concurrency_mod

        mock_chat.return_value = {"answer": "x", "sources": []}
        self._fill_gate()
        with mock.patch.object(concurrency_mod, "DEFAULT_BUSY_WAIT_S", 0.05):
            resp = client.post("/api/chat", json={"question": "hi", "history": []})
        assert resp.status_code == 429
        assert resp.headers.get("retry-after") == "1"

    @mock.patch("lilbee.server.handlers.ask_stream")
    def test_sequential_streams_succeed(self, mock_stream, client):
        """After the first stream completes, its slot frees and the second is allowed."""
        from lilbee.server.chat_dispatch.concurrency import chat_gate

        mock_stream.return_value = mock_async_gen("event: token\ndata: {}\n\n")
        resp1 = client.post("/api/ask/stream", json={"question": "first"})
        assert resp1.status_code == 201
        assert b"event: token" in resp1.content
        # The slot must have been released by the gated_stream's finally.
        assert chat_gate().in_flight == 0
        mock_stream.return_value = mock_async_gen("event: token\ndata: {}\n\n")
        resp2 = client.post("/api/ask/stream", json={"question": "second"})
        assert resp2.status_code == 201

    @mock.patch("lilbee.server.handlers.ask_stream")
    def test_slot_released_on_provider_error(self, mock_stream, client):
        """When the SSE generator raises, the slot is still freed for the next caller."""
        from lilbee.server.chat_dispatch.concurrency import chat_gate

        async def _raising_gen():
            yield "event: token\ndata: {}\n\n"
            raise RuntimeError("provider blew up")

        mock_stream.return_value = _raising_gen()
        import contextlib

        # The TestClient sees the partial response; the generator's finally
        # block still runs and frees the slot.
        with contextlib.suppress(Exception):
            client.post("/api/ask/stream", json={"question": "boom"})
        assert chat_gate().in_flight == 0


class TestEmbeddingMismatchSurfacing:
    """A downloaded index built with a different embedder surfaces as an actionable
    error, not a generic 503/stream failure. The store raises
    EmbeddingModelMismatchError; non-stream routes translate it to a 409 whose
    ``extra`` carries the index's embedder so the client can offer to adopt it,
    and the streaming path emits an SSE error carrying the INDEX_EMBEDDER_MISMATCH
    code plus that embedder in ``detail``."""

    PERSISTED = "orgA/repoA/modelA.gguf"

    def _mismatch(self, *, dims_match: bool = True):
        from lilbee.data.store import EmbeddingModelMismatchError

        return EmbeddingModelMismatchError(
            persisted_model=self.PERSISTED,
            persisted_dim=768,
            current_model="orgB/repoB/modelB.gguf",
            current_dim=768 if dims_match else 384,
        )

    @mock.patch("lilbee.server.handlers.search", new_callable=AsyncMock)
    def test_search_route_returns_structured_409(self, mock_search, client):
        mock_search.side_effect = self._mismatch()
        resp = client.get("/api/search", params={"q": "x"})
        assert resp.status_code == 409
        body = resp.json()
        assert self.PERSISTED in body["detail"]
        assert body["extra"]["persisted_model"] == self.PERSISTED
        assert body["extra"]["adoptable"] is True

    @mock.patch("lilbee.server.handlers.ask", new_callable=AsyncMock)
    def test_ask_route_returns_structured_409(self, mock_ask, client):
        mock_ask.side_effect = self._mismatch()
        resp = client.post("/api/ask", json={"question": "q"})
        assert resp.status_code == 409
        assert resp.json()["extra"]["persisted_model"] == self.PERSISTED

    @mock.patch("lilbee.server.handlers.chat", new_callable=AsyncMock)
    def test_chat_route_returns_structured_409(self, mock_chat, client):
        mock_chat.side_effect = self._mismatch()
        resp = client.post("/api/chat", json={"question": "q", "history": []})
        assert resp.status_code == 409
        assert resp.json()["extra"]["persisted_model"] == self.PERSISTED

    @mock.patch("lilbee.server.handlers.search", new_callable=AsyncMock)
    def test_dim_incompatible_index_is_not_adoptable(self, mock_search, client):
        mock_search.side_effect = self._mismatch(dims_match=False)
        resp = client.get("/api/search", params={"q": "x"})
        assert resp.status_code == 409
        assert resp.json()["extra"]["adoptable"] is False

    def test_chat_stream_emits_mismatch_code_and_embedder(self):
        """The chat stream catches the mismatch and emits a coded SSE error."""
        self._assert_stream_emits_mismatch(lambda rag: rag.chat_stream(question="q", history=[]))

    def test_ask_stream_emits_mismatch_code_and_embedder(self):
        """The ask stream catches the mismatch and emits the same coded SSE error."""
        self._assert_stream_emits_mismatch(lambda rag: rag.ask_stream(question="q"))

    def _assert_stream_emits_mismatch(self, make_stream):
        from lilbee.runtime.progress import SseErrorCode
        from lilbee.server.handlers import rag

        async def _collect():
            with mock.patch.object(rag, "get_services") as mock_services:
                mock_services.return_value.searcher.build_rag_context.side_effect = self._mismatch()
                return [event async for event in make_stream(rag)]

        events = asyncio.run(_collect())
        payloads = [
            json.loads(e.split("data: ", 1)[1]) for e in events if e.startswith("event: error")
        ]
        assert len(payloads) == 1
        assert payloads[0]["code"] == SseErrorCode.INDEX_EMBEDDER_MISMATCH
        assert payloads[0]["detail"] == self.PERSISTED


class TestSyncRoute:
    @mock.patch("lilbee.server.handlers.sync_stream")
    def test_returns_sse(self, mock_stream, client):
        mock_stream.return_value = mock_async_gen("event: done\ndata: {}\n\n")
        resp = client.post("/api/sync")
        assert resp.status_code == 201
        assert b"event: done" in resp.content
        mock_stream.assert_called_once_with(
            enable_ocr=None, force_rebuild=False, retry_skipped=False
        )

    @mock.patch("lilbee.server.handlers.sync_stream")
    def test_enable_ocr(self, mock_stream, client):
        mock_stream.return_value = mock_async_gen("event: done\ndata: {}\n\n")
        client.post("/api/sync", json={"enable_ocr": True})
        mock_stream.assert_called_once_with(
            enable_ocr=True, force_rebuild=False, retry_skipped=False
        )

    @mock.patch("lilbee.server.handlers.sync_stream")
    def test_force_rebuild_plumbs_through(self, mock_stream, client):
        """REST callers can request a full rebuild via {"force_rebuild": true}."""
        mock_stream.return_value = mock_async_gen("event: done\ndata: {}\n\n")
        resp = client.post("/api/sync", json={"force_rebuild": True})
        assert resp.status_code == 201
        mock_stream.assert_called_once_with(
            enable_ocr=None, force_rebuild=True, retry_skipped=False
        )

    @mock.patch("lilbee.server.handlers.sync_stream")
    def test_retry_skipped_plumbs_through(self, mock_stream, client):
        """REST callers can retry previously-skipped files via {"retry_skipped": true}."""
        mock_stream.return_value = mock_async_gen("event: done\ndata: {}\n\n")
        resp = client.post("/api/sync", json={"retry_skipped": True})
        assert resp.status_code == 201
        mock_stream.assert_called_once_with(
            enable_ocr=None, force_rebuild=False, retry_skipped=True
        )


class TestModelsListRoute:
    @mock.patch(
        "lilbee.server.handlers.list_models",
        new_callable=AsyncMock,
    )
    def test_returns_json(self, mock_patched, client):
        from lilbee.server.handlers import ModelCatalogSection, ModelsResponse

        empty_section = ModelCatalogSection(active="", catalog=[], installed=[])
        mock_patched.return_value = ModelsResponse(
            chat=empty_section,
            embedding=empty_section,
            vision=empty_section,
            reranker=empty_section,
        )
        resp = client.get("/api/models")
        assert resp.status_code == 200
        body = resp.json()
        assert "chat" in body
        assert "embedding" in body
        assert "vision" in body
        assert "reranker" in body


class TestModelsExternalRoute:
    @mock.patch(
        "lilbee.server.handlers.list_external_models",
        new_callable=AsyncMock,
        return_value={"models": ["model-large", "model-small"]},
    )
    def test_returns_json(self, mock_patched, client):
        resp = client.get("/api/models/external")
        assert resp.status_code == 200
        assert resp.json()["models"] == ["model-large", "model-small"]


class TestModelsSetChatRoute:
    @mock.patch(
        "lilbee.server.handlers.set_chat_model",
        new_callable=AsyncMock,
        return_value={"model": "llama3:8b"},
    )
    def test_returns_model(self, mock_set, client):
        resp = client.put("/api/models/chat", json={"model": "llama3:8b"})
        assert resp.status_code == 200
        assert resp.json()["model"] == "llama3:8b"

    @mock.patch(
        "lilbee.server.handlers.set_chat_model",
        new_callable=AsyncMock,
        side_effect=ValueError("Model 'bogus:latest' is not available."),
    )
    def test_returns_422_for_unavailable_model(self, mock_set, client):
        resp = client.put("/api/models/chat", json={"model": "bogus"})
        assert resp.status_code == 422
        assert "not available" in resp.json()["detail"]


class TestSetEmbeddingModelRoute:
    @mock.patch(
        "lilbee.server.handlers.set_embedding_model",
        new_callable=AsyncMock,
        side_effect=ValueError("Model 'bogus:latest' is not available."),
    )
    def test_returns_422_for_unavailable_embedding(self, mock_set, client):
        resp = client.put("/api/models/embedding", json={"model": "bogus"})
        assert resp.status_code == 422
        assert "not available" in resp.json()["detail"]


class TestSetVisionModelRoute:
    @mock.patch(
        "lilbee.server.handlers.set_vision_model",
        new_callable=AsyncMock,
        return_value={"model": "lightonocr:2-1b"},
    )
    def test_returns_model(self, mock_set, client):
        resp = client.put("/api/models/vision", json={"model": "lightonocr:2-1b"})
        assert resp.status_code == 200
        assert resp.json()["model"] == "lightonocr:2-1b"

    @mock.patch(
        "lilbee.server.handlers.set_vision_model",
        new_callable=AsyncMock,
        return_value={"model": ""},
    )
    def test_accepts_empty_string_to_unset(self, mock_set, client):
        resp = client.put("/api/models/vision", json={"model": ""})
        assert resp.status_code == 200
        assert resp.json()["model"] == ""

    @mock.patch(
        "lilbee.server.handlers.set_vision_model",
        new_callable=AsyncMock,
        side_effect=TaskMismatchError("qwen3:0.6b", ModelTask.CHAT, ModelTask.VISION),
    )
    def test_returns_422_for_task_mismatch(self, mock_set, client):
        resp = client.put("/api/models/vision", json={"model": "qwen3:0.6b"})
        assert resp.status_code == 422
        assert "not vision" in resp.json()["detail"]
        assert "/api/models/chat" in resp.json()["detail"]


class TestSetRerankerModelRoute:
    @mock.patch(
        "lilbee.server.handlers.set_reranker_model",
        new_callable=AsyncMock,
        return_value={"model": "bge-reranker-v2-m3:latest"},
    )
    def test_returns_model(self, mock_set, client):
        resp = client.put("/api/models/reranker", json={"model": "bge-reranker-v2-m3:latest"})
        assert resp.status_code == 200
        assert resp.json()["model"] == "bge-reranker-v2-m3:latest"

    @mock.patch(
        "lilbee.server.handlers.set_reranker_model",
        new_callable=AsyncMock,
        return_value={"model": ""},
    )
    def test_accepts_empty_string_to_unset(self, mock_set, client):
        resp = client.put("/api/models/reranker", json={"model": ""})
        assert resp.status_code == 200
        assert resp.json()["model"] == ""

    @mock.patch(
        "lilbee.server.handlers.set_reranker_model",
        new_callable=AsyncMock,
        side_effect=TaskMismatchError("qwen3:0.6b", ModelTask.CHAT, ModelTask.RERANK),
    )
    def test_returns_422_for_task_mismatch(self, mock_set, client):
        resp = client.put("/api/models/reranker", json={"model": "qwen3:0.6b"})
        assert resp.status_code == 422
        assert "not rerank" in resp.json()["detail"]

    @mock.patch("lilbee.server.handlers.models._require_model_available")
    @mock.patch("lilbee.server.handlers.models._set_model", new_callable=AsyncMock)
    def test_route_resolves_bare_repo_to_installed_quant(self, mock_set, mock_available, client):
        """POSTing a bare ``hf_repo`` resolves to whichever quant is installed.

        Locks the Litestar pipeline (route serialization + handler +
        ``_require_model_for_task`` resolution) so the response body
        carries the canonical full ref of the installed quant, not the
        bare repo the user posted.
        """
        full_ref = "gpustack/bge-reranker-v2-m3-GGUF/bge-reranker-Q4_K_M.gguf"
        mock_available.return_value = full_ref
        mock_set.return_value = {"model": full_ref}
        resp = client.put(
            "/api/models/reranker",
            json={"model": "gpustack/bge-reranker-v2-m3-GGUF"},
        )
        assert resp.status_code == 200
        assert resp.json()["model"] == full_ref
        assert mock_set.call_args.args[1] == full_ref


class TestModelsCatalogRoute:
    @mock.patch(
        "lilbee.server.handlers.models_catalog",
        new_callable=AsyncMock,
        return_value={"total": 0, "limit": 20, "offset": 0, "models": []},
    )
    def test_returns_json(self, mock_cat, client):
        resp = client.get("/api/models/catalog")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_invalid_sort_returns_422(self, client):
        """An unknown sort value must surface as 422, not 500."""
        resp = client.get("/api/models/catalog?sort=random&featured=true")
        assert resp.status_code == 422

    def test_invalid_size_returns_422(self, client):
        """An unknown size bucket must surface as 422, not 500."""
        resp = client.get("/api/models/catalog?size=huge&featured=true")
        assert resp.status_code == 422

    def test_invalid_task_returns_422(self, client):
        """An unknown task must surface as 422, not 500."""
        resp = client.get("/api/models/catalog?task=bogus&featured=true")
        assert resp.status_code == 422


class TestModelsInstalledRoute:
    @mock.patch(
        "lilbee.server.handlers.models_installed",
        new_callable=AsyncMock,
        return_value={"models": []},
    )
    def test_returns_json(self, mock_inst, client):
        resp = client.get("/api/models/installed")
        assert resp.status_code == 200
        assert resp.json()["models"] == []


class TestModelsPullRoute:
    @mock.patch("lilbee.server.handlers.models_pull")
    def test_returns_sse(self, mock_pull, client):
        mock_pull.return_value = mock_async_gen("event: progress\ndata: {}\n\n")
        resp = client.post("/api/models/pull", json={"model": "test", "source": "native"})
        assert resp.status_code == 201


class TestModelsShowRoute:
    @mock.patch(
        "lilbee.server.handlers.models_show",
        new_callable=AsyncMock,
        return_value={"parameters": "temp 0.7"},
    )
    def test_returns_json(self, mock_show, client):
        resp = client.post("/api/models/show", json={"model": "test"})
        assert resp.status_code == 201
        assert resp.json()["parameters"] == "temp 0.7"


class TestModelsDeleteRoute:
    @mock.patch(
        "lilbee.server.handlers.models_delete",
        new_callable=AsyncMock,
        return_value={"deleted": True, "model": "test", "freed_gb": 0.0},
    )
    def test_returns_json(self, mock_del, client):
        resp = client.delete("/api/models/test")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True


class TestConfigRoute:
    @mock.patch(
        "lilbee.server.handlers.get_config",
        new_callable=AsyncMock,
        return_value={"chat_model": "qwen3:8b", "rag_system_prompt": "You are helpful."},
    )
    def test_returns_json(self, mock_cfg, client):
        resp = client.get("/api/config")
        assert resp.status_code == 200
        assert resp.json()["chat_model"] == "qwen3:8b"
        assert "rag_system_prompt" in resp.json()


class TestConfigDefaultsRoute:
    def test_returns_writable_defaults(self, client):
        from lilbee.core.config import DEFAULT_CRAWL_EXCLUDE_PATTERNS

        resp = client.get("/api/config/defaults")
        assert resp.status_code == 200
        data = resp.json()
        # Scalar defaults present.
        assert data["chunk_size"] == 512
        assert data["top_k"] == 12
        # Nullable defaults come through as null.
        assert data["crawl_max_depth"] is None
        assert data["crawl_max_pages"] is None
        # default_factory fields come through as their evaluated list.
        assert data["crawl_exclude_patterns"] == list(DEFAULT_CRAWL_EXCLUDE_PATTERNS)
        # Model role defaults surface so UI reset affordances can restore
        # them via PUT /api/models/<role>, even though the fields are
        # non-writable on PATCH /api/config.
        assert data["chat_model"] == "Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"
        # Wiki cfg fields are writable and appear with their declared defaults.
        assert data["wiki_prune_raw"] is False
        assert data["wiki_embedding_faithfulness_threshold"] == 0.5


class TestConfigUpdateRoute:
    @mock.patch(
        "lilbee.server.handlers.update_config",
        new_callable=AsyncMock,
        return_value={"updated": ["temperature"], "reindex_required": False},
    )
    def test_returns_json(self, mock_update, client):
        resp = client.patch("/api/config", json={"temperature": 0.7})
        assert resp.status_code == 200
        assert resp.json()["updated"] == ["temperature"]

    @mock.patch(
        "lilbee.server.handlers.update_config",
        new_callable=AsyncMock,
        side_effect=ValueError("Unknown or read-only config field: bogus"),
    )
    def test_unknown_field_returns_error(self, mock_update, client):
        resp = client.patch("/api/config", json={"bogus": 1})
        assert resp.status_code == 400

    def test_pydantic_validation_error_returns_400(self, client):
        from pydantic import ValidationError

        @mock.patch(
            "lilbee.server.handlers.update_config",
            new_callable=AsyncMock,
            side_effect=ValidationError.from_exception_data(
                "Config",
                [
                    {
                        "type": "int_parsing",
                        "loc": ("chunk_size",),
                        "msg": "Input should be a valid integer",
                        "input": "not_a_number",
                    }
                ],
            ),
        )
        def _inner(mock_update):
            resp = client.patch("/api/config", json={"chunk_size": "not_a_number"})
            assert resp.status_code == 400

        _inner()

    def test_crawl_exclude_patterns_rejects_invalid_regex(self, client):
        # Invalid character class: `p-a` is a backwards range; should be rejected
        # at PATCH time rather than crashing the next crawl with an opaque error.
        resp = client.patch("/api/config", json={"crawl_exclude_patterns": ["[wp-a]"]})
        assert resp.status_code == 400
        body = resp.text
        assert "crawl_exclude_patterns" in body or "regex" in body or "[0]" in body

    def test_crawl_exclude_patterns_accepts_valid_regex(self, client):
        resp = client.patch(
            "/api/config", json={"crawl_exclude_patterns": [r"/page/\d+/?$", r"/tag/"]}
        )
        assert resp.status_code == 200
        assert "crawl_exclude_patterns" in resp.json()["updated"]


class TestModelsSetEmbeddingRoute:
    @mock.patch(
        "lilbee.server.handlers.set_embedding_model",
        new_callable=AsyncMock,
        return_value={"model": "nomic-embed-text:latest"},
    )
    def test_returns_model(self, mock_set, client):
        resp = client.put("/api/models/embedding", json={"model": "nomic-embed-text:latest"})
        assert resp.status_code == 200
        assert resp.json()["model"] == "nomic-embed-text:latest"


class TestDocumentsListRoute:
    @mock.patch(
        "lilbee.server.handlers.list_documents",
        new_callable=AsyncMock,
        return_value={"documents": [], "total": 0},
    )
    def test_returns_json(self, mock_list, client):
        resp = client.get("/api/documents")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0


class TestDocumentsRemoveRoute:
    @mock.patch(
        "lilbee.server.handlers.delete_documents",
        new_callable=AsyncMock,
        return_value={"removed": ["a.md"], "not_found": []},
    )
    def test_returns_json(self, mock_remove, client):
        resp = client.post("/api/documents/remove", json={"names": ["a.md"]})
        assert resp.status_code == 201
        assert resp.json()["removed"] == ["a.md"]


class TestOpenAPISchema:
    def test_schema_endpoint_returns_json(self, client):
        resp = client.get("/schema/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert schema["info"]["title"] == "lilbee"
        assert "/api/health" in schema["paths"]
        assert "/api/ask" in schema["paths"]

    def test_redoc_endpoint(self, client):
        resp = client.get("/schema/redoc")
        assert resp.status_code == 200
        assert b"redoc" in resp.content.lower()

    def test_swagger_endpoint(self, client):
        resp = client.get("/schema/swagger")
        assert resp.status_code == 200


class TestCors:
    @mock.patch(
        "lilbee.server.handlers.health",
        new_callable=AsyncMock,
        return_value={"status": "ok", "version": "1.0.0"},
    )
    def test_configured_origin_allowed(self, mock_patched):
        from litestar.testing import TestClient

        cfg.cors_origins = ["app://custom.example"]
        from lilbee.server.app import create_app

        with TestClient(create_app()) as c:
            resp = c.options(
                "/api/health",
                headers={
                    "Origin": "app://custom.example",
                    "Access-Control-Request-Method": "GET",
                },
            )
        assert resp.headers.get("access-control-allow-origin") == "app://custom.example"

    @mock.patch(
        "lilbee.server.handlers.health",
        new_callable=AsyncMock,
        return_value={"status": "ok", "version": "1.0.0"},
    )
    def test_multiple_origins_allowed(self, mock_patched):
        from litestar.testing import TestClient

        cfg.cors_origins = ["app://obsidian.md", "https://my-app.com"]
        from lilbee.server.app import create_app

        with TestClient(create_app()) as c:
            for origin in cfg.cors_origins:
                resp = c.options(
                    "/api/health",
                    headers={
                        "Origin": origin,
                        "Access-Control-Request-Method": "GET",
                    },
                )
                assert resp.headers.get("access-control-allow-origin") == origin

    @mock.patch(
        "lilbee.server.handlers.health",
        new_callable=AsyncMock,
        return_value={"status": "ok", "version": "1.0.0"},
    )
    def test_localhost_origin_allowed(self, mock_patched):
        from litestar.testing import TestClient

        cfg.cors_origins = ["http://localhost:7433"]
        from lilbee.server.app import create_app

        with TestClient(create_app()) as c:
            resp = c.options(
                "/api/health",
                headers={
                    "Origin": "http://localhost:7433",
                    "Access-Control-Request-Method": "GET",
                },
            )
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:7433"


class TestCorsDefaultRegex:
    """Default cors_origin_regex should allow Obsidian (desktop + mobile) and any
    localhost origin out of the box, without any config or env var."""

    @staticmethod
    def _preflight(origin: str) -> str | None:
        from lilbee.server.app import create_app

        with TestClient(create_app()) as c:
            resp = c.options(
                "/api/health",
                headers={
                    "Origin": origin,
                    "Access-Control-Request-Method": "GET",
                },
            )
        return resp.headers.get("access-control-allow-origin")

    @mock.patch(
        "lilbee.server.handlers.health",
        new_callable=AsyncMock,
        return_value={"status": "ok", "version": "1.0.0"},
    )
    def test_allows_obsidian_desktop(self, mock_patched):
        assert self._preflight("app://obsidian.md") == "app://obsidian.md"

    @mock.patch(
        "lilbee.server.handlers.health",
        new_callable=AsyncMock,
        return_value={"status": "ok", "version": "1.0.0"},
    )
    def test_allows_obsidian_mobile_capacitor(self, mock_patched):
        assert self._preflight("capacitor://localhost") == "capacitor://localhost"

    @mock.patch(
        "lilbee.server.handlers.health",
        new_callable=AsyncMock,
        return_value={"status": "ok", "version": "1.0.0"},
    )
    def test_allows_http_localhost_any_port(self, mock_patched):
        assert self._preflight("http://localhost:3000") == "http://localhost:3000"
        assert self._preflight("http://localhost:8080") == "http://localhost:8080"
        assert self._preflight("https://localhost:8443") == "https://localhost:8443"

    @mock.patch(
        "lilbee.server.handlers.health",
        new_callable=AsyncMock,
        return_value={"status": "ok", "version": "1.0.0"},
    )
    def test_allows_loopback_ipv4(self, mock_patched):
        assert self._preflight("http://127.0.0.1:7433") == "http://127.0.0.1:7433"

    @mock.patch(
        "lilbee.server.handlers.health",
        new_callable=AsyncMock,
        return_value={"status": "ok", "version": "1.0.0"},
    )
    def test_allows_loopback_ipv6(self, mock_patched):
        assert self._preflight("http://[::1]:7433") == "http://[::1]:7433"

    @mock.patch(
        "lilbee.server.handlers.health",
        new_callable=AsyncMock,
        return_value={"status": "ok", "version": "1.0.0"},
    )
    def test_rejects_random_remote(self, mock_patched):
        assert self._preflight("https://evil.example.com") is None
        assert self._preflight("app://some-other-app.md") is None

    @mock.patch(
        "lilbee.server.handlers.health",
        new_callable=AsyncMock,
        return_value={"status": "ok", "version": "1.0.0"},
    )
    def test_regex_and_explicit_list_combine(self, mock_patched):
        # User adds an explicit remote origin; default regex is untouched.
        cfg.cors_origins = ["https://my-remote-app.example"]
        assert self._preflight("https://my-remote-app.example") == "https://my-remote-app.example"
        # Default regex still applies on top.
        assert self._preflight("app://obsidian.md") == "app://obsidian.md"

    @mock.patch(
        "lilbee.server.handlers.health",
        new_callable=AsyncMock,
        return_value={"status": "ok", "version": "1.0.0"},
    )
    def test_match_nothing_regex_disables_default(self, mock_patched):
        # Documented opt-out: set regex to ^$ so only the explicit list is consulted.
        cfg.cors_origin_regex = "^$"
        cfg.cors_origins = ["https://only-this.example"]
        assert self._preflight("https://only-this.example") == "https://only-this.example"
        assert self._preflight("app://obsidian.md") is None
        assert self._preflight("http://localhost:3000") is None


class TestCrawlRoute:
    @mock.patch("lilbee.server.handlers.crawl_stream")
    def test_post_crawl_streams_sse(self, mock_stream, client):
        mock_stream.return_value = mock_async_gen(
            "event: crawl_start\ndata: {}\n\n",
            "event: done\ndata: {}\n\n",
        )
        resp = client.post("/api/crawl", json={"url": "https://example.com", "depth": 1})
        assert resp.status_code == 201
        assert "text/event-stream" in resp.headers["content-type"]
        assert b"crawl_start" in resp.content

    @mock.patch(
        "lilbee.server.handlers.crawl_stream",
        side_effect=ValueError("URL must start with http:// or https://"),
    )
    def test_post_crawl_invalid_url(self, mock_stream, client):
        resp = client.post("/api/crawl", json={"url": "ftp://bad.com"})
        assert resp.status_code == 400

    @mock.patch("lilbee.server.handlers.crawl_stream")
    def test_post_crawl_accepts_null_max_pages(self, mock_stream, client):
        """null max_pages / depth passes through as None (unbounded)."""
        mock_stream.return_value = mock_async_gen("event: done\ndata: {}\n\n")
        resp = client.post(
            "/api/crawl",
            json={"url": "https://example.com", "depth": None, "max_pages": None},
        )
        assert resp.status_code == 201
        kwargs = mock_stream.call_args.kwargs
        assert kwargs["depth"] is None
        assert kwargs["max_pages"] is None

    @mock.patch("lilbee.server.handlers.crawl_stream")
    def test_post_crawl_omitted_fields_default_to_none(self, mock_stream, client):
        """When depth/max_pages are omitted, the handler receives None (unbounded)."""
        mock_stream.return_value = mock_async_gen("event: done\ndata: {}\n\n")
        resp = client.post("/api/crawl", json={"url": "https://example.com"})
        assert resp.status_code == 201
        kwargs = mock_stream.call_args.kwargs
        assert kwargs["depth"] is None
        assert kwargs["max_pages"] is None

    def test_post_crawl_rejects_zero_max_pages(self, client):
        """max_pages=0 is invalid (use null for unbounded)."""
        resp = client.post("/api/crawl", json={"url": "https://example.com", "max_pages": 0})
        assert resp.status_code == 400


class TestSetupCrawlerRoutes:
    """bb-wq8g: GET /setup/crawler/status + POST /setup/crawler."""

    @mock.patch("lilbee.server.routes.setup.crawler_available", return_value=True)
    @mock.patch("lilbee.server.routes.setup.chromium_installed", return_value=True)
    def test_status_when_installed(self, _mock_chromium, _mock_pkg, client):
        resp = client.get("/setup/crawler/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["installed"] is True
        assert body["package_installed"] is True
        assert body["chromium_installed"] is True
        assert body["component"] == "chromium"
        assert "browsers_path" in body

    @mock.patch("lilbee.server.routes.setup.chromium_installed", return_value=False)
    def test_status_when_missing(self, _mock, client):
        resp = client.get("/setup/crawler/status")
        assert resp.status_code == 200
        assert resp.json()["installed"] is False

    def test_post_setup_crawler_streams_setup_events(self, client):
        """Stub bootstrap_chromium to emit a setup_done event via on_progress."""
        from lilbee.runtime.progress import EventType, SetupDoneEvent, SetupStartEvent

        async def _fake_bootstrap(on_progress=None):
            if on_progress is not None:
                on_progress(EventType.SETUP_START, SetupStartEvent(component="chromium"))
                on_progress(
                    EventType.SETUP_DONE,
                    SetupDoneEvent(component="chromium", success=True, error=None),
                )

        with mock.patch("lilbee.server.routes.setup.bootstrap_chromium", new=_fake_bootstrap):
            resp = client.post("/setup/crawler")
            assert resp.status_code == 201
            assert "text/event-stream" in resp.headers["content-type"]
            assert b"event: setup_start" in resp.content
            assert b"event: setup_done" in resp.content

    def test_post_setup_crawler_emits_error_when_bootstrap_raises(self, client):
        """Non-zero-exit bootstrap routes through sse_error before done."""
        from lilbee.crawler import CrawlerBrowserError

        async def _fake_bootstrap(on_progress=None):
            raise CrawlerBrowserError("network unreachable")

        with mock.patch("lilbee.server.routes.setup.bootstrap_chromium", new=_fake_bootstrap):
            resp = client.post("/setup/crawler")
            assert resp.status_code == 201
            assert b"event: error" in resp.content
            assert b"network unreachable" in resp.content


class TestCreateAppReexport:
    @mock.patch("lilbee.server.app.create_app")
    def test_lazy_import(self, mock_create):
        from lilbee.server import create_app

        create_app()
        mock_create.assert_called_once()


class TestLifespan:
    @mock.patch("lilbee.server.app.get_services")
    async def test_calls_get_services(self, mock_get_svc):
        mock_svc = mock.MagicMock()
        mock_get_svc.return_value = mock_svc
        from lilbee.server.app import _lifespan

        async with _lifespan(mock.MagicMock()):
            pass
        mock_get_svc.assert_called()
        mock_svc.embedder.validate_model.assert_called_once()

    @mock.patch("lilbee.server.app.get_services", side_effect=RuntimeError("no provider"))
    async def test_provider_failure_does_not_block(self, mock_get_svc):
        from lilbee.server.app import _lifespan

        async with _lifespan(mock.MagicMock()):
            pass

    @mock.patch("lilbee.server.app.get_services")
    async def test_validate_model_failure_does_not_block(self, mock_get_svc):
        mock_svc = mock.MagicMock()
        mock_svc.embedder.validate_model.side_effect = RuntimeError("no model")
        mock_get_svc.return_value = mock_svc
        from lilbee.server.app import _lifespan

        async with _lifespan(mock.MagicMock()):
            pass
        mock_get_svc.assert_called()


class TestSessionManagerPersistence:
    """Cover SessionManager.load_or_generate token-reuse semantics."""

    @pytest.fixture()
    def fresh_manager(self):
        """Yield a new SessionManager + clean server.json path between tests."""
        from lilbee.server.auth import SessionManager, server_json_path

        path = server_json_path()
        path.unlink(missing_ok=True)
        yield SessionManager()
        path.unlink(missing_ok=True)

    def test_creates_new_token_when_file_missing(self, fresh_manager):
        from lilbee.server.auth import server_json_path

        token = fresh_manager.load_or_generate()
        assert isinstance(token, str)
        assert len(token) >= 32
        assert fresh_manager.token == token
        assert server_json_path().exists()

    def test_reuses_existing_valid_token(self, fresh_manager):
        first = fresh_manager.load_or_generate()
        # Simulate a server restart: drop the in-memory token, keep file.
        fresh_manager.token = None
        second = fresh_manager.load_or_generate()
        assert second == first
        assert fresh_manager.token == first

    def test_regenerates_when_file_is_empty(self, fresh_manager):
        from lilbee.server.auth import server_json_path

        path = server_json_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        token = fresh_manager.load_or_generate()
        assert isinstance(token, str)
        assert len(token) >= 32

    def test_regenerates_when_json_invalid(self, fresh_manager):
        from lilbee.server.auth import server_json_path

        path = server_json_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not-json")
        token = fresh_manager.load_or_generate()
        assert isinstance(token, str)
        assert len(token) >= 32

    def test_regenerates_when_payload_not_dict(self, fresh_manager):
        from lilbee.server.auth import server_json_path

        path = server_json_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(["array", "not", "dict"]))
        token = fresh_manager.load_or_generate()
        assert isinstance(token, str)
        assert len(token) >= 32

    def test_regenerates_when_token_field_missing(self, fresh_manager):
        from lilbee.server.auth import server_json_path

        path = server_json_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"other": "value"}))
        token = fresh_manager.load_or_generate()
        assert isinstance(token, str)
        assert len(token) >= 32

    def test_regenerates_when_token_too_short(self, fresh_manager):
        from lilbee.server.auth import server_json_path

        path = server_json_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"token": "short"}))
        token = fresh_manager.load_or_generate()
        assert token != "short"
        assert len(token) >= 32

    def test_regenerates_when_token_not_string(self, fresh_manager):
        from lilbee.server.auth import server_json_path

        path = server_json_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"token": 12345}))
        token = fresh_manager.load_or_generate()
        assert isinstance(token, str)
        assert len(token) >= 32

    def test_returns_none_when_path_unreadable(self, fresh_manager, tmp_path):
        # _read_persisted_token swallows OSError and returns None, covered
        # by passing a path that is a directory (read_text raises IsADirectoryError,
        # an OSError subclass).
        from lilbee.server.auth import SessionManager

        unreadable = tmp_path / "as_dir"
        unreadable.mkdir()
        assert SessionManager._read_persisted_token(unreadable) is None

    @mock.patch("lilbee.server.app.get_services")
    async def test_lifespan_reuses_token_across_consecutive_runs(self, mock_get_svc):
        """Two _lifespan invocations against the same data_dir reuse the token.

        Simulates ``lilbee serve`` restart: after the first lifespan exits,
        ``cleanup()`` removes server.json, so by design a fresh token is
        generated next time. This regression test pins the contract: if the
        cleanup hook is suppressed (e.g. crash exit), the next startup
        REUSES the persisted token rather than rotating.
        """
        from lilbee.server import auth as auth_mod
        from lilbee.server.app import _lifespan

        mock_svc = mock.MagicMock()
        mock_get_svc.return_value = mock_svc

        # First "serve": generate fresh.
        async with _lifespan(mock.MagicMock()):
            first = auth_mod.session_manager.token
        assert first is not None
        # Simulate a crash exit: cleanup did run normally, but the token
        # file remains on disk to prove reuse semantics under intact files.
        # Manually rewrite the file to the same token (mimicking a clean
        # write that survived the crash) and clear the in-memory state.
        auth_mod.server_json_path().parent.mkdir(parents=True, exist_ok=True)
        auth_mod.server_json_path().write_text(json.dumps({"token": first}))
        auth_mod.session_manager.token = None

        # Second "serve": must reuse the same token.
        async with _lifespan(mock.MagicMock()):
            second = auth_mod.session_manager.token
        assert second == first


class TestAuthMiddleware:
    @pytest.fixture()
    def middleware(self):
        from lilbee.server.auth import AuthMiddleware

        app = AsyncMock()
        return AuthMiddleware(app)

    @pytest.mark.asyncio
    async def test_non_http_scope_passes_through(self, middleware):
        scope = {"type": "websocket"}
        await middleware(scope, AsyncMock(), AsyncMock())
        middleware.app.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_options_method_passes_through(self, middleware):
        import lilbee.server.auth as auth_mod

        old = auth_mod.session_manager.token
        auth_mod.session_manager.token = "secret"
        try:
            scope = {"type": "http", "method": "OPTIONS", "headers": []}
            await middleware(scope, AsyncMock(), AsyncMock())
            middleware.app.assert_awaited_once()
        finally:
            auth_mod.session_manager.token = old

    @pytest.mark.asyncio
    async def test_read_only_handler_passes_through(self, middleware):
        import lilbee.server.auth as auth_mod

        old = auth_mod.session_manager.token
        auth_mod.session_manager.token = "secret"
        try:
            handler = mock.MagicMock()

            @auth_mod.read_only
            def _ro_route() -> None: ...

            handler.fn = _ro_route
            scope = {"type": "http", "method": "GET", "headers": [], "route_handler": handler}
            await middleware(scope, AsyncMock(), AsyncMock())
            middleware.app.assert_awaited_once()
        finally:
            auth_mod.session_manager.token = old

    @pytest.mark.asyncio
    async def test_invalid_token_raises(self, middleware):
        import lilbee.server.auth as auth_mod

        old = auth_mod.session_manager.token
        auth_mod.session_manager.token = "valid_token"
        try:
            scope = {
                "type": "http",
                "method": "POST",
                "headers": [(b"authorization", b"Bearer wrong_token")],
            }
            with pytest.raises(NotAuthorizedException):
                await middleware(scope, AsyncMock(), AsyncMock())
        finally:
            auth_mod.session_manager.token = old

    @pytest.mark.asyncio
    async def test_empty_token_raises(self, middleware):
        """When session token is empty string, requests are denied."""
        import lilbee.server.auth as auth_mod

        old = auth_mod.session_manager.token
        auth_mod.session_manager.token = ""
        try:
            scope = {
                "type": "http",
                "method": "POST",
                "headers": [(b"authorization", b"Bearer anything")],
            }
            with pytest.raises(NotAuthorizedException, match="not initialized"):
                await middleware(scope, AsyncMock(), AsyncMock())
        finally:
            auth_mod.session_manager.token = old


class TestAuthRequiredRoutes:
    """Verify mutating endpoints return 401 without a valid bearer token."""

    @pytest.fixture()
    def auth_client(self):
        import lilbee.server.auth as auth_mod
        from lilbee.server.app import create_app

        auth_mod.session_manager.token = "test-secret"
        app = create_app()
        yield TestClient(app)
        auth_mod.session_manager.token = None

    def test_patch_config_requires_auth(self, auth_client):
        resp = auth_client.patch("/api/config", json={"temperature": 0.5})
        assert resp.status_code == 401

    def test_put_models_embedding_requires_auth(self, auth_client):
        resp = auth_client.put("/api/models/embedding", json={"model": "nomic-embed-text:latest"})
        assert resp.status_code == 401


class _StubEmbedder:
    truncated_total = 0

    def embed_batch(self, texts, source="", on_progress=None, **_kwargs):
        if on_progress is not None:
            on_progress(
                EventType.EMBED,
                EmbedEvent(file=source, chunk=len(texts), total_chunks=len(texts)),
            )
        return [[0.1] * cfg.embedding_dim for _ in texts]


@pytest.fixture()
def dataset_store(tmp_path):
    """Real store with one indexed page wired into services."""
    import lilbee.app.services as svc_mod
    from lilbee.data.store import Store
    from tests.conftest import make_mock_services

    store = Store(cfg.model_copy(update={"lancedb_dir": tmp_path / "lancedb"}))
    store.add_page_texts(
        [{"source": "doc.pdf", "page": 1, "text": "page one", "content_type": "pdf"}]
    )
    store.upsert_source("doc.pdf", "h", 1)
    svc_mod.set_services(make_mock_services(store=store, embedder=_StubEmbedder()))
    yield store
    svc_mod.set_services(None)


class TestExportRoute:
    def test_parquet_download_by_default(self, dataset_store, client):
        resp = client.get("/api/export")
        assert resp.status_code == 200
        assert resp.content.startswith(b"PAR1")
        assert resp.headers["content-disposition"] == 'attachment; filename="pages.parquet"'

    def test_jsonl_format(self, dataset_store, client):
        resp = client.get("/api/export", params={"format": "jsonl"})
        assert resp.status_code == 200
        row = json.loads(resp.content.decode().splitlines()[0])
        assert row["source"] == "doc.pdf"
        assert resp.headers["content-disposition"] == 'attachment; filename="pages.jsonl"'

    def test_read_only_without_token(self, dataset_store):
        import lilbee.server.auth as auth_mod
        from lilbee.server.app import create_app

        auth_mod.session_manager.token = "test-secret"
        try:
            resp = TestClient(create_app()).get("/api/export")
        finally:
            auth_mod.session_manager.token = None
        assert resp.status_code == 200

    def test_unknown_source_is_400(self, dataset_store, client):
        resp = client.get("/api/export", params={"source": "missing.pdf"})
        assert resp.status_code == 400
        assert "Source not found" in resp.json()["detail"]

    def test_bad_format_is_400(self, dataset_store, client):
        resp = client.get("/api/export", params={"format": "csv"})
        assert resp.status_code == 400


class TestImportRoute:
    def test_round_trip_streams_progress_and_done(self, dataset_store, client):
        data = client.get("/api/export", params={"format": "jsonl"}).content
        dataset_store.delete_by_source("doc.pdf")

        resp = client.post("/api/import", params={"format": "jsonl"}, content=data)
        assert resp.status_code == 201
        assert "text/event-stream" in resp.headers["content-type"]
        events = parse_sse_events(resp.content)
        assert any(name == "embed" for name, _ in events), events
        done = next(payload for name, payload in events if name == "done")
        assert done["command"] == "import"
        assert done["sources"] == ["doc.pdf"]
        assert done["pages"] == 1
        assert done["chunks"] > 0

    def test_format_required(self, dataset_store, client):
        resp = client.post("/api/import", content=b"{}")
        assert resp.status_code == 400
        assert "format is required" in resp.json()["detail"]

    def test_bad_format_is_400(self, dataset_store, client):
        resp = client.post("/api/import", params={"format": "csv"}, content=b"{}")
        assert resp.status_code == 400

    def test_empty_body_streams_error(self, dataset_store, client):
        resp = client.post("/api/import", params={"format": "jsonl"}, content=b"")
        assert resp.status_code == 201
        error = next(p for name, p in parse_sse_events(resp.content) if name == "error")
        assert "no pages" in error["message"]

    def test_requires_auth(self, dataset_store):
        import lilbee.server.auth as auth_mod
        from lilbee.server.app import create_app

        auth_mod.session_manager.token = "test-secret"
        try:
            resp = TestClient(create_app()).post(
                "/api/import", params={"format": "jsonl"}, content=b""
            )
        finally:
            auth_mod.session_manager.token = None
        assert resp.status_code == 401
