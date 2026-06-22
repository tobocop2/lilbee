"""Tests for server review-fix findings.

Covers: chunk_type decoder unification, stop-sequence forwarding, crawl
max_pages sentinel, one-shot memory extraction, chat search-unavailable
refusal parity, the catalog installed filter, ?source= 422 mapping, the
streaming model echo, and the add-files lock partition.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from litestar.testing import AsyncTestClient
from pydantic import ValidationError

import lilbee.server.handlers.rag as rag
from lilbee.app.services import set_services
from lilbee.data.store import ChunkType
from lilbee.providers.base import filter_options
from lilbee.retrieval.query.searcher import SEARCH_NEEDS_EMBEDDER, AskResult
from lilbee.server import auth as _auth_mod
from lilbee.server.models import CrawlRequest, decode_chunk_type


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_auth_mod.session_manager.token}"}


class TestStopForwarding:
    def test_filter_options_keeps_stop(self) -> None:
        """A stop sequence survives the LLMOptions allowlist instead of being dropped."""
        assert filter_options({"stop": ["END"], "temperature": 0.5})["stop"] == ["END"]

    def test_filter_options_omits_absent_stop(self) -> None:
        assert "stop" not in filter_options({"temperature": 0.5})


class TestChunkTypeDecoder:
    def test_raw_and_wiki_decode(self) -> None:
        assert decode_chunk_type("raw") == ChunkType.RAW
        assert decode_chunk_type("wiki") == ChunkType.WIKI

    def test_both_and_none_mean_no_filter(self) -> None:
        assert decode_chunk_type("both") is None
        assert decode_chunk_type(None) is None

    def test_invalid_gives_friendly_error(self) -> None:
        with pytest.raises(ValueError, match="'raw', 'wiki', 'both'"):
            decode_chunk_type("nonsense")


class TestCrawlMaxPagesSentinel:
    def test_zero_means_unlimited(self) -> None:
        assert CrawlRequest(url="https://x", max_pages=0).max_pages == 0

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CrawlRequest(url="https://x", max_pages=-1)


class TestStoreExtractedMemories:
    async def test_empty_answer_returns_empty(self, monkeypatch) -> None:
        monkeypatch.setattr(rag, "auto_extract_enabled", lambda: True)
        called = MagicMock()
        monkeypatch.setattr(rag, "auto_extract", called)
        assert await rag._store_extracted_memories("q", "") == []
        called.assert_not_called()

    async def test_off_returns_empty(self, monkeypatch) -> None:
        monkeypatch.setattr(rag, "auto_extract_enabled", lambda: False)
        called = MagicMock()
        monkeypatch.setattr(rag, "auto_extract", called)
        assert await rag._store_extracted_memories("q", "a") == []
        called.assert_not_called()

    async def test_runs_when_enabled(self, monkeypatch) -> None:
        monkeypatch.setattr(rag, "auto_extract_enabled", lambda: True)
        sentinel = [MagicMock()]
        monkeypatch.setattr(rag, "auto_extract", lambda q, a: sentinel)
        assert await rag._store_extracted_memories("q", "a") is sentinel


def _mock_searcher_services(*, search_unavailable: bool):
    from tests.conftest import make_mock_services

    searcher = MagicMock()
    searcher.search_unavailable.return_value = search_unavailable
    services = make_mock_services(searcher=searcher)
    return services, searcher


class TestAskMemoryExtraction:
    async def test_ask_extracts_when_search_available(self, monkeypatch) -> None:
        services, searcher = _mock_searcher_services(search_unavailable=False)
        searcher.ask_raw.return_value = AskResult(answer="grounded", sources=[])
        set_services(services)
        extracted: list[tuple[str, str]] = []
        monkeypatch.setattr(rag, "_store_extracted_memories", _record(extracted))
        try:
            resp = await rag.ask("q")
        finally:
            set_services(None)
        assert resp.answer == "grounded"
        assert extracted == [("q", "grounded")]

    async def test_ask_skips_extraction_on_refusal(self, monkeypatch) -> None:
        services, searcher = _mock_searcher_services(search_unavailable=True)
        searcher.ask_raw.return_value = AskResult(answer=SEARCH_NEEDS_EMBEDDER, sources=[])
        set_services(services)
        extracted: list[tuple[str, str]] = []
        monkeypatch.setattr(rag, "_store_extracted_memories", _record(extracted))
        try:
            resp = await rag.ask("q")
        finally:
            set_services(None)
        assert resp.answer == SEARCH_NEEDS_EMBEDDER
        assert extracted == []


def _record(sink: list[tuple[str, str]]):
    async def _store(question: str, answer: str) -> list:
        sink.append((question, answer))
        return []

    return _store


class TestChatSearchUnavailableParity:
    async def test_chat_refuses_when_search_unavailable(self) -> None:
        services, _ = _mock_searcher_services(search_unavailable=True)
        set_services(services)
        try:
            resp = await rag.chat("q", [])
        finally:
            set_services(None)
        assert resp.answer == SEARCH_NEEDS_EMBEDDER
        assert resp.sources == []
        assert resp.cited_sources == []

    async def test_chat_stream_refuses_when_search_unavailable(self) -> None:
        services, _ = _mock_searcher_services(search_unavailable=True)
        set_services(services)
        try:
            frames = [f async for f in rag.chat_stream("q", [])]
        finally:
            set_services(None)
        joined = "".join(frames)
        assert SEARCH_NEEDS_EMBEDDER in joined
        assert '"done"' in joined or "done" in joined


class TestSourceErrorMapping:
    async def test_delete_unknown_source_is_422(self) -> None:
        from lilbee.server import app as app_module

        async with AsyncTestClient(app_module.create_app()) as client:
            resp = await client.delete(
                "/api/models/whatever", params={"source": "bogus"}, headers=_auth_headers()
            )
        assert resp.status_code == 422

    async def test_pull_unknown_source_is_422(self) -> None:
        from lilbee.server import app as app_module

        async with AsyncTestClient(app_module.create_app()) as client:
            resp = await client.post(
                "/api/models/pull",
                json={"model": "x", "source": "bogus"},
                headers=_auth_headers(),
            )
        assert resp.status_code == 422


class TestCatalogInstalledFilter:
    async def test_route_forwards_installed_filter(self, monkeypatch) -> None:
        """The REST catalog route forwards installed + model_manager to get_catalog."""
        from lilbee.server.handlers import models as models_handler

        captured: dict[str, object] = {}

        def _fake_get_catalog(**kwargs):
            captured.update(kwargs)
            return MagicMock(total=0, limit=20, offset=0, has_more=False, models=[])

        monkeypatch.setattr(models_handler, "get_catalog", _fake_get_catalog)
        monkeypatch.setattr(models_handler, "enrich_catalog", lambda result, refs: [])
        services = MagicMock()
        services.registry.list_installed.return_value = []
        services.model_manager = MagicMock()
        monkeypatch.setattr(models_handler, "get_services", lambda: services)

        await models_handler.models_catalog(installed=True)
        assert captured["installed"] is True
        assert captured["model_manager"] is services.model_manager


class TestStreamingModelEcho:
    async def test_stream_passes_resolved_model_to_translator(self, monkeypatch) -> None:
        """The streamed chunks are built with the resolved model, not the request model."""
        from lilbee.server.chat_completions_api import routes
        from lilbee.server.chat_dispatch.canonical import CanonicalChatRequest, CanonicalMessage
        from lilbee.server.chat_dispatch.concurrency import ChatSlotGuard

        captured: dict[str, object] = {}

        async def _fake_stream(req: object):
            return
            yield  # pragma: no cover -- makes this an async generator

        def _fake_chunks(events, *, model, response_id, include_usage):
            captured["model"] = model

            async def _empty():
                return
                yield  # pragma: no cover

            return _empty()

        monkeypatch.setattr(routes, "dispatch_chat_stream", _fake_stream)
        monkeypatch.setattr(routes, "canonical_stream_to_completions_chunks", _fake_chunks)
        req = CanonicalChatRequest(
            model="requested/alias",
            messages=(CanonicalMessage(role="user", content="hi"),),
            stream=True,
        )
        async for _ in routes._gated_completions_stream(
            req, ChatSlotGuard(), model="resolved/canonical"
        ):
            pass
        assert captured["model"] == "resolved/canonical"
