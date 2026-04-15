"""Tests for the Litestar HTTP adapter."""

from unittest import mock
from unittest.mock import AsyncMock

import pytest
from litestar.exceptions import NotAuthorizedException
from litestar.testing import TestClient

from lilbee.config import cfg


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
        mock_ask.assert_awaited_once_with(question="meaning?", top_k=0, options=None)

    @mock.patch(
        "lilbee.server.handlers.ask",
        new_callable=AsyncMock,
        return_value={"answer": "yes", "sources": []},
    )
    def test_forwards_top_k(self, mock_ask, client):
        resp = client.post("/api/ask", json={"question": "q", "top_k": 10})
        assert resp.status_code == 201
        mock_ask.assert_awaited_once_with(question="q", top_k=10, options=None)

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


class TestAskStreamRoute:
    @mock.patch("lilbee.server.handlers.ask_stream")
    def test_returns_sse(self, mock_stream, client):
        mock_stream.return_value = mock_async_gen("event: token\ndata: {}\n\n")
        resp = client.post("/api/ask/stream", json={"question": "hi"})
        assert resp.status_code == 201
        assert "text/event-stream" in resp.headers["content-type"]
        assert b"event: token" in resp.content


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
        mock_chat.assert_awaited_once_with(question="q", history=history, top_k=0, options=None)

    @mock.patch(
        "lilbee.server.handlers.chat",
        new_callable=AsyncMock,
        return_value={"answer": "a", "sources": []},
    )
    def test_default_empty_history(self, mock_chat, client):
        client.post("/api/chat", json={"question": "q"})
        mock_chat.assert_awaited_once_with(question="q", history=[], top_k=0, options=None)


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


class TestSyncRoute:
    @mock.patch("lilbee.server.handlers.sync_stream")
    def test_returns_sse(self, mock_stream, client):
        mock_stream.return_value = mock_async_gen("event: done\ndata: {}\n\n")
        resp = client.post("/api/sync")
        assert resp.status_code == 201
        assert b"event: done" in resp.content
        mock_stream.assert_called_once_with(enable_ocr=None)

    @mock.patch("lilbee.server.handlers.sync_stream")
    def test_enable_ocr(self, mock_stream, client):
        mock_stream.return_value = mock_async_gen("event: done\ndata: {}\n\n")
        client.post("/api/sync", json={"enable_ocr": True})
        mock_stream.assert_called_once_with(enable_ocr=True)


class TestModelsListRoute:
    @mock.patch(
        "lilbee.server.handlers.list_models",
        new_callable=AsyncMock,
        return_value={"chat": {}, "vision": {}},
    )
    def test_returns_json(self, mock_patched, client):
        resp = client.get("/api/models")
        assert resp.status_code == 200
        assert "chat" in resp.json()


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
        return_value={"chat_model": "qwen3:8b", "system_prompt": "You are helpful."},
    )
    def test_returns_json(self, mock_cfg, client):
        resp = client.get("/api/config")
        assert resp.status_code == 200
        assert resp.json()["chat_model"] == "qwen3:8b"
        assert "system_prompt" in resp.json()


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
            handler.fn._lilbee_read_only = True
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
