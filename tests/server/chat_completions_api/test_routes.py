"""Route tests for the OpenAI-shaped chat-completions surface."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
from litestar import Litestar
from litestar.testing import AsyncTestClient

from lilbee.app.services import set_services
from lilbee.providers.base import (
    ChatResult,
    FinishReason,
    ToolCall,
    ToolCallDelta,
)
from lilbee.server import auth as _auth_mod
from lilbee.server.chat_completions_api.routes import completions_router
from lilbee.server.chat_dispatch.concurrency import ChatSlotGuard, chat_gate

INSTALLED_REF = "vendor/Model-GGUF/model-Q4.gguf"


def _h() -> dict[str, str]:
    """Auth header carrying the active session token."""
    return {"Authorization": f"Bearer {_auth_mod.session_manager.token}"}


def _build_app() -> Litestar:
    return Litestar(route_handlers=[completions_router])


def _installed_chat_model(ref: str = INSTALLED_REF) -> MagicMock:
    manifest = MagicMock()
    manifest.ref = ref
    manifest.task = "chat"
    manifest.downloaded_at = "2026-05-15T00:00:00+00:00"
    return manifest


def _services_with(provider: MagicMock, installed: list[MagicMock]) -> Any:
    from tests.conftest import make_mock_services

    # The chat route admits against this and advertises this window; give bare
    # mocks valid scalars so the gate math and /v1/models shape stay correct.
    provider.max_concurrent_chats.return_value = 1
    provider.served_chat_ctx.return_value = None
    services = make_mock_services(provider=provider)
    services.registry.list_installed = MagicMock(return_value=installed)
    _populate_known_models(services, installed)
    return services


def _populate_known_models(services: Any, installed: list[MagicMock]) -> None:
    """Mirror installed refs into the KnownModelCache mock the route consults.

    Production builds compose the cache from the registry + Ollama tags +
    frontier APIs; the route layer only sees the unified set. Route tests
    that pre-load the registry must also pre-load the cache so resolve()
    finds the same refs.
    """
    refs = {m.ref for m in installed}
    services.known_models.refs = MagicMock(return_value=refs)

    def _resolve(model: str) -> str | None:
        if model in refs:
            return model
        if "/" not in model and ":" in model:
            prefixed = f"ollama/{model}"
            if prefixed in refs:
                return prefixed
        return None

    services.known_models.resolve = MagicMock(side_effect=_resolve)


@pytest.fixture
def services_with_chat_model(monkeypatch):
    from lilbee.core.config import cfg

    monkeypatch.setattr(cfg, "chat_model", INSTALLED_REF)
    provider = MagicMock()
    provider.chat.return_value = ChatResult(
        text="hello", tool_calls=(), finish_reason=FinishReason.STOP
    )
    provider.supports_tools.return_value = False
    services = _services_with(provider, [_installed_chat_model()])
    set_services(services)
    yield services
    set_services(None)


@pytest.fixture
def _auth_token():
    """Install an auth token for the duration of the test."""
    previous = _auth_mod.session_manager.token
    previous_init = _auth_mod.session_manager._initialized
    _auth_mod.session_manager.token = "test-token-" + "x" * 40
    _auth_mod.session_manager._initialized = True
    yield
    _auth_mod.session_manager.token = previous
    _auth_mod.session_manager._initialized = previous_init


@pytest.fixture(autouse=True)
def _clear_chat_lock():
    """Drop the cached chat lock between tests so each test starts clean."""
    chat_gate.cache_clear()
    yield
    chat_gate.cache_clear()


class FakeProviderStream:
    """Async iterator that mimics the provider streaming protocol."""

    def __init__(self, frames: list[Any]) -> None:
        self._frames = list(frames)
        self.closed = False

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        if not self._frames:
            raise StopAsyncIteration
        return self._frames.pop(0)

    def close(self) -> None:
        self.closed = True


def _sse_to_chunks(body: bytes) -> list[Any]:
    """Parse an OpenAI-style SSE body into a list of dicts plus the [DONE] sentinel."""
    out: list[Any] = []
    for line in body.decode().split("\n\n"):
        line = line.strip()
        if not line:
            continue
        assert line.startswith("data: "), line
        payload = line.removeprefix("data: ")
        if payload == "[DONE]":
            out.append(payload)
        else:
            out.append(json.loads(payload))
    return out


class TestListModelsEndpoint:
    async def test_returns_installed_chat_models(self, services_with_chat_model, _auth_token):
        services_with_chat_model.registry.list_installed = MagicMock(
            return_value=[
                _installed_chat_model("a/Model/a.gguf"),
                _installed_chat_model("b/Model/b.gguf"),
            ]
        )
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.get("/v1/models", headers=_h())
        body = resp.json()
        assert resp.status_code == 200
        assert body["object"] == "list"
        assert [m["id"] for m in body["data"]] == ["a/Model/a.gguf", "b/Model/b.gguf"]
        assert all(m["object"] == "model" for m in body["data"])
        assert all(m["owned_by"] == "lilbee" for m in body["data"])

    async def test_skips_non_chat_models(self, services_with_chat_model, _auth_token):
        chat = _installed_chat_model("a/Model/a.gguf")
        embed = _installed_chat_model("b/Embed/e.gguf")
        embed.task = "embedding"
        services_with_chat_model.registry.list_installed = MagicMock(return_value=[chat, embed])
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.get("/v1/models", headers=_h())
        assert [m["id"] for m in resp.json()["data"]] == ["a/Model/a.gguf"]

    async def test_created_field_uses_downloaded_at_when_set(
        self, services_with_chat_model, _auth_token
    ):
        chat = _installed_chat_model()
        chat.downloaded_at = "2026-05-15T00:00:00+00:00"
        services_with_chat_model.registry.list_installed = MagicMock(return_value=[chat])
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.get("/v1/models", headers=_h())
        expected = int(datetime.fromisoformat("2026-05-15T00:00:00+00:00").timestamp())
        assert resp.json()["data"][0]["created"] == expected

    async def test_context_window_advertised_for_active_model_only(
        self, services_with_chat_model, _auth_token, monkeypatch
    ):
        """The served window is advertised on the active chat model so a client
        trims history to fit; other listed models carry no window."""
        from lilbee.core.config import cfg

        other = _installed_chat_model("z/Other/o.gguf")
        services_with_chat_model.registry.list_installed = MagicMock(
            return_value=[_installed_chat_model(INSTALLED_REF), other]
        )
        monkeypatch.setattr(cfg, "chat_model", INSTALLED_REF)
        services_with_chat_model.provider.served_chat_ctx.return_value = 40960
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.get("/v1/models", headers=_h())
        by_id = {m["id"]: m for m in resp.json()["data"]}
        assert by_id[INSTALLED_REF]["context_window"] == 40960
        assert by_id["z/Other/o.gguf"]["context_window"] is None

    async def test_remote_configured_chat_model_is_listed_first(
        self, services_with_chat_model, _auth_token, monkeypatch
    ):
        """A remote-configured chat model is served without a registry entry, so
        the listing includes it (same rule the launcher applies to its picker)."""
        from lilbee.core.config import cfg

        remote_ref = "ollama/qwen3:8b"
        monkeypatch.setattr(cfg, "chat_model", remote_ref)
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.get("/v1/models", headers=_h())
        ids = [m["id"] for m in resp.json()["data"]]
        assert ids == [remote_ref, INSTALLED_REF]
        by_id = {m["id"]: m for m in resp.json()["data"]}
        # The unregistered remote ref carries the newest native timestamp so
        # clients sorting by created desc don't bury the model lilbee serves.
        newest_native = int(datetime.fromisoformat("2026-05-15T00:00:00+00:00").timestamp())
        assert by_id[remote_ref]["created"] == newest_native

    async def test_local_configured_chat_model_is_listed_first(
        self, services_with_chat_model, _auth_token, monkeypatch
    ):
        """A configured local model leads the listing even when it is not
        alphabetically first, mirroring the launcher's picker order."""
        from lilbee.core.config import cfg

        configured = "z/Last/z.gguf"
        services_with_chat_model.registry.list_installed = MagicMock(
            return_value=[
                _installed_chat_model("a/Model/a.gguf"),
                _installed_chat_model(configured),
            ]
        )
        monkeypatch.setattr(cfg, "chat_model", configured)
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.get("/v1/models", headers=_h())
        assert [m["id"] for m in resp.json()["data"]] == [configured, "a/Model/a.gguf"]

    async def test_remote_configured_chat_model_is_not_duplicated(
        self, services_with_chat_model, _auth_token, monkeypatch
    ):
        from lilbee.core.config import cfg

        remote_ref = "ollama/qwen3:8b"
        monkeypatch.setattr(cfg, "chat_model", remote_ref)
        services_with_chat_model.registry.list_installed = MagicMock(
            return_value=[_installed_chat_model(remote_ref)]
        )
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.get("/v1/models", headers=_h())
        assert [m["id"] for m in resp.json()["data"]] == [remote_ref]

    async def test_subdir_native_model_listed_and_servable(self, _auth_token, monkeypatch):
        """F6: a registered subdir-filename giant is advertised by /v1/models and
        resolves for a completion (its abs-path inconsistency was a symptom of it
        not being registerable; once F2 registers it, the surface is consistent).
        """
        from lilbee.core.config import cfg

        subdir_ref = "unsloth/MiniMax-M2-GGUF/Q4_K_M/MiniMax-M2-Q4_K_M-00001-of-00003.gguf"
        monkeypatch.setattr(cfg, "chat_model", subdir_ref)
        provider = MagicMock()
        provider.chat.return_value = ChatResult(
            text="ok", tool_calls=(), finish_reason=FinishReason.STOP
        )
        provider.supports_tools.return_value = False
        services = _services_with(provider, [_installed_chat_model(subdir_ref)])
        set_services(services)
        try:
            async with AsyncTestClient(_build_app()) as client:
                listed = await client.get("/v1/models", headers=_h())
                assert subdir_ref in [m["id"] for m in listed.json()["data"]]
                completion = await client.post(
                    "/v1/chat/completions",
                    headers=_h(),
                    json={
                        "model": subdir_ref,
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
            assert completion.status_code == 200
            assert completion.json()["choices"][0]["message"]["content"] == "ok"
        finally:
            set_services(None)

    async def test_created_is_zero_when_downloaded_at_unparseable(
        self, services_with_chat_model, _auth_token
    ):
        chat = _installed_chat_model()
        chat.downloaded_at = "not-a-date"
        services_with_chat_model.registry.list_installed = MagicMock(return_value=[chat])
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.get("/v1/models", headers=_h())
        assert resp.json()["data"][0]["created"] == 0

    async def test_created_is_zero_when_downloaded_at_empty(
        self, services_with_chat_model, _auth_token
    ):
        chat = _installed_chat_model()
        chat.downloaded_at = ""
        services_with_chat_model.registry.list_installed = MagicMock(return_value=[chat])
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.get("/v1/models", headers=_h())
        assert resp.json()["data"][0]["created"] == 0


class TestNonStreamingCompletion:
    async def test_text_only_response(self, services_with_chat_model, _auth_token):
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers=_h(),
                json={
                    "model": INSTALLED_REF,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "chat.completion"
        assert body["model"] == INSTALLED_REF
        assert body["choices"][0]["message"]["content"] == "hello"
        assert body["choices"][0]["finish_reason"] == "stop"

    async def test_unknown_model_returns_404_openai_error_body(
        self, services_with_chat_model, _auth_token
    ):
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers=_h(),
                json={
                    "model": "missing/model.gguf",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert resp.status_code == 404
        body = resp.json()
        assert body["error"]["code"] == "model_not_found"
        assert body["error"]["type"] == "invalid_request_error"
        assert "missing/model.gguf" in body["error"]["message"]

    async def test_tools_against_non_tool_model_returns_400(
        self, services_with_chat_model, _auth_token
    ):
        services_with_chat_model.provider.supports_tools.return_value = False
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers=_h(),
                json={
                    "model": INSTALLED_REF,
                    "messages": [{"role": "user", "content": "x"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "search",
                                "description": "",
                                "parameters": {},
                            },
                        }
                    ],
                },
            )
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["code"] == "model_does_not_support_tools"

    async def test_with_tools_returns_tool_call_response(
        self, services_with_chat_model, _auth_token
    ):
        services_with_chat_model.provider.supports_tools.return_value = True
        services_with_chat_model.provider.chat.return_value = ChatResult(
            text="",
            tool_calls=(ToolCall(id="c1", name="search", arguments='{"q":"foo"}'),),
            finish_reason=FinishReason.TOOL_CALLS,
        )
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers=_h(),
                json={
                    "model": INSTALLED_REF,
                    "messages": [{"role": "user", "content": "search foo"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "search",
                                "description": "Search docs",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                    "tool_choice": "auto",
                },
            )
        body = resp.json()
        choice = body["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["tool_calls"][0]["function"]["name"] == "search"
        assert choice["message"]["tool_calls"][0]["function"]["arguments"] == '{"q": "foo"}'

    async def test_invalid_payload_returns_400_with_openai_envelope(
        self, services_with_chat_model, _auth_token
    ):
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers=_h(),
                json={"messages": [{"role": "user", "content": "x"}]},
            )
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["type"] == "invalid_request_error"
        # The translator's ``ValidationError`` message lists the missing field.
        assert "model" in body["error"]["message"]

    async def test_wrong_type_payload_returns_400_with_openai_envelope(
        self, services_with_chat_model, _auth_token
    ):
        # ``temperature`` must be a float; passing a string trips
        # pydantic ``ValidationError`` inside the handler and surfaces
        # the same OpenAI envelope as a missing-field error.
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers=_h(),
                json={
                    "model": INSTALLED_REF,
                    "messages": [{"role": "user", "content": "hi"}],
                    "temperature": "hot",
                },
            )
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["type"] == "invalid_request_error"
        assert "temperature" in body["error"]["message"]

    async def test_image_content_part_returns_400_with_openai_envelope(
        self, services_with_chat_model, _auth_token
    ):
        """Image content parts in a user message surface as a 400 INVALID_REQUEST.

        The translate layer raises ValueError because lilbee cannot route image
        data to a chat model yet; the route catches that and returns a clean
        400 instead of letting it bubble up as a 500.
        """
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers=_h(),
                json={
                    "model": INSTALLED_REF,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "describe"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": "data:image/png;base64,aGVsbG8=",
                                    },
                                },
                            ],
                        }
                    ],
                },
            )
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["code"] == "invalid_request"
        assert "Image content" in body["error"]["message"]

    async def test_unknown_role_payload_returns_400(self, services_with_chat_model, _auth_token):
        # ``role`` is a Literal of four OpenAI roles; ``developer`` is rejected.
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers=_h(),
                json={
                    "model": INSTALLED_REF,
                    "messages": [{"role": "developer", "content": "x"}],
                },
            )
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["type"] == "invalid_request_error"

    async def test_unknown_extra_fields_are_tolerated(self, services_with_chat_model, _auth_token):
        # Pydantic's default ``extra="ignore"`` lets unknown top-level
        # fields through so older OpenAI clients keep working.
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers=_h(),
                json={
                    "model": INSTALLED_REF,
                    "messages": [{"role": "user", "content": "hi"}],
                    "frequency_penalty": 0.3,
                    "user": "tobias",
                },
            )
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "hello"

    async def test_unknown_tool_choice_mode_returns_400_invalid_request(
        self, services_with_chat_model, _auth_token
    ):
        # ``tool_choice: "bogus"`` is rejected at request validation by
        # the ``ToolChoiceMode`` enum before the translator ever sees it.
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers=_h(),
                json={
                    "model": INSTALLED_REF,
                    "messages": [{"role": "user", "content": "hi"}],
                    "tool_choice": "bogus",
                },
            )
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["code"] == "invalid_request"
        assert "tool_choice" in body["error"]["message"]

    async def test_non_dict_body_returns_400_via_validation_handler(
        self, services_with_chat_model, _auth_token
    ):
        # Litestar parses the body as ``dict[str, Any]`` and raises
        # ``ValidationException`` for non-dict JSON; the custom handler
        # wraps it in the OpenAI 400 envelope.
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers=_h(),
                json=["not", "a", "dict"],
            )
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["code"] == "invalid_request"

    async def test_provider_exception_returns_500_envelope_and_releases_lock(
        self, services_with_chat_model, _auth_token
    ):
        services_with_chat_model.provider.chat.side_effect = RuntimeError("kaboom")
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers=_h(),
                json={
                    "model": INSTALLED_REF,
                    "messages": [{"role": "user", "content": "x"}],
                },
            )
        assert resp.status_code == 500
        body = resp.json()
        assert body["error"]["code"] == "internal_error"
        assert body["error"]["type"] == "api_error"
        assert chat_gate().in_flight == 0

    async def test_context_window_exceeded_returns_400_with_openai_envelope(
        self, services_with_chat_model, _auth_token
    ):
        """When the provider raises ``ContextWindowExceededError`` (prompt
        too large for the loaded model's context window), the wire surface
        is a 400 with ``context_length_exceeded``, not a generic 500.
        """
        from lilbee.providers.base import ProviderError, ProviderErrorKind

        services_with_chat_model.provider.chat.side_effect = ProviderError(
            f"Prompt of 161000 tokens exceeds the 40960-token context window of {INSTALLED_REF!r}.",
            kind=ProviderErrorKind.CONTEXT_OVERFLOW,
        )
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers=_h(),
                json={
                    "model": INSTALLED_REF,
                    "messages": [{"role": "user", "content": "x"}],
                },
            )
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["code"] == "context_length_exceeded"
        assert body["error"]["type"] == "invalid_request_error"
        assert "161000" in body["error"]["message"]
        assert chat_gate().in_flight == 0

    async def test_missing_role_model_returns_404_not_500(
        self, services_with_chat_model, _auth_token
    ):
        """A NOT_FOUND ProviderError (e.g. the embed role model isn't installed)
        surfaces as a clear 404 naming the model, not a generic 500. (F3)
        """
        from lilbee.providers.base import ProviderError, ProviderErrorKind

        services_with_chat_model.provider.chat.side_effect = ProviderError(
            "Model 'nomic-ai/embed/embed.gguf' is not installed. "
            "Run 'lilbee model pull nomic-ai/embed/embed.gguf' to download it.",
            kind=ProviderErrorKind.NOT_FOUND,
        )
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers=_h(),
                json={
                    "model": INSTALLED_REF,
                    "messages": [{"role": "user", "content": "x"}],
                },
            )
        assert resp.status_code == 404
        body = resp.json()
        assert body["error"]["code"] == "model_not_found"
        assert body["error"]["type"] == "invalid_request_error"
        assert "nomic-ai/embed/embed.gguf" in body["error"]["message"]
        assert "lilbee model pull" in body["error"]["message"]
        assert chat_gate().in_flight == 0

    async def test_usage_tokens_populated_from_provider_result(
        self, services_with_chat_model, _auth_token
    ):
        """Usage counts come from the provider's ChatResult, not hardcoded 0. (F4)"""
        from lilbee.providers.base import TokenUsage

        services_with_chat_model.provider.chat.return_value = ChatResult(
            text="hello",
            tool_calls=(),
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(prompt_tokens=12, completion_tokens=5),
        )
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers=_h(),
                json={
                    "model": INSTALLED_REF,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        usage = resp.json()["usage"]
        assert usage["prompt_tokens"] == 12
        assert usage["completion_tokens"] == 5
        assert usage["total_tokens"] == 17

    async def test_unknown_provider_error_returns_500_envelope(
        self, services_with_chat_model, _auth_token
    ):
        """A ProviderError the backend couldn't classify (kind UNKNOWN) is an
        internal_error 500 on this surface, distinct from the mapped 4xx kinds.
        """
        from lilbee.providers.base import ProviderError, ProviderErrorKind

        services_with_chat_model.provider.chat.side_effect = ProviderError(
            "something inscrutable happened",
            kind=ProviderErrorKind.UNKNOWN,
        )
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers=_h(),
                json={
                    "model": INSTALLED_REF,
                    "messages": [{"role": "user", "content": "x"}],
                },
            )
        assert resp.status_code == 500
        body = resp.json()
        assert body["error"]["code"] == "internal_error"
        assert body["error"]["type"] == "api_error"
        assert chat_gate().in_flight == 0

    @pytest.mark.parametrize(
        ("kind", "status", "code"),
        [
            ("auth", 401, "invalid_api_key"),
            ("rate_limit", 429, "rate_limit_exceeded"),
            ("bad_request", 400, "invalid_request"),
            ("connection", 503, "internal_error"),
            ("server", 502, "internal_error"),
        ],
    )
    async def test_classified_provider_error_returns_mapped_envelope(
        self, services_with_chat_model, _auth_token, kind, status, code
    ):
        """A classified remote-provider failure keeps its kind's status and message."""
        from lilbee.providers.base import ProviderError, ProviderErrorKind

        services_with_chat_model.provider.chat.side_effect = ProviderError(
            "the provider said no, here is what to do about it",
            kind=ProviderErrorKind(kind),
        )
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers=_h(),
                json={
                    "model": INSTALLED_REF,
                    "messages": [{"role": "user", "content": "x"}],
                },
            )
        assert resp.status_code == status
        body = resp.json()
        assert body["error"]["code"] == code
        assert body["error"]["message"] == "the provider said no, here is what to do about it"
        assert chat_gate().in_flight == 0

    async def test_installed_but_not_configured_model_returns_actionable_400(
        self, services_with_chat_model, _auth_token
    ):
        """The fleet's configured-model mismatch surfaces as a 400 with its message."""
        from lilbee.providers.fleet.provider import FleetProvider
        from lilbee.providers.roles import WorkerRole

        fleet = FleetProvider.__new__(FleetProvider)
        with pytest.raises(Exception) as excinfo:
            fleet._require_configured_model(
                INSTALLED_REF, "vendor/Other-GGUF/o.gguf", WorkerRole.CHAT
            )
        services_with_chat_model.provider.chat.side_effect = excinfo.value
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers=_h(),
                json={
                    "model": INSTALLED_REF,
                    "messages": [{"role": "user", "content": "x"}],
                },
            )
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["code"] == "invalid_request"
        assert "set it as the chat model in lilbee settings" in body["error"]["message"]
        assert chat_gate().in_flight == 0

    async def test_non_stream_not_configured_model_rejected_at_preflight(
        self, services_with_chat_model, _auth_token
    ):
        """The preflight rejects an installed-but-not-configured local model
        before the provider is ever called, mirroring the fleet's own guard."""
        other_ref = "vendor/Other-GGUF/other-Q4.gguf"
        installed = [_installed_chat_model(), _installed_chat_model(other_ref)]
        services_with_chat_model.registry.list_installed = MagicMock(return_value=installed)
        _populate_known_models(services_with_chat_model, installed)
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers=_h(),
                json={
                    "model": other_ref,
                    "messages": [{"role": "user", "content": "x"}],
                },
            )
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["code"] == "invalid_request"
        assert "set it as the chat model in lilbee settings" in body["error"]["message"]
        services_with_chat_model.provider.chat.assert_not_called()
        assert chat_gate().in_flight == 0

    async def test_remote_ref_not_subject_to_configured_model_preflight(
        self, services_with_chat_model, _auth_token
    ):
        """A remote ref differing from the configured chat model dispatches normally."""
        remote_ref = "ollama/qwen3:8b"
        refs = {INSTALLED_REF, remote_ref}
        services_with_chat_model.known_models.refs = MagicMock(return_value=refs)
        services_with_chat_model.known_models.resolve = MagicMock(
            side_effect=lambda model: model if model in refs else None
        )
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers=_h(),
                json={
                    "model": remote_ref,
                    "messages": [{"role": "user", "content": "x"}],
                },
            )
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "hello"


class TestStreamingCompletion:
    async def test_stream_emits_role_content_done(self, services_with_chat_model, _auth_token):
        services_with_chat_model.provider.chat.return_value = FakeProviderStream(["he", "llo"])
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers=_h(),
                json={
                    "model": INSTALLED_REF,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        chunks = _sse_to_chunks(resp.content)
        # role chunk, then two content deltas, then finish, then [DONE].
        assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
        assert chunks[1]["choices"][0]["delta"]["content"] == "he"
        assert chunks[2]["choices"][0]["delta"]["content"] == "llo"
        assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
        assert chunks[-1] == "[DONE]"

    async def test_stream_releases_lock_when_finished(self, services_with_chat_model, _auth_token):
        services_with_chat_model.provider.chat.return_value = FakeProviderStream(["x"])
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers=_h(),
                json={
                    "model": INSTALLED_REF,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
        # Drain the body so the generator's finally clause fires.
        assert resp.content
        assert chat_gate().in_flight == 0

    async def test_stream_unknown_model_returns_404_preflush(
        self, services_with_chat_model, _auth_token
    ):
        """Pre-flush validation surfaces unknown-model as a real 404, not a 200 SSE body.

        Once headers flush at 200, downstream errors only travel via SSE
        frames; clients that don't parse OpenAI's chunk-shape (or that
        treat unknown chunks as transport failures) end up in retry loops.
        Returning 404 here keeps the path debuggable for every client.
        """
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers=_h(),
                json={
                    "model": "missing/model.gguf",
                    "messages": [{"role": "user", "content": "x"}],
                    "stream": True,
                },
            )
        assert resp.status_code == 404
        body = resp.json()
        assert body["error"]["code"] == "model_not_found"
        assert chat_gate().in_flight == 0

    async def test_stream_installed_but_not_configured_model_returns_400_preflush(
        self, services_with_chat_model, _auth_token
    ):
        """A local model that is installed but not the configured chat model is
        rejected with the actionable 400 before headers flush, not a 200 SSE body."""
        other_ref = "vendor/Other-GGUF/other-Q4.gguf"
        installed = [_installed_chat_model(), _installed_chat_model(other_ref)]
        services_with_chat_model.registry.list_installed = MagicMock(return_value=installed)
        _populate_known_models(services_with_chat_model, installed)
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers=_h(),
                json={
                    "model": other_ref,
                    "messages": [{"role": "user", "content": "x"}],
                    "stream": True,
                },
            )
        assert resp.status_code == 400
        assert resp.headers["content-type"].startswith("application/json")
        body = resp.json()
        assert body["error"]["code"] == "invalid_request"
        assert "set it as the chat model in lilbee settings" in body["error"]["message"]
        assert INSTALLED_REF in body["error"]["message"]
        services_with_chat_model.provider.chat.assert_not_called()
        assert chat_gate().in_flight == 0

    async def test_stream_tools_against_non_tool_model_returns_400_preflush(
        self, services_with_chat_model, _auth_token
    ):
        """Pre-flush validation surfaces tool-incapable model as 400 instead of SSE body."""
        services_with_chat_model.provider.supports_tools.return_value = False
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers=_h(),
                json={
                    "model": INSTALLED_REF,
                    "messages": [{"role": "user", "content": "x"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "search",
                                "description": "",
                                "parameters": {},
                            },
                        }
                    ],
                    "stream": True,
                },
            )
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["code"] == "model_does_not_support_tools"
        assert chat_gate().in_flight == 0

    async def test_stream_provider_exception_emits_internal_error_frame(
        self, services_with_chat_model, _auth_token
    ):
        services_with_chat_model.provider.chat.side_effect = RuntimeError("boom")
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers=_h(),
                json={
                    "model": INSTALLED_REF,
                    "messages": [{"role": "user", "content": "x"}],
                    "stream": True,
                },
            )
        chunks = _sse_to_chunks(resp.content)
        assert chunks[0]["error"]["code"] == "internal_error"
        assert chunks[0]["error"]["type"] == "api_error"
        assert chunks[-1] == "[DONE]"
        assert chat_gate().in_flight == 0

    async def test_stream_context_window_exceeded_emits_400_error_frame(
        self, services_with_chat_model, _auth_token
    ):
        """Mid-stream context overflow surfaces as one SSE error frame with
        ``context_length_exceeded`` followed by ``[DONE]``, not a generic
        ``internal_error``.
        """
        from lilbee.providers.base import ProviderError, ProviderErrorKind

        services_with_chat_model.provider.chat.side_effect = ProviderError(
            f"Prompt of 161000 tokens exceeds the 40960-token context window of {INSTALLED_REF!r}.",
            kind=ProviderErrorKind.CONTEXT_OVERFLOW,
        )
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers=_h(),
                json={
                    "model": INSTALLED_REF,
                    "messages": [{"role": "user", "content": "x"}],
                    "stream": True,
                },
            )
        chunks = _sse_to_chunks(resp.content)
        # The error chunk follows OpenAI's chat.completion.chunk shape so the
        # opencode / aider / ai-sdk family of clients can parse it cleanly
        # instead of seeing a bare {"error": ...} frame and entering a
        # reconnect/retry loop.
        chunk = chunks[0]
        assert chunk["object"] == "chat.completion.chunk"
        # finish_reason stays null in error frames; "length" would invite clients
        # to auto-continue a "truncated" answer that actually failed.
        assert chunk["choices"][0]["finish_reason"] is None
        assert chunk["error"]["code"] == "context_length_exceeded"
        assert chunk["error"]["type"] == "invalid_request_error"
        assert chunks[-1] == "[DONE]"
        assert chat_gate().in_flight == 0

    async def test_stream_auth_provider_error_emits_invalid_api_key_frame(
        self, services_with_chat_model, _auth_token
    ):
        """An AUTH ProviderError mid-stream emits its mapped invalid_api_key frame
        carrying the provider's user-facing message."""
        from lilbee.providers.base import ProviderError, ProviderErrorKind

        services_with_chat_model.provider.chat.side_effect = ProviderError(
            "gemini/m rejected your API key.",
            kind=ProviderErrorKind.AUTH,
        )
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers=_h(),
                json={
                    "model": INSTALLED_REF,
                    "messages": [{"role": "user", "content": "x"}],
                    "stream": True,
                },
            )
        chunks = _sse_to_chunks(resp.content)
        assert chunks[0]["error"]["code"] == "invalid_api_key"
        assert chunks[0]["error"]["type"] == "authentication_error"
        assert chunks[0]["error"]["message"] == "gemini/m rejected your API key."
        assert chunks[-1] == "[DONE]"
        assert chat_gate().in_flight == 0

    async def test_stream_emits_usage_chunk_when_include_usage_set(
        self, services_with_chat_model, _auth_token
    ):
        """With stream_options.include_usage, a trailing TokenUsage frame becomes a
        final usage chunk (empty choices, populated totals) before [DONE].
        """
        from lilbee.providers.base import TokenUsage

        services_with_chat_model.provider.chat.return_value = FakeProviderStream(
            ["he", "llo", TokenUsage(prompt_tokens=9, completion_tokens=2)]
        )
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers=_h(),
                json={
                    "model": INSTALLED_REF,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
            )
        chunks = _sse_to_chunks(resp.content)
        assert chunks[-1] == "[DONE]"
        usage_chunk = chunks[-2]
        assert usage_chunk["choices"] == []
        assert usage_chunk["usage"]["prompt_tokens"] == 9
        assert usage_chunk["usage"]["completion_tokens"] == 2
        assert usage_chunk["usage"]["total_tokens"] == 11

    async def test_stream_omits_usage_chunk_without_include_usage(
        self, services_with_chat_model, _auth_token
    ):
        """Without stream_options.include_usage, no usage chunk is emitted even when
        the provider reports a trailing usage frame (OpenAI contract)."""
        from lilbee.providers.base import TokenUsage

        services_with_chat_model.provider.chat.return_value = FakeProviderStream(
            ["he", "llo", TokenUsage(prompt_tokens=9, completion_tokens=2)]
        )
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers=_h(),
                json={
                    "model": INSTALLED_REF,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
        chunks = _sse_to_chunks(resp.content)
        assert chunks[-1] == "[DONE]"
        assert all(not isinstance(c, dict) or c.get("usage") is None for c in chunks)

    async def test_stream_missing_role_model_emits_model_not_found_frame(
        self, services_with_chat_model, _auth_token
    ):
        """A NOT_FOUND ProviderError mid-stream emits a model_not_found error
        frame rather than a generic internal_error. (F3 streaming)
        """
        from lilbee.providers.base import ProviderError, ProviderErrorKind

        services_with_chat_model.provider.chat.side_effect = ProviderError(
            "Model 'nomic-ai/embed/embed.gguf' is not installed. "
            "Run 'lilbee model pull nomic-ai/embed/embed.gguf' to download it.",
            kind=ProviderErrorKind.NOT_FOUND,
        )
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers=_h(),
                json={
                    "model": INSTALLED_REF,
                    "messages": [{"role": "user", "content": "x"}],
                    "stream": True,
                },
            )
        chunks = _sse_to_chunks(resp.content)
        assert chunks[0]["error"]["code"] == "model_not_found"
        assert "nomic-ai/embed/embed.gguf" in chunks[0]["error"]["message"]
        assert chunks[-1] == "[DONE]"
        assert chat_gate().in_flight == 0

    async def test_stream_with_tools_emits_tool_call_chunks(
        self, services_with_chat_model, _auth_token
    ):
        services_with_chat_model.provider.supports_tools.return_value = True
        services_with_chat_model.provider.chat.return_value = FakeProviderStream(
            [
                ToolCallDelta(index=0, id="c1", name="search", arguments_delta=None),
                ToolCallDelta(index=0, id=None, name=None, arguments_delta='{"q":'),
                ToolCallDelta(index=0, id=None, name=None, arguments_delta='"foo"}'),
            ]
        )
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers=_h(),
                json={
                    "model": INSTALLED_REF,
                    "messages": [{"role": "user", "content": "search foo"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "search",
                                "description": "",
                                "parameters": {},
                            },
                        }
                    ],
                    "stream": True,
                },
            )
        chunks = _sse_to_chunks(resp.content)
        # First non-DONE chunk opens the tool call with id/name.
        first = chunks[0]["choices"][0]["delta"]
        assert first["tool_calls"][0]["id"] == "c1"
        assert first["tool_calls"][0]["function"]["name"] == "search"
        # Finish reason is tool_calls.
        assert chunks[-2]["choices"][0]["finish_reason"] == "tool_calls"


class TestAuth:
    async def test_missing_auth_returns_401_openai_envelope(
        self, services_with_chat_model, _auth_token
    ):
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": INSTALLED_REF,
                    "messages": [{"role": "user", "content": "x"}],
                },
            )
        assert resp.status_code == 401
        body = resp.json()
        assert body["error"]["type"] == "authentication_error"
        assert body["error"]["code"] == "invalid_api_key"

    async def test_wrong_token_returns_401_openai_envelope(
        self, services_with_chat_model, _auth_token
    ):
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer wrong-token"},
                json={
                    "model": INSTALLED_REF,
                    "messages": [{"role": "user", "content": "x"}],
                },
            )
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "invalid_api_key"

    async def test_missing_auth_on_models_endpoint_returns_401_envelope(
        self, services_with_chat_model, _auth_token
    ):
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.get("/v1/models")
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "invalid_api_key"

    async def test_no_token_set_means_auth_disabled(self, services_with_chat_model):
        # Auth explicitly disabled (the test-harness path) accepts requests
        # without a bearer header; this is distinct from the uninitialized
        # state, which now fails closed.
        previous = _auth_mod.session_manager.token
        previous_init = _auth_mod.session_manager._initialized
        _auth_mod.session_manager.disable()
        try:
            async with AsyncTestClient(_build_app()) as client:
                resp = await client.get("/v1/models")
            assert resp.status_code == 200
        finally:
            _auth_mod.session_manager.token = previous
            _auth_mod.session_manager._initialized = previous_init

    async def test_uninitialized_auth_returns_401_envelope(self, services_with_chat_model):
        # Uninitialized (pre-lifespan) state fails closed: validate() raises
        # NotAuthorizedException, which _auth_failure maps to the 401 OpenAI
        # envelope rather than letting it escape as a 500.
        previous = _auth_mod.session_manager.token
        previous_init = _auth_mod.session_manager._initialized
        _auth_mod.session_manager.token = None
        _auth_mod.session_manager._initialized = False
        try:
            async with AsyncTestClient(_build_app()) as client:
                resp = await client.get("/v1/models")
            assert resp.status_code == 401
            assert resp.json()["error"]["code"] == "invalid_api_key"
        finally:
            _auth_mod.session_manager.token = previous
            _auth_mod.session_manager._initialized = previous_init


class TestBusy:
    async def test_busy_backend_returns_429_only_after_wait_timeout(
        self, services_with_chat_model, _auth_token, monkeypatch
    ):
        """The route holds the request on the chat lock up to the configured
        timeout before returning 429. Verifies the wait-then-429 contract that
        replaced the immediate-bounce behaviour. (bb-2x6j)
        """
        from lilbee.server.chat_dispatch import concurrency as concurrency_mod

        # Tight timeout so the test runs fast; the production default is 60s.
        monkeypatch.setattr(concurrency_mod, "DEFAULT_BUSY_WAIT_S", 0.05)

        # Fill the single chat slot so the route sees the backend busy.
        gate = chat_gate()
        await gate.acquire(1, 60)
        try:
            async with AsyncTestClient(_build_app()) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    headers=_h(),
                    json={
                        "model": INSTALLED_REF,
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
        finally:
            await gate.release()
        assert resp.status_code == 429
        body = resp.json()
        assert body["error"]["code"] == "rate_limit_exceeded"
        assert body["error"]["type"] == "rate_limit_error"
        assert resp.headers.get("retry-after") == "1"


class TestStreamSlotReleaseOnEarlyDisconnect:
    """A disconnect between admission and the first SSE chunk must not leak the slot."""

    def _stream_request_body(self) -> Any:
        from lilbee.server.chat_completions_api.models import CompletionsRequest

        return CompletionsRequest(
            model=INSTALLED_REF,
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )

    def _authed_request(self) -> MagicMock:
        request = MagicMock()
        request.headers = {"authorization": f"Bearer {_auth_mod.session_manager.token}"}
        return request

    @pytest.mark.asyncio
    async def test_never_started_stream_releases_slot_via_after_send_hook(
        self, services_with_chat_model, _auth_token
    ) -> None:
        from lilbee.server.chat_completions_api.routes import chat_completions_endpoint

        response = await chat_completions_endpoint.fn(
            request=self._authed_request(), data=self._stream_request_body()
        )
        assert chat_gate().in_flight == 1
        # Litestar never starts the generator when the disconnect lands first;
        # its response cleanup still runs the background task.
        assert response.background is not None
        await response.background()
        assert chat_gate().in_flight == 0
        # Late generator cleanup must stay a no-op, not a double release.
        await response.iterator.aclose()
        assert chat_gate().in_flight == 0

    @pytest.mark.asyncio
    async def test_completed_stream_releases_slot_exactly_once(
        self, services_with_chat_model, _auth_token
    ) -> None:
        from lilbee.server.chat_completions_api.routes import chat_completions_endpoint
        from lilbee.server.chat_dispatch.concurrency import acquire_chat_slot_or_busy

        services_with_chat_model.provider.max_concurrent_chats.return_value = 2
        services_with_chat_model.provider.chat.return_value = FakeProviderStream(["hi"])
        await acquire_chat_slot_or_busy(2)  # second slot detects a double release
        response = await chat_completions_endpoint.fn(
            request=self._authed_request(), data=self._stream_request_body()
        )
        assert chat_gate().in_flight == 2
        frames = [frame async for frame in response.iterator]
        assert frames
        assert chat_gate().in_flight == 1
        await response.background()
        assert chat_gate().in_flight == 1


class TestRouteDispatchErrorBranches:
    """The route layer catches model-known errors raised by dispatch_chat post-preflight.

    Preflight normally filters these, but if dispatch_chat itself raises one we
    must still surface the canonical 4xx envelope rather than collapsing to 500.
    """

    @pytest.mark.asyncio
    async def test_non_stream_returns_404_when_dispatch_raises_model_not_found(
        self, monkeypatch
    ) -> None:
        from lilbee.server.chat_completions_api.routes import _run_non_stream
        from lilbee.server.chat_dispatch.canonical import CanonicalChatRequest, CanonicalMessage
        from lilbee.server.chat_dispatch.dispatch import ModelNotFoundError

        def _raise(req: object) -> None:
            raise ModelNotFoundError("vendor/missing")

        monkeypatch.setattr("lilbee.server.chat_completions_api.routes.dispatch_chat", _raise)
        req = CanonicalChatRequest(
            model="vendor/missing",
            messages=(CanonicalMessage(role="user", content="hi"),),
        )
        response = await _run_non_stream(req, ChatSlotGuard())
        assert response.status_code == 404
        assert response.content["error"]["code"] == "model_not_found"

    @pytest.mark.asyncio
    async def test_non_stream_returns_400_when_dispatch_raises_tools_unsupported(
        self, monkeypatch
    ) -> None:
        from lilbee.server.chat_completions_api.routes import _run_non_stream
        from lilbee.server.chat_dispatch.canonical import CanonicalChatRequest, CanonicalMessage
        from lilbee.server.chat_dispatch.dispatch import ModelDoesNotSupportToolsError

        def _raise(req: object) -> None:
            raise ModelDoesNotSupportToolsError("vendor/notools")

        monkeypatch.setattr("lilbee.server.chat_completions_api.routes.dispatch_chat", _raise)
        req = CanonicalChatRequest(
            model="vendor/notools",
            messages=(CanonicalMessage(role="user", content="hi"),),
        )
        response = await _run_non_stream(req, ChatSlotGuard())
        assert response.status_code == 400
        assert response.content["error"]["code"] == "model_does_not_support_tools"

    @pytest.mark.asyncio
    async def test_stream_emits_404_sse_frame_when_dispatch_raises_model_not_found(
        self, monkeypatch
    ) -> None:
        from lilbee.server.chat_completions_api.routes import _gated_completions_stream
        from lilbee.server.chat_dispatch.canonical import CanonicalChatRequest, CanonicalMessage
        from lilbee.server.chat_dispatch.dispatch import ModelNotFoundError

        async def _raising_stream(req: object) -> Any:
            raise ModelNotFoundError("vendor/missing")
            yield  # pragma: no cover  -- unreachable, makes this an async generator

        monkeypatch.setattr(
            "lilbee.server.chat_completions_api.routes.dispatch_chat_stream", _raising_stream
        )
        req = CanonicalChatRequest(
            model="vendor/missing",
            messages=(CanonicalMessage(role="user", content="hi"),),
            stream=True,
        )
        frames = [frame async for frame in _gated_completions_stream(req, ChatSlotGuard())]
        joined = b"".join(frames).decode()
        assert "model_not_found" in joined

    @pytest.mark.asyncio
    async def test_stream_emits_400_sse_frame_when_dispatch_raises_tools_unsupported(
        self, monkeypatch
    ) -> None:
        from lilbee.server.chat_completions_api.routes import _gated_completions_stream
        from lilbee.server.chat_dispatch.canonical import CanonicalChatRequest, CanonicalMessage
        from lilbee.server.chat_dispatch.dispatch import ModelDoesNotSupportToolsError

        async def _raising_stream(req: object) -> Any:
            raise ModelDoesNotSupportToolsError("vendor/notools")
            yield  # pragma: no cover

        monkeypatch.setattr(
            "lilbee.server.chat_completions_api.routes.dispatch_chat_stream", _raising_stream
        )
        req = CanonicalChatRequest(
            model="vendor/notools",
            messages=(CanonicalMessage(role="user", content="hi"),),
            stream=True,
        )
        frames = [frame async for frame in _gated_completions_stream(req, ChatSlotGuard())]
        joined = b"".join(frames).decode()
        assert "model_does_not_support_tools" in joined

    @pytest.mark.asyncio
    async def test_preflush_reraises_unclassified_preflight_error(self, monkeypatch) -> None:
        # Preflight only raises classifiable typed errors today; if it ever
        # raised something else, the route re-raises rather than masking it.
        from lilbee.server.chat_completions_api.routes import _preflush_or_none
        from lilbee.server.chat_dispatch.canonical import CanonicalChatRequest, CanonicalMessage

        def _raise(req: object) -> str:
            raise RuntimeError("unexpected preflight failure")

        monkeypatch.setattr(
            "lilbee.server.chat_completions_api.routes.preflight_chat_request", _raise
        )
        req = CanonicalChatRequest(
            model="vendor/m",
            messages=(CanonicalMessage(role="user", content="hi"),),
        )
        with pytest.raises(RuntimeError, match="unexpected preflight failure"):
            await _preflush_or_none(req)

    @pytest.mark.asyncio
    async def test_preflush_runs_preflight_off_the_event_loop(self, monkeypatch) -> None:
        # Discovery probes inside the preflight are blocking HTTP; the async
        # boundary must push them to a worker thread.
        import threading

        from lilbee.server.chat_completions_api.routes import _preflush_or_none
        from lilbee.server.chat_dispatch.canonical import CanonicalChatRequest, CanonicalMessage

        loop_thread = threading.current_thread()
        seen_threads: list[threading.Thread] = []

        def _record(req: object) -> str:
            seen_threads.append(threading.current_thread())
            return "vendor/m"

        monkeypatch.setattr(
            "lilbee.server.chat_completions_api.routes.preflight_chat_request", _record
        )
        req = CanonicalChatRequest(
            model="vendor/m",
            messages=(CanonicalMessage(role="user", content="hi"),),
        )
        assert await _preflush_or_none(req) is None
        assert seen_threads and seen_threads[0] is not loop_thread
