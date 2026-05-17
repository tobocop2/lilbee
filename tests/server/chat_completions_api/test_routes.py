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
from lilbee.providers.worker.transport import (
    ChatResult,
    FinishReason,
    ToolCall,
    ToolCallDelta,
)
from lilbee.server import auth as _auth_mod
from lilbee.server.chat_completions_api.routes import completions_router
from lilbee.server.chat_dispatch.concurrency import chat_lock

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

    services = make_mock_services(provider=provider)
    services.registry.list_installed = MagicMock(return_value=installed)
    return services


@pytest.fixture
def services_with_chat_model():
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
    _auth_mod.session_manager.token = "test-token-" + "x" * 40
    yield
    _auth_mod.session_manager.token = previous


@pytest.fixture(autouse=True)
def _clear_chat_lock():
    """Drop the cached chat lock between tests so each test starts clean."""
    chat_lock.cache_clear()
    yield
    chat_lock.cache_clear()


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
        # ``tool_choice: "bogus"`` parses (it is a string) but fails in
        # the translator with ``ValueError``; the route maps it to a
        # 400 invalid-request envelope.
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
        assert "bogus" in body["error"]["message"]

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
        assert not chat_lock().locked()


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
        assert not chat_lock().locked()

    async def test_stream_unknown_model_emits_error_frame_then_done(
        self, services_with_chat_model, _auth_token
    ):
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
        assert resp.status_code == 200
        chunks = _sse_to_chunks(resp.content)
        # Single error frame followed by [DONE].
        assert chunks[0]["error"]["code"] == "model_not_found"
        assert chunks[-1] == "[DONE]"
        assert not chat_lock().locked()

    async def test_stream_tools_against_non_tool_model_emits_error_frame(
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
                    "stream": True,
                },
            )
        chunks = _sse_to_chunks(resp.content)
        assert chunks[0]["error"]["code"] == "model_does_not_support_tools"
        assert chunks[-1] == "[DONE]"

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
        assert not chat_lock().locked()

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
        # In test mode where session_manager.token is None, requests succeed
        # without a bearer header (matches AuthMiddleware.validate semantics).
        previous = _auth_mod.session_manager.token
        _auth_mod.session_manager.token = None
        try:
            async with AsyncTestClient(_build_app()) as client:
                resp = await client.get("/v1/models")
            assert resp.status_code == 200
        finally:
            _auth_mod.session_manager.token = previous


class TestBusy:
    async def test_busy_backend_returns_429(self, services_with_chat_model, _auth_token):
        # Pre-acquire the chat lock so the second request sees it held.
        lock = chat_lock()
        await lock.acquire()
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
            lock.release()
        assert resp.status_code == 429
        body = resp.json()
        assert body["error"]["code"] == "rate_limit_exceeded"
        assert body["error"]["type"] == "rate_limit_error"
        assert resp.headers.get("retry-after") == "1"
