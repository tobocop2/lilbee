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

    async def test_subdir_native_model_listed_and_servable(self, _auth_token):
        """F6: a registered subdir-filename giant is advertised by /v1/models and
        resolves for a completion (its abs-path inconsistency was a symptom of it
        not being registerable; once F2 registers it, the surface is consistent).
        """
        subdir_ref = "unsloth/MiniMax-M2-GGUF/Q4_K_M/MiniMax-M2-Q4_K_M-00001-of-00003.gguf"
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
        assert not chat_lock().locked()

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
        assert not chat_lock().locked()

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
        assert not chat_lock().locked()

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

    async def test_non_overflow_provider_error_returns_500_envelope(
        self, services_with_chat_model, _auth_token
    ):
        """A ProviderError whose kind is NOT context-overflow (e.g. auth) is an
        internal_error 500 on this surface, distinct from the 400 overflow case.
        """
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
        assert not chat_lock().locked()

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
        assert not chat_lock().locked()

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
        assert chunk["choices"][0]["finish_reason"] == "length"
        assert chunk["error"]["code"] == "context_length_exceeded"
        assert chunk["error"]["type"] == "invalid_request_error"
        assert chunks[-1] == "[DONE]"
        assert not chat_lock().locked()

    async def test_stream_non_overflow_provider_error_emits_internal_error_frame(
        self, services_with_chat_model, _auth_token
    ):
        """A non-overflow ProviderError mid-stream (e.g. auth) emits an
        internal_error frame, not the context_length_exceeded one."""
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
        assert chunks[0]["error"]["code"] == "internal_error"
        assert chunks[0]["error"]["type"] == "api_error"
        assert chunks[-1] == "[DONE]"
        assert not chat_lock().locked()

    async def test_stream_emits_usage_chunk_from_final_usage_frame(
        self, services_with_chat_model, _auth_token
    ):
        """A trailing TokenUsage frame from the provider becomes a final usage
        chunk (empty choices, populated totals) before [DONE]. (F4 streaming)
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
                },
            )
        chunks = _sse_to_chunks(resp.content)
        assert chunks[-1] == "[DONE]"
        usage_chunk = chunks[-2]
        assert usage_chunk["choices"] == []
        assert usage_chunk["usage"]["prompt_tokens"] == 9
        assert usage_chunk["usage"]["completion_tokens"] == 2
        assert usage_chunk["usage"]["total_tokens"] == 11

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

        # Pre-acquire the chat lock so the route sees it held.
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
        import asyncio

        lock = asyncio.Lock()
        await lock.acquire()
        response = await _run_non_stream(req, lock)
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
        import asyncio

        lock = asyncio.Lock()
        await lock.acquire()
        response = await _run_non_stream(req, lock)
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
        import asyncio

        lock = asyncio.Lock()
        await lock.acquire()
        frames = [frame async for frame in _gated_completions_stream(req, lock)]
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
        import asyncio

        lock = asyncio.Lock()
        await lock.acquire()
        frames = [frame async for frame in _gated_completions_stream(req, lock)]
        joined = b"".join(frames).decode()
        assert "model_does_not_support_tools" in joined

    def test_preflush_reraises_unclassified_preflight_error(self, monkeypatch) -> None:
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
            _preflush_or_none(req)
