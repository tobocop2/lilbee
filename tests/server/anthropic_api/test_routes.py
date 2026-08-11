"""Route tests for the Anthropic-shaped ``/v1/messages`` surface."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest
from litestar import Litestar
from litestar.testing import AsyncTestClient

from lilbee.app.services import set_services
from lilbee.providers.base import ChatResult, FinishReason, ToolCall
from lilbee.server import auth as _auth_mod
from lilbee.server.anthropic_api.routes import anthropic_router
from lilbee.server.chat_dispatch.concurrency import chat_gate

INSTALLED_REF = "vendor/Model-GGUF/model-Q4.gguf"


def _h() -> dict[str, str]:
    """Auth header carrying the active session token (Claude Code's shape)."""
    return {"Authorization": f"Bearer {_auth_mod.session_manager.token}"}


def _build_app() -> Litestar:
    return Litestar(route_handlers=[anthropic_router])


def _body(**overrides) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": INSTALLED_REF,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
    }
    body.update(overrides)
    return body


def _installed_chat_model(ref: str = INSTALLED_REF) -> MagicMock:
    manifest = MagicMock()
    manifest.ref = ref
    manifest.task = "chat"
    manifest.downloaded_at = "2026-05-15T00:00:00+00:00"
    return manifest


def _services_with(provider: MagicMock, installed: list[MagicMock]) -> Any:
    from tests.conftest import make_mock_services

    provider.max_concurrent_chats.return_value = 1
    provider.served_chat_ctx.return_value = None
    services = make_mock_services(provider=provider)
    services.registry.list_installed = MagicMock(return_value=installed)
    refs = {m.ref for m in installed}
    services.known_models.refs = MagicMock(return_value=refs)
    services.known_models.resolve = MagicMock(
        side_effect=lambda model: model if model in refs else None
    )
    return services


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
    previous = _auth_mod.session_manager.token
    previous_init = _auth_mod.session_manager._initialized
    _auth_mod.session_manager.token = "test-token-" + "x" * 40
    _auth_mod.session_manager._initialized = True
    yield
    _auth_mod.session_manager.token = previous
    _auth_mod.session_manager._initialized = previous_init


@pytest.fixture(autouse=True)
def _clear_chat_lock():
    chat_gate.cache_clear()
    yield
    chat_gate.cache_clear()


class FakeProviderStream:
    """``ClosableIterator`` mimicking the provider streaming protocol."""

    def __init__(self, frames: list[Any]) -> None:
        self._frames = list(frames)
        self.closed = False

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        if not self._frames:
            raise StopIteration
        return self._frames.pop(0)

    def close(self) -> None:
        self.closed = True


def _sse_events(body: bytes) -> list[tuple[str, Any]]:
    """Parse an Anthropic SSE body into (event_type, payload) pairs."""
    out: list[tuple[str, Any]] = []
    for frame in body.decode().split("\n\n"):
        frame = frame.strip()
        if not frame:
            continue
        lines = frame.split("\n")
        assert lines[0].startswith("event: "), frame
        assert lines[1].startswith("data: "), frame
        out.append((lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))))
    return out


class TestAuth:
    async def test_missing_token_is_401_in_anthropic_envelope(
        self, services_with_chat_model, _auth_token
    ):
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post("/v1/messages", json=_body())
        assert resp.status_code == 401
        body = resp.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "authentication_error"

    async def test_auth_wins_over_body_validation(self, services_with_chat_model, _auth_token):
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post("/v1/messages", json={"model": 5})
        assert resp.status_code == 401


class TestValidation:
    async def test_missing_max_tokens_is_400_envelope(
        self, services_with_chat_model, _auth_token
    ):
        body = _body()
        del body["max_tokens"]
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post("/v1/messages", json=body, headers=_h())
        assert resp.status_code == 400
        assert resp.json()["error"]["type"] == "invalid_request_error"
        assert "max_tokens" in resp.json()["error"]["message"]

    async def test_image_content_is_400(self, services_with_chat_model, _auth_token):
        body = _body(
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "image", "source": {"type": "base64", "data": "x"}}],
                }
            ]
        )
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post("/v1/messages", json=body, headers=_h())
        assert resp.status_code == 400
        assert "Image content" in resp.json()["error"]["message"]

    async def test_unknown_model_is_404_not_found_error(
        self, services_with_chat_model, _auth_token
    ):
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/messages", json=_body(model="nope/missing"), headers=_h()
            )
        assert resp.status_code == 404
        assert resp.json()["error"]["type"] == "not_found_error"

    async def test_thinking_param_is_ignored_not_rejected(
        self, services_with_chat_model, _auth_token
    ):
        body = _body(thinking={"type": "adaptive"}, output_config={"effort": "high"})
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post("/v1/messages", json=body, headers=_h())
        assert resp.status_code == 200


class TestNonStreaming:
    async def test_happy_path_returns_message(self, services_with_chat_model, _auth_token):
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post("/v1/messages", json=_body(), headers=_h())
        assert resp.status_code == 200
        body = resp.json()
        assert body["type"] == "message"
        assert body["role"] == "assistant"
        assert body["id"].startswith("msg_")
        assert body["model"] == INSTALLED_REF
        assert body["content"] == [{"type": "text", "text": "hello"}]
        assert body["stop_reason"] == "end_turn"
        assert body["stop_sequence"] is None
        assert set(body["usage"]) == {"input_tokens", "output_tokens"}

    async def test_tool_call_response(self, services_with_chat_model, _auth_token):
        services_with_chat_model.provider.supports_tools.return_value = True
        services_with_chat_model.provider.chat.return_value = ChatResult(
            text="",
            tool_calls=(ToolCall(id="c1", name="search", arguments='{"q": "foo"}'),),
            finish_reason=FinishReason.TOOL_CALLS,
        )
        body = _body(
            tools=[{"name": "search", "description": "s", "input_schema": {"type": "object"}}]
        )
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post("/v1/messages", json=body, headers=_h())
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["stop_reason"] == "tool_use"
        tool_blocks = [b for b in payload["content"] if b["type"] == "tool_use"]
        assert tool_blocks == [{"type": "tool_use", "id": "c1", "name": "search", "input": {"q": "foo"}}]


class TestStreaming:
    async def test_stream_event_sequence(self, services_with_chat_model, _auth_token):
        services_with_chat_model.provider.chat.return_value = FakeProviderStream(["he", "llo"])
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post("/v1/messages", json=_body(stream=True), headers=_h())
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = _sse_events(resp.content)
        types = [t for t, _ in events]
        assert types[0] == "message_start"
        assert types[-1] == "message_stop"
        assert "content_block_start" in types
        assert "content_block_delta" in types
        assert "content_block_stop" in types
        assert "message_delta" in types
        text = "".join(
            p["delta"]["text"]
            for t, p in events
            if t == "content_block_delta" and p["delta"]["type"] == "text_delta"
        )
        assert text == "hello"
        start = next(p for t, p in events if t == "message_start")
        assert start["message"]["role"] == "assistant"
        assert start["message"]["model"] == INSTALLED_REF

    async def test_mid_stream_error_emits_error_event(
        self, services_with_chat_model, _auth_token, monkeypatch
    ):
        def _raising_stream(*_args, **_kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(
            "lilbee.server.anthropic_api.routes.dispatch_chat_stream", _raising_stream
        )
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post("/v1/messages", json=_body(stream=True), headers=_h())
        assert resp.status_code == 200
        events = _sse_events(resp.content)
        error_events = [p for t, p in events if t == "error"]
        assert len(error_events) == 1
        assert error_events[0]["error"]["type"] == "api_error"
