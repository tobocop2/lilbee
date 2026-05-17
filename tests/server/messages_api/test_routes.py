"""Route-level tests for ``POST /v1/messages``."""

from __future__ import annotations

import json
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
from lilbee.server.messages_api.routes import messages_router


@pytest.fixture
def services_with_model():
    """Install a mock services container with one chat model in the registry."""
    from tests.conftest import make_mock_services

    provider = MagicMock()
    provider.chat.return_value = ChatResult(
        text="hello there",
        tool_calls=(),
        finish_reason=FinishReason.STOP,
    )
    provider.supports_tools.return_value = False

    services = make_mock_services(provider=provider)
    installed = MagicMock()
    installed.ref = "vendor/model::Q4"
    services.registry.list_installed = MagicMock(return_value=[installed])

    set_services(services)
    yield services
    set_services(None)


@pytest.fixture
def app() -> Litestar:
    return Litestar(route_handlers=[messages_router])


def _h() -> dict[str, str]:
    return {"x-api-key": _auth_mod.session_manager.token or "test"}


class _FakeStream:
    """Mimics the provider's ClosableIterator: async iter + sync close()."""

    def __init__(self, items):
        self._items = list(items)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)

    def close(self) -> None:
        self.closed = True


def _payload(**overrides) -> dict:
    base = {
        "model": "vendor/model::Q4",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
    }
    base.update(overrides)
    return base


class TestNonStream:
    async def test_returns_anthropic_message_envelope(self, app, services_with_model) -> None:
        async with AsyncTestClient(app) as client:
            resp = await client.post("/v1/messages", json=_payload(), headers=_h())
        assert resp.status_code == 200
        body = resp.json()
        assert body["type"] == "message"
        assert body["role"] == "assistant"
        assert body["model"] == "vendor/model::Q4"
        assert body["content"] == [{"type": "text", "text": "hello there"}]
        assert body["stop_reason"] == "end_turn"
        assert body["usage"] == {"input_tokens": 0, "output_tokens": 0}
        assert body["id"].startswith("msg_")

    async def test_unknown_model_returns_404_with_anthropic_body(
        self, app, services_with_model
    ) -> None:
        async with AsyncTestClient(app) as client:
            resp = await client.post(
                "/v1/messages",
                json=_payload(model="who/dis::Q4"),
                headers=_h(),
            )
        assert resp.status_code == 404
        body = resp.json()
        assert body == {
            "type": "error",
            "error": {"type": "not_found_error", "message": "Model 'who/dis::Q4' not found"},
        }

    async def test_tool_request_against_non_tool_model_returns_400(
        self, app, services_with_model
    ) -> None:
        async with AsyncTestClient(app) as client:
            resp = await client.post(
                "/v1/messages",
                json=_payload(
                    tools=[
                        {
                            "name": "search",
                            "description": "Search",
                            "input_schema": {"type": "object"},
                        }
                    ],
                ),
                headers=_h(),
            )
        assert resp.status_code == 400
        body = resp.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "invalid_request_error"
        assert "vendor/model::Q4" in body["error"]["message"]

    async def test_with_tools_round_trips_tool_use_block(self, app, services_with_model) -> None:
        services_with_model.provider.supports_tools.return_value = True
        services_with_model.provider.chat.return_value = ChatResult(
            text="",
            tool_calls=(ToolCall(id="call_1", name="search", arguments='{"q": "foo"}'),),
            finish_reason=FinishReason.TOOL_CALLS,
        )
        async with AsyncTestClient(app) as client:
            resp = await client.post(
                "/v1/messages",
                json=_payload(
                    tools=[
                        {
                            "name": "search",
                            "description": "Search",
                            "input_schema": {"type": "object"},
                        }
                    ],
                ),
                headers=_h(),
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["stop_reason"] == "tool_use"
        assert {
            "type": "tool_use",
            "id": "call_1",
            "name": "search",
            "input": {"q": "foo"},
        } in body["content"]

    async def test_invalid_request_body_returns_400(self, app, services_with_model) -> None:
        async with AsyncTestClient(app) as client:
            resp = await client.post(
                "/v1/messages",
                json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
                headers=_h(),
            )
        assert resp.status_code == 400
        body = resp.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "invalid_request_error"


class TestAuth:
    async def test_missing_x_api_key_when_token_set_returns_401(
        self, app, services_with_model, monkeypatch
    ) -> None:
        monkeypatch.setattr(_auth_mod.session_manager, "token", "secret-token")
        async with AsyncTestClient(app) as client:
            resp = await client.post("/v1/messages", json=_payload())
        assert resp.status_code == 401
        body = resp.json()
        assert body == {
            "type": "error",
            "error": {"type": "authentication_error", "message": "Missing or invalid API key"},
        }

    async def test_authorization_bearer_accepted(
        self, app, services_with_model, monkeypatch
    ) -> None:
        monkeypatch.setattr(_auth_mod.session_manager, "token", "secret-token")
        async with AsyncTestClient(app) as client:
            resp = await client.post(
                "/v1/messages",
                json=_payload(),
                headers={"Authorization": "Bearer secret-token"},
            )
        assert resp.status_code == 200

    async def test_x_api_key_accepted(self, app, services_with_model, monkeypatch) -> None:
        monkeypatch.setattr(_auth_mod.session_manager, "token", "secret-token")
        async with AsyncTestClient(app) as client:
            resp = await client.post(
                "/v1/messages",
                json=_payload(),
                headers={"x-api-key": "secret-token"},
            )
        assert resp.status_code == 200

    async def test_wrong_x_api_key_returns_401(self, app, services_with_model, monkeypatch) -> None:
        monkeypatch.setattr(_auth_mod.session_manager, "token", "secret-token")
        async with AsyncTestClient(app) as client:
            resp = await client.post(
                "/v1/messages",
                json=_payload(),
                headers={"x-api-key": "wrong"},
            )
        assert resp.status_code == 401
        assert resp.json()["error"]["type"] == "authentication_error"


class TestBusy:
    async def test_second_concurrent_request_gets_429(self, app, services_with_model) -> None:
        from lilbee.server.chat_dispatch.concurrency import chat_lock

        lock = chat_lock()
        await lock.acquire()
        try:
            async with AsyncTestClient(app) as client:
                resp = await client.post("/v1/messages", json=_payload(), headers=_h())
            assert resp.status_code == 429
            body = resp.json()
            assert body["type"] == "error"
            assert body["error"]["type"] == "overloaded_error"
            assert resp.headers.get("retry-after") == "1"
        finally:
            lock.release()


class TestStreaming:
    async def test_full_event_sequence(self, app, services_with_model) -> None:
        services_with_model.provider.chat.return_value = _FakeStream(["hel", "lo"])

        async with AsyncTestClient(app) as client:
            resp = await client.post(
                "/v1/messages",
                json=_payload(stream=True),
                headers=_h(),
            )
        assert resp.status_code == 200
        body = resp.text
        events = [
            line[len("event: ") :] for line in body.splitlines() if line.startswith("event: ")
        ]
        assert events == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        # No [DONE] terminator.
        assert "[DONE]" not in body

    async def test_stream_with_tool_call_round_trip(self, app, services_with_model) -> None:
        services_with_model.provider.supports_tools.return_value = True
        services_with_model.provider.chat.return_value = _FakeStream(
            [
                ToolCallDelta(
                    index=0,
                    id="call_1",
                    name="search",
                    arguments_delta='{"q":"foo"}',
                )
            ]
        )

        async with AsyncTestClient(app) as client:
            resp = await client.post(
                "/v1/messages",
                json=_payload(
                    stream=True,
                    tools=[
                        {
                            "name": "search",
                            "description": "Search",
                            "input_schema": {"type": "object"},
                        }
                    ],
                ),
                headers=_h(),
            )
        body = resp.text
        # content_block_start should carry a tool_use block.
        for line in body.splitlines():
            if line.startswith("data: ") and '"content_block_start"' in line:
                payload = json.loads(line[len("data: ") :])
                assert payload["content_block"]["type"] == "tool_use"
                assert payload["content_block"]["name"] == "search"
                break
        else:
            raise AssertionError("no content_block_start event found")

    async def test_stream_releases_lock_on_completion(self, app, services_with_model) -> None:
        from lilbee.server.chat_dispatch.concurrency import chat_lock

        services_with_model.provider.chat.return_value = _FakeStream(["x"])

        async with AsyncTestClient(app) as client:
            resp = await client.post(
                "/v1/messages",
                json=_payload(stream=True),
                headers=_h(),
            )
        assert resp.status_code == 200
        assert not chat_lock().locked()

    async def test_stream_unknown_model_returns_404(self, app, services_with_model) -> None:
        async with AsyncTestClient(app) as client:
            resp = await client.post(
                "/v1/messages",
                json=_payload(model="who/dis::Q4", stream=True),
                headers=_h(),
            )
        assert resp.status_code == 404
        assert resp.json()["error"]["type"] == "not_found_error"

    async def test_stream_tools_against_non_tool_model_returns_400(
        self, app, services_with_model
    ) -> None:
        async with AsyncTestClient(app) as client:
            resp = await client.post(
                "/v1/messages",
                json=_payload(
                    stream=True,
                    tools=[
                        {
                            "name": "search",
                            "description": "Search",
                            "input_schema": {"type": "object"},
                        }
                    ],
                ),
                headers=_h(),
            )
        assert resp.status_code == 400
        assert resp.json()["error"]["type"] == "invalid_request_error"

    async def test_stream_busy_returns_429(self, app, services_with_model) -> None:
        from lilbee.server.chat_dispatch.concurrency import chat_lock

        lock = chat_lock()
        await lock.acquire()
        try:
            async with AsyncTestClient(app) as client:
                resp = await client.post(
                    "/v1/messages",
                    json=_payload(stream=True),
                    headers=_h(),
                )
            assert resp.status_code == 429
            body = resp.json()
            assert body["error"]["type"] == "overloaded_error"
            assert resp.headers.get("retry-after") == "1"
        finally:
            lock.release()
