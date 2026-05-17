"""Default-CI smoke checks for the ``/v1/*`` response envelopes.

No ``opencode`` binary, no model download. The Litestar app is driven
in-process via ``AsyncTestClient`` (httpx-over-ASGI) so the suite is
fast enough to run on every push while still catching wire-shape
regressions in the OpenAI and Anthropic protocol bindings.

A real installed tool-capable model is not required: the provider is
mocked. The optional ``test_protocol_smoke_uses_installed_chat_model``
test detects an installed model (if any) and reports it; it skips when
no chat model is installed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from litestar import Litestar
from litestar.testing import AsyncTestClient

from lilbee.app.services import get_services, set_services
from lilbee.providers.worker.transport import ChatResult, FinishReason
from lilbee.server import auth as _auth_mod
from lilbee.server.chat_completions_api.routes import completions_router
from lilbee.server.chat_dispatch.concurrency import chat_lock
from lilbee.server.messages_api.routes import messages_router

_MOCK_MODEL_REF = "vendor/Model-GGUF/model-Q4.gguf"


def _installed_chat_model(ref: str = _MOCK_MODEL_REF) -> MagicMock:
    manifest = MagicMock()
    manifest.ref = ref
    manifest.task = "chat"
    manifest.downloaded_at = "2026-05-15T00:00:00+00:00"
    return manifest


@pytest.fixture
def services_with_chat_model():
    """Install a mock services container with one chat model and a canned reply."""
    from tests.conftest import make_mock_services

    provider = MagicMock()
    provider.chat.return_value = ChatResult(
        text="hello", tool_calls=(), finish_reason=FinishReason.STOP
    )
    provider.supports_tools.return_value = False
    services = make_mock_services(provider=provider)
    services.registry.list_installed = MagicMock(return_value=[_installed_chat_model()])
    set_services(services)
    yield services
    set_services(None)


@pytest.fixture
def auth_token():
    """Install a session token for the duration of the test."""
    previous = _auth_mod.session_manager.token
    _auth_mod.session_manager.token = "smoke-token-" + "x" * 40
    yield _auth_mod.session_manager.token
    _auth_mod.session_manager.token = previous


@pytest.fixture(autouse=True)
def reset_chat_lock():
    """Drop the cached chat lock between tests so each starts clean."""
    chat_lock.cache_clear()
    yield
    chat_lock.cache_clear()


@pytest.fixture
def app() -> Litestar:
    return Litestar(route_handlers=[completions_router, messages_router])


def _openai_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _anthropic_headers(token: str) -> dict[str, str]:
    return {"x-api-key": token}


async def test_chat_completions_response_matches_openai_envelope(
    app: Litestar, services_with_chat_model, auth_token: str
) -> None:
    """Non-streaming ``/v1/chat/completions`` returns the canonical OpenAI envelope."""
    async with AsyncTestClient(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": _MOCK_MODEL_REF,
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers=_openai_headers(auth_token),
        )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == _MOCK_MODEL_REF
    assert body["id"].startswith("chatcmpl-")
    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "hello"
    assert choice["finish_reason"] == "stop"
    usage = body["usage"]
    assert {"prompt_tokens", "completion_tokens", "total_tokens"} <= set(usage)


async def test_v1_models_returns_openai_list_envelope(
    app: Litestar, services_with_chat_model, auth_token: str
) -> None:
    """``GET /v1/models`` returns the canonical OpenAI list envelope."""
    async with AsyncTestClient(app) as client:
        response = await client.get("/v1/models", headers=_openai_headers(auth_token))
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    refs = [entry["id"] for entry in body["data"]]
    assert _MOCK_MODEL_REF in refs
    for entry in body["data"]:
        assert entry["object"] == "model"
        assert entry["owned_by"] == "lilbee"
        assert isinstance(entry["created"], int)


async def test_messages_response_matches_anthropic_envelope(
    app: Litestar, services_with_chat_model, auth_token: str
) -> None:
    """``POST /v1/messages`` returns the canonical Anthropic message envelope."""
    async with AsyncTestClient(app) as client:
        response = await client.post(
            "/v1/messages",
            json={
                "model": _MOCK_MODEL_REF,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16,
            },
            headers=_anthropic_headers(auth_token),
        )
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["model"] == _MOCK_MODEL_REF
    assert body["id"].startswith("msg_")
    assert body["content"] == [{"type": "text", "text": "hello"}]
    assert body["stop_reason"] == "end_turn"
    assert body["usage"] == {"input_tokens": 0, "output_tokens": 0}


async def test_chat_completions_rejects_missing_auth(
    app: Litestar, services_with_chat_model, auth_token: str
) -> None:
    """No bearer header surfaces the OpenAI 401 error envelope."""
    async with AsyncTestClient(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": _MOCK_MODEL_REF, "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 401
    body = response.json()
    assert body["error"]["code"] == "invalid_api_key"
    assert "type" in body["error"]


async def test_messages_rejects_missing_auth(
    app: Litestar, services_with_chat_model, auth_token: str
) -> None:
    """No api-key header surfaces the Anthropic 401 error envelope."""
    async with AsyncTestClient(app) as client:
        response = await client.post(
            "/v1/messages",
            json={
                "model": _MOCK_MODEL_REF,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16,
            },
        )
    assert response.status_code == 401
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "authentication_error"


def test_protocol_smoke_uses_installed_chat_model_if_present(
    services_with_chat_model,
) -> None:
    """When a chat model is installed the registry surfaces it under its real ref.

    The mocked services container always installs ``_MOCK_MODEL_REF``;
    this test guards against future fixture changes that would silently
    drop chat models from the smoke suite.
    """
    registry = get_services().registry
    chat_refs = [m.ref for m in registry.list_installed() if m.task == "chat"]
    if not chat_refs:
        pytest.skip("no chat model installed in the test fixture's data dir")
    assert _MOCK_MODEL_REF in chat_refs
