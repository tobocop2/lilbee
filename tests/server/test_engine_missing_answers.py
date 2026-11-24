"""A missing engine binary and a missing model answer differently over HTTP."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import pytest
from litestar.testing import TestClient

from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet.provider import FleetProvider
from lilbee.providers.roles import WorkerRole
from lilbee.server import auth as auth_mod
from lilbee.server.app import create_app

_BODY = {"model": "m", "messages": [{"role": "user", "content": "x"}]}
_NO_ENGINE = "llama-server binary not found. Install it."


@pytest.fixture()
def client() -> Iterator[TestClient]:
    auth_mod.session_manager.disable()
    yield TestClient(create_app(), raise_server_exceptions=False)
    auth_mod.session_manager.cleanup()


def _fleet_error(engine_reason: str | None) -> ProviderError:
    """The error a real fleet raises for a role with no server."""
    provider = FleetProvider()
    with (
        mock.patch.object(FleetProvider, "_unusable_engine_reason", return_value=engine_reason),
        pytest.raises(ProviderError) as raised,
    ):
        provider._raise_no_clients(WorkerRole.CHAT)
    return raised.value


def _preflight_raises(error: Exception) -> Any:
    return mock.patch(
        "lilbee.server.chat_completions_api.routes.preflight_chat_request", side_effect=error
    )


def test_a_missing_engine_binary_answers_503_with_its_own_code(client: TestClient) -> None:
    with _preflight_raises(_fleet_error(_NO_ENGINE)):
        resp = client.post("/v1/chat/completions", json=_BODY)

    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "engine_missing"
    assert "llama-server binary not found" in resp.json()["error"]["message"]
    assert "Retry-After" not in resp.headers


def test_a_model_that_is_not_installed_still_answers_404(client: TestClient) -> None:
    with _preflight_raises(_fleet_error(None)):
        resp = client.post("/v1/chat/completions", json=_BODY)

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "model_not_found"


def test_a_missing_engine_on_the_anthropic_route_is_an_api_error(client: TestClient) -> None:
    error = _fleet_error(_NO_ENGINE)
    with mock.patch("lilbee.server.anthropic_api.routes.preflight_chat_request", side_effect=error):
        resp = client.post("/v1/messages", json={**_BODY, "max_tokens": 8})

    assert resp.status_code == 503
    assert resp.json()["error"]["type"] == "api_error"


def test_the_api_chat_route_answers_a_missing_engine_as_503(client: TestClient) -> None:
    error = _fleet_error(_NO_ENGINE)
    with mock.patch("lilbee.server.handlers.chat", new_callable=AsyncMock, side_effect=error):
        resp = client.post("/api/chat", json={"question": "q"})

    assert resp.status_code == 503
    assert "llama-server binary not found" in resp.text


@pytest.mark.parametrize(
    ("kind", "status"),
    [
        ("not_found", 404),
        ("bad_request", 400),
        ("engine_missing", 503),
    ],
)
def test_the_app_handler_gives_each_kind_its_status(
    client: TestClient, kind: str, status: int
) -> None:
    error = ProviderError("the reason", kind=ProviderErrorKind(kind))
    with mock.patch("lilbee.server.handlers.health", new_callable=AsyncMock, side_effect=error):
        resp = client.get("/api/health")

    assert resp.status_code == status
    assert "the reason" in resp.text
    assert "Retry-After" not in resp.headers
