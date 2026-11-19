"""A 503 that clears by itself carries Retry-After; a 503 that needs a user change does not."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import pytest
from litestar.testing import TestClient

from lilbee.core.security import FileLockUnavailableError
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.server_registry import ConfigReadError
from lilbee.server import auth as auth_mod
from lilbee.server.app import _UPGRADE_RETRY_AFTER_SECONDS, create_app
from lilbee.server.chat_completions_api.errors import classify_provider_error
from lilbee.server.provider_status import UNAVAILABLE_RETRY_AFTER_SECONDS, retry_after_headers

_RETRY_AFTER = str(UNAVAILABLE_RETRY_AFTER_SECONDS)
_CLEARING = (
    ProviderErrorKind.CONNECTION,
    ProviderErrorKind.CAPACITY,
    ProviderErrorKind.PORT_CONFLICT,
    ProviderErrorKind.STARTING,
)
_PERMANENT = (
    ProviderErrorKind.MISSING_LIBRARY,
    ProviderErrorKind.CPU_UNSUPPORTED,
    ProviderErrorKind.MODEL_FILE_BROKEN,
)
_COMPLETIONS_BODY = {"model": "m", "messages": [{"role": "user", "content": "x"}]}


@pytest.fixture()
def client() -> Iterator[TestClient]:
    auth_mod.session_manager.disable()
    yield TestClient(create_app(), raise_server_exceptions=False)
    auth_mod.session_manager.cleanup()


def _health_raises(error: Exception) -> AbstractContextManager[Any]:
    return mock.patch("lilbee.server.handlers.health", new_callable=AsyncMock, side_effect=error)


def _completions_preflight_raises(kind: ProviderErrorKind) -> AbstractContextManager[Any]:
    return mock.patch(
        "lilbee.server.chat_completions_api.routes.preflight_chat_request",
        side_effect=ProviderError("x", kind=kind),
    )


def _api_chat_raises(error: Exception) -> AbstractContextManager[Any]:
    return mock.patch("lilbee.server.handlers.chat", new_callable=AsyncMock, side_effect=error)


@pytest.mark.parametrize("kind", ProviderErrorKind)
def test_only_the_kinds_that_clear_by_themselves_get_a_retry_hint(kind: ProviderErrorKind) -> None:
    expected = {"Retry-After": _RETRY_AFTER} if kind in _CLEARING else {}
    assert retry_after_headers(kind) == expected


@pytest.mark.parametrize("kind", ProviderErrorKind)
def test_a_classified_error_carries_the_hint_only_with_a_503(kind: ProviderErrorKind) -> None:
    classified = classify_provider_error(ProviderError("x", kind=kind))
    if classified is None:
        return
    assert bool(classified.headers) == (classified.http_status == 503 and kind in _CLEARING)


def test_a_failure_with_no_kind_gets_a_retry_hint() -> None:
    assert retry_after_headers() == {"Retry-After": _RETRY_AFTER}


@pytest.mark.parametrize(
    "error",
    [
        FileLockUnavailableError("another lilbee process held the lock"),
        ProviderError("The engines are still starting.", kind=ProviderErrorKind.STARTING),
    ],
    ids=["config-locked", "engines-starting"],
)
def test_a_503_from_an_app_level_handler_that_clears_carries_retry_after(
    client: TestClient, error: Exception
) -> None:
    with _health_raises(error):
        response = client.get("/api/health")
    assert response.status_code == 503
    assert response.headers["retry-after"] == _RETRY_AFTER


def test_a_503_for_an_unreadable_config_carries_no_retry_after(client: TestClient) -> None:
    with _health_raises(ConfigReadError("Could not read config.toml: denied")):
        response = client.get("/api/health")
    assert response.status_code == 503
    assert "retry-after" not in response.headers


@pytest.mark.parametrize("kind", _CLEARING)
def test_an_api_chat_503_of_a_clearing_kind_carries_retry_after(
    client: TestClient, kind: ProviderErrorKind
) -> None:
    with _api_chat_raises(ProviderError("busy", kind=kind)):
        response = client.post("/api/chat", json={"question": "q", "history": []})
    assert response.status_code == 503
    assert response.headers["retry-after"] == _RETRY_AFTER


@pytest.mark.parametrize("error", [ProviderError("x", kind=k) for k in _PERMANENT])
def test_an_api_chat_503_of_a_permanent_kind_carries_no_retry_after(
    client: TestClient, error: ProviderError
) -> None:
    with _api_chat_raises(error):
        response = client.post("/api/chat", json={"question": "q", "history": []})
    assert response.status_code == 503
    assert "retry-after" not in response.headers


def test_an_api_chat_503_with_no_known_cause_carries_no_retry_after(client: TestClient) -> None:
    with _api_chat_raises(RuntimeError("broke")):
        response = client.post("/api/chat", json={"question": "q", "history": []})
    assert response.status_code == 503
    assert "retry-after" not in response.headers


@pytest.mark.parametrize("kind", _CLEARING)
def test_a_completions_503_of_a_clearing_kind_carries_retry_after(
    client: TestClient, kind: ProviderErrorKind
) -> None:
    with _completions_preflight_raises(kind):
        response = client.post("/v1/chat/completions", json=_COMPLETIONS_BODY)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "internal_error"
    assert response.headers["retry-after"] == _RETRY_AFTER


@pytest.mark.parametrize(
    ("kind", "carries"),
    [(ProviderErrorKind.CONNECTION, True), (ProviderErrorKind.MISSING_LIBRARY, False)],
)
def test_a_completions_503_from_the_chat_call_carries_retry_after_only_when_it_clears(
    client: TestClient, kind: ProviderErrorKind, carries: bool
) -> None:
    with (
        mock.patch(
            "lilbee.server.chat_completions_api.routes.preflight_chat_request",
            return_value="m",
        ),
        mock.patch(
            "lilbee.server.chat_completions_api.routes.cap_aware_chat",
            side_effect=ProviderError("x", kind=kind),
        ),
    ):
        response = client.post("/v1/chat/completions", json=_COMPLETIONS_BODY)
    assert response.status_code == 503
    assert ("retry-after" in response.headers) is carries


@pytest.mark.parametrize("kind", _PERMANENT)
def test_a_completions_503_of_a_permanent_kind_carries_no_retry_after(
    client: TestClient, kind: ProviderErrorKind
) -> None:
    with _completions_preflight_raises(kind):
        response = client.post("/v1/chat/completions", json=_COMPLETIONS_BODY)
    assert response.status_code == 503
    assert "retry-after" not in response.headers


@pytest.mark.parametrize(
    ("kind", "carries"),
    [(ProviderErrorKind.CONNECTION, True), (ProviderErrorKind.MISSING_LIBRARY, False)],
)
def test_a_messages_503_carries_retry_after_only_when_it_clears(
    client: TestClient, kind: ProviderErrorKind, carries: bool
) -> None:
    with mock.patch(
        "lilbee.server.anthropic_api.routes.resolve_served_model",
        side_effect=ProviderError("x", kind=kind),
    ):
        response = client.post("/v1/messages/count_tokens", json=_COMPLETIONS_BODY)
    assert response.status_code == 503
    assert ("retry-after" in response.headers) is carries


def test_a_503_that_sets_its_own_retry_after_keeps_it(client: TestClient) -> None:
    assert _UPGRADE_RETRY_AFTER_SECONDS != UNAVAILABLE_RETRY_AFTER_SECONDS
    store = mock.MagicMock(is_open=False)
    store.wait_for_open.return_value = False
    store.upgrade_progress.return_value = None
    with mock.patch("lilbee.server.app.peek_services", return_value=mock.MagicMock(store=store)):
        response = client.get("/api/status")
    assert response.status_code == 503
    assert response.headers.get_list("retry-after") == [str(_UPGRADE_RETRY_AFTER_SECONDS)]


def test_an_answer_that_is_not_a_503_carries_no_retry_after(client: TestClient) -> None:
    with _health_raises(PermissionError("denied")):
        failed = client.get("/api/health")
    assert failed.status_code == 500
    assert "retry-after" not in failed.headers
