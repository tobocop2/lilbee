"""Each surface answers engines that are still starting with a readable 503."""

from __future__ import annotations

from unittest import mock

import pytest
from litestar import get
from litestar.testing import create_test_client

from lilbee.app.placement import _apply_to_running
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.server.app import _provider_error_response, create_app
from lilbee.server.chat_completions_api.errors import classify_provider_error
from lilbee.server.handlers.rag import _classify_stream_error
from lilbee.server.routes import search as search_routes

_TEXT = "The engines are still starting after 60 s."
_STARTING = ProviderError(_TEXT, kind=ProviderErrorKind.STARTING)


def _route_status(error: ProviderError) -> tuple[int, str]:
    @get("/read", sync_to_thread=False)
    def read() -> None:
        raise error

    with create_test_client(
        route_handlers=[read], exception_handlers={ProviderError: _provider_error_response}
    ) as client:
        answer = client.get("/read")
    return answer.status_code, answer.json()["detail"]


@pytest.mark.parametrize(
    ("failure", "kind", "reason"),
    [
        (_STARTING, ProviderErrorKind.STARTING, "The engines are still starting after 60 s"),
        (OSError("the disk is full"), ProviderErrorKind.UNKNOWN, "the disk is full"),
    ],
)
def test_a_placement_that_did_not_apply_says_it_is_saved(
    failure: Exception, kind: ProviderErrorKind, reason: str
) -> None:
    services = mock.MagicMock()
    services.provider.reload_placement.side_effect = failure
    with pytest.raises(ProviderError) as raised:
        _apply_to_running(services)
    assert str(raised.value) == (
        "The placement is saved, but the running server could not apply it: "
        f"{reason}. Restart lilbee to use the placement."
    )
    assert raised.value.kind is kind
    services.provider.reload_placement.assert_called_once_with(wait=True)


def test_a_route_with_no_handler_of_its_own_answers_503_with_the_text() -> None:
    assert _route_status(_STARTING) == (503, _TEXT)


def test_any_other_provider_failure_on_such_a_route_stays_a_500() -> None:
    assert _route_status(ProviderError("boom"))[0] == 500


def test_the_completions_surface_returns_the_text() -> None:
    classified = classify_provider_error(_STARTING)
    assert classified is not None
    assert (classified.http_status, classified.message) == (503, _TEXT)


def test_the_stream_surface_returns_the_kind_and_the_text() -> None:
    assert _classify_stream_error(_STARTING) == (ProviderErrorKind.STARTING, _TEXT)


def test_the_chat_routes_answer_503_with_the_text() -> None:
    assert search_routes._api_chat_error_status(_STARTING) == 503


@pytest.mark.parametrize(
    ("error", "detail"),
    [(_STARTING, _TEXT), (ProviderError("boom"), "Search is temporarily unavailable.")],
)
def test_the_search_route_returns_only_this_text(
    monkeypatch: pytest.MonkeyPatch, error: ProviderError, detail: str
) -> None:
    async def fails(*_args: object, **_kwargs: object) -> None:
        raise error

    monkeypatch.setattr(search_routes.handlers, "search", fails)
    with create_test_client(route_handlers=[search_routes.search_route]) as client:
        answer = client.get("/api/search", params={"q": "x"})
    assert (answer.status_code, answer.json()["detail"]) == (503, detail)


def test_the_app_registers_the_handler_for_provider_failures() -> None:
    assert create_app().exception_handlers[ProviderError] is _provider_error_response
