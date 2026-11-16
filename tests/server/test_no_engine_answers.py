"""A request for a model that no installed engine loads answers a readable 400."""

from __future__ import annotations

import pytest
from litestar import get
from litestar.testing import create_test_client

from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet.engines.choice import NoEngineError
from lilbee.server.app import create_app
from lilbee.server.routes import search as search_routes

_TEXT = "For chat: No installed engine loads the mlx format. Run lilbee engine list."
_NO_ENGINE = NoEngineError(_TEXT, kind=ProviderErrorKind.BAD_REQUEST)


def _app_status(error: Exception) -> tuple[int, str]:
    """Raise *error* from a route with no handler of its own, under the real app's handlers."""

    @get("/read", sync_to_thread=False)
    def read() -> None:
        raise error

    handlers = create_app().exception_handlers
    with create_test_client(route_handlers=[read], exception_handlers=handlers) as client:
        answer = client.get("/read")
    return answer.status_code, answer.json()["detail"]


def test_a_route_with_no_handler_of_its_own_answers_400_with_the_text() -> None:
    assert _app_status(_NO_ENGINE) == (400, _TEXT)


def test_another_bad_request_provider_failure_stays_a_500() -> None:
    status, _ = _app_status(ProviderError("boom", kind=ProviderErrorKind.BAD_REQUEST))
    assert status == 500


def test_another_provider_failure_on_the_chat_route_stays_503() -> None:
    assert search_routes._api_chat_error_status(ProviderError("boom")) == 503


@pytest.mark.parametrize(
    ("kind", "status"),
    [
        (ProviderErrorKind.CONTEXT_OVERFLOW, 400),
        (ProviderErrorKind.NOT_FOUND, 404),
        (ProviderErrorKind.AUTH, 503),
    ],
)
def test_the_chat_route_keeps_the_status_of_each_other_kind(
    kind: ProviderErrorKind, status: int
) -> None:
    assert search_routes._api_chat_error_status(ProviderError("boom", kind=kind)) == status


@pytest.mark.parametrize("kind", [ProviderErrorKind.BAD_REQUEST, ProviderErrorKind.UNKNOWN])
def test_the_chat_route_maps_by_error_class_not_kind(kind: ProviderErrorKind) -> None:
    assert search_routes._api_chat_error_status(NoEngineError(_TEXT, kind=kind)) == 400
