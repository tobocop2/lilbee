"""A bad-request provider failure answers a readable 400 on every route."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from litestar import get
from litestar.testing import create_test_client

from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet.engines.choice import NoEngineError
from lilbee.server import app as app_module
from lilbee.server.app import create_app
from lilbee.server.routes import search as search_routes

_TEXT = "For chat: No installed engine loads the mlx format. Run lilbee engine list."
_NO_ENGINE = NoEngineError(_TEXT, kind=ProviderErrorKind.BAD_REQUEST)
_REFUSAL = "The configured model refuses this request."


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


def test_any_bad_request_provider_failure_answers_400_with_its_text() -> None:
    error = ProviderError(_REFUSAL, kind=ProviderErrorKind.BAD_REQUEST)
    assert _app_status(error) == (400, _REFUSAL)


@pytest.mark.parametrize(
    "kind",
    [
        ProviderErrorKind.AUTH,
        ProviderErrorKind.RATE_LIMIT,
        ProviderErrorKind.CONTEXT_OVERFLOW,
        ProviderErrorKind.CONNECTION,
        ProviderErrorKind.SERVER,
        ProviderErrorKind.UNKNOWN,
    ],
)
def test_the_other_provider_kinds_keep_the_500_on_a_route_with_no_handler(
    kind: ProviderErrorKind,
) -> None:
    status, _ = _app_status(ProviderError("boom", kind=kind))
    assert status == 500


def test_a_model_that_is_not_found_answers_404_on_a_route_with_no_handler() -> None:
    status, _ = _app_status(ProviderError("boom", kind=ProviderErrorKind.NOT_FOUND))
    assert status == 404


def test_an_unmapped_provider_kind_logs_its_traceback() -> None:
    error = ProviderError("boom", kind=ProviderErrorKind.UNKNOWN)
    with patch.object(app_module.log, "error") as logged:
        _app_status(error)
    logged.assert_called_once()
    assert logged.call_args.kwargs["exc_info"] is error


def test_engines_still_starting_keep_the_503() -> None:
    status, _ = _app_status(ProviderError("warming", kind=ProviderErrorKind.STARTING))
    assert status == 503


def test_another_provider_failure_on_the_chat_route_stays_503() -> None:
    assert search_routes._api_chat_error_status(ProviderError("boom")) == 503


@pytest.mark.parametrize(
    ("kind", "status"),
    [
        (ProviderErrorKind.CONTEXT_OVERFLOW, 400),
        (ProviderErrorKind.NOT_FOUND, 404),
        (ProviderErrorKind.BAD_REQUEST, 400),
        (ProviderErrorKind.AUTH, 503),
        (ProviderErrorKind.RATE_LIMIT, 503),
        (ProviderErrorKind.SERVER, 503),
    ],
)
def test_the_chat_route_maps_each_provider_kind(kind: ProviderErrorKind, status: int) -> None:
    assert search_routes._api_chat_error_status(ProviderError("boom", kind=kind)) == status


def test_the_chat_route_and_the_app_agree_on_a_bad_request() -> None:
    error = ProviderError("boom", kind=ProviderErrorKind.BAD_REQUEST)
    assert search_routes._api_chat_error_status(error) == _app_status(error)[0] == 400
