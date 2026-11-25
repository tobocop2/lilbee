"""A command in an error message gets its REST route where the server answers, and only there.

The CLI and the TUI show the messages as they are. The server adds the route where an error
becomes an HTTP response, an SSE error event or a /v1 error body.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from litestar import get
from litestar.exceptions import HTTPException

from lilbee.crawler.messages import CHROMIUM_MISSING_MESSAGE
from lilbee.providers import engine_params
from lilbee.providers.base import ProviderError, ProviderErrorKind, WorkerRole, require_role_ref
from lilbee.providers.fleet import first_use
from lilbee.providers.fleet.client import _upstream_death_message
from lilbee.providers.fleet.engine_install import engine_install_hint
from lilbee.providers.fleet.engine_store import EngineFolderError
from lilbee.providers.fleet.engines import choice
from lilbee.server.app import _json_error_response, _provider_error_response
from lilbee.server.chat_completions_api.errors import classify_provider_error
from lilbee.server.chat_dispatch.dispatch import ModelNotFoundError
from lilbee.server.handlers.sse import sse_error
from lilbee.server.route_names import name_routes
from tests._litestar_test_client import create_test_client

PULL = "POST /api/models/pull"
LIST = "GET /api/models/installed"
ENGINE_INSTALL = "POST /api/engines/install, which needs allow_http_engines"
ENGINE_LIST = "GET /api/engines"
CRAWLER_SETUP = "POST /setup/crawler"


def _not_installed() -> ProviderError:
    registry = MagicMock()
    registry.resolve.side_effect = KeyError("absent")
    with pytest.raises(ProviderError) as refused:
        engine_params.resolve_model_path("org/repo/model.gguf", registry)
    return refused.value


def _shared_texts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, str]:
    """Each message that the CLI, the TUI and the server all show, built from the real code."""
    monkeypatch.setattr(first_use, "pack_lock", lambda: (None, "cell"))
    monkeypatch.setattr(choice, "install_offered", lambda *_a: True)
    with pytest.raises(ProviderError) as unset:
        require_role_ref("", WorkerRole.CHAT)
    with pytest.raises(ProviderError) as empty:
        engine_params.resolve_model_path("")
    return {
        "list hint": choice.LIST_HINT,
        "no model": str(empty.value),
        "unset role": str(unset.value),
        "not installed": str(_not_installed()),
        "unknown model": str(ModelNotFoundError("org/repo/model.gguf")),
        "damaged file": _upstream_death_message(
            ProviderErrorKind.MODEL_FILE_BROKEN, 500, "org/repo/model.gguf", None
        ),
        "damaged file, no name": _upstream_death_message(
            ProviderErrorKind.MODEL_FILE_BROKEN, 500, None, None
        ),
        "install hint": engine_install_hint(bundled=False),
        "damaged folder": str(EngineFolderError(tmp_path / "tei", "tei")),
        "no browser": CHROMIUM_MISSING_MESSAGE,
    }


def test_the_messages_that_the_cli_and_the_tui_show_name_no_route(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name, text in _shared_texts(monkeypatch, tmp_path).items():
        assert "/api/" not in text, name
        assert "/setup/" not in text, name
        assert "lilbee " in text, name


@pytest.mark.parametrize(
    ("command", "route"),
    [
        pytest.param("lilbee model pull org/m", PULL, id="pull"),
        pytest.param("lilbee model list", LIST, id="list"),
        pytest.param("lilbee engine install tei", ENGINE_INSTALL, id="engine-install"),
        pytest.param("lilbee engine list", ENGINE_LIST, id="engine-list"),
        pytest.param("lilbee setup crawler", CRAWLER_SETUP, id="crawler"),
    ],
)
def test_a_command_gets_the_route_that_does_the_same_work(command: str, route: str) -> None:
    text = name_routes(f"Run '{command}' first.")
    assert text.startswith(f"Run '{command}' first.")
    assert f"Over HTTP, call {route} in place of '" in text


def test_the_model_remove_command_gets_no_route() -> None:
    text = "Run 'lilbee model rm org/repo/m.gguf'."
    assert name_routes(text) == text


def test_a_text_without_a_command_stays_as_it_is() -> None:
    assert name_routes("The model server stopped.") == "The model server stopped."


def test_a_route_that_the_text_already_names_is_not_named_again() -> None:
    text = "Run 'lilbee model pull m' (or POST /api/models/pull)."
    assert name_routes(text) == text


def test_naming_the_routes_twice_adds_them_once() -> None:
    once = name_routes("Run 'lilbee model pull m' or 'lilbee model list'.")
    assert name_routes(once) == once


def _error_body(raised: Exception, handlers: dict) -> tuple[int, dict]:
    @get("/boom")
    async def boom() -> None:
        raise raised

    with create_test_client([boom], exception_handlers=handlers, debug=False) as client:
        response = client.get("/boom")
    return response.status_code, response.json()


def test_a_rest_error_for_an_unknown_model_names_the_list_and_pull_routes() -> None:
    error = HTTPException(status_code=404, detail=str(ModelNotFoundError("org/repo/m.gguf")))
    status, body = _error_body(error, {HTTPException: _json_error_response})
    assert status == 404
    assert LIST in body["detail"]
    assert PULL in body["detail"]
    assert "lilbee model pull org/repo/m.gguf" in body["detail"]


def test_a_rest_error_for_a_missing_engine_names_the_install_route_and_its_setting() -> None:
    error = ProviderError(
        f"llama-server binary not found. {engine_install_hint(bundled=False)}",
        kind=ProviderErrorKind.ENGINE_MISSING,
    )
    status, body = _error_body(error, {ProviderError: _provider_error_response})
    assert status == 503
    assert ENGINE_INSTALL in body["detail"]


def test_a_rest_error_for_an_engine_that_is_not_installed_names_the_list_route() -> None:
    detail = f"Engine tei is not installed. {choice.LIST_HINT}"
    status, body = _error_body(
        HTTPException(status_code=422, detail=detail), {HTTPException: _json_error_response}
    )
    assert status == 422
    assert ENGINE_LIST in body["detail"]


def test_a_rest_error_with_no_detail_names_no_route() -> None:
    status, body = _error_body(
        HTTPException(status_code=404), {HTTPException: _json_error_response}
    )
    assert status == 404
    assert "/api/" not in json.dumps(body)


def test_an_sse_error_for_a_missing_browser_names_the_crawler_setup_route() -> None:
    event = sse_error(CHROMIUM_MISSING_MESSAGE)
    assert CRAWLER_SETUP in event
    assert "lilbee setup crawler" in event


def test_a_completions_error_for_an_unknown_model_names_the_routes() -> None:
    classified = classify_provider_error(ModelNotFoundError("org/repo/m.gguf"))
    assert classified is not None
    assert LIST in classified.message
    assert PULL in classified.message


def test_a_completions_error_for_a_model_that_is_not_installed_names_the_pull_route() -> None:
    classified = classify_provider_error(_not_installed())
    assert classified is not None
    assert PULL in classified.message


@pytest.mark.parametrize(
    ("model", "route"),
    [pytest.param("org/repo/m.gguf", PULL, id="named"), pytest.param(None, LIST, id="unnamed")],
)
def test_a_damaged_model_file_gets_a_route_and_no_remove_route(
    model: str | None, route: str
) -> None:
    text = _upstream_death_message(ProviderErrorKind.MODEL_FILE_BROKEN, 500, model, None)
    named = name_routes(text)
    assert route in named
    assert "DELETE" not in named
