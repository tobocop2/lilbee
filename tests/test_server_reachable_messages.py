"""A command in an error message gets its REST route or MCP tool where the server answers.

The CLI and the TUI show the messages as they are. The server adds the route where an error
becomes an HTTP response, an SSE error event or a /v1 error body. The MCP server adds the tool.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NoReturn
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from litestar import get
from litestar.exceptions import HTTPException

import lilbee.mcp_server as mcp_server
from lilbee.app import engine_service
from lilbee.app.engine_service import RoleView
from lilbee.core.config import cfg
from lilbee.core.health_warnings import HealthWarning, WarningCode
from lilbee.crawler.messages import CHROMIUM_MISSING_MESSAGE
from lilbee.data.store import EmbeddingEngineChangedError, EmbeddingModelMismatchError
from lilbee.providers import engine_params
from lilbee.providers.base import ProviderError, ProviderErrorKind, WorkerRole, require_role_ref
from lilbee.providers.fleet import first_use
from lilbee.providers.fleet.client import _upstream_death_message
from lilbee.providers.fleet.engine_install import engine_install_hint
from lilbee.providers.fleet.engine_store import EngineFolderError
from lilbee.providers.fleet.engines import choice
from lilbee.server import handlers
from lilbee.server.app import _json_error_response, _provider_error_response
from lilbee.server.chat_completions_api.errors import classify_provider_error
from lilbee.server.chat_dispatch.dispatch import ModelNotFoundError
from lilbee.server.handlers.rag import SearchResponse
from lilbee.server.handlers.sse import sse_error
from lilbee.server.route_names import name_routes, name_tools, name_warnings
from lilbee.server.routes import engines as engine_routes
from lilbee.server.routes.search import (
    SEARCH_WARNINGS_HEADER,
    _embedding_mismatch_http,
    search_route,
)
from tests._litestar_test_client import create_test_client

PULL = "POST /api/models/pull"
LIST = "GET /api/models/installed"
ENGINE_INSTALL = "POST /api/engines/install, which needs allow_http_engines"
ENGINE_LIST = "GET /api/engines"
CRAWLER_SETUP = "POST /setup/crawler"
REBUILD = "POST /api/sync, with force_rebuild true"


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
        pytest.param("lilbee rebuild", REBUILD, id="rebuild"),
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


TOOL_NAMED = [
    pytest.param("lilbee model pull org/m", "the model_pull tool in place of", id="pull"),
    pytest.param("lilbee model list", "the model_list tool in place of", id="list"),
    pytest.param(
        "lilbee engine install tei",
        "the engine_install tool, which needs mcp_engines_enabled in place of",
        id="engine-install",
    ),
    pytest.param(
        "lilbee engine list",
        "the engine_list tool, which needs mcp_engines_enabled in place of",
        id="engine-list",
    ),
    pytest.param(
        "lilbee rebuild",
        "the sync tool, with force_rebuild true in place of",
        id="rebuild",
    ),
]


@pytest.mark.parametrize(("command", "tool"), TOOL_NAMED)
def test_a_command_gets_the_mcp_tool_that_does_the_same_work(command: str, tool: str) -> None:
    text = name_tools(f"Run '{command}' first.")
    assert text.startswith(f"Run '{command}' first.")
    assert f"Over MCP, call {tool} '" in text
    assert "/api/" not in text


@pytest.mark.parametrize("command", ["lilbee model rm m", "lilbee setup crawler"])
def test_a_command_with_no_mcp_tool_gets_none(command: str) -> None:
    assert name_tools(f"Run '{command}'.") == f"Run '{command}'."


def test_the_mcp_tools_are_named_once() -> None:
    once = name_tools("Run 'lilbee model pull m' or 'lilbee model list'.")
    assert name_tools(once) == once
    named = "Run 'lilbee model pull m' (or the model_pull tool)."
    assert name_tools(named) == named


def _roles_with(notice: str | None) -> list[RoleView]:
    return [RoleView(WorkerRole.CHAT, "org/repo/m.gguf", "gguf", None, notice)]


def test_the_rest_roles_notice_names_the_engine_list_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine_service, "roles", lambda *_a: _roles_with(choice.LIST_HINT))
    with create_test_client([engine_routes.engine_roles_route], debug=False) as client:
        notice = client.get("/api/engines/roles").json()["roles"][0]["notice"]
    assert notice == name_routes(choice.LIST_HINT)
    assert ENGINE_LIST in notice


def test_the_rest_roles_route_keeps_a_role_with_no_notice_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine_service, "roles", lambda *_a: _roles_with(None))
    with create_test_client([engine_routes.engine_roles_route], debug=False) as client:
        assert client.get("/api/engines/roles").json()["roles"][0]["notice"] is None


def test_the_mcp_roles_notice_names_the_engine_list_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cfg, "mcp_engines_enabled", True)
    monkeypatch.setattr(engine_service, "roles", lambda *_a: _roles_with(choice.LIST_HINT))
    notice = mcp_server.engine_roles_tool()["roles"][0]["notice"]
    assert notice == name_tools(choice.LIST_HINT)
    assert "the engine_list tool" in notice
    assert "/api/" not in notice


def test_the_mcp_roles_tool_keeps_a_role_with_no_notice_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cfg, "mcp_engines_enabled", True)
    monkeypatch.setattr(engine_service, "roles", lambda *_a: _roles_with(None))
    assert mcp_server.engine_roles_tool()["roles"][0]["notice"] is None


def test_an_mcp_error_for_an_unknown_model_names_the_list_and_pull_tools() -> None:
    error = mcp_server._error(str(ModelNotFoundError("org/repo/m.gguf")))["error"]
    assert "the model_list tool" in error
    assert "the model_pull tool" in error
    assert "/api/" not in error


def test_an_mcp_engine_refusal_names_the_engine_list_tool() -> None:
    def refuse() -> NoReturn:
        raise engine_service.EngineRefusedError(f"Engine tei is not installed. {choice.LIST_HINT}")

    assert "the engine_list tool" in mcp_server._engine_result(refuse)["error"]


def test_a_rest_error_leaves_the_raised_exception_as_it_was() -> None:
    error = HTTPException(status_code=404, detail=str(ModelNotFoundError("org/repo/m.gguf")))
    plain = error.detail
    _error_body(error, {HTTPException: _json_error_response})
    assert error.detail == plain


def _mismatches() -> list[EmbeddingModelMismatchError]:
    same_dim = EmbeddingModelMismatchError(
        persisted_model="a", persisted_dim=8, current_model="b", current_dim=8
    )
    other_dim = EmbeddingModelMismatchError(
        persisted_model="a", persisted_dim=8, current_model="b", current_dim=16
    )
    engine = EmbeddingEngineChangedError(model="a", dim=8, distance=0.5, verified=True)
    return [same_dim, other_dim, engine]


@pytest.mark.parametrize("mismatch", _mismatches(), ids=["same-dim", "other-dim", "engine"])
def test_an_mcp_embedding_mismatch_names_the_sync_tool(
    mismatch: EmbeddingModelMismatchError,
) -> None:
    error = mcp_server._embedding_mismatch_error(mismatch)["error"]
    assert "lilbee rebuild" in error
    assert "Over MCP, call the sync tool, with force_rebuild true in place of" in error
    assert "/api/" not in error


@pytest.mark.parametrize("mismatch", _mismatches(), ids=["same-dim", "other-dim", "engine"])
def test_a_rest_embedding_mismatch_names_the_sync_route(
    mismatch: EmbeddingModelMismatchError,
) -> None:
    status, body = _error_body(
        _embedding_mismatch_http(mismatch), {HTTPException: _json_error_response}
    )
    assert status == 409
    assert REBUILD in body["detail"]


def test_the_install_tool_over_http_names_both_settings_that_gate_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mcp_server._transport, "http_mounted", True)
    error = mcp_server._error("Run 'lilbee engine install tei' first.")["error"]
    both = "mcp_engines_enabled and allow_http_engines"
    assert f"the engine_install tool, which needs {both} in place of" in error


def test_the_install_tool_over_stdio_names_only_the_tools_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mcp_server._transport, "http_mounted", False)
    error = mcp_server._error("Run 'lilbee engine install tei' first.")["error"]
    assert "the engine_install tool, which needs mcp_engines_enabled in place of" in error
    assert "allow_http_engines" not in error


def test_the_list_tool_over_http_names_only_the_tools_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mcp_server._transport, "http_mounted", True)
    error = mcp_server._error("Run 'lilbee engine list' first.")["error"]
    assert "the engine_list tool, which needs mcp_engines_enabled in place of" in error


BOTH_SETTINGS = "mcp_engines_enabled and allow_http_engines"
INSTALL_COMMAND = "Run 'lilbee engine install tei' first."
REBUILD_REMEDY = "Run `lilbee rebuild` to rebuild the search index."


def _rebuild_warning() -> HealthWarning:
    return HealthWarning(
        code=WarningCode.FTS_UNAVAILABLE, message="keyword search is down", remedy=REBUILD_REMEDY
    )


def test_a_warning_gets_the_namer_applied_to_its_message_and_its_remedy() -> None:
    plain = HealthWarning(code=WarningCode.FTS_UNAVAILABLE, message="keyword search is down")
    named = name_warnings([_rebuild_warning(), plain], name_routes)
    assert named[0].remedy == name_routes(REBUILD_REMEDY)
    assert (named[1].message, named[1].remedy) == ("keyword search is down", None)
    assert name_warnings([_rebuild_warning()], str.upper)[0].message == "KEYWORD SEARCH IS DOWN"


def test_the_search_warnings_header_names_the_sync_route(monkeypatch: pytest.MonkeyPatch) -> None:
    outcome = SearchResponse([], (_rebuild_warning(),))
    monkeypatch.setattr(handlers, "search", AsyncMock(return_value=outcome))
    with create_test_client([search_route], debug=False) as client:
        header = client.get("/api/search", params={"q": "x"}).headers[SEARCH_WARNINGS_HEADER]
    assert REBUILD in json.loads(header)[0]["remedy"]


def test_the_mcp_search_warnings_name_the_sync_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    report = SimpleNamespace(rows=[], degraded=[_rebuild_warning()])
    services = SimpleNamespace(searcher=SimpleNamespace(search_report=lambda *_a, **_k: report))
    monkeypatch.setattr(mcp_server, "get_services", lambda: services)
    result = mcp_server.search("x")
    sentence = "Over MCP, call the sync tool, with force_rebuild true in place of"
    assert sentence in result.structured_content["warnings"][0]["remedy"]
    assert sentence in result.content[-1].text


async def test_a_refused_rebuild_names_both_install_settings_over_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def refused(**_kwargs: object) -> NoReturn:
        raise ProviderError(INSTALL_COMMAND)

    monkeypatch.setattr("lilbee.data.ingest.sync", refused)
    monkeypatch.setattr(mcp_server._transport, "http_mounted", True)
    result = await mcp_server._rebuild_when(True)
    assert BOTH_SETTINGS in result["rebuild_error"]


def test_the_mcp_roles_notice_names_both_install_settings_over_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cfg, "mcp_engines_enabled", True)
    monkeypatch.setattr(mcp_server._transport, "http_mounted", True)
    monkeypatch.setattr(engine_service, "roles", lambda *_a: _roles_with(INSTALL_COMMAND))
    assert BOTH_SETTINGS in mcp_server.engine_roles_tool()["roles"][0]["notice"]


def test_the_mcp_list_names_both_install_settings_beside_a_damaged_folder_over_http(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(first_use, "pack_lock", lambda: (None, "cell"))
    monkeypatch.setattr(choice, "install_offered", lambda *_a: True)
    monkeypatch.setattr(cfg, "mcp_engines_enabled", True)
    monkeypatch.setattr(mcp_server._transport, "http_mounted", True)
    damaged = [EngineFolderError(tmp_path / "tei", "tei")]
    listed = engine_service.EngineListView([], damaged, [], tmp_path)
    monkeypatch.setattr(engine_service, "engine_list", lambda *_a: listed)
    assert BOTH_SETTINGS in mcp_server.engine_list_tool()["damaged"][0]["error"]
