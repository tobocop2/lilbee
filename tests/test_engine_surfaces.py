"""Parity of the engine choice across the service, CLI, REST, MCP and the slash command.

The service is the oracle: a refusal is the text the service raises, and every other surface must
show that same text. The stored value is also read back from config.toml.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import lilbee.mcp_server as mcp_server
from lilbee.app import engine_choice, engine_service
from lilbee.app.engine_choice import engine_for_ref
from lilbee.cli import app as cli_app
from lilbee.cli.tui.engine_commands import USAGE, run_engine_command
from lilbee.core import settings
from lilbee.core.config import Config, cfg
from lilbee.core.config.enums import EngineSetting
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.choice import HostFacts, NoEngineError
from lilbee.providers.roles import WorkerRole
from lilbee.providers.routing_provider import RoutingProvider
from lilbee.server.routes import engines as routes
from tests._litestar_test_client import create_test_client
from tests.test_engine_service import (
    GGUF_ENGINE,
    GGUF_REF,
    MLX_ENGINE,
    MLX_REF,
    NOT_INSTALLED,
    UNREADABLE_NOTICE,
)

ALL_ROUTES = [routes.engine_roles_route, routes.engine_use_route, routes.engine_prefer_route]
SURFACES = ("cli", "rest", "mcp", "slash")
FALLBACK = (
    "Engine mlx for chat is not installed. Using llama.cpp. "
    "Run lilbee engine list to see the installed engines."
)


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    snapshot = cfg.model_copy()
    monkeypatch.setenv("LILBEE_DATA", str(tmp_path))
    cfg.data_root = tmp_path
    cfg.allow_http_engines = True
    cfg.mcp_engines_enabled = True
    cfg.chat_model = GGUF_REF
    cfg.preferred_engine = EngineSetting.AUTO
    cfg.engine_choices = None
    monkeypatch.setattr(
        engine_choice, "host_facts", lambda: HostFacts(system="darwin", machine="arm64")
    )
    monkeypatch.setitem(choice.ADAPTERS, EngineName.LLAMA, GGUF_ENGINE)  # type: ignore[arg-type]
    monkeypatch.setitem(choice.ADAPTERS, EngineName.MLX, MLX_ENGINE)  # type: ignore[arg-type]
    yield
    mcp_server.set_http_mounted(False)
    for name in Config.model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _stored(key: str) -> object:
    return settings.load(cfg.data_root).get(key)


def _run_cli(*argv: str) -> tuple[int, dict[str, Any]]:
    result = CliRunner().invoke(cli_app, ["--json", "engine", *argv])
    return result.exit_code, json.loads(result.stdout.strip().splitlines()[-1])


def _rest(method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, Any]:
    with create_test_client(ALL_ROUTES, debug=False) as client:
        res = client.request(method, path, json=body)
        return res.status_code, res.json()


def _refusal(surface: str, role: str, model: str, engine: str) -> str:
    """The refusal text a surface shows for *engine* chosen for *model* of *role*."""
    if surface == "cli":
        code, body = _run_cli("use", role, model, engine)
        assert code == 1
        return body["error"]
    if surface == "rest":
        code, body = _rest("PUT", f"/api/engines/roles/{role}", {"model": model, "engine": engine})
        assert code == 422
        return body["detail"]
    if surface == "mcp":
        return mcp_server.engine_use_tool(WorkerRole(role), model, engine)["error"]
    out = run_engine_command(f"use {role} {model} {engine}")
    assert not out.ok
    return out.text


def _service_refusal(role: str, model: str, engine: str) -> str:
    with pytest.raises(engine_service.EngineRefusedError) as exc:
        engine_service.use_for_role(WorkerRole(role), model, engine)
    return str(exc.value)


@pytest.fixture
def no_mlx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(choice.ADAPTERS, EngineName.MLX)


REFUSALS = [
    pytest.param("chat", GGUF_REF, "nonesuch", id="unknown-engine"),
    pytest.param("chat", GGUF_REF, "mlx", id="cannot-load-format"),
    pytest.param("chat", MLX_REF, "mlx", id="model-is-not-the-role-model"),
]


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize(("role", "model", "engine"), REFUSALS)
def test_every_surface_shows_the_service_refusal_and_stores_nothing(
    surface: str, role: str, model: str, engine: str
) -> None:
    expected = _service_refusal(role, model, engine)
    assert expected
    assert _refusal(surface, role, model, engine) == expected
    assert _stored("engine_choices") is None


@pytest.mark.parametrize("surface", SURFACES)
def test_a_choice_that_is_not_installed_is_refused_with_the_same_text(
    surface: str, no_mlx: None
) -> None:
    cfg.chat_model = MLX_REF
    expected = _service_refusal("chat", MLX_REF, "mlx")
    assert expected == NOT_INSTALLED.format("mlx")
    assert _refusal(surface, "chat", MLX_REF, "mlx") == expected


def test_the_refused_preference_has_the_same_text_on_every_surface(no_mlx: None) -> None:
    with pytest.raises(engine_service.EngineRefusedError) as exc:
        engine_service.prefer("mlx")
    expected = str(exc.value)
    assert expected == NOT_INSTALLED.format("mlx")
    code, body = _run_cli("prefer", "mlx")
    assert (code, body["error"]) == (1, expected)
    code, body = _rest("PUT", "/api/engines/preferred", {"engine": "mlx"})
    assert (code, body["detail"]) == (422, expected)
    assert mcp_server.engine_prefer_tool("mlx") == {"error": expected}
    assert run_engine_command("prefer mlx").text == expected
    assert _stored("preferred_engine") is None


def _write(surface: str, engine: str) -> None:
    if surface == "cli":
        assert _run_cli("use", "chat", GGUF_REF, engine)[0] == 0
    elif surface == "rest":
        body = {"model": GGUF_REF, "engine": engine}
        assert _rest("PUT", "/api/engines/roles/chat", body)[0] == 200
    elif surface == "mcp":
        assert mcp_server.engine_use_tool(WorkerRole.CHAT, GGUF_REF, engine) == {"engine": engine}
    else:
        assert run_engine_command(f"use chat {GGUF_REF} {engine}").ok


@pytest.mark.parametrize("surface", SURFACES)
def test_a_write_stores_the_choice_and_auto_removes_it(surface: str) -> None:
    _write(surface, "llama.cpp")
    assert json.loads(str(_stored("engine_choices"))) == {GGUF_REF: "llama.cpp"}
    _write(surface, "auto")
    assert _stored("engine_choices") is None


@pytest.mark.parametrize("surface", SURFACES)
def test_the_preference_is_stored_by_each_surface(surface: str) -> None:
    if surface == "cli":
        assert _run_cli("prefer", "llama.cpp") == (
            0,
            {"command": "engine prefer", "engine": "llama.cpp"},
        )
    elif surface == "rest":
        res = _rest("PUT", "/api/engines/preferred", {"engine": "llama.cpp"})
        assert res == (200, {"engine": "llama.cpp"})
    elif surface == "mcp":
        assert mcp_server.engine_prefer_tool("llama.cpp") == {"engine": "llama.cpp"}
    else:
        assert run_engine_command("prefer llama.cpp").ok
    assert _stored("preferred_engine") == "llama.cpp"


def _gated_calls() -> dict[str, Callable[[], object]]:
    return {
        "rest use": lambda: _rest(
            "PUT", "/api/engines/roles/chat", {"model": GGUF_REF, "engine": "llama.cpp"}
        ),
        "rest prefer": lambda: _rest("PUT", "/api/engines/preferred", {"engine": "llama.cpp"}),
    }


def test_rest_writes_are_refused_while_allow_http_engines_is_off() -> None:
    cfg.allow_http_engines = False
    for name, call in _gated_calls().items():
        code, body = call()  # type: ignore[misc]
        assert (code, body["detail"]) == (409, engine_service.http_refused_message()), name
    assert _stored("engine_choices") is None
    assert _stored("preferred_engine") is None


def test_rest_reads_stay_open_while_the_flag_is_off() -> None:
    cfg.allow_http_engines = False
    code, body = _rest("GET", "/api/engines/roles")
    assert code == 200
    assert {r["role"] for r in body["roles"]} == {str(r) for r in WorkerRole}


def test_mcp_writes_on_the_http_mount_are_refused_while_the_flag_is_off() -> None:
    cfg.allow_http_engines = False
    mcp_server.set_http_mounted(True)
    refusal = {"error": engine_service.http_refused_message()}
    assert mcp_server.engine_use_tool(WorkerRole.CHAT, GGUF_REF, "llama.cpp") == refusal
    assert mcp_server.engine_prefer_tool("llama.cpp") == refusal
    assert mcp_server.engine_roles_tool()["roles"]
    assert _stored("engine_choices") is None


def test_mcp_writes_work_off_the_http_mount_with_the_flag_off() -> None:
    cfg.allow_http_engines = False
    assert mcp_server.engine_prefer_tool("llama.cpp") == {"engine": "llama.cpp"}


def test_an_unknown_role_is_refused_on_every_surface_and_stores_nothing() -> None:
    with pytest.raises(engine_service.EngineRefusedError) as exc:
        engine_service.parse_role("nonesuch")
    assert run_engine_command(f"use nonesuch {GGUF_REF} llama.cpp").text == str(exc.value)
    assert "chat" in str(exc.value)
    result = CliRunner().invoke(cli_app, ["engine", "use", "nonesuch", GGUF_REF, "llama.cpp"])
    assert result.exit_code == 2
    body = {"model": GGUF_REF, "engine": "llama.cpp"}
    assert _rest("PUT", "/api/engines/roles/nonesuch", body)[0] == 400
    assert _stored("engine_choices") is None


def test_the_slash_command_with_no_or_bad_words_shows_roles_or_usage() -> None:
    assert run_engine_command("").ok
    for bad in ("nonesuch", "use chat", "prefer", "roles extra", "use 'unclosed"):
        out = run_engine_command(bad)
        assert (out.ok, out.text) == (False, USAGE), bad


def test_the_tools_and_routes_are_registered() -> None:
    import asyncio

    tools = {t.name for t in asyncio.run(mcp_server.build_mcp_server().list_tools())}
    assert {"engine_roles", "engine_use", "engine_prefer"} <= tools


ENGINE_TOOLS = {"engine_roles", "engine_use", "engine_prefer"}


def _wire_names() -> set[str]:
    return {tool.name for tool in mcp_server.build_mcp_server()._tool_manager.list_tools()}


def test_the_engine_tools_are_off_the_wire_by_default() -> None:
    assert Config.model_fields["mcp_engines_enabled"].default is False
    cfg.mcp_engines_enabled = Config.model_fields["mcp_engines_enabled"].default
    names = _wire_names()
    assert "search" in names
    assert names.isdisjoint(ENGINE_TOOLS)


def test_the_setting_on_registers_exactly_the_three_engine_tools() -> None:
    cfg.mcp_engines_enabled = False
    before = _wire_names()
    cfg.mcp_engines_enabled = True
    assert _wire_names() - before == ENGINE_TOOLS


def test_every_engine_tool_of_a_live_server_refuses_once_the_setting_goes_off() -> None:
    cfg.mcp_engines_enabled = False
    refusal = {"error": mcp_server.ENGINES_DISABLED_ERROR}
    assert "mcp_engines_enabled" in refusal["error"]
    assert mcp_server.engine_roles_tool() == refusal
    assert mcp_server.engine_use_tool(WorkerRole.CHAT, GGUF_REF, "llama.cpp") == refusal
    assert mcp_server.engine_prefer_tool("llama.cpp") == refusal
    assert _stored("engine_choices") is None
    assert _stored("preferred_engine") is None


def test_the_http_guard_still_refuses_with_the_setting_on() -> None:
    cfg.mcp_engines_enabled = True
    cfg.allow_http_engines = False
    mcp_server.set_http_mounted(True)
    refusal = {"error": engine_service.http_refused_message()}
    assert mcp_server.engine_prefer_tool("llama.cpp") == refusal
    assert _stored("preferred_engine") is None


def test_no_surface_holds_a_validator() -> None:
    root = Path(engine_service.__file__).parents[1]
    files = [
        root / "cli/engine.py",
        root / "server/routes/engines.py",
        root / "cli/tui/engine_commands.py",
    ]
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "def " in text, path
        for forbidden in ("installed_engines", "EngineName(", "model_format", "ADAPTERS"):
            assert forbidden not in text, (path.name, forbidden)
    mcp_text = (root / "mcp_server.py").read_text(encoding="utf-8")
    assert "installed_engines" not in mcp_text


def _roles_texts() -> dict[str, str]:
    """The text each surface shows for the roles view."""
    code, body = _run_cli("roles")
    assert code == 0
    cli_text = "\n".join(r["notice"] or "" for r in body["roles"])
    rest_code, rest_body = _rest("GET", "/api/engines/roles")
    assert rest_code == 200
    mcp_body = mcp_server.engine_roles_tool()
    plain = CliRunner().invoke(cli_app, ["engine", "roles"]).stdout
    return {
        "cli json": cli_text,
        "cli text": plain,
        "rest": "\n".join(r["notice"] or "" for r in rest_body["roles"]),
        "mcp": "\n".join(r["notice"] or "" for r in mcp_body["roles"]),
        "slash": run_engine_command("roles").text,
    }


def test_every_role_view_shows_the_fallback_notice(no_mlx: None) -> None:
    cfg.preferred_engine = EngineSetting.MLX
    for surface, text in _roles_texts().items():
        assert FALLBACK in " ".join(text.split()), surface


def test_every_role_view_shows_the_unreadable_choices_notice() -> None:
    cfg.engine_choices = "not json"
    for surface, text in _roles_texts().items():
        assert UNREADABLE_NOTICE in " ".join(text.split()), surface


def test_a_role_with_no_notice_shows_none_on_the_json_surfaces() -> None:
    _, body = _run_cli("roles")
    assert [r["notice"] for r in body["roles"] if r["model"] == GGUF_REF] == [None]


def test_a_pull_refusal_holds_only_the_no_engine_text(no_mlx: None) -> None:
    cfg.engine_choices = "not json"
    with pytest.raises(NoEngineError) as exc:
        engine_for_ref(MLX_REF, cfg)
    assert UNREADABLE_NOTICE not in str(exc.value)
    assert str(exc.value).startswith("No installed engine loads the mlx format of")


def _provider() -> RoutingProvider:
    return RoutingProvider()


def test_the_first_request_of_a_role_shows_the_notice_once(
    no_mlx: None, caplog: pytest.LogCaptureFixture
) -> None:
    cfg.preferred_engine = EngineSetting.MLX
    provider = _provider()
    with caplog.at_level(logging.WARNING):
        provider._warn_once(WorkerRole.CHAT)
        provider._warn_once(WorkerRole.CHAT)
        provider._warn_once(WorkerRole.EMBED)
    messages = [r.getMessage() for r in caplog.records]
    assert messages.count(FALLBACK) == 1
    assert any(m.startswith("Engine mlx for embed is not installed") for m in messages)


def test_the_first_request_shows_the_unreadable_choices_notice_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg.engine_choices = "not json"
    provider = _provider()
    with caplog.at_level(logging.WARNING):
        provider._warn_once(WorkerRole.CHAT)
        provider._warn_once(WorkerRole.CHAT)
    assert [r.getMessage() for r in caplog.records] == [UNREADABLE_NOTICE]


def test_a_role_with_no_notice_says_nothing_and_asks_again(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = _provider()
    with caplog.at_level(logging.WARNING):
        provider._warn_once(WorkerRole.CHAT)
    assert caplog.records == []
    cfg.engine_choices = "not json"
    with caplog.at_level(logging.WARNING):
        provider._warn_once(WorkerRole.CHAT)
    assert [r.getMessage() for r in caplog.records] == [UNREADABLE_NOTICE]


def test_every_request_path_checks_the_notice(
    no_mlx: None, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The product path: _announce, used by embed, chat and vision, warns on the first call."""
    from lilbee.providers import routing_provider
    from lilbee.providers.model_ref import parse_model_ref

    cfg.preferred_engine = EngineSetting.MLX
    monkeypatch.setattr(routing_provider, "role_serving", lambda role, ref: _SERVED)
    provider = _provider()
    with caplog.at_level(logging.WARNING):
        provider._announce(WorkerRole.CHAT, parse_model_ref(GGUF_REF))
    assert FALLBACK in [r.getMessage() for r in caplog.records]


class _Served:
    engine = "llama.cpp"
    version = "1"


_SERVED = _Served()


def test_the_cli_prints_plain_lines_for_a_write_and_a_refusal() -> None:
    runner = CliRunner()
    use = runner.invoke(cli_app, ["engine", "use", "chat", GGUF_REF, "llama.cpp"])
    assert (use.exit_code, " ".join(use.stdout.split())) == (
        0,
        f"chat model {GGUF_REF} uses engine llama.cpp.",
    )
    prefer = runner.invoke(cli_app, ["engine", "prefer", "llama.cpp"])
    assert (prefer.exit_code, " ".join(prefer.stdout.split())) == (
        0,
        "The preferred engine is llama.cpp.",
    )
    refused = runner.invoke(cli_app, ["engine", "prefer", "nonesuch"])
    assert refused.exit_code == 1
    assert "No engine is named 'nonesuch'" in refused.stderr + refused.stdout


def test_the_first_rerank_request_shows_the_notice_once(
    no_mlx: None, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest import mock

    from lilbee.providers import routing_provider

    cfg.preferred_engine = EngineSetting.MLX
    cfg.reranker_model = GGUF_REF
    monkeypatch.setattr(routing_provider, "role_engine", lambda role: GGUF_ENGINE)
    monkeypatch.setattr(routing_provider, "is_native_rerank_ref", lambda model, engine: True)
    monkeypatch.setattr(routing_provider, "served_by", lambda *a, **k: _SERVED)
    provider = _provider()
    local = mock.Mock()
    local.rerank.return_value = [0.5]
    monkeypatch.setattr(provider, "_get_local", lambda: local)
    with caplog.at_level(logging.WARNING):
        assert provider.rerank("q", ["a"]) == [0.5]
        assert provider.rerank("q", ["a"]) == [0.5]
    rerank_notice = (
        "Engine mlx for rerank is not installed. Using llama.cpp. "
        "Run lilbee engine list to see the installed engines."
    )
    assert [r.getMessage() for r in caplog.records].count(rerank_notice) == 1


SECRET_PATH = "/private/secret-data-root/config.tmp"


@pytest.fixture
def unwritable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The settings file cannot be written; the error names an absolute path."""

    def refuse(_root: Path, _settings: dict[str, Any]) -> None:
        raise PermissionError(13, "Permission denied", SECRET_PATH)

    monkeypatch.setattr(settings, "save", refuse)


def _write_calls() -> dict[str, Callable[[], object]]:
    return {
        "prefer": lambda: engine_service.prefer("llama.cpp"),
        "use": lambda: engine_service.use(GGUF_REF, "llama.cpp"),
        "use-auto": lambda: engine_service.use(GGUF_REF, "auto"),
    }


@pytest.mark.parametrize("call", ["prefer", "use", "use-auto"])
def test_a_failed_write_leaves_the_live_choice_equal_to_the_file(
    call: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine_service.use(GGUF_REF, "llama.cpp")
    stored_before = (_stored("engine_choices"), _stored("preferred_engine"))
    live_before = (cfg.engine_choices, cfg.preferred_engine)
    assert live_before[0] is not None
    monkeypatch.setattr(
        settings, "save", lambda *_a: (_ for _ in ()).throw(PermissionError(13, "no", SECRET_PATH))
    )
    with pytest.raises(engine_service.EngineNotSavedError):
        _write_calls()[call]()
    assert (cfg.engine_choices, cfg.preferred_engine) == live_before
    assert (_stored("engine_choices"), _stored("preferred_engine")) == stored_before


def _unsaved_text(surface: str) -> tuple[str, int | None]:
    if surface == "cli":
        code, body = _run_cli("prefer", "llama.cpp")
        assert code == 1
        return body["error"], None
    if surface == "rest":
        code, body = _rest("PUT", "/api/engines/preferred", {"engine": "llama.cpp"})
        return body["detail"], code
    if surface == "mcp":
        return mcp_server.engine_prefer_tool("llama.cpp")["error"], None
    out = run_engine_command("prefer llama.cpp")
    assert not out.ok
    return out.text, None


@pytest.mark.parametrize("surface", SURFACES)
def test_every_surface_says_the_choice_was_not_saved_without_a_path(
    surface: str, unwritable: None
) -> None:
    text, status = _unsaved_text(surface)
    assert text.startswith("The engine choice was not saved.")
    assert "secret" not in text and "/" not in text
    assert status in (None, 503)
    assert cfg.preferred_engine is EngineSetting.AUTO


def test_the_cli_exits_with_a_refusal_not_a_traceback_when_use_cannot_save(
    unwritable: None,
) -> None:
    result = CliRunner().invoke(cli_app, ["engine", "use", "chat", GGUF_REF, "llama.cpp"])
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "The engine choice was not saved." in result.stderr + result.stdout
    assert SECRET_PATH not in result.stderr + result.stdout


def test_a_role_with_no_model_says_that_no_model_is_set() -> None:
    cfg.reranker_model = ""
    with pytest.raises(engine_service.EngineRefusedError) as exc:
        engine_service.use_for_role(WorkerRole.RERANK, "other", "llama.cpp")
    assert str(exc.value) == (
        "No rerank model is set. Choose a rerank model before you choose its engine."
    )
    code, body = _rest(
        "PUT", "/api/engines/roles/rerank", {"model": "other", "engine": "llama.cpp"}
    )
    assert (code, body["detail"]) == (422, str(exc.value))


def test_an_unknown_role_on_rest_names_the_valid_roles() -> None:
    code, body = _rest(
        "PUT", "/api/engines/roles/nonesuch", {"model": GGUF_REF, "engine": "llama.cpp"}
    )
    assert code == 400
    assert (
        body["detail"] == "No role is named 'nonesuch'. Choose one of: embed, rerank, chat, vision."
    )


def test_the_success_text_carries_no_restart_note() -> None:
    use = run_engine_command(f"use chat {GGUF_REF} llama.cpp")
    prefer = run_engine_command("prefer llama.cpp")
    assert use.text == f"chat model {GGUF_REF} uses engine llama.cpp."
    assert prefer.text == "The preferred engine is llama.cpp."


def test_a_rest_write_runs_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    where: list[str] = []

    def record(name: str) -> str:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            where.append("worker thread")
        else:
            where.append("event loop")
        return name

    monkeypatch.setattr(engine_service, "prefer", record)
    monkeypatch.setattr(engine_service, "use_for_role", lambda _r, _m, name: record(name))
    monkeypatch.setattr(engine_service, "roles", lambda: record("") and [])
    assert _rest("PUT", "/api/engines/preferred", {"engine": "auto"})[0] == 200
    body = {"model": GGUF_REF, "engine": "auto"}
    assert _rest("PUT", "/api/engines/roles/chat", body)[0] == 200
    assert _rest("GET", "/api/engines/roles")[0] == 200
    assert where == ["worker thread"] * 3


def test_the_health_list_carries_the_fallback_notice(no_mlx: None) -> None:
    from lilbee.core.health_warnings import WarningCode

    assert _provider().health_warnings() == []
    cfg.preferred_engine = EngineSetting.MLX
    warnings = _provider().health_warnings()
    assert {w.code for w in warnings} == {WarningCode.ENGINE_FALLBACK}
    assert FALLBACK in [w.message for w in warnings]
    assert len(warnings) == 4


def test_a_different_notice_for_a_role_is_logged_again(
    no_mlx: None, caplog: pytest.LogCaptureFixture
) -> None:
    cfg.preferred_engine = EngineSetting.MLX
    provider = _provider()
    with caplog.at_level(logging.WARNING):
        provider._warn_once(WorkerRole.CHAT)
        provider._warn_once(WorkerRole.CHAT)
        cfg.engine_choices = "not json"
        provider._warn_once(WorkerRole.CHAT)
    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 2
    assert messages[0] == FALLBACK
    assert messages[1].startswith(UNREADABLE_NOTICE)
