"""Parity of engine list and remove across the service, CLI, REST, MCP and the slash command."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest
from litestar.testing import AsyncTestClient
from typer.testing import CliRunner

import lilbee.mcp_server as mcp_server
from lilbee.app import engine_service
from lilbee.cli import app as cli_app
from lilbee.cli.tui.engine_commands import USAGE, run_engine_command
from lilbee.core.config import Config, cfg
from lilbee.providers.fleet.engine_store import EngineStore, engines_root
from lilbee.server.app import create_app
from lilbee.server.auth import session_manager
from lilbee.server.routes import engines as routes
from tests._litestar_test_client import create_test_client
from tests.cli.test_engine_builds import install

ALL_ROUTES = [
    routes.engine_builds_route,
    routes.engine_remove_route,
    routes.engine_roles_route,
]
BUILD = {"engine": "fx", "version": "1", "backend": "cpu"}
BUNDLED_REFUSAL = "inside the executable"
MISSING_REFUSAL = "Engine fx 9 (cpu) is not installed."


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    snapshot = cfg.model_copy()
    monkeypatch.setenv("LILBEE_DATA", str(tmp_path))
    monkeypatch.setenv("LILBEE_ENGINE_DIR", str(tmp_path / "slot"))
    cfg.data_root = tmp_path
    cfg.models_dir = tmp_path / "models"
    cfg.allow_http_engines = True
    cfg.mcp_engines_enabled = True
    yield
    mcp_server.set_http_mounted(False)
    for name in Config.model_fields:
        setattr(cfg, name, getattr(snapshot, name))


ENGINE_ROUTES = [
    ("GET", "/api/engines", None, 200),
    ("DELETE", "/api/engines/fx/1/cpu", None, 409),
    ("GET", "/api/engines/roles", None, 200),
    ("PUT", "/api/engines/roles/chat", {"model": "m", "engine": "e"}, 409),
    ("PUT", "/api/engines/preferred", {"engine": "e"}, 409),
]


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {session_manager.token}"}


@pytest.mark.parametrize(("method", "path", "body", "status"), ENGINE_ROUTES)
async def test_the_real_app_serves_every_engine_route(
    method: str, path: str, body: dict[str, str] | None, status: int
) -> None:
    cfg.allow_http_engines = False
    async with AsyncTestClient(create_app()) as client:
        res = await client.request(method, path, json=body, headers=_auth_headers())
    assert res.status_code == status


def _rest(method: str, path: str) -> tuple[int, Any]:
    with create_test_client(ALL_ROUTES, debug=False) as client:
        res = client.request(method, path)
        return res.status_code, res.json()


def _cli(*argv: str) -> tuple[int, dict[str, Any]]:
    result = CliRunner().invoke(cli_app, ["--json", "engine", *argv])
    return result.exit_code, json.loads(result.output.strip().splitlines()[-1])


def _installed() -> list[Any]:
    return EngineStore(engines_root(cfg.models_dir)).installed()


def test_every_surface_lists_the_same_builds() -> None:
    install(cfg.models_dir)
    views = engine_service.builds()
    assert [(v.engine, v.version, v.backend) for v in views] == [("fx", "1", "cpu")]
    code, listed = _cli("list")
    assert (code, listed["command"], listed["builds"]) == (0, "engine list", [BUILD])
    assert _rest("GET", "/api/engines") == (200, {"builds": [BUILD]})
    assert mcp_server.engine_list_tool() == {"builds": [BUILD]}
    assert run_engine_command("list").text == "fx 1 (cpu)"


def test_every_surface_lists_nothing_the_same_way() -> None:
    assert engine_service.builds() == []
    assert _cli("list")[1]["builds"] == []
    assert _rest("GET", "/api/engines") == (200, {"builds": []})
    assert mcp_server.engine_list_tool() == {"builds": []}
    assert run_engine_command("list").text == "No engine builds are installed."


def test_every_surface_removes_a_build_and_names_it_the_same_way() -> None:
    label = "fx 1 (cpu)"
    calls = {
        "service": lambda: engine_service.remove_build("fx", "1", "cpu"),
        "cli": lambda: _cli("remove", "fx", "1", "cpu")[1]["removed"],
        "rest": lambda: _rest("DELETE", "/api/engines/fx/1/cpu")[1]["removed"],
        "mcp": lambda: mcp_server.engine_remove_tool("fx", "1", "cpu")["removed"],
        "slash": lambda: run_engine_command("remove fx 1 cpu").text,
    }
    for surface, call in calls.items():
        install(cfg.models_dir)
        assert len(_installed()) == 1, surface
        got = call()
        assert label in got, surface
        assert _installed() == [], surface


def test_every_surface_shows_the_service_refusal_for_a_missing_build() -> None:
    with pytest.raises(engine_service.EngineRefusedError, match="is not installed") as exc:
        engine_service.remove_build("fx", "9", "cpu")
    text = str(exc.value)
    assert text == MISSING_REFUSAL
    code, body = _cli("remove", "fx", "9", "cpu")
    assert (code, body["error"]) == (1, text)
    code, body = _rest("DELETE", "/api/engines/fx/9/cpu")
    assert (code, body["detail"]) == (422, text)
    assert mcp_server.engine_remove_tool("fx", "9", "cpu") == {"error": text}
    slash = run_engine_command("remove fx 9 cpu")
    assert (slash.ok, slash.text) == (False, text)


def test_the_bundled_engine_is_refused_on_every_surface() -> None:
    with pytest.raises(engine_service.EngineRefusedError, match=BUNDLED_REFUSAL) as exc:
        engine_service.remove_build("llama.cpp", "bundled", "bundled")
    text = str(exc.value)
    assert _cli("remove", "llama.cpp", "bundled", "bundled")[1]["error"] == text
    code, body = _rest("DELETE", "/api/engines/llama.cpp/bundled/bundled")
    assert (code, body["detail"]) == (422, text)
    assert mcp_server.engine_remove_tool("llama.cpp", "bundled", "bundled") == {"error": text}
    assert run_engine_command("remove llama.cpp bundled bundled").text == text


def test_a_build_in_use_is_refused_and_kept_on_every_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(cfg.models_dir)
    monkeypatch.setattr(engine_service, "launch_uses_engine", lambda *_: True)
    with pytest.raises(engine_service.EngineRefusedError, match="is in use") as exc:
        engine_service.remove_build("fx", "1", "cpu")
    text = str(exc.value)
    assert _cli("remove", "fx", "1", "cpu")[1]["error"] == text
    assert _rest("DELETE", "/api/engines/fx/1/cpu")[1]["detail"] == text
    assert mcp_server.engine_remove_tool("fx", "1", "cpu") == {"error": text}
    assert run_engine_command("remove fx 1 cpu").text == text
    assert len(_installed()) == 1


def test_a_name_that_leaves_the_engines_folder_is_refused_and_removes_nothing() -> None:
    install(cfg.models_dir)
    with pytest.raises(engine_service.EngineRefusedError, match="cannot name an engine build"):
        engine_service.remove_build("..", "1", "cpu")
    assert mcp_server.engine_remove_tool("fx", "1", "../cpu")["error"].endswith(
        "cannot name an engine build."
    )
    assert len(_installed()) == 1


def test_rest_remove_is_refused_while_allow_http_engines_is_off() -> None:
    install(cfg.models_dir)
    cfg.allow_http_engines = False
    code, body = _rest("DELETE", "/api/engines/fx/1/cpu")
    assert (code, body["detail"]) == (409, engine_service.http_refused_message())
    assert len(_installed()) == 1


def test_rest_list_stays_open_while_the_flag_is_off() -> None:
    install(cfg.models_dir)
    cfg.allow_http_engines = False
    assert _rest("GET", "/api/engines") == (200, {"builds": [BUILD]})


def test_mcp_remove_on_the_http_mount_is_refused_while_the_flag_is_off() -> None:
    install(cfg.models_dir)
    cfg.allow_http_engines = False
    mcp_server.set_http_mounted(True)
    assert mcp_server.engine_remove_tool("fx", "1", "cpu") == {
        "error": engine_service.http_refused_message()
    }
    assert mcp_server.engine_list_tool() == {"builds": [BUILD]}
    assert len(_installed()) == 1


def test_mcp_remove_works_off_the_http_mount_with_the_flag_off() -> None:
    install(cfg.models_dir)
    cfg.allow_http_engines = False
    assert mcp_server.engine_remove_tool("fx", "1", "cpu") == {"removed": "fx 1 (cpu)"}


def test_the_engine_tools_of_a_live_server_refuse_once_the_setting_goes_off() -> None:
    install(cfg.models_dir)
    cfg.mcp_engines_enabled = False
    refusal = {"error": mcp_server.ENGINES_DISABLED_ERROR}
    assert mcp_server.engine_list_tool() == refusal
    assert mcp_server.engine_remove_tool("fx", "1", "cpu") == refusal
    assert len(_installed()) == 1


def test_the_setting_on_registers_the_list_and_remove_tools() -> None:
    cfg.mcp_engines_enabled = False
    before = {t.name for t in mcp_server.build_mcp_server()._tool_manager.list_tools()}
    cfg.mcp_engines_enabled = True
    after = {t.name for t in mcp_server.build_mcp_server()._tool_manager.list_tools()}
    assert {"engine_list", "engine_remove"} <= after - before


def test_a_rest_remove_does_not_block_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    released = threading.Event()
    waited: list[bool] = []

    def blocking_remove(*_args: str) -> str:
        waited.append(released.wait(timeout=3))
        return "fx 1 (cpu)"

    async def release_from_the_loop() -> None:
        await asyncio.sleep(0.05)
        released.set()

    async def scenario() -> int:
        app = create_app()
        transport = httpx.ASGITransport(app=app)
        async with (
            app.lifespan(),
            httpx.AsyncClient(transport=transport, base_url="http://app") as client,
        ):
            ticker = asyncio.create_task(release_from_the_loop())
            res = await client.delete("/api/engines/fx/1/cpu", headers=_auth_headers())
            await ticker
        return res.status_code

    monkeypatch.setattr(engine_service, "remove_build", blocking_remove)
    assert asyncio.run(scenario()) == 200
    assert waited == [True]


def test_the_slash_command_usage_names_list_and_remove() -> None:
    assert "/engine list" in USAGE
    assert "/engine remove" in USAGE
    for bad in ("remove fx 1", "list extra"):
        out = run_engine_command(bad)
        assert (out.ok, out.text) == (False, USAGE), bad
