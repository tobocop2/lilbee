"""Installing an engine: one service call behind the CLI, REST, MCP and slash command."""

from __future__ import annotations

import hashlib
import json
import re
import socket
import stat
import time
import zipfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from huggingface_hub import constants as hub_constants
from typer.testing import CliRunner

import lilbee.mcp_server as mcp_server
from lilbee.app import engine_choice, engine_service
from lilbee.catalog import download, runnable
from lilbee.catalog.models import DownloadProgress
from lilbee.cli import app as cli_app
from lilbee.cli.tui.engine_commands import run_engine_command
from lilbee.cli.tui.screens.chat import ChatScreen
from lilbee.cli.tui.task_queue import TaskStatus
from lilbee.core.config import Config, cfg
from lilbee.core.config.enums import EngineSetting
from lilbee.providers.fleet import engine_manifest, engine_store, host_caps
from lilbee.providers.fleet.engine_store import EngineStore, engines_root
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.choice import installed_engines
from lilbee.providers.fleet.engines.host_gate import PASS, HostGate
from lilbee.providers.fleet.host_caps import Feature, HostArch, HostCaps, HostOS, collect
from lilbee.server.routes import engines as routes
from tests._async_wait import wait_until
from tests._lilbee_app_test_host import await_chat
from tests._litestar_test_client import create_test_client
from tests._release_server import ReleaseServer, release_server
from tests.conformance.fake_adapter import FakeAdapter
from tests.test_engine_service import GGUF_ENGINE
from tests.test_tui_engine_choice import _ChatApp

ENGINE = EngineName.IK_LLAMA
VERSION = "git-0123456789ab"
NO_AVX2 = "This host has no AVX2. ik_llama.cpp needs it."
STAYS = "llama.cpp stays in use."
BAD_PACK = (
    "The file for engine ik_llama.cpp git-0123456789ab (cpu) does not match its pin, "
    "so lilbee did not install it. Run the install again. If this happens again, report it."
)
ROUTES = [routes.engine_install_route]


def make_wheel(path: Path, package: str = "lilbee_engine_fx") -> str:
    """A wheel with one executable and an MIT licence; returns its sha256."""
    with zipfile.ZipFile(path, "w") as archive:
        tool = zipfile.ZipInfo(f"{package}/bin/server")
        tool.external_attr = (stat.S_IFREG | 0o755) << 16
        archive.writestr(tool, "#!/bin/sh\n")
        archive.writestr(f"{package}-1.dist-info/METADATA", "License-Expression: MIT\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StandIn(FakeAdapter):
    """A managed engine with a known pin and a gate the test sets."""

    name = ENGINE.value
    formats = frozenset({"gguf"})

    def __init__(
        self,
        sha: str | None,
        gate: HostGate = PASS,
        licences: tuple[str, ...] = ("MIT",),
        cells: Mapping[str, Any] | None = None,
    ):
        self.sha, self.gate, self.licences = sha, gate, licences
        self.cells = cells or {}

    def host_gate(self, host: Any) -> HostGate:
        return self.gate

    def pin_record(self) -> Mapping[str, Any]:
        return {
            "engine": self.name,
            "version": VERSION,
            "backend": "cpu",
            "platforms": self.cells,
            "sha256": self.sha,
            "licenses": list(self.licences),
            "tool": "server",
        }


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    snapshot = cfg.model_copy()
    monkeypatch.setenv("LILBEE_DATA", str(tmp_path))
    cfg.data_root = tmp_path
    cfg.models_dir = tmp_path / "models"
    cfg.allow_http_engines = True
    cfg.mcp_engines_enabled = True
    cfg.preferred_engine = EngineSetting.AUTO
    cfg.engine_choices = None
    monkeypatch.setattr(
        engine_choice,
        "session_host",
        lambda: collect(
            system="Linux", machine="x86_64", cpu_info={"flags": ["avx2"]}, nvidia=None
        ),
    )
    monkeypatch.setitem(choice.ADAPTERS, EngineName.LLAMA, GGUF_ENGINE)  # type: ignore[arg-type]
    monkeypatch.setattr(runnable, "detect", _live_host)
    yield
    mcp_server.set_http_mounted(False)
    for name in Config.model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _live_host() -> host_caps.HostCaps:
    return host_caps.collect(
        system="Linux",
        machine="x86_64",
        cpu_info={"flags": ["avx2"], "brand_raw": "Test CPU"},
        nvidia=None,
    )


@pytest.fixture
def release(monkeypatch: pytest.MonkeyPatch) -> Iterator[ReleaseServer]:
    """A loopback release server; the hub's offline switch is off while it runs."""
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(engine_store, "PACKAGES_URL", "http://127.0.0.1:")
    with release_server() as server:
        yield server


def _cell(
    url: str, sha: str = "a" * 64, min_os: str | None = None, size: int = 1
) -> dict[str, Any]:
    return {"linux-x86_64-cpu": {"url": url, "sha256": sha, "size": size, "min_os": min_os}}


@pytest.fixture
def good(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, release: ReleaseServer) -> Path:
    """Register the stand-in with the sha of a good wheel; its cell is a 404 on the server."""
    wheel = tmp_path / "good.whl"
    sha = make_wheel(wheel)
    cells = _cell(f"{release.url}/gone.whl", sha)
    monkeypatch.setitem(choice.ADAPTERS, ENGINE, StandIn(sha, cells=cells))  # type: ignore[arg-type]
    return wheel


@pytest.fixture
def root() -> Path:
    return engines_root(cfg.models_dir)


def _builds() -> list[tuple[str, str, str]]:
    return [
        (b.engine, b.version, b.backend)
        for b in EngineStore(engines_root(cfg.models_dir)).installed()
    ]


def test_a_good_wheel_is_installed_and_the_engine_is_listed(good: Path) -> None:
    assert ENGINE not in installed_engines()
    built = engine_service.install(ENGINE.value, good)
    assert (built.engine, built.version, built.backend) == (ENGINE.value, VERSION, "cpu")
    assert _builds() == [(ENGINE.value, VERSION, "cpu")]
    assert ENGINE in installed_engines()


def test_a_wrong_sha256_is_refused_with_the_store_text_and_leaves_nothing(
    good: Path, tmp_path: Path, root: Path
) -> None:
    other = tmp_path / "other.whl"
    make_wheel(other, package="lilbee_engine_other")
    with pytest.raises(engine_service.EngineRefusedError) as exc:
        engine_service.install(ENGINE.value, other)
    assert str(exc.value) == BAD_PACK
    assert _builds() == []
    assert not root.exists()


def test_no_pack_and_no_network_gives_the_missing_pack_text(good: Path, root: Path) -> None:
    with pytest.raises(engine_service.EngineRefusedError) as exc:
        engine_service.install(ENGINE.value)
    assert str(exc.value) == (
        f"Engine {ENGINE} {VERSION} (cpu) is not installed and cannot be downloaded. "
        f"Copy {ENGINE}-{VERSION}-cpu.whl into {root / 'packs'} and run the install again."
    )
    assert _builds() == []


def test_a_download_from_the_release_url_installs_the_build(
    good: Path, release: ReleaseServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    release.files["/fx.whl"] = good.read_bytes()
    sha = hashlib.sha256(good.read_bytes()).hexdigest()
    cells = _cell(f"{release.url}/fx.whl", sha, size=good.stat().st_size)
    monkeypatch.setitem(choice.ADAPTERS, ENGINE, StandIn(sha, cells=cells))  # type: ignore[arg-type]
    built = engine_service.install(ENGINE.value)
    assert (built.engine, built.version, built.backend) == (ENGINE.value, VERSION, "cpu")
    assert _builds() == [(ENGINE.value, VERSION, "cpu")]
    assert len(release.requests) == 1


def test_a_host_with_no_cell_is_refused_and_no_request_is_sent(
    good: Path, release: ReleaseServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = {"macos-arm64-metal": _cell(f"{release.url}/fx.whl")["linux-x86_64-cpu"]}
    monkeypatch.setitem(choice.ADAPTERS, ENGINE, StandIn("a" * 64, cells=other))  # type: ignore[arg-type]
    with pytest.raises(engine_service.EngineRefusedError) as exc:
        engine_service.install(ENGINE.value)
    assert str(exc.value) == (
        f"Engine {ENGINE} has no cpu build for linux-x86_64. Builds for this computer: none."
    )
    assert (release.requests, _builds()) == ([], [])


@pytest.mark.parametrize("with_wheel", [False, True])
def test_a_system_below_the_floor_is_refused_on_the_download_and_the_wheel_route(
    good: Path, release: ReleaseServer, monkeypatch: pytest.MonkeyPatch, with_wheel: bool
) -> None:
    cells = _cell(f"{release.url}/fx.whl", min_os="2.34")
    monkeypatch.setitem(choice.ADAPTERS, ENGINE, StandIn("a" * 64, cells=cells))  # type: ignore[arg-type]
    monkeypatch.setattr(engine_manifest, "host_os_version", lambda _os: "2.31")
    with pytest.raises(engine_service.EngineRefusedError) as exc:
        engine_service.install(ENGINE.value, good if with_wheel else None)
    head = f"Engine {ENGINE} {VERSION} (cpu) needs glibc 2.34 or later. "
    tail = "Update the system."
    assert str(exc.value) == f"{head}This computer has glibc 2.31. {tail}"
    assert (release.requests, _builds()) == ([], [])


def test_a_host_the_manifest_cannot_name_is_refused(
    good: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runnable,
        "detect",
        lambda: host_caps.collect(system="Plan9", machine="x86_64", cpu_info={}, nvidia=None),
    )
    with pytest.raises(engine_service.EngineRefusedError) as exc:
        engine_service.install(ENGINE.value, good)
    assert str(exc.value) == f"Engine {ENGINE} has no cpu build for this computer."


def test_a_download_the_network_keeps_cutting_gives_the_missing_pack_text(
    good: Path, release: ReleaseServer, monkeypatch: pytest.MonkeyPatch, root: Path
) -> None:
    release.files["/fx.whl"] = good.read_bytes()
    release.reset_next = 10
    sha = hashlib.sha256(good.read_bytes()).hexdigest()
    cells = _cell(f"{release.url}/fx.whl", sha, size=good.stat().st_size)
    monkeypatch.setitem(choice.ADAPTERS, ENGINE, StandIn(sha, cells=cells))  # type: ignore[arg-type]
    with (
        mock.patch.object(download.time, "sleep"),
        pytest.raises(engine_service.EngineRefusedError) as exc,
    ):
        engine_service.install(ENGINE.value)
    assert str(exc.value) == (
        f"Engine {ENGINE} {VERSION} (cpu) is not installed and cannot be downloaded. "
        f"Copy {ENGINE}-{VERSION}-cpu.whl into {root / 'packs'} and run the install again."
    )
    assert len(release.requests) == 3
    assert _builds() == []


def _downloadable(good: Path, release: ReleaseServer, monkeypatch: pytest.MonkeyPatch) -> None:
    release.files["/fx.whl"] = good.read_bytes()
    sha = hashlib.sha256(good.read_bytes()).hexdigest()
    cells = _cell(f"{release.url}/fx.whl", sha, size=good.stat().st_size)
    monkeypatch.setitem(choice.ADAPTERS, ENGINE, StandIn(sha, cells=cells))  # type: ignore[arg-type]


def _leftovers(root: Path) -> list[str]:
    """Every download folder, lock file and stage folder still under the engines folder."""
    return sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() or path.name.startswith((".download-", ".stage-"))
    )


async def test_cancelling_the_slash_install_mid_download_leaves_no_partial_file_and_no_lock(
    good: Path, release: ReleaseServer, monkeypatch: pytest.MonkeyPatch, root: Path
) -> None:
    _downloadable(good, release, monkeypatch)
    app = _ChatApp()
    real = engine_service.make_download_callback

    def cancelling(on_update: Any) -> Any:
        inner = real(on_update)

        def on_bytes(done: int, total: int) -> None:
            app.task_bar.cancel_task(app.task_bar.queue.active_tasks[0].task_id)
            inner(done, total)

        return on_bytes

    async with app.run_test(size=(120, 40)) as pilot:
        chat = await await_chat(app, pilot)
        assert isinstance(chat, ChatScreen)
        monkeypatch.setattr(engine_service, "make_download_callback", cancelling)
        chat._cmd_engine(f"install {ENGINE.value}")
        assert await wait_until(
            pilot,
            lambda: bool(app.task_bar.queue.history) and not app.task_bar._workers,
            timeout=10,
        )
        (row,) = app.task_bar.queue.history
        assert row.status is TaskStatus.CANCELLED
    assert len(release.requests) == 1
    assert _builds() == []
    assert _leftovers(root) == []


async def test_cancelling_the_slash_install_during_the_backoff_ends_it_before_the_next_connect(
    good: Path, release: ReleaseServer, monkeypatch: pytest.MonkeyPatch, root: Path
) -> None:
    _downloadable(good, release, monkeypatch)
    release.reset_next = 10
    monkeypatch.setattr(download, "_RETRY_BACKOFF_SECONDS", 60)
    app = _ChatApp()
    async with app.run_test(size=(120, 40)) as pilot:
        chat = await await_chat(app, pilot)
        assert isinstance(chat, ChatScreen)
        chat._cmd_engine(f"install {ENGINE.value}")
        assert await wait_until(
            pilot,
            lambda: any("trying again" in str(n.message) for n in app._notifications),
            timeout=10,
        )
        started = time.monotonic()
        app.task_bar.cancel_task(app.task_bar.queue.active_tasks[0].task_id)
        assert await wait_until(
            pilot,
            lambda: bool(app.task_bar.queue.history) and not app.task_bar._workers,
            timeout=10,
        )
        (row,) = app.task_bar.queue.history
        assert row.status is TaskStatus.CANCELLED
        assert time.monotonic() - started < 5
    assert len(release.requests) == 1
    assert _builds() == []
    assert _leftovers(root) == []


def test_a_pin_with_no_cells_installs_a_wheel_on_a_host_the_manifest_cannot_name(
    good: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        choice.ADAPTERS, ENGINE, StandIn(hashlib.sha256(good.read_bytes()).hexdigest())
    )  # type: ignore[arg-type]
    monkeypatch.setattr(
        runnable,
        "detect",
        lambda: host_caps.collect(system="Plan9", machine="riscv64", cpu_info={}, nvidia=None),
    )
    engine_service.install(ENGINE.value, good)
    assert _builds() == [(ENGINE.value, VERSION, "cpu")]


def test_a_pack_in_the_packs_folder_is_installed_without_a_wheel_argument(
    good: Path, root: Path
) -> None:
    (root / "packs").mkdir(parents=True)
    (root / "packs" / f"{ENGINE}-{VERSION}-cpu.whl").write_bytes(good.read_bytes())
    engine_service.install(ENGINE.value)
    assert _builds() == [(ENGINE.value, VERSION, "cpu")]


def test_a_host_that_fails_the_gate_is_refused_before_the_store_is_touched(
    good: Path, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(choice.ADAPTERS, ENGINE, StandIn("a" * 64, HostGate(False, NO_AVX2)))  # type: ignore[arg-type]
    with pytest.raises(engine_service.EngineRefusedError) as exc:
        engine_service.install(ENGINE.value, good)
    assert str(exc.value) == f"Engine {ENGINE} cannot run on this host. {NO_AVX2} {STAYS}"
    assert not root.exists()


def test_an_unknown_name_is_refused_naming_the_engines() -> None:
    with pytest.raises(engine_service.EngineRefusedError, match="No engine is named 'nonesuch'"):
        engine_service.install("nonesuch")


def test_an_engine_with_no_adapter_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    without = {name: adapter for name, adapter in choice.ADAPTERS.items() if name != ENGINE}
    assert len(without) == len(choice.ADAPTERS) - 1
    monkeypatch.setattr(choice, "ADAPTERS", without)
    with pytest.raises(engine_service.EngineRefusedError, match="has no build to install"):
        engine_service.install(ENGINE.value)


def test_the_engine_in_the_package_is_refused(tmp_path: Path) -> None:
    server = tmp_path / "llama-server"
    server.write_text("#!/bin/sh\n", encoding="utf-8")
    cfg.llama_server_path = str(server)
    with pytest.raises(engine_service.EngineRefusedError, match="ships with lilbee"):
        engine_service.install(EngineName.LLAMA.value)


def test_a_licence_off_the_permissive_list_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel = tmp_path / "good.whl"
    stand_in = StandIn(make_wheel(wheel), licences=("GPL-3.0-only",))
    monkeypatch.setitem(choice.ADAPTERS, ENGINE, stand_in)  # type: ignore[arg-type]
    with pytest.raises(engine_service.EngineRefusedError, match=re.escape("GPL-3.0-only")):
        engine_service.install(ENGINE.value, wheel)
    assert _builds() == []


def test_a_build_of_another_wheel_is_refused_not_replaced(
    good: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine_service.install(ENGINE.value, good)
    other = tmp_path / "other.whl"
    sha = make_wheel(other, package="lilbee_engine_other")
    monkeypatch.setitem(choice.ADAPTERS, ENGINE, StandIn(sha))  # type: ignore[arg-type]
    with pytest.raises(engine_service.EngineRefusedError, match="different wheel than its pin"):
        engine_service.install(ENGINE.value, other)
    assert _builds() == [(ENGINE.value, VERSION, "cpu")]


def test_a_pin_that_names_no_backend_is_refused(
    good: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class NoBackend(StandIn):
        def pin_record(self) -> Mapping[str, Any]:
            return {**super().pin_record(), "backend": None}

    monkeypatch.setitem(choice.ADAPTERS, ENGINE, NoBackend("a" * 64))  # type: ignore[arg-type]
    with pytest.raises(engine_service.EngineRefusedError, match="pin record is not usable"):
        engine_service.install(ENGINE.value, good)


def test_a_missing_wheel_and_a_pinned_file_that_is_not_a_wheel_are_refused(
    good: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(engine_service.EngineRefusedError, match="was not installed"):
        engine_service.install(ENGINE.value, tmp_path / "absent.whl")
    junk = tmp_path / "junk.whl"
    junk.write_text("not a zip", encoding="utf-8")
    pinned = hashlib.sha256(junk.read_bytes()).hexdigest()
    monkeypatch.setitem(choice.ADAPTERS, ENGINE, StandIn(pinned))  # type: ignore[arg-type]
    with pytest.raises(engine_service.EngineRefusedError, match="was not installed"):
        engine_service.install(ENGINE.value, junk)


# ------------------------------------------------------------------ surfaces


def _cli(*argv: str) -> tuple[int, dict[str, Any]]:
    result = CliRunner().invoke(cli_app, ["--json", "engine", *argv])
    return result.exit_code, json.loads(result.stdout.strip().splitlines()[-1])


def _rest(body: dict[str, Any]) -> tuple[int, Any]:
    with create_test_client(ROUTES, debug=False) as client:
        res = client.post("/api/engines/install", json=body)
        return res.status_code, res.json()


def test_the_cli_installs_and_lists_the_engine(good: Path) -> None:
    code, body = _cli("install", ENGINE.value, "--wheel", str(good))
    assert (code, body) == (
        0,
        {
            "command": "engine install",
            "installed": f"{ENGINE} {VERSION} (cpu)",
            "note": "",
            "not_run_on": [],
        },
    )
    listing = CliRunner().invoke(cli_app, ["engine", "list"])
    assert f"{ENGINE} {VERSION} (cpu)" in listing.output


def test_the_cli_lists_a_build_that_differs_from_the_pin_on_one_line(
    good: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine_service.install(ENGINE.value, good)
    monkeypatch.setitem(choice.ADAPTERS, ENGINE, StandIn("0" * 64))  # type: ignore[arg-type]
    output = " ".join(CliRunner().invoke(cli_app, ["engine", "list"]).output.split())
    assert output.count(ENGINE.value) == 1
    assert f"{ENGINE} not installed: the installed build {VERSION} (cpu) differs from the pin" in (
        output
    )


def test_the_cli_text_mode_says_what_was_installed(good: Path) -> None:
    result = CliRunner().invoke(cli_app, ["engine", "install", ENGINE.value, "--wheel", str(good)])
    assert result.exit_code == 0
    assert f"Installed engine {ENGINE} {VERSION} (cpu)." in result.output


def test_the_cli_refuses_with_the_service_text(good: Path, tmp_path: Path) -> None:
    other = tmp_path / "other.whl"
    make_wheel(other, package="lilbee_engine_other")
    assert _cli("install", ENGINE.value, "--wheel", str(other)) == (
        1,
        {"command": "engine install", "error": BAD_PACK},
    )
    result = CliRunner().invoke(cli_app, ["engine", "install", ENGINE.value, "--wheel", str(other)])
    assert result.exit_code == 1


def test_rest_installs_a_pack_from_the_packs_folder(good: Path, root: Path) -> None:
    (root / "packs").mkdir(parents=True)
    (root / "packs" / f"{ENGINE}-{VERSION}-cpu.whl").write_bytes(good.read_bytes())
    assert _rest({"engine": ENGINE.value}) == (
        200,
        {
            "engine": ENGINE.value,
            "version": VERSION,
            "backend": "cpu",
            "note": "",
            "not_run_on": [],
        },
    )
    assert _builds() == [(ENGINE.value, VERSION, "cpu")]


def test_rest_refuses_with_a_422_and_the_service_text(good: Path) -> None:
    code, body = _rest({"engine": ENGINE.value})
    assert code == 422
    assert "cannot be downloaded" in body["detail"]


def test_rest_install_is_gated_like_the_other_writes(good: Path, root: Path) -> None:
    cfg.allow_http_engines = False
    (root / "packs").mkdir(parents=True)
    (root / "packs" / f"{ENGINE}-{VERSION}-cpu.whl").write_bytes(good.read_bytes())
    code, body = _rest({"engine": ENGINE.value})
    assert (code, body["detail"]) == (409, engine_service.http_refused_message())
    assert _builds() == []


def test_mcp_installs_and_refuses(good: Path, root: Path) -> None:
    refused = mcp_server.engine_install_tool(ENGINE.value)
    assert "cannot be downloaded" in refused["error"]
    (root / "packs").mkdir(parents=True)
    (root / "packs" / f"{ENGINE}-{VERSION}-cpu.whl").write_bytes(good.read_bytes())
    assert mcp_server.engine_install_tool(ENGINE.value) == {
        "engine": ENGINE.value,
        "version": VERSION,
        "backend": "cpu",
        "note": "",
        "not_run_on": [],
    }
    assert _builds() == [(ENGINE.value, VERSION, "cpu")]


def test_the_mcp_tool_is_off_the_wire_and_refuses_while_the_setting_is_off(good: Path) -> None:
    cfg.mcp_engines_enabled = False
    names = {t.name for t in mcp_server.build_mcp_server()._tool_manager.list_tools()}
    assert "engine_install" not in names
    assert mcp_server.engine_install_tool(ENGINE.value) == {
        "error": mcp_server.ENGINES_DISABLED_ERROR
    }
    cfg.mcp_engines_enabled = True
    names = {t.name for t in mcp_server.build_mcp_server()._tool_manager.list_tools()}
    assert "engine_install" in names
    assert _builds() == []


def test_the_mcp_tool_is_refused_on_the_http_server_unless_allowed(good: Path) -> None:
    cfg.allow_http_engines = False
    mcp_server.set_http_mounted(True)
    assert mcp_server.engine_install_tool(ENGINE.value) == {
        "error": engine_service.http_refused_message()
    }


def test_the_slash_command_installs_from_the_packs_folder(good: Path, root: Path) -> None:
    (root / "packs").mkdir(parents=True)
    (root / "packs" / f"{ENGINE}-{VERSION}-cpu.whl").write_bytes(good.read_bytes())
    out = run_engine_command(f"install {ENGINE.value}")
    assert out.ok
    assert f"{ENGINE} {VERSION} (cpu)" in out.text
    assert _builds() == [(ENGINE.value, VERSION, "cpu")]


def test_the_slash_command_with_no_name_shows_usage() -> None:
    out = run_engine_command("install")
    assert (out.ok, "/engine install ENGINE" in out.text) == (False, True)


NO_PACK = "No pack of ik_llama.cpp is published yet. llama.cpp stays in use."


def test_an_engine_with_no_published_pack_is_refused_on_every_surface_and_touches_nothing(
    tmp_path: Path, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(choice.ADAPTERS, ENGINE, StandIn(None))  # type: ignore[arg-type]
    reads: list[str] = []

    def spy(self: object, *args: object, **kwargs: object) -> None:
        reads.append("store")

    monkeypatch.setattr(EngineStore, "ensure", spy)
    monkeypatch.setattr(EngineStore, "install_from_wheel", spy)
    real_connect = socket.socket.connect

    def connect_spy(sock: socket.socket, address: Any, *args: object) -> None:
        # The event loop of the REST client joins its own socket pair over loopback on Windows.
        if not isinstance(address, tuple) or address[0] in ("127.0.0.1", "::1"):
            real_connect(sock, address, *args)
        else:
            reads.append("network")

    monkeypatch.setattr(socket.socket, "connect", connect_spy)
    pack = tmp_path / "pack.whl"
    pack.write_bytes(b"not read")
    with pytest.raises(engine_service.EngineRefusedError) as exc:
        engine_service.install(ENGINE.value, pack)
    assert str(exc.value) == NO_PACK
    assert _cli("install", ENGINE.value, "--wheel", str(pack))[1]["error"] == NO_PACK
    assert _cli("install", ENGINE.value)[1]["error"] == NO_PACK
    code, body = _rest({"engine": ENGINE.value})
    assert (code, body["detail"]) == (422, NO_PACK)
    assert mcp_server.engine_install_tool(ENGINE.value) == {"error": NO_PACK}
    assert run_engine_command(f"install {ENGINE.value}").text == NO_PACK
    assert (reads, root.exists()) == ([], False)


def test_an_engine_with_no_published_pack_is_not_installed_and_logs_no_warning(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(choice.ADAPTERS, ENGINE, StandIn(None))  # type: ignore[arg-type]
    assert ENGINE not in installed_engines()
    assert caplog.records == []


@pytest.mark.parametrize("name", ["nonesuch", ENGINE.value])
def test_every_surface_shows_the_service_refusal(
    name: str, good: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if name == ENGINE.value:
        monkeypatch.setitem(choice.ADAPTERS, ENGINE, StandIn("a" * 64, HostGate(False, NO_AVX2)))  # type: ignore[arg-type]
    with pytest.raises(engine_service.EngineRefusedError) as exc:
        engine_service.install(name)
    expected = str(exc.value)
    assert expected
    assert expected.endswith(STAYS) is (name == ENGINE.value)
    assert _cli("install", name)[1]["error"] == expected
    code, body = _rest({"engine": name})
    assert (code, body["detail"]) == (422, expected)
    assert mcp_server.engine_install_tool(name) == {"error": expected}
    assert run_engine_command(f"install {name}").text == expected


def test_the_server_app_serves_the_install_route() -> None:
    from lilbee.server.app import create_app

    paths = {route.path for route in create_app().routes}
    assert "/api/engines/install" in paths


# ------------------------------------------------------------------ review answers

WHEEL_CLI_ONLY = "The wheel option works only with the lilbee engine install command."


def _pinned(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setitem(choice.ADAPTERS, ENGINE, StandIn(sha))  # type: ignore[arg-type]


def _damaged_wheel(path: Path, kind: str) -> None:
    """A wheel that is a zip file but fails when the store reads it."""
    with zipfile.ZipFile(path, "w") as archive:
        data = b"\xff\xfe" if kind == "non-utf8" else b"License-Expression: MIT\n"
        archive.writestr("lilbee_engine_fx-1.dist-info/METADATA", data)
    patches = {"encrypted": ((6, 1), (8, 1)), "compression": ((8, 99), (10, 99))}
    if kind in patches:
        raw = bytearray(path.read_bytes())
        markers = (b"PK\x03\x04", b"PK\x01\x02")
        for marker, (offset, value) in zip(markers, patches[kind], strict=True):
            at = raw.find(marker) + offset
            raw[at : at + 2] = value.to_bytes(2, "little")
        path.write_bytes(bytes(raw))


DAMAGED = ["non-utf8", "encrypted", "compression"]


@pytest.mark.parametrize("kind", DAMAGED)
def test_a_pinned_wheel_the_store_cannot_read_is_refused_on_every_surface(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel = tmp_path / "damaged.whl"
    _damaged_wheel(wheel, kind)
    _pinned(monkeypatch, wheel)
    with pytest.raises(engine_service.EngineRefusedError, match="was not installed") as exc:
        engine_service.install(ENGINE.value, wheel)
    expected = str(exc.value)
    assert _cli("install", ENGINE.value, "--wheel", str(wheel)) == (
        1,
        {"command": "engine install", "error": expected},
    )
    root_dir = engines_root(cfg.models_dir)
    (root_dir / "packs").mkdir(parents=True)
    (root_dir / "packs" / f"{ENGINE}-{VERSION}-cpu.whl").write_bytes(wheel.read_bytes())
    code, body = _rest({"engine": ENGINE.value})
    assert (code, body["detail"]) == (422, expected)
    assert mcp_server.engine_install_tool(ENGINE.value) == {"error": expected}
    slash = run_engine_command(f"install {ENGINE.value}")
    assert (slash.ok, slash.text) == (False, expected)
    assert _builds() == []


def test_auto_is_refused_as_not_installable_and_is_left_out_of_the_list() -> None:
    with pytest.raises(engine_service.EngineRefusedError) as exc:
        engine_service.install("auto")
    text = str(exc.value)
    assert "auto" in text.split("Choose one of:")[0]
    assert "cannot be installed" in text
    assert "auto" not in text.split("Choose one of:")[1]
    assert _cli("install", "auto")[1]["error"] == text
    assert _rest({"engine": "auto"})[1]["detail"] == text
    assert mcp_server.engine_install_tool("auto") == {"error": text}
    assert run_engine_command("install auto").text == text


def test_the_unknown_name_list_for_install_leaves_out_auto() -> None:
    with pytest.raises(engine_service.EngineRefusedError) as exc:
        engine_service.install("nonesuch")
    assert "llama.cpp" in str(exc.value)
    assert "auto" not in str(exc.value)


def test_the_cli_takes_only_a_regular_file_as_the_wheel(good: Path, tmp_path: Path) -> None:
    missing = tmp_path / "absent.whl"
    code, body = _cli("install", ENGINE.value, "--wheel", str(missing))
    assert (code, body) == (
        1,
        {"command": "engine install", "error": f"{missing} is not a file. Name a wheel file."},
    )
    assert "Errno" not in body["error"]
    folder = tmp_path / "folder"
    folder.mkdir()
    assert _cli("install", ENGINE.value, "--wheel", str(folder))[1]["error"] == (
        f"{folder} is not a file. Name a wheel file."
    )


@pytest.mark.skipif(not hasattr(__import__("os"), "mkfifo"), reason="needs a POSIX fifo")
def test_the_cli_returns_at_once_for_a_fifo_as_the_wheel(good: Path, tmp_path: Path) -> None:
    import os

    fifo = tmp_path / "pipe.whl"
    os.mkfifo(fifo)
    held = os.open(fifo, os.O_RDWR | os.O_NONBLOCK)
    try:
        assert _cli("install", ENGINE.value, "--wheel", str(fifo))[1]["error"] == (
            f"{fifo} is not a file. Name a wheel file."
        )
    finally:
        os.close(held)


def test_rest_refuses_a_wheel_field_and_installs_nothing(good: Path, root: Path) -> None:
    (root / "packs").mkdir(parents=True)
    (root / "packs" / f"{ENGINE}-{VERSION}-cpu.whl").write_bytes(good.read_bytes())
    code, body = _rest({"engine": ENGINE.value, "wheel": str(good)})
    assert (code, body["detail"]) == (422, WHEEL_CLI_ONLY)
    assert _builds() == []


def test_mcp_refuses_a_wheel_argument_and_installs_nothing(good: Path, root: Path) -> None:
    (root / "packs").mkdir(parents=True)
    (root / "packs" / f"{ENGINE}-{VERSION}-cpu.whl").write_bytes(good.read_bytes())
    assert mcp_server.engine_install_tool(ENGINE.value, wheel=str(good)) == {
        "error": WHEEL_CLI_ONLY
    }
    assert _builds() == []


NOTE = "This host has no NVIDIA GPU, so the CPU archive runs."


def test_a_note_from_the_host_gate_reaches_every_install_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = engines_root(cfg.models_dir)
    (root / "packs").mkdir(parents=True)
    wheel = tmp_path / "noted.whl"
    monkeypatch.setitem(
        choice.ADAPTERS,
        ENGINE,
        StandIn(make_wheel(wheel), HostGate(True, NOTE)),  # type: ignore[arg-type]
    )
    assert engine_service.install(ENGINE.value, wheel).note == NOTE
    code, body = _cli("install", ENGINE.value, "--wheel", str(wheel))
    assert (code, body["note"]) == (0, NOTE)
    text = CliRunner().invoke(cli_app, ["engine", "install", ENGINE.value, "--wheel", str(wheel)])
    assert NOTE in text.output
    (root / "packs" / f"{ENGINE}-{VERSION}-cpu.whl").write_bytes(wheel.read_bytes())
    assert _rest({"engine": ENGINE.value})[1]["note"] == NOTE
    assert mcp_server.engine_install_tool(ENGINE.value)["note"] == NOTE
    assert NOTE in run_engine_command(f"install {ENGINE.value}").text


def _two_digest_pin(
    monkeypatch: pytest.MonkeyPatch, url: str, cell_sha: str, top_sha: str, size: int = 1
) -> None:
    cells = _cell(url, cell_sha, size=size)
    monkeypatch.setitem(choice.ADAPTERS, ENGINE, StandIn(top_sha, cells=cells))  # type: ignore[arg-type]


def test_a_pin_whose_top_digest_is_another_cells_installs_a_wheel_of_this_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel = tmp_path / "cell.whl"
    cell_sha = make_wheel(wheel)
    _two_digest_pin(monkeypatch, "https://example.invalid/x.whl", cell_sha, "b" * 64)
    engine_service.install(ENGINE.value, wheel)
    (installed,) = EngineStore(engines_root(cfg.models_dir)).installed()
    assert installed.sha256 == cell_sha
    assert _builds() == [(ENGINE.value, VERSION, "cpu")]


def test_a_pin_whose_top_digest_is_another_cells_installs_the_download_of_this_cell(
    release: ReleaseServer, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    wheel = tmp_path / "cell.whl"
    cell_sha = make_wheel(wheel)
    release.files["/fx.whl"] = wheel.read_bytes()
    _two_digest_pin(monkeypatch, f"{release.url}/fx.whl", cell_sha, "b" * 64, wheel.stat().st_size)
    engine_service.install(ENGINE.value)
    assert _builds() == [(ENGINE.value, VERSION, "cpu")]


def test_a_wheel_of_the_top_digest_is_refused_when_the_cell_digest_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel = tmp_path / "top.whl"
    top_sha = make_wheel(wheel)
    _two_digest_pin(monkeypatch, "https://example.invalid/x.whl", "b" * 64, top_sha)
    with pytest.raises(engine_service.EngineRefusedError) as exc:
        engine_service.install(ENGINE.value, wheel)
    assert str(exc.value) == BAD_PACK
    assert _builds() == []


def test_a_pack_url_outside_the_index_is_refused_and_no_request_is_sent(
    good: Path, release: ReleaseServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(engine_store, "PACKAGES_URL", "https://index.test/packs/")
    with pytest.raises(engine_service.EngineRefusedError, match="not on the pack index") as exc:
        engine_service.install(ENGINE.value)
    assert "https://index.test/packs/" in str(exc.value)
    assert (release.requests, _builds()) == ([], [])


def _refusing_host(os: HostOS, arch: HostArch, avx2: Feature | None = None) -> HostCaps:
    return HostCaps(
        os=os,
        arch=arch,
        avx2=avx2 or Feature(True),
        neon=Feature(False, "This is not an ARM CPU."),
        gpus=(),
        nvidia_driver_cuda=None,
        nvidia_reason="",
        nvidia_known=True,
    )


@pytest.mark.parametrize(
    ("name", "host", "reason"),
    [
        (
            "ik_llama.cpp",
            _refusing_host(HostOS.LINUX, HostArch.X86_64, Feature(False, "Xeon lacks AVX2.")),
            "Xeon lacks AVX2. ik_llama.cpp needs a CPU with AVX2 or NEON.",
        ),
        (
            "tei",
            _refusing_host(HostOS.WINDOWS, HostArch.X86_64),
            "No build exists for Windows.",
        ),
        (
            "mistral.rs",
            _refusing_host(HostOS.MACOS, HostArch.X86_64),
            "mistral.rs is not offered on macos x86_64.",
        ),
    ],
)
def test_each_refusing_engine_gives_its_reason_and_the_install_sentence(
    name: str, host: HostCaps, reason: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(engine_service, "refresh_session_host", lambda: host)
    with pytest.raises(engine_service.EngineRefusedError) as exc:
        engine_service.install(name)
    assert str(exc.value) == f"Engine {name} cannot run on this host. {reason} {STAYS}"


STALL_TIMEOUT = 0.3


@pytest.fixture
def stalling(good: Path, release: ReleaseServer, monkeypatch: pytest.MonkeyPatch) -> ReleaseServer:
    """A release whose file the engine's cell points at; the client gives up after 0.3 s."""
    release.files["/fx.whl"] = good.read_bytes() * 50
    sha = hashlib.sha256(good.read_bytes()).hexdigest()
    cells = _cell(f"{release.url}/fx.whl", sha, size=len(release.files["/fx.whl"]))
    monkeypatch.setitem(choice.ADAPTERS, ENGINE, StandIn(sha, cells=cells))  # type: ignore[arg-type]
    monkeypatch.setattr(hub_constants, "HF_HUB_DOWNLOAD_TIMEOUT", STALL_TIMEOUT)
    monkeypatch.setattr(hub_constants, "DOWNLOAD_CHUNK_SIZE", 1024)
    return release


def test_a_silent_server_names_each_failed_attempt_with_its_cause(
    stalling: ReleaseServer,
) -> None:
    stalling.silent_next = 10
    retries: list[str] = []
    with (
        mock.patch.object(download.time, "sleep"),
        pytest.raises(engine_service.EngineRefusedError, match="cannot be downloaded"),
    ):
        engine_service.install(ENGINE.value, on_retry=retries.append)
    assert len(stalling.requests) == 3
    assert [t.split(". ")[-1] for t in retries] == [
        "Attempt 1 of 3 failed; trying again.",
        "Attempt 2 of 3 failed; trying again.",
    ]
    assert all(t.startswith("Transfer failed: ") for t in retries)


def test_a_download_that_stops_mid_body_reports_the_bytes_that_arrived(
    stalling: ReleaseServer,
) -> None:
    stalling.stall_next = 10
    updates: list[tuple[float, str]] = []
    with (
        mock.patch.object(download.time, "sleep"),
        pytest.raises(engine_service.EngineRefusedError),
    ):
        engine_service.install(
            ENGINE.value, on_update=lambda p: updates.append((p.percent, p.detail))
        )
    assert 0 < updates[0][0] < 100


def test_a_healthy_download_reports_no_retry_and_completes(
    good: Path, release: ReleaseServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    release.files["/fx.whl"] = good.read_bytes()
    sha = hashlib.sha256(good.read_bytes()).hexdigest()
    cells = _cell(f"{release.url}/fx.whl", sha, size=good.stat().st_size)
    monkeypatch.setitem(choice.ADAPTERS, ENGINE, StandIn(sha, cells=cells))  # type: ignore[arg-type]
    retries: list[str] = []
    built = engine_service.install(ENGINE.value, on_retry=retries.append)
    assert (built.engine, retries) == (ENGINE.value, [])
    assert len(release.requests) == 1


@pytest.mark.parametrize("flags", [[], ["--json"]])
def test_the_cli_tells_the_user_about_each_retry_on_stderr(
    stalling: ReleaseServer, flags: list[str]
) -> None:
    stalling.silent_next = 10
    with mock.patch.object(download.time, "sleep"):
        result = CliRunner().invoke(cli_app, [*flags, "engine", "install", ENGINE.value])
    assert result.exit_code == 1
    assert result.stderr.count("failed; trying again.") == 2
    if flags:
        assert json.loads(result.stdout)["command"] == "engine install"
    assert "failed; trying again." not in result.stdout


def test_the_slash_command_hands_each_retry_to_the_screen(stalling: ReleaseServer) -> None:
    stalling.silent_next = 10
    retries: list[str] = []
    with mock.patch.object(download.time, "sleep"):
        out = run_engine_command(f"install {ENGINE.value}", on_retry=retries.append)
    assert not out.ok
    assert len(retries) == 2


def test_the_cli_shows_download_progress_on_stderr_and_keeps_stdout_clean(
    good: Path,
) -> None:
    def install(
        _name: str, _wheel: Path | None, on_update: Any, on_retry: Any
    ) -> engine_service.InstalledBuild:
        on_update(DownloadProgress(percent=40.0, detail="4/10 MB", is_cache_hit=False))
        return engine_service.InstalledBuild(ENGINE.value, VERSION, "cpu", "")

    with mock.patch.object(engine_service, "install", side_effect=install):
        result = CliRunner().invoke(cli_app, ["--json", "engine", "install", ENGINE.value])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["installed"] == f"{ENGINE.value} {VERSION} (cpu)"
