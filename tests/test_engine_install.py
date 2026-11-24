"""Installing an engine: one service call behind the CLI, REST, MCP and slash command."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import socket
import stat
import threading
import time
import zipfile
from collections.abc import Callable, Iterator, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from huggingface_hub import constants as hub_constants
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel
from rich.text import Text
from typer.testing import CliRunner

import lilbee.mcp_server as mcp_server
from lilbee.app import engine_service
from lilbee.catalog import download, runnable
from lilbee.catalog.models import DownloadProgress
from lilbee.cli import app as cli_app
from lilbee.cli import engine as cli_engine
from lilbee.cli import theme
from lilbee.cli.tui.engine_commands import run_engine_command
from lilbee.cli.tui.screens.chat import ChatScreen
from lilbee.cli.tui.task_queue import TaskStatus
from lilbee.core.config import Config, cfg
from lilbee.core.config.enums import EngineSetting
from lilbee.providers.fleet import engine_manifest, engine_store, host_caps
from lilbee.providers.fleet.build_use import BuildHolds
from lilbee.providers.fleet.engine_store import LOCK_DIR, EngineStore, engines_root
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.choice import installed_engines
from lilbee.providers.fleet.engines.host_gate import PASS, HostGate
from lilbee.providers.fleet.host_caps import Feature, HostArch, HostCaps, HostOS
from lilbee.providers.fleet.router_install import RouterDownloadError
from lilbee.runtime.console import PlainConsole
from lilbee.server.routes import engines as routes
from tests._async_wait import wait_until
from tests._lilbee_app_test_host import await_chat
from tests._litestar_test_client import create_test_client
from tests._release_server import ReleaseServer, release_server
from tests._toasts import posted_toasts
from tests.conformance.fake_adapter import FakeAdapter
from tests.test_engine_service import GGUF_ENGINE
from tests.test_tui_engine_choice import _ChatApp

ENGINE = EngineName.IK_LLAMA
VERSION = "git-0123456789ab"
NO_AVX2 = "This host has no AVX2. ik_llama.cpp needs it."
STAYS = "llama.cpp stays in use."


def bad_copy(path: Path) -> str:
    """The refusal for a file the user copied that does not match its pin."""
    text = (
        f"The file {path} for engine ik_llama.cpp git-0123456789ab (cpu) does not match its "
        f"pin, so lilbee did not install it. Copy {path.name} again from the lilbee release "
        "that matches this version and run the install again."
    )
    if path.parent.name == "packs":
        text += f" Or remove it from {path.parent} to download it."
    return text


BAD_DOWNLOAD = (
    "The file for engine ik_llama.cpp git-0123456789ab (cpu) does not match its pin, "
    "so lilbee did not install it. Run the install again. If this happens again, report it."
)
ROUTES = [routes.engine_install_route]
INSTALL_REFUSED = "engine_install_refused"


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
    monkeypatch.setenv("LILBEE_MCP_ENGINES_ENABLED", "true")
    cfg.data_root = tmp_path
    cfg.models_dir = tmp_path / "models"
    cfg.allow_http_engines = True
    cfg.mcp_engines_enabled = True
    cfg.preferred_engine = EngineSetting.AUTO
    cfg.engine_choices = None
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


def test_a_wheel_that_does_not_match_its_pin_is_refused_with_the_text_to_copy_it_again(
    good: Path, tmp_path: Path, root: Path
) -> None:
    other = tmp_path / "other.whl"
    make_wheel(other, package="lilbee_engine_other")
    with pytest.raises(engine_service.EngineRefusedError) as exc:
        engine_service.install(ENGINE.value, other)
    assert str(exc.value) == bad_copy(other)
    assert _builds() == []
    assert [p.name for p in root.glob("*") if p.name != LOCK_DIR] == []


def test_no_pack_and_no_network_gives_the_missing_pack_text(good: Path, root: Path) -> None:
    with pytest.raises(engine_service.EngineRefusedError) as exc:
        engine_service.install(ENGINE.value)
    assert str(exc.value) == (
        f"Engine {ENGINE} {VERSION} (cpu) is not installed and cannot be downloaded. "
        f"Copy {ENGINE}-{VERSION}-cpu.whl into {root / 'packs'} and run the install again."
    )
    assert _builds() == []


def test_a_damaged_pack_in_the_packs_folder_is_refused_with_the_text_to_copy_it_again(
    good: Path, root: Path
) -> None:
    damaged = good.read_bytes()[:-8]
    (root / "packs").mkdir(parents=True)
    pack = root / "packs" / f"{ENGINE}-{VERSION}-cpu.whl"
    pack.write_bytes(damaged)
    with pytest.raises(engine_service.EngineRefusedError) as exc:
        engine_service.install(ENGINE.value)
    assert str(exc.value) == (
        f"The file {pack} for engine {ENGINE} {VERSION} (cpu) does not match its pin, "
        f"so lilbee did not install it. Copy {pack.name} again from the lilbee release "
        f"that matches this version and run the install again. "
        f"Or remove it from {pack.parent} to download it."
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


def test_a_download_that_does_not_match_its_pin_says_to_run_the_install_again(
    good: Path, release: ReleaseServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    release.files["/fx.whl"] = good.read_bytes()
    sha = "0" * 64
    cells = _cell(f"{release.url}/fx.whl", sha, size=good.stat().st_size)
    monkeypatch.setitem(choice.ADAPTERS, ENGINE, StandIn(sha, cells=cells))  # type: ignore[arg-type]
    with pytest.raises(engine_service.EngineRefusedError) as exc:
        engine_service.install(ENGINE.value)
    assert str(exc.value) == BAD_DOWNLOAD
    assert _builds() == []


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
    assert str(exc.value) == (
        f"Engine {ENGINE} cannot be installed: lilbee could not read the operating system "
        "of this computer. Run lilbee on Windows, macOS or Linux with an x86-64 or ARM64 "
        "processor."
    )


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
            lambda: any("trying again" in str(n.message) for n in posted_toasts()),
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


def test_the_engines_tab_row_of_a_build_of_another_wheel_removes_it_so_install_works(
    good: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine_service.install(ENGINE.value, good)
    other = tmp_path / "other.whl"
    sha = make_wheel(other, package="lilbee_engine_other")
    monkeypatch.setitem(choice.ADAPTERS, ENGINE, StandIn(sha))  # type: ignore[arg-type]
    rows = [r for r in engine_service.engine_rows() if r.engine == ENGINE.value]
    assert [(r.state, r.version, r.backend) for r in rows] == [
        (engine_service.EngineState.NOT_INSTALLED, "", "cpu"),
        (engine_service.EngineState.DIFFERS_FROM_PIN, VERSION, "cpu"),
    ]
    stale = rows[1]
    assert stale.size_bytes is not None and stale.size_bytes > 0
    # The row names the build that `lilbee engine list` names on its line.
    listing = " ".join(CliRunner().invoke(cli_app, ["engine", "list"]).output.split())
    assert (
        f"{ENGINE} not installed: the installed build {stale.version} ({stale.backend}) "
        "differs from the pin"
    ) in listing
    engine_service.remove_build(stale.engine, stale.version, stale.backend)
    built = engine_service.install(ENGINE.value, other)
    assert (built.engine, built.version, built.backend) == (ENGINE.value, VERSION, "cpu")
    rows = [r for r in engine_service.engine_rows() if r.engine == ENGINE.value]
    assert [r.state for r in rows] == [engine_service.EngineState.INSTALLED]


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


class _PairState(threading.local):
    """Whether the current thread is inside a socket pair."""

    active = False


def _spy_connects(monkeypatch: pytest.MonkeyPatch, allow: tuple[str, ...] = ()) -> list[Any]:
    """Record every socket connect to a host the test does not allow by name.

    An allowed host connects for real. Any other connect, loopback included, is
    recorded and not made. The connect inside a socket pair is left out: on Windows
    the event loop joins its own pair over loopback.
    """
    seen: list[Any] = []
    real_connect = socket.socket.connect
    real_pair = socket.socketpair
    pairing = _PairState()

    def connect_spy(sock: socket.socket, address: Any, *args: object) -> None:
        if pairing.active or (isinstance(address, tuple) and address[0] in allow):
            real_connect(sock, address, *args)
        else:
            seen.append(address)

    def pair_spy() -> tuple[socket.socket, socket.socket]:
        pairing.active = True
        try:
            return real_pair()
        finally:
            pairing.active = False

    monkeypatch.setattr(socket.socket, "connect", connect_spy)
    monkeypatch.setattr(socket, "socketpair", pair_spy)
    return seen


def test_the_cli_installs_and_lists_the_engine(good: Path) -> None:
    code, body = _cli("install", ENGINE.value, "--wheel", str(good))
    assert (code, body) == (
        0,
        {
            "command": "engine install",
            "installed": f"{ENGINE} {VERSION} (cpu)",
            "note": "",
            "not_run_on": [],
            "engines_folder": str(cfg.models_dir.parent / "engines"),
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


def test_with_lilbee_data_set_the_cli_names_the_shared_engines_folder(
    good: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    for variable in ("HOME", "XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(home))
    monkeypatch.setenv("LILBEE_DATA", str(tmp_path / "library"))
    resolved = Config()
    cfg.models_dir = resolved.models_dir
    folder = resolved.models_dir.parent / "engines"
    assert not folder.is_relative_to(resolved.data_root)

    installed = CliRunner().invoke(
        cli_app, ["engine", "install", ENGINE.value, "--wheel", str(good)]
    )
    listed = CliRunner().invoke(cli_app, ["engine", "list"])

    assert (folder / ENGINE.value).is_dir()
    for result in (installed, listed):
        assert result.exit_code == 0
        assert (
            f"Engines folder: {folder} (shared by every data directory; LILBEE_MODELS_DIR moves it)"
            in result.output
        )


def test_the_cli_refuses_with_the_service_text(good: Path, tmp_path: Path) -> None:
    other = tmp_path / "other.whl"
    make_wheel(other, package="lilbee_engine_other")
    assert _cli("install", ENGINE.value, "--wheel", str(other)) == (
        1,
        {"command": "engine install", "error": bad_copy(other), "code": INSTALL_REFUSED},
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
            "engines_folder": str(cfg.models_dir.parent / "engines"),
        },
    )
    assert _builds() == [(ENGINE.value, VERSION, "cpu")]


def test_rest_refuses_with_a_422_and_the_service_text(good: Path) -> None:
    code, body = _rest({"engine": ENGINE.value})
    assert code == 422
    assert "cannot be downloaded" in body["detail"]
    assert body["extra"] == {"code": INSTALL_REFUSED}


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
    assert refused["code"] == INSTALL_REFUSED
    (root / "packs").mkdir(parents=True)
    (root / "packs" / f"{ENGINE}-{VERSION}-cpu.whl").write_bytes(good.read_bytes())
    assert mcp_server.engine_install_tool(ENGINE.value) == {
        "engine": ENGINE.value,
        "version": VERSION,
        "backend": "cpu",
        "note": "",
        "not_run_on": [],
        "engines_folder": str(cfg.models_dir.parent / "engines"),
    }
    assert _builds() == [(ENGINE.value, VERSION, "cpu")]


def test_every_install_surface_names_the_folder_the_list_names(good: Path, root: Path) -> None:
    (root / "packs").mkdir(parents=True)
    (root / "packs" / f"{ENGINE}-{VERSION}-cpu.whl").write_bytes(good.read_bytes())
    expected = str(cfg.models_dir.parent / "engines")
    folders = {
        "cli": _cli("install", ENGINE.value, "--wheel", str(good))[1]["engines_folder"],
        "rest": _rest({"engine": ENGINE.value})[1]["engines_folder"],
        "mcp": mcp_server.engine_install_tool(ENGINE.value)["engines_folder"],
        "list": mcp_server.engine_list_tool()["engines_folder"],
    }
    assert folders == dict.fromkeys(("cli", "rest", "mcp", "list"), expected)


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
    folder = engine_service.FOLDER_LINE.format(folder=engine_service.engines_folder())
    assert out.text == f"Installed engine {ENGINE} {VERSION} (cpu).\n{folder}"
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
    connects = _spy_connects(monkeypatch)
    pack = tmp_path / "pack.whl"
    pack.write_bytes(b"not read")
    with pytest.raises(engine_service.EngineRefusedError) as exc:
        engine_service.install(ENGINE.value, pack)
    assert str(exc.value) == NO_PACK
    assert _cli("install", ENGINE.value, "--wheel", str(pack))[1] == {
        "command": "engine install",
        "error": NO_PACK,
        "code": INSTALL_REFUSED,
    }
    assert _cli("install", ENGINE.value)[1]["code"] == INSTALL_REFUSED
    code, body = _rest({"engine": ENGINE.value})
    assert (code, body["detail"], body["extra"]) == (422, NO_PACK, {"code": INSTALL_REFUSED})
    assert mcp_server.engine_install_tool(ENGINE.value) == {
        "error": NO_PACK,
        "code": INSTALL_REFUSED,
    }
    assert run_engine_command(f"install {ENGINE.value}").text == NO_PACK
    assert (reads, connects, root.exists()) == ([], [], False)


def test_the_connect_spy_records_a_loopback_connect_unless_the_test_allows_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        address = server.getsockname()
        with monkeypatch.context() as refusing:
            refused = _spy_connects(refusing)
            with socket.socket() as client:
                client.connect(address)
        assert refused == [address]
        with monkeypatch.context() as allowing:
            allowed = _spy_connects(allowing, allow=("127.0.0.1",))
            with socket.socket() as client:
                client.connect(address)
                assert client.getpeername() == address
        assert allowed == []


def test_the_connect_spy_ignores_the_connect_a_socket_pair_makes_inside_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def windows_style_pair() -> tuple[socket.socket, socket.socket]:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            listener.settimeout(5)
            client = socket.socket()
            client.connect(listener.getsockname())
            server, _ = listener.accept()
            return client, server

    monkeypatch.setattr(socket, "socketpair", windows_style_pair)
    connects = _spy_connects(monkeypatch)
    left, right = socket.socketpair()
    left.close()
    right.close()
    assert connects == []


def test_the_connect_spy_records_a_connect_from_another_thread_during_a_socket_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inside_pair, leave_pair = threading.Event(), threading.Event()

    def held_pair() -> tuple[socket.socket, socket.socket]:
        inside_pair.set()
        assert leave_pair.wait(timeout=5)
        return socket.socket(), socket.socket()

    monkeypatch.setattr(socket, "socketpair", held_pair)
    connects = _spy_connects(monkeypatch)
    pairs: list[tuple[socket.socket, socket.socket]] = []
    maker = threading.Thread(target=lambda: pairs.append(socket.socketpair()))
    maker.start()
    try:
        assert inside_pair.wait(timeout=5)
        with socket.socket() as server, socket.socket() as client:
            server.bind(("127.0.0.1", 0))
            server.listen()
            address = server.getsockname()
            client.connect(address)
        assert connects == [address]
    finally:
        leave_pair.set()
        maker.join(timeout=5)
    for sock in pairs[0]:
        sock.close()


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
    refused = name == ENGINE.value
    coded = {"code": INSTALL_REFUSED} if refused else {}
    assert _cli("install", name)[1] == {"command": "engine install", "error": expected, **coded}
    code, body = _rest({"engine": name})
    assert (code, body["detail"], body.get("extra")) == (422, expected, coded or None)
    assert mcp_server.engine_install_tool(name) == {"error": expected, **coded}
    assert run_engine_command(f"install {name}").text == expected


def test_the_advice_names_the_command_only_for_an_engine_with_a_published_pack() -> None:
    linux = _refusing_host(HostOS.LINUX, HostArch.X86_64)
    assert choice.install_advice(StandIn("a" * 64), linux) == f"Run lilbee engine install {ENGINE}."
    assert choice.install_advice(StandIn(None), linux) == (
        f"No pack of {ENGINE} is published yet, so there is nothing to install."
    )


def _any_advice(monkeypatch: pytest.MonkeyPatch, adapters: dict[EngineName, StandIn]) -> str:
    monkeypatch.setattr(choice, "ADAPTERS", adapters)
    return choice.install_any_advice(_refusing_host(HostOS.LINUX, HostArch.X86_64))


def test_the_advice_for_no_engine_names_the_command_of_each_engine_that_serves_the_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Second(StandIn):
        name = EngineName.MLX.value

    advice = _any_advice(
        monkeypatch,
        {ENGINE: StandIn("a" * 64), EngineName.MLX: Second("b" * 64)},
    )
    assert advice == f"Run lilbee engine install {ENGINE} or lilbee engine install mlx."


def test_the_advice_for_no_engine_skips_an_engine_the_host_refuses_or_with_no_pack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoPack(StandIn):
        name = EngineName.MLX.value

    advice = _any_advice(
        monkeypatch,
        {ENGINE: StandIn("a" * 64, HostGate(False, NO_AVX2)), EngineName.MLX: NoPack(None)},
    )
    assert advice == choice.LIST_HINT


def test_the_advice_names_an_unusable_pin_record() -> None:
    class NoBackend(StandIn):
        def pin_record(self) -> Mapping[str, Any]:
            return {**super().pin_record(), "backend": None}

    linux = _refusing_host(HostOS.LINUX, HostArch.X86_64)
    assert choice.install_advice(NoBackend("a" * 64), linux).startswith(
        f"Engine {ENGINE} cannot be installed: the pin record is not usable"
    )


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
        {"command": "engine install", "error": expected, "code": INSTALL_REFUSED},
    )
    root_dir = engines_root(cfg.models_dir)
    (root_dir / "packs").mkdir(parents=True)
    (root_dir / "packs" / f"{ENGINE}-{VERSION}-cpu.whl").write_bytes(wheel.read_bytes())
    code, body = _rest({"engine": ENGINE.value})
    assert (code, body["detail"]) == (422, expected)
    assert mcp_server.engine_install_tool(ENGINE.value) == {
        "error": expected,
        "code": INSTALL_REFUSED,
    }
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
    assert _cli("install", "auto")[1] == {"command": "engine install", "error": text}
    assert "extra" not in _rest({"engine": "auto"})[1]
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


@pytest.mark.parametrize("as_null", [False, True], ids=["path", "null"])
def test_rest_refuses_a_wheel_field_and_installs_nothing(
    good: Path, root: Path, as_null: bool
) -> None:
    (root / "packs").mkdir(parents=True)
    (root / "packs" / f"{ENGINE}-{VERSION}-cpu.whl").write_bytes(good.read_bytes())
    code, body = _rest({"engine": ENGINE.value, "wheel": None if as_null else str(good)})
    assert code == 400
    assert WHEEL_CLI_ONLY in json.dumps(body["extra"])
    assert _builds() == []


def test_rest_still_ignores_an_unknown_field_other_than_wheel(good: Path, root: Path) -> None:
    (root / "packs").mkdir(parents=True)
    (root / "packs" / f"{ENGINE}-{VERSION}-cpu.whl").write_bytes(good.read_bytes())
    code, body = _rest({"engine": ENGINE.value, "client_tag": "plugin"})
    assert (code, body["engine"]) == (200, ENGINE.value)
    assert _builds() == [(ENGINE.value, VERSION, "cpu")]


def test_the_rest_install_schema_lists_no_wheel_field() -> None:
    with create_test_client(ROUTES, debug=False) as client:
        schema = client.get("/schema/openapi.json").json()
    body = schema["components"]["schemas"]["EngineInstallBody"]
    assert list(body["properties"]) == ["engine"]


@pytest.mark.parametrize("as_null", [False, True], ids=["path", "null"])
def test_mcp_refuses_a_wheel_argument_and_installs_nothing(
    good: Path, root: Path, as_null: bool
) -> None:
    (root / "packs").mkdir(parents=True)
    (root / "packs" / f"{ENGINE}-{VERSION}-cpu.whl").write_bytes(good.read_bytes())
    server = mcp_server.build_mcp_server()
    arguments = {"engine": ENGINE.value, "wheel": None if as_null else str(good)}
    with pytest.raises(ToolError, match=re.escape(WHEEL_CLI_ONLY)):
        asyncio.run(server.call_tool("engine_install", arguments))
    assert _builds() == []


def test_mcp_still_ignores_an_unknown_argument_other_than_wheel(good: Path, root: Path) -> None:
    (root / "packs").mkdir(parents=True)
    (root / "packs" / f"{ENGINE}-{VERSION}-cpu.whl").write_bytes(good.read_bytes())
    server = mcp_server.build_mcp_server()
    asyncio.run(server.call_tool("engine_install", {"engine": ENGINE.value, "client_tag": "x"}))
    assert _builds() == [(ENGINE.value, VERSION, "cpu")]


def test_the_mcp_install_schema_lists_no_wheel_argument() -> None:
    tools = asyncio.run(mcp_server.build_mcp_server().list_tools())
    schema = next(t for t in tools if t.name == "engine_install").input_schema
    assert list(schema["properties"]) == ["engine"]


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
    assert str(exc.value) == bad_copy(wheel)
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
# One attempt is one request: a timeout ends the attempt and the retry loop reports it.
ATTEMPTS = 3


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
    stalling.silent_next = ATTEMPTS
    retries: list[str] = []
    with (
        mock.patch.object(download.time, "sleep"),
        pytest.raises(engine_service.EngineRefusedError, match="cannot be downloaded"),
    ):
        engine_service.install(ENGINE.value, on_retry=retries.append)
    assert len(stalling.requests) == ATTEMPTS
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
    stalling.silent_next = ATTEMPTS
    with mock.patch.object(download.time, "sleep"):
        result = CliRunner().invoke(cli_app, [*flags, "engine", "install", ENGINE.value])
    assert result.exit_code == 1
    assert result.stderr.count("failed; trying again.") == 2
    if flags:
        assert json.loads(result.stdout)["command"] == "engine install"
    assert "failed; trying again." not in result.stdout


def test_the_slash_command_hands_each_retry_to_the_screen(stalling: ReleaseServer) -> None:
    stalling.silent_next = ATTEMPTS
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


def test_an_engine_install_also_installs_the_router_that_starts_it(
    good: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    heard: list[list[str]] = []

    def router(**signals: object) -> Path:
        heard.append(sorted(signals))
        return Path("llama-swap")

    monkeypatch.setattr(engine_service, "install_router", router)
    engine_service.install(ENGINE.value, good)
    assert _builds() == [(ENGINE.value, VERSION, "cpu")]
    assert heard == [["cancel", "on_progress", "on_retry"]]


def test_a_router_that_cannot_install_names_the_engine_that_did(
    good: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(**_signals: object) -> Path:
        raise RouterDownloadError("The router cannot be downloaded.")

    monkeypatch.setattr(engine_service, "install_router", refuse)
    with pytest.raises(engine_service.EngineRefusedError) as exc:
        engine_service.install(ENGINE.value, good)
    assert str(exc.value) == (
        f"Engine {ENGINE} is installed, but the router that starts it is not. "
        "The router cannot be downloaded."
    )
    assert _builds() == [(ENGINE.value, VERSION, "cpu")]


def test_a_refused_engine_install_never_reaches_the_router(
    good: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = tmp_path / "other.whl"
    make_wheel(other, package="lilbee_engine_other")

    def fail(**_signals: object) -> Path:
        raise AssertionError("no engine was installed, so no router is installed")

    monkeypatch.setattr(engine_service, "install_router", fail)
    with pytest.raises(engine_service.EngineRefusedError, match="does not match its pin"):
        engine_service.install(ENGINE.value, other)


def test_the_router_install_hears_the_progress_the_retries_and_the_cancel(
    good: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    heard: dict[str, Any] = {}

    def router(**signals: Any) -> Path:
        heard.update(signals)
        return tmp_path / "llama-swap"

    monkeypatch.setattr(engine_service, "install_router", router)
    cancel = threading.Event()
    retries: list[str] = []
    updates: list[DownloadProgress] = []
    engine_service.install(
        ENGINE.value, good, on_update=updates.append, on_retry=retries.append, cancel=cancel
    )
    assert heard["cancel"] is cancel
    heard["on_progress"](50, 100)
    assert [update.percent for update in updates] == [50.0]
    heard["on_retry"](2, 3, "the link dropped")
    assert len(retries) == 1
    assert "the link dropped" in retries[0]


class _Report(BaseModel):
    folder: Path
    stamp: datetime
    names: set[str]
    blob: bytes


def test_the_mcp_result_is_plain_json_text() -> None:
    report = _Report(
        folder=Path("/engines"), stamp=datetime(2026, 1, 2, 3, 4, 5), names={"a"}, blob=b"ab"
    )
    assert mcp_server._engine_result(lambda: report) == {
        "folder": str(Path("/engines")),
        "stamp": "2026-01-02T03:04:05",
        "names": ["a"],
        "blob": "ab",
    }


def _raises(error: Exception) -> Callable[..., Any]:
    def install(*_args: object, **_kwargs: object) -> None:
        raise error

    return install


def test_the_mcp_tool_says_the_choice_is_saved_when_the_server_did_not_apply_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = engine_service.EngineNotAppliedError("the engine did not start")
    monkeypatch.setattr(engine_service, "install", _raises(error))
    assert mcp_server.engine_install_tool(ENGINE.value) == {
        "error": "the engine did not start",
        "code": engine_service.NOT_APPLIED_CODE,
        "saved": True,
    }


def test_the_mcp_tool_asks_for_the_confirmation_of_an_embed_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = engine_service.EmbedConfirmRequiredError("docs", 3)
    monkeypatch.setattr(engine_service, "install", _raises(error))
    assert mcp_server.engine_install_tool(ENGINE.value) == {
        "error": f"{error} Pass confirm_embed_change=true to confirm.",
        "code": engine_service.CONFIRM_CODE,
        "library": "docs",
        "documents": 3,
    }


def test_rest_answers_a_failed_save_with_a_503_and_the_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = engine_service.EngineNotSavedError("the settings file is read-only")
    monkeypatch.setattr(engine_service, "install", _raises(error))
    code, body = _rest({"engine": ENGINE.value})
    assert code == 503
    assert body["detail"] == "the settings file is read-only"


def test_the_cli_exits_1_and_prints_a_refusal_whole_to_stderr_in_the_error_style(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    printed: list[Text] = []

    class Recorder(PlainConsole):
        def print(self, *objects: Any, **kwargs: Any) -> None:
            printed.extend(objects)
            super().print(*objects, **kwargs)

    monkeypatch.setattr(cli_engine, "PlainConsole", Recorder)
    monkeypatch.setattr(cfg, "json_mode", False)
    missing = tmp_path / ("a-long-name-" * 12 + "missing.whl")
    message = f"{missing} is not a file. Name a wheel file."
    result = CliRunner().invoke(
        cli_app, ["engine", "install", ENGINE.value, "--wheel", str(missing)]
    )
    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == message + "\n"
    assert [(text.plain, text.style) for text in printed] == [(message, theme.ERROR)]


def test_rest_asks_for_the_confirmation_of_an_embed_change_with_a_409_and_the_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = engine_service.EmbedConfirmRequiredError("docs", 3)
    monkeypatch.setattr(engine_service, "install", _raises(error))
    code, body = _rest({"engine": ENGINE.value})
    assert code == 409
    assert body["detail"] == f"{error} Send confirm_embed_change: true to confirm."
    assert body["extra"] == {
        "code": engine_service.CONFIRM_CODE,
        "library": "docs",
        "documents": 3,
    }


HELD_REFUSAL = (
    f"Engine {ENGINE.value} {VERSION} (cpu) is in use. "
    "Stop the engine with 'lilbee engine stop' and install it again."
)
SURVIVOR = "survivor.txt"
SURVIVOR_TEXT = "a running launch reads this"


def _damaged_build(good: Path, root: Path) -> Path:
    """Install the build, then drop its record so the store reads it as damaged."""
    (root / "packs").mkdir(parents=True)
    (root / "packs" / f"{ENGINE}-{VERSION}-cpu.whl").write_bytes(good.read_bytes())
    engine_service.install(ENGINE.value)
    build = root / ENGINE.value / f"{VERSION}-cpu"
    (build / engine_store.RECORD_NAME).unlink()
    (build / SURVIVOR).write_text(SURVIVOR_TEXT, encoding="utf-8")
    assert _builds() == []
    return build


def _service_install() -> str:
    with pytest.raises(engine_service.EngineInstallRefusedError) as exc:
        engine_service.install(ENGINE.value)
    return str(exc.value)


def _cli_install() -> str:
    code, body = _cli("install", ENGINE.value)
    assert (code, body["code"]) == (1, INSTALL_REFUSED)
    return str(body["error"])


def _rest_install() -> str:
    code, body = _rest({"engine": ENGINE.value})
    assert (code, body["extra"]) == (422, {"code": INSTALL_REFUSED})
    return str(body["detail"])


def _mcp_install() -> str:
    refused = mcp_server.engine_install_tool(ENGINE.value)
    assert refused["code"] == INSTALL_REFUSED
    return str(refused["error"])


def _slash_install() -> str:
    result = run_engine_command(f"install {ENGINE.value}")
    assert result.ok is False
    return result.text


HELD_INSTALLS = {
    "service": _service_install,
    "cli": _cli_install,
    "rest": _rest_install,
    "mcp": _mcp_install,
    "slash": _slash_install,
}


@pytest.mark.parametrize("surface", sorted(HELD_INSTALLS))
def test_install_refuses_a_damaged_build_that_a_launch_holds(
    surface: str, good: Path, root: Path
) -> None:
    build = _damaged_build(good, root)
    hold = BuildHolds([build])
    try:
        text = HELD_INSTALLS[surface]()
    finally:
        hold.close()
    assert text == HELD_REFUSAL
    assert (build / SURVIVOR).read_text(encoding="utf-8") == SURVIVOR_TEXT
    assert _builds() == []


def test_install_and_remove_refuse_a_held_build_in_the_same_words(
    good: Path, root: Path
) -> None:
    build = _damaged_build(good, root)
    hold = BuildHolds([build])
    try:
        with pytest.raises(engine_service.EngineRefusedError) as removed:
            engine_service.remove_build(ENGINE.value, VERSION, "cpu")
        installed = _service_install()
    finally:
        hold.close()
    assert str(removed.value).replace("remove it", "install it") == installed


def test_install_replaces_a_damaged_build_that_no_launch_holds(
    good: Path, root: Path
) -> None:
    build = _damaged_build(good, root)
    engine_service.install(ENGINE.value)
    assert not (build / SURVIVOR).exists()
    assert _builds() == [(ENGINE.value, VERSION, "cpu")]


def test_install_succeeds_once_the_launch_releases_the_build(
    good: Path, root: Path
) -> None:
    build = _damaged_build(good, root)
    hold = BuildHolds([build])
    assert _service_install() == HELD_REFUSAL
    hold.close()
    engine_service.install(ENGINE.value)
    assert not (build / SURVIVOR).exists()
    assert _builds() == [(ENGINE.value, VERSION, "cpu")]
