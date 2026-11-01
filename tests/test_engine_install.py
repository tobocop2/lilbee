"""Installing an engine: one service call behind the CLI, REST, MCP and slash command."""

from __future__ import annotations

import hashlib
import json
import re
import socket
import stat
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import lilbee.mcp_server as mcp_server
from lilbee.app import engine_choice, engine_service
from lilbee.catalog import runnable
from lilbee.cli import app as cli_app
from lilbee.cli.tui.engine_commands import run_engine_command
from lilbee.core.config import Config, cfg
from lilbee.core.config.enums import EngineSetting
from lilbee.providers.fleet import host_caps
from lilbee.providers.fleet.engine_store import EngineStore, engines_root
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.choice import installed_engines
from lilbee.providers.fleet.engines.host_gate import PASS, HostGate
from lilbee.providers.fleet.host_caps import collect
from lilbee.server.routes import engines as routes
from tests._litestar_test_client import create_test_client
from tests.conformance.fake_adapter import FakeAdapter
from tests.test_engine_service import GGUF_ENGINE

ENGINE = EngineName.IK_LLAMA
VERSION = "git-0123456789ab"
NO_AVX2 = "This host has no AVX2. ik_llama.cpp needs it."
STAYS = "llama.cpp stays in use."
BAD_PACK = "The wheel for engine ik_llama.cpp git-0123456789ab does not match its pin."
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
        self, sha: str | None, gate: HostGate = PASS, licences: tuple[str, ...] = ("MIT",)
    ):
        self.sha, self.gate, self.licences = sha, gate, licences

    def host_gate(self, host: Any) -> HostGate:
        return self.gate

    def pin_record(self) -> Mapping[str, Any]:
        return {
            "engine": self.name,
            "version": VERSION,
            "backend": "cpu",
            "sha256": self.sha,
            "licenses": list(self.licences),
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
        lambda: collect(system="Linux", machine="x86_64", cpu_info={"flags": ["avx2"]}, nvml=None),
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
        nvml=None,
    )


@pytest.fixture
def good(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Register the stand-in with the sha of a good wheel; return that wheel."""
    wheel = tmp_path / "good.whl"
    monkeypatch.setitem(choice.ADAPTERS, ENGINE, StandIn(make_wheel(wheel)))  # type: ignore[arg-type]
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


def test_the_engine_in_the_package_is_refused() -> None:
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
        {"command": "engine install", "installed": f"{ENGINE} {VERSION} (cpu)", "note": ""},
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
        {"engine": ENGINE.value, "version": VERSION, "backend": "cpu", "note": ""},
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
