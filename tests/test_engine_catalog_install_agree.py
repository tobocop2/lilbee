"""The engine list and the engine install ask one gate, so they agree on an unreadable host."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import lilbee.mcp_server as mcp_server
from lilbee.app import engine_service
from lilbee.app.engine_service import EngineRefusedError
from lilbee.catalog import runnable
from lilbee.cli import app as cli_app
from lilbee.cli.tui.engine_commands import run_engine_command
from lilbee.core.config import cfg
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.choice import ADAPTERS, install_answer
from lilbee.providers.fleet.engines.host_gate import PASS, HostGate
from lilbee.providers.fleet.host_caps import HostCaps, collect
from lilbee.server.routes import engines as routes
from tests._litestar_test_client import create_test_client
from tests._published import publish
from tests.conformance.fake_adapter import FakeAdapter

_UNREADABLE = {"driver": None, "cuda_driver": None, "gpus": None}


class _Reached(Exception):
    """The install passed the host gate and the pin, and went to fetch the pack."""


def _unreadable_host(monkeypatch: pytest.MonkeyPatch) -> HostCaps:
    host = collect(system="", machine="", cpu_info={}, nvidia=_UNREADABLE)
    monkeypatch.setattr(engine_service, "session_host", lambda: host)
    monkeypatch.setattr(runnable, "session_host", lambda: host)
    monkeypatch.setattr(engine_service, "refresh_session_host", lambda: host)
    return host


def _install_verdict(monkeypatch: pytest.MonkeyPatch, name: str, tmp_path: Path) -> str | None:
    """The refusal text of an install of *name*, or None when it reaches the fetch."""

    def reach(*_args: object, **_kwargs: object) -> None:
        raise _Reached

    monkeypatch.setattr(engine_service.EngineStore, "ensure", reach)
    try:
        engine_service.install(name)
    except _Reached:
        return None
    except EngineRefusedError as exc:
        return str(exc)
    raise AssertionError("install neither refused nor reached the fetch")


_NAMES = [str(n) for n, a in ADAPTERS.items() if not a.bundled]


def test_every_engine_to_check_is_found() -> None:
    assert len(_NAMES) >= 3


@pytest.mark.parametrize("name", _NAMES)
def test_catalog_and_install_agree_on_an_unreadable_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str
) -> None:
    _unreadable_host(monkeypatch)
    listed = name in engine_service.not_installed()
    refusal = _install_verdict(monkeypatch, name, tmp_path)
    assert listed == (refusal is None), f"{name}: listed={listed} install={refusal!r}"


@pytest.mark.parametrize("name", _NAMES)
def test_install_names_the_unreadable_host_as_the_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str
) -> None:
    _unreadable_host(monkeypatch)
    refusal = _install_verdict(monkeypatch, name, tmp_path)
    assert refusal == (
        f"Engine {name} cannot be installed: lilbee could not read the operating system "
        "and the processor of this computer. Run lilbee on Windows, macOS or Linux with an "
        "x86-64 or ARM64 processor."
    )


def test_every_install_surface_shows_the_unreadable_host_sentence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _unreadable_host(monkeypatch)
    monkeypatch.setattr(cfg, "allow_http_engines", True)
    monkeypatch.setattr(cfg, "mcp_engines_enabled", True)
    monkeypatch.setattr(cfg, "models_dir", tmp_path / "models")
    monkeypatch.setattr(cfg, "json_mode", False)
    expected = (
        "Engine tei cannot be installed: lilbee could not read the operating system and the "
        "processor of this computer. Run lilbee on Windows, macOS or Linux with an x86-64 or "
        "ARM64 processor."
    )
    assert _install_verdict(monkeypatch, "tei", tmp_path) == expected
    cli = CliRunner().invoke(cli_app, ["--json", "engine", "install", "tei"])
    assert (cli.exit_code, json.loads(cli.output.strip().splitlines()[-1])["error"]) == (
        1,
        expected,
    )
    with create_test_client([routes.engine_install_route], debug=False) as client:
        res = client.post("/api/engines/install", json={"engine": "tei"})
    assert (res.status_code, res.json()["detail"]) == (422, expected)
    monkeypatch.setattr(cfg, "mcp_engines_enabled", True)  # the CLI run reloads the settings
    assert mcp_server.engine_install_tool("tei") == {
        "error": expected,
        "code": "engine_install_refused",
    }
    slash = run_engine_command("install tei")
    assert (slash.ok, slash.text) == (False, expected)


@pytest.mark.parametrize("name", _NAMES)
def test_the_install_gate_of_an_unread_host_is_a_plain_unknown_refusal(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    host = _unreadable_host(monkeypatch)
    gate = install_answer(ADAPTERS[EngineName(name)], host)
    assert (gate.ok, gate.known, gate.reason.startswith(f"Engine {name} cannot be installed")) == (
        False,
        False,
        True,
    )


def test_a_readable_host_still_lists_and_installs_tei(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    host = collect(system="Linux", machine="x86_64", cpu_info={"flags": ["avx2"]}, nvidia=None)
    monkeypatch.setattr(engine_service, "session_host", lambda: host)
    monkeypatch.setattr(engine_service, "refresh_session_host", lambda: host)
    monkeypatch.setattr(runnable, "session_host", lambda: host)
    publish(monkeypatch, EngineName.TEI, "0" * 64)
    assert "tei" in engine_service.not_installed()
    assert _install_verdict(monkeypatch, "tei", tmp_path) is None


@pytest.mark.parametrize(("system", "machine"), [("Linux", ""), ("", "x86_64")])
def test_a_half_unread_host_lists_only_the_ungated_engine_and_install_refuses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, system: str, machine: str
) -> None:
    host = collect(system=system, machine=machine, cpu_info={"flags": ["avx2"]}, nvidia=None)
    monkeypatch.setattr(engine_service, "session_host", lambda: host)
    monkeypatch.setattr(engine_service, "refresh_session_host", lambda: host)
    monkeypatch.setattr(engine_service, "llama_server_ready", lambda: False)
    assert engine_service.not_installed() == ["llama.cpp"]
    refusal = _install_verdict(monkeypatch, "tei", tmp_path)
    unread = "operating system" if system == "" else "processor"
    assert refusal == (
        f"Engine tei cannot be installed: lilbee could not read the {unread} of this "
        "computer. Run lilbee on Windows, macOS or Linux with an x86-64 or ARM64 processor."
    )


def test_a_bundled_engine_with_no_install_stays_listed_on_an_unreadable_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _unreadable_host(monkeypatch)
    monkeypatch.setattr(engine_service, "installed_engines", lambda: {})
    assert engine_service.not_installed() == ["llama.cpp"]


def test_the_list_logs_why_it_hides_engines_on_an_unreadable_host(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _unreadable_host(monkeypatch)
    with caplog.at_level("WARNING", logger=engine_service.log.name):
        engine_service.not_installed()
    messages = [r.getMessage() for r in caplog.records if r.name == engine_service.log.name]
    assert len(messages) == 1
    names, _, cause = messages[0].removeprefix("The engine list hides ").partition(": ")
    assert set(names.split(", ")) == set(_NAMES)
    assert cause == "the OS or CPU of this host could not be read."


def test_the_list_logs_the_hidden_engines_when_only_the_os_is_unreadable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    host = collect(system="", machine="x86_64", cpu_info={"flags": ["avx2"]}, nvidia=_UNREADABLE)
    assert host.os is None and host.arch is not None
    monkeypatch.setattr(engine_service, "session_host", lambda: host)
    monkeypatch.setattr(runnable, "session_host", lambda: host)
    with caplog.at_level("WARNING", logger=engine_service.log.name):
        engine_service.not_installed()
    messages = [r.getMessage() for r in caplog.records if r.name == engine_service.log.name]
    assert len(messages) == 1
    assert messages[0].endswith(": the OS or CPU of this host could not be read.")


def test_a_readable_host_logs_nothing_about_hidden_engines(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    host = collect(system="Linux", machine="x86_64", cpu_info={"flags": ["avx2"]}, nvidia=None)
    monkeypatch.setattr(engine_service, "session_host", lambda: host)
    with caplog.at_level("WARNING", logger=engine_service.log.name):
        engine_service.not_installed()
    assert [r for r in caplog.records if r.name == engine_service.log.name] == []


class _OneWheel(FakeAdapter):
    """A managed engine whose pin has no cells: one wheel serves every host."""

    name = "one-wheel"
    formats = frozenset({"gguf"})

    def host_gate(self, host: HostCaps) -> HostGate:
        return PASS

    def pin_record(self) -> dict[str, object]:
        return {
            "engine": self.name,
            "version": "1.0",
            "backend": "cpu",
            "platforms": {},
            "sha256": "a" * 64,
            "licenses": ["MIT"],
            "tool": "server",
        }


def test_a_pin_with_no_cells_is_listed_and_installable_on_an_unreadable_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _unreadable_host(monkeypatch)
    monkeypatch.setitem(ADAPTERS, EngineName.TEI, _OneWheel())  # type: ignore[arg-type]
    assert "tei" in engine_service.not_installed()
    assert _install_verdict(monkeypatch, "tei", tmp_path) is None
