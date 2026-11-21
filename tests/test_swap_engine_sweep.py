"""The orphan sweep finds the server of every shipped engine from what its adapter declares."""

from __future__ import annotations

import psutil
import pytest

from lilbee.providers.fleet import swap_manager as sm
from lilbee.providers.fleet.engines.base import EngineName, ServerProcess
from lilbee.providers.fleet.engines.choice import ADAPTERS
from tests._fake_process import (
    FakeProcess,
    Hop,
    UnreadableCmdline,
    UnreadableName,
    process_table,
)

_PORT = 5800
_INTERPRETER = "python3.12"
_LLAMA = EngineName.LLAMA
_SCRIPT_ENGINE = EngineName.MLX


def _server(name: str, port: int, ancestors: list[Hop] | None = None) -> FakeProcess:
    return FakeProcess(name, [f"/opt/engine/{name}", "--port", str(port)], ancestors)


def _table(monkeypatch: pytest.MonkeyPatch, procs: list[FakeProcess]) -> None:
    monkeypatch.setattr(sm.psutil, "process_iter", process_table(procs))


def _declared(name: EngineName) -> str:
    return ADAPTERS[name].server_process.name


def _script_server(ancestors: list[Hop] | None = None, port: int = _PORT) -> FakeProcess:
    """A server of the engine that runs as a script: the interpreter's name, the script in argv."""
    cmdline = [_INTERPRETER, f"/venv/bin/{_declared(_SCRIPT_ENGINE)}", "--port", str(port)]
    return FakeProcess(_INTERPRETER, cmdline, ancestors)


def test_the_census_covers_every_engine_the_fleet_can_start() -> None:
    assert set(ADAPTERS) == set(EngineName)


@pytest.mark.parametrize("engine", list(EngineName))
def test_an_orphan_of_each_engine_is_found_by_its_declared_name_and_port(
    engine: EngineName, monkeypatch: pytest.MonkeyPatch
) -> None:
    orphan = _script_server() if engine == _SCRIPT_ENGINE else _server(_declared(engine), _PORT)
    _table(monkeypatch, [orphan])
    assert sm._find_orphan_servers((_PORT,)) == [orphan]


@pytest.mark.parametrize("engine", list(EngineName))
def test_a_server_of_each_engine_on_another_port_is_spared(
    engine: EngineName, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = (
        _script_server(port=_PORT + 1)
        if engine == _SCRIPT_ENGINE
        else _server(_declared(engine), _PORT + 1)
    )
    _table(monkeypatch, [other])
    assert sm._find_orphan_servers((_PORT,)) == []


@pytest.mark.parametrize("engine", list(EngineName))
def test_a_server_of_each_engine_below_a_live_swap_is_spared(
    engine: EngineName, monkeypatch: pytest.MonkeyPatch
) -> None:
    ancestors: list[Hop] = ["llama-swap"]
    live = (
        _script_server(ancestors)
        if engine == _SCRIPT_ENGINE
        else _server(_declared(engine), _PORT, ancestors)
    )
    _table(monkeypatch, [live])
    assert sm._find_orphan_servers((_PORT,)) == []


@pytest.mark.parametrize("engine", list(EngineName))
@pytest.mark.parametrize("pattern", ["my-{}", "{}-wrapper", "x{}"])
def test_a_process_whose_name_only_contains_a_declared_name_is_spared(
    engine: EngineName, pattern: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _table(monkeypatch, [_server(pattern.format(_declared(engine)), _PORT)])
    assert sm._find_orphan_servers((_PORT,)) == []


def test_a_windows_executable_name_is_the_declared_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orphan = _server(f"{_declared(_LLAMA)}.exe", _PORT)
    _table(monkeypatch, [orphan])
    assert sm._find_orphan_servers((_PORT,)) == [orphan]


def test_a_process_of_no_declared_name_on_a_member_port_is_spared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    declared = {adapter.server_process.name for adapter in ADAPTERS.values()}
    assert "python3" not in declared
    _table(monkeypatch, [_server("python3", _PORT)])
    assert sm._find_orphan_servers((_PORT,)) == []


@pytest.mark.parametrize(
    ("cmdline", "found"),
    [
        (["python3.12", "/venv/bin/{name}", "--port", "5800"], True),
        (["python3.12", "-u", "/venv/bin/{name}", "--port", "5800"], True),
        (["python3.12", "{name}", "--port", "5800"], True),
        (["python3.12", "serve.py", "--port", "5800"], False),
        (["python3.12", "serve.py", "{name}", "--port", "5800"], False),
        (["python3.12", "-m", "http.server", "--port", "5800"], False),
        (["python3.12", "/venv/bin/{name}", "--port", "5801"], False),
        (["python3.12"], False),
    ],
)
def test_a_script_server_is_found_by_the_script_after_the_interpreter(
    cmdline: list[str], found: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = _declared(_SCRIPT_ENGINE)
    proc = FakeProcess(_INTERPRETER, [arg.format(name=name) for arg in cmdline])
    _table(monkeypatch, [proc])
    assert sm._find_orphan_servers((_PORT,)) == ([proc] if found else [])


def test_the_process_that_the_mlx_chat_launcher_leaves_is_found_by_its_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The launcher hands over to the interpreter, so the table shows its module, not itself."""
    cmdline = ["/e/python/bin/python3", "-I", "-m", "mlx_lm.server", "--port", "5800"]
    proc = FakeProcess(_INTERPRETER, cmdline)
    _table(monkeypatch, [proc])
    assert sm._find_orphan_servers((_PORT,)) == [proc]


def test_a_script_named_by_no_engine_is_spared_on_a_member_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc = FakeProcess(_INTERPRETER, [_INTERPRETER, "-I", "/e/other_server.py", "--port", "5800"])
    _table(monkeypatch, [proc])
    assert sm._find_orphan_servers((_PORT,)) == []


def test_an_interpreter_that_runs_a_native_server_name_is_not_a_script_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cmdline = [_INTERPRETER, f"/venv/bin/{_declared(_LLAMA)}", "--port", str(_PORT)]
    _table(monkeypatch, [FakeProcess(_INTERPRETER, cmdline)])
    assert sm._find_orphan_servers((_PORT,)) == []


def test_a_script_server_below_a_live_swap_is_spared(monkeypatch: pytest.MonkeyPatch) -> None:
    _table(monkeypatch, [_script_server(["llama-swap"])])
    assert sm._find_orphan_servers((_PORT,)) == []


def test_a_live_swap_is_found_although_an_ancestor_above_it_vanishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc = _server(_declared(_LLAMA), _PORT, ["llama-swap", psutil.NoSuchProcess(1)])
    _table(monkeypatch, [proc])
    assert sm._find_orphan_servers((_PORT,)) == []


def test_a_live_swap_is_found_although_an_ancestor_above_it_is_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc = _server(_declared(_LLAMA), _PORT, ["llama-swap", psutil.AccessDenied(1)])
    _table(monkeypatch, [proc])
    assert sm._find_orphan_servers((_PORT,)) == []


def test_a_live_swap_higher_in_the_chain_spares_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    _table(monkeypatch, [_server(_declared(_LLAMA), _PORT, ["bash", "llama-swap"])])
    assert sm._find_orphan_servers((_PORT,)) == []


@pytest.mark.parametrize("hop", [psutil.NoSuchProcess(1), psutil.AccessDenied(1)])
def test_a_hop_that_cannot_be_read_before_any_swap_leaves_the_server_an_orphan(
    hop: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    orphan = _server(_declared(_LLAMA), _PORT, ["bash", hop, "llama-swap"])
    _table(monkeypatch, [orphan])
    assert sm._find_orphan_servers((_PORT,)) == [orphan]


def test_a_process_that_cannot_be_named_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    name = _declared(_LLAMA)
    _table(monkeypatch, [UnreadableName(name, [name, "--port", str(_PORT)])])
    assert sm._find_orphan_servers((_PORT,)) == []


def test_a_process_that_cannot_be_read_does_not_hide_the_ones_after_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = _declared(_LLAMA)
    orphan = _server(name, _PORT)
    _table(monkeypatch, [UnreadableCmdline(name, []), orphan])
    assert sm._find_orphan_servers((_PORT,)) == [orphan]


def test_an_interpreter_that_runs_another_script_does_not_hide_the_ones_after_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    other = FakeProcess(_INTERPRETER, [_INTERPRETER, "serve.py", "--port", str(_PORT)])
    orphan = _script_server()
    _table(monkeypatch, [other, orphan])
    assert sm._find_orphan_servers((_PORT,)) == [orphan]


def test_engines_that_share_a_server_name_scan_it_once() -> None:
    servers = sm._server_processes()
    assert len(servers) == len(set(servers))
    assert len(servers) < len(ADAPTERS)
    assert all(isinstance(server, ServerProcess) for server in servers)


def test_every_adapter_declares_a_server_process() -> None:
    adapters = list(ADAPTERS.values())
    assert adapters
    assert all(adapter.server_process.name for adapter in adapters)
