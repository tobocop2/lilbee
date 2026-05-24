"""Tests for the multi-GPU fleet supervisor (spawn, reap, monitor, restart)."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest

from lilbee.providers import base as base_mod
from lilbee.providers.multi_gpu import fleet as fleet_mod
from lilbee.providers.multi_gpu.fleet import (
    Fleet,
    FleetServer,
    InstanceLaunch,
    pick_free_port,
    reap_orphans,
)
from lilbee.providers.worker.transport import WorkerRole


class FakeProc:
    _next_pid = 5000

    def __init__(self, *, alive: bool = True, wait_raises: bool = False) -> None:
        FakeProc._next_pid += 1
        self.pid = FakeProc._next_pid
        self._alive = alive
        self._wait_raises = wait_raises
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return None if self._alive else 0

    def wait(self, timeout: float | None = None) -> int:
        if self._wait_raises:
            raise subprocess.TimeoutExpired("llama-server", timeout)
        self._alive = False
        return 0

    def terminate(self) -> None:
        self.terminated = True
        self._alive = False

    def kill(self) -> None:
        self.killed = True
        self._alive = False


def _launch(tmp_path: Path, role: WorkerRole = WorkerRole.CHAT) -> InstanceLaunch:
    return InstanceLaunch(
        role=role,
        argv=["/bin/llama-server", "--model", "m.gguf"],
        env_overrides={"CUDA_VISIBLE_DEVICES": "0"},
        model="m.gguf",
        port_file=tmp_path / f"llama-server-{role.value}.port",
    )


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    killed: list[tuple[int, int]] = []
    procs: list[FakeProc] = []

    def _popen(*_a: object, **_k: object) -> FakeProc:
        proc = FakeProc()
        procs.append(proc)
        return proc

    monkeypatch.setattr(fleet_mod.subprocess, "Popen", _popen)
    monkeypatch.setattr(fleet_mod.sys, "platform", "linux")
    # raising=False: os.getpgid/killpg are absent on Windows CI runners.
    monkeypatch.setattr(fleet_mod.os, "getpgid", lambda pid: pid, raising=False)
    monkeypatch.setattr(
        fleet_mod.os, "killpg", lambda pgid, sig: killed.append((pgid, sig)), raising=False
    )
    monkeypatch.setattr(fleet_mod.LlamaServerClient, "health", lambda self: True)
    monkeypatch.setattr(fleet_mod.LlamaServerClient, "close", lambda self: None)
    return {"killed": killed, "procs": procs}


def test_pick_free_port_returns_int() -> None:
    assert isinstance(pick_free_port(), int)


def test_spawn_claims_port_writes_pid_file_and_creates_client(
    tmp_path: Path, patched: dict
) -> None:
    server = FleetServer(_launch(tmp_path))
    server.spawn()
    record = json.loads((tmp_path / "llama-server-chat.port").read_text())
    assert record["port"] > 0
    assert record["pid"] == patched["procs"][-1].pid
    assert server.client is not None
    assert server.is_alive()


def test_spawn_appends_port_to_argv(tmp_path: Path, patched: dict, monkeypatch) -> None:
    seen: dict[str, list] = {}
    monkeypatch.setattr(fleet_mod, "pick_free_port", lambda: 42999)

    def _popen(argv, **_k):
        seen["argv"] = argv
        return FakeProc()

    monkeypatch.setattr(fleet_mod.subprocess, "Popen", _popen)
    FleetServer(_launch(tmp_path)).spawn()
    assert seen["argv"][-2:] == ["--port", "42999"]


def test_wait_ready_true_when_health_ok(tmp_path: Path, patched: dict) -> None:
    server = FleetServer(_launch(tmp_path))
    server.spawn()
    assert server.wait_ready(timeout=1.0) is True


def test_wait_ready_false_on_timeout(
    tmp_path: Path, patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fleet_mod.LlamaServerClient, "health", lambda self: False)
    monkeypatch.setattr(fleet_mod.time, "sleep", lambda _s: None)
    server = FleetServer(_launch(tmp_path))
    server.spawn()
    assert server.wait_ready(timeout=0.05) is False


def test_wait_ready_respawns_on_port_bind_death(
    tmp_path: Path, patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Process dies immediately (e.g. port stolen); wait_ready retries on a fresh
    # port, then a live one becomes ready.
    states = iter([False, False, True])
    monkeypatch.setattr(fleet_mod.subprocess, "Popen", lambda *a, **k: FakeProc(alive=next(states)))
    monkeypatch.setattr(fleet_mod.time, "sleep", lambda _s: None)
    server = FleetServer(_launch(tmp_path))
    server.spawn()
    assert server.wait_ready(timeout=0.05) is True


def test_stop_terminates_group_and_cleans_up(tmp_path: Path, patched: dict) -> None:
    import signal

    server = FleetServer(_launch(tmp_path))
    server.spawn()
    server.stop()
    assert (patched["procs"][-1].pid, signal.SIGTERM) in patched["killed"]
    assert not (tmp_path / "llama-server-chat.port").exists()


def test_stop_escalates_to_sigkill_on_timeout(
    tmp_path: Path, patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fleet_mod.subprocess, "Popen", lambda *a, **k: FakeProc(wait_raises=True))
    server = FleetServer(_launch(tmp_path))
    server.spawn()
    server.stop()
    assert any(sig == fleet_mod._SIGKILL for _pgid, sig in patched["killed"])


def test_stop_on_windows_hard_terminates(
    tmp_path: Path, patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc = FakeProc()
    monkeypatch.setattr(fleet_mod.sys, "platform", "win32")
    monkeypatch.setattr(fleet_mod.subprocess, "Popen", lambda *a, **k: proc)
    server = FleetServer(_launch(tmp_path))
    server.spawn()
    server.stop()
    assert proc.terminated


def test_restart_respawns_dead_server(tmp_path: Path, patched: dict) -> None:
    server = FleetServer(_launch(tmp_path))
    server.spawn()
    patched["procs"][-1]._alive = False  # simulate a crash
    assert server.restart() is True
    assert server.restarts == 1
    assert server.is_alive()


def test_reap_orphans_kills_recorded_pids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    killed: list[int] = []
    monkeypatch.setattr(fleet_mod, "_kill_pid_group", lambda pid: killed.append(pid))
    (tmp_path / "llama-server-chat.port").write_text(json.dumps({"pid": 4242, "port": 9000}))
    (tmp_path / "llama-server-embed.port").write_text("not-json")  # malformed -> just cleaned
    reap_orphans(tmp_path)
    assert killed == [4242]
    assert list(tmp_path.glob("llama-server-*.port")) == []


def test_kill_pid_group_handles_missing_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fleet_mod.sys, "platform", "linux")

    def _missing(_pid: int) -> int:
        raise ProcessLookupError

    monkeypatch.setattr(fleet_mod.os, "getpgid", _missing, raising=False)
    fleet_mod._kill_pid_group(123)  # must not raise


def test_fleet_start_reaps_then_serves_healthy_clients(
    tmp_path: Path, patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    reaped: list[Path] = []
    monkeypatch.setattr(fleet_mod, "reap_orphans", reaped.append)
    fleet = Fleet(data_dir=tmp_path)
    fleet.start([_launch(tmp_path, WorkerRole.CHAT), _launch(tmp_path, WorkerRole.EMBED)])
    try:
        assert reaped == [tmp_path]  # prior orphans reaped before spawning
        assert len(fleet.healthy_clients(WorkerRole.CHAT)) == 1
        assert len(fleet.healthy_clients(WorkerRole.EMBED)) == 1
    finally:
        fleet.shutdown()
    assert fleet.healthy_clients(WorkerRole.CHAT) == []


def test_fleet_start_raises_and_tears_down_when_not_ready(
    tmp_path: Path, patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fleet_mod.LlamaServerClient, "health", lambda self: False)
    monkeypatch.setattr(fleet_mod.time, "sleep", lambda _s: None)
    monkeypatch.setattr(fleet_mod.subprocess, "Popen", lambda *a, **k: FakeProc(alive=True))
    fleet = Fleet(ready_timeout=0.02, data_dir=tmp_path)
    with pytest.raises(base_mod.ProviderError, match="failed to become ready"):
        fleet.start([_launch(tmp_path)])
    assert fleet.healthy_clients(WorkerRole.CHAT) == []


def test_monitor_restarts_a_dead_server(
    tmp_path: Path, patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fleet_mod, "_RESTART_BACKOFF_S", (0.0,))
    fleet = Fleet(data_dir=tmp_path)
    fleet.start([_launch(tmp_path)])
    try:
        server = fleet._servers[0]
        server._proc._alive = False  # crash it
        fleet._restart_dead()  # the monitor loop's step, driven directly
        assert server.restarts == 1
        assert server.is_alive()
    finally:
        fleet.shutdown()


def test_monitor_thread_runs_and_stops(tmp_path: Path, patched: dict, monkeypatch) -> None:
    monkeypatch.setattr(fleet_mod, "_MONITOR_POLL_S", 0.01)
    fleet = Fleet(data_dir=tmp_path)
    fleet.start([_launch(tmp_path)])
    time.sleep(0.05)  # let the monitor tick at least once with all servers alive
    fleet.shutdown()
    assert fleet._monitor is None
