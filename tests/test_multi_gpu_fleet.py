"""Tests for the multi-GPU fleet supervisor (spawn, reap, monitor, restart)."""

from __future__ import annotations

import json
import os
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
    assert record["parent_pid"] == os.getpid()
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


def test_stop_on_windows_escalates_to_kill(
    tmp_path: Path, patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc = FakeProc(wait_raises=True)  # terminate doesn't take; wait times out
    monkeypatch.setattr(fleet_mod.sys, "platform", "win32")
    monkeypatch.setattr(fleet_mod.subprocess, "Popen", lambda *a, **k: proc)
    server = FleetServer(_launch(tmp_path))
    server.spawn()
    server.stop()
    assert proc.terminated and proc.killed


def test_restart_respawns_dead_server(tmp_path: Path, patched: dict) -> None:
    server = FleetServer(_launch(tmp_path))
    server.spawn()
    patched["procs"][-1]._alive = False  # simulate a crash
    assert server.restart() is True
    assert server.restarts == 1
    assert server.is_alive()


def test_reap_orphans_kills_only_dead_parents_servers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    killed: list[int] = []
    monkeypatch.setattr(fleet_mod, "_kill_pid_group", lambda pid: killed.append(pid))
    monkeypatch.setattr(fleet_mod, "_is_pid_alive", lambda pid: pid == 111)  # 111 alive
    dead_owner = tmp_path / "llama-server-chat-999.port"
    live_owner = tmp_path / "llama-server-embed-111.port"
    dead_owner.write_text(json.dumps({"parent_pid": 999, "pid": 4242, "port": 9000}))
    live_owner.write_text(json.dumps({"parent_pid": 111, "pid": 5555, "port": 9001}))
    (tmp_path / "llama-server-bad-0.port").write_text("not-json")  # malformed -> cleaned
    reap_orphans(tmp_path)
    assert killed == [4242]  # only the crashed parent's server
    assert live_owner.exists()  # a concurrent live instance's server is untouched
    assert not dead_owner.exists()
    assert not (tmp_path / "llama-server-bad-0.port").exists()


def test_is_pid_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fleet_mod.sys, "platform", "linux")
    assert fleet_mod._is_pid_alive(os.getpid()) is True  # this process
    assert fleet_mod._is_pid_alive(1) is True  # exists, EPERM -> assume alive
    assert fleet_mod._is_pid_alive(2**31 - 1) is False  # no such pid
    monkeypatch.setattr(fleet_mod.sys, "platform", "win32")
    assert fleet_mod._is_pid_alive(2**31 - 1) is True  # Windows: never reap


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


def test_restart_dead_respawns_and_remarks_ready(
    tmp_path: Path, patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Drive _restart_dead directly in the main thread (no monitor thread), so the
    # behavior and its coverage are deterministic.
    monkeypatch.setattr(fleet_mod, "_RESTART_BACKOFF_S", (0.0,))
    fleet = Fleet(data_dir=tmp_path)
    server = FleetServer(_launch(tmp_path))
    server.spawn()
    server.ready = True
    fleet._servers.append(server)
    server._proc._alive = False  # crash it
    fleet._restart_dead()
    assert server.restarts == 1
    assert server.is_alive()
    assert server.ready is True  # re-marked ready after a successful restart


def test_restart_dead_aborts_when_stopping(
    tmp_path: Path, patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fleet_mod, "_RESTART_BACKOFF_S", (0.0,))
    fleet = Fleet(data_dir=tmp_path)
    server = FleetServer(_launch(tmp_path))
    server.spawn()
    fleet._servers.append(server)
    server._proc._alive = False
    fleet._stop_monitor.set()  # shutting down: must abort, not restart
    fleet._restart_dead()
    assert server.restarts == 0


def test_restart_dead_stops_respawn_if_shutdown_raced(
    tmp_path: Path, patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    # shutdown sets the stop flag while a restart's respawn is in flight; the
    # monitor must stop the just-spawned server rather than leave it running.
    monkeypatch.setattr(fleet_mod, "_RESTART_BACKOFF_S", (0.0,))
    fleet = Fleet(data_dir=tmp_path)
    server = FleetServer(_launch(tmp_path))
    server.spawn()
    fleet._servers.append(server)
    server._proc._alive = False
    real_restart = server.restart

    def _restart_then_shutdown_races() -> bool:
        result = real_restart()
        fleet._stop_monitor.set()  # shutdown lands during the respawn
        return result

    monkeypatch.setattr(server, "restart", _restart_then_shutdown_races)
    fleet._restart_dead()
    assert server.ready is False
    assert not server._launch.port_file.exists()  # stop() cleaned it up


def test_monitor_loop_runs_until_stopped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Cover the loop body in the main thread: one tick, then stop.
    monkeypatch.setattr(fleet_mod, "_MONITOR_POLL_S", 0.0)
    fleet = Fleet(data_dir=tmp_path)
    calls = {"n": 0}

    def _tick() -> None:
        calls["n"] += 1
        fleet._stop_monitor.set()

    monkeypatch.setattr(fleet, "_restart_dead", _tick)
    fleet._monitor_loop()
    assert calls["n"] == 1


def test_monitor_thread_runs_and_stops(tmp_path: Path, patched: dict, monkeypatch) -> None:
    monkeypatch.setattr(fleet_mod, "_MONITOR_POLL_S", 0.01)
    fleet = Fleet(data_dir=tmp_path)
    fleet.start([_launch(tmp_path)])
    time.sleep(0.05)  # let the monitor tick at least once with all servers alive
    fleet.shutdown()
    assert fleet._monitor is None
