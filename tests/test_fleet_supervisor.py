"""Tests for the multi-GPU fleet supervisor (spawn, reap, monitor, restart)."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from lilbee.providers.fleet import fleet as fleet_mod
from lilbee.providers.fleet.fleet import (
    _GIB,
    _READY_TIMEOUT_PER_GIB_S,
    _READY_TIMEOUT_S,
    Fleet,
    FleetServer,
    InstanceLaunch,
    _ready_timeout_for,
    pick_free_port,
    reap_orphans,
)
from lilbee.providers.roles import WorkerRole


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


def _ready_server(tmp_path: Path, role: WorkerRole, *, slots: int, ctx: int) -> FleetServer:
    launch = InstanceLaunch(
        role=role,
        argv=["/bin/llama-server"],
        env_overrides={},
        model="m.gguf",
        port_file=tmp_path / f"{role.value}-{slots}-{ctx}.port",
        slots=slots,
        ctx=ctx,
    )
    server = FleetServer(launch)
    server.ready = True
    return server


def test_chat_slot_capacity_sums_ready_chat_servers(tmp_path: Path) -> None:
    fleet = fleet_mod.Fleet()
    fleet._servers = [
        _ready_server(tmp_path, WorkerRole.CHAT, slots=4, ctx=16384),
        _ready_server(tmp_path, WorkerRole.EMBED, slots=8, ctx=512),
    ]
    assert fleet.chat_slot_capacity() == 4  # embed slots do not count toward chat
    assert fleet.chat_served_ctx() == 16384


def test_chat_capacity_ignores_unready_chat_server(tmp_path: Path) -> None:
    fleet = fleet_mod.Fleet()
    cold = _ready_server(tmp_path, WorkerRole.CHAT, slots=4, ctx=16384)
    cold.ready = False
    fleet._servers = [cold]
    assert fleet.chat_slot_capacity() == 1  # floor, never zero
    assert fleet.chat_served_ctx() is None


def test_chat_capacity_defaults_with_no_servers() -> None:
    fleet = fleet_mod.Fleet()
    assert fleet.chat_slot_capacity() == 1
    assert fleet.chat_served_ctx() is None


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


def test_spawn_creates_missing_port_file_parent_dir(tmp_path: Path, patched: dict) -> None:
    """spawn() must create the port-file/stderr-log parent dir when it is missing.

    The fleet warm-up can spawn the embed server at startup before the data dir
    exists (indexing creates it later). Opening the stderr log in a missing dir
    raised FileNotFoundError that silently failed warm-up, so the first search hit
    a cold embed engine and 503'd instead of returning results (bb-1ldh).
    """
    missing = tmp_path / "data"  # parent dir does NOT exist yet
    launch = InstanceLaunch(
        role=WorkerRole.EMBED,
        argv=["/bin/llama-server", "--model", "m.gguf"],
        env_overrides={"CUDA_VISIBLE_DEVICES": "0"},
        model="m.gguf",
        port_file=missing / "llama-server-embed.port",
    )
    FleetServer(launch).spawn()
    assert (missing / "llama-server-embed.port").exists()
    assert (missing / "llama-server-embed.log").exists()


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


def test_wait_ready_false_when_dead_on_every_retry(
    tmp_path: Path, patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every (re)spawn dies at once -> exhaust the bind retries -> False.
    monkeypatch.setattr(fleet_mod.subprocess, "Popen", lambda *a, **k: FakeProc(alive=False))
    monkeypatch.setattr(fleet_mod.time, "sleep", lambda _s: None)
    server = FleetServer(_launch(tmp_path))
    server.spawn()
    assert server.wait_ready(timeout=0.01) is False


def test_restart_terminates_a_still_alive_process(tmp_path: Path, patched: dict) -> None:
    # restart() defensively kills a server whose process is still alive.
    server = FleetServer(_launch(tmp_path))
    server.spawn()  # alive
    assert server.restart() is True
    assert server.restarts == 1


def test_restart_dead_skips_a_recovered_server(
    tmp_path: Path, patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Dead in the snapshot but alive again by the time the loop reaches it -> skip.
    monkeypatch.setattr(fleet_mod, "_RESTART_BACKOFF_S", (0.0,))
    fleet = Fleet(data_dir=tmp_path)
    server = FleetServer(_launch(tmp_path))
    server.spawn()
    fleet._servers.append(server)
    calls = {"n": 0}

    def _is_alive() -> bool:
        calls["n"] += 1
        return calls["n"] > 1  # False on the snapshot, True on the recheck

    monkeypatch.setattr(server, "is_alive", _is_alive)
    fleet._restart_dead()
    assert server.restarts == 0  # recovered, so not restarted


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
    # Exercise the POSIX branch by mocking os.kill's outcomes -- calling the real
    # syscall is non-portable (Windows os.kill has different error semantics and
    # can even target the live process), which is why this used to fail on Windows.
    monkeypatch.setattr(fleet_mod.os, "kill", lambda _pid, _sig: None)
    assert fleet_mod._is_pid_alive(123) is True  # signalable -> alive

    def _no_such(_pid: int, _sig: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(fleet_mod.os, "kill", _no_such)
    assert fleet_mod._is_pid_alive(123) is False  # no such pid

    def _eperm(_pid: int, _sig: int) -> None:
        raise PermissionError  # an OSError subclass

    monkeypatch.setattr(fleet_mod.os, "kill", _eperm)
    assert fleet_mod._is_pid_alive(123) is True  # exists but not signalable

    monkeypatch.setattr(fleet_mod.sys, "platform", "win32")
    assert fleet_mod._is_pid_alive(2**31 - 1) is True  # Windows: never reap (no os.kill)


def test_kill_pid_group_handles_missing_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fleet_mod.sys, "platform", "linux")
    # os.killpg does not exist on Windows; provide it so forcing the POSIX branch
    # doesn't raise AttributeError on the runner before getpgid is reached.
    monkeypatch.setattr(fleet_mod.os, "killpg", lambda _pgid, _sig: None, raising=False)

    def _missing(_pid: int) -> int:
        raise ProcessLookupError

    monkeypatch.setattr(fleet_mod.os, "getpgid", _missing, raising=False)
    fleet_mod._kill_pid_group(123)  # getpgid raises -> swallowed, must not propagate


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


def test_empty_placement_starts_no_monitor(tmp_path: Path, patched: dict) -> None:
    # Every role degraded to in-process -> no servers -> no supervisor thread.
    fleet = Fleet(data_dir=tmp_path)
    fleet.start([])
    try:
        assert fleet._monitor is None
        assert fleet.healthy_clients(WorkerRole.CHAT) == []
    finally:
        fleet.shutdown()


def test_start_with_empty_eager_roles_spawns_nothing(
    tmp_path: Path, patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Deferred build: launches are planned (joint placement) but no server spawns
    # until a role is used, so an embed-only ingest never loads chat's VRAM.
    reaped: list[Path] = []
    monkeypatch.setattr(fleet_mod, "reap_orphans", reaped.append)
    fleet = Fleet(data_dir=tmp_path)
    fleet.start(
        [_launch(tmp_path, WorkerRole.CHAT), _launch(tmp_path, WorkerRole.EMBED)],
        eager_roles=frozenset(),
    )
    try:
        assert reaped == [tmp_path]  # orphans still reaped at build
        assert fleet._monitor is None  # nothing spawned -> no supervisor yet
        assert fleet.healthy_clients(WorkerRole.CHAT) == []
        assert fleet.healthy_clients(WorkerRole.EMBED) == []
    finally:
        fleet.shutdown()


def test_ensure_role_brings_up_only_that_role(tmp_path: Path, patched: dict) -> None:
    fleet = Fleet(data_dir=tmp_path)
    fleet.start(
        [_launch(tmp_path, WorkerRole.CHAT), _launch(tmp_path, WorkerRole.EMBED)],
        eager_roles=frozenset(),
    )
    try:
        fleet.ensure_role(WorkerRole.EMBED)
        assert len(fleet.healthy_clients(WorkerRole.EMBED)) == 1
        assert fleet.healthy_clients(WorkerRole.CHAT) == []  # chat stays deferred
        assert fleet._monitor is not None  # monitor starts with the first server
    finally:
        fleet.shutdown()


def test_ensure_all_brings_up_remaining_roles(tmp_path: Path, patched: dict) -> None:
    fleet = Fleet(data_dir=tmp_path)
    fleet.start(
        [_launch(tmp_path, WorkerRole.CHAT), _launch(tmp_path, WorkerRole.EMBED)],
        eager_roles=frozenset(),
    )
    try:
        fleet.ensure_role(WorkerRole.EMBED)
        fleet.ensure_all()
        assert len(fleet.healthy_clients(WorkerRole.CHAT)) == 1
        assert len(fleet.healthy_clients(WorkerRole.EMBED)) == 1
    finally:
        fleet.shutdown()


def test_ensure_role_is_idempotent(tmp_path: Path, patched: dict) -> None:
    fleet = Fleet(data_dir=tmp_path)
    fleet.start([_launch(tmp_path, WorkerRole.EMBED)], eager_roles=frozenset())
    try:
        fleet.ensure_role(WorkerRole.EMBED)
        fleet.ensure_role(WorkerRole.EMBED)  # second call must not spawn a duplicate
        assert len(fleet.healthy_clients(WorkerRole.EMBED)) == 1
    finally:
        fleet.shutdown()


def test_ensure_role_for_unplanned_role_is_noop(tmp_path: Path, patched: dict) -> None:
    # A role with nothing pending (unconfigured / not installed) brings up nothing.
    fleet = Fleet(data_dir=tmp_path)
    fleet.start([_launch(tmp_path, WorkerRole.EMBED)], eager_roles=frozenset())
    try:
        fleet.ensure_role(WorkerRole.CHAT)  # not planned -> no-op, no raise
        assert fleet.healthy_clients(WorkerRole.CHAT) == []
        assert fleet._monitor is None
    finally:
        fleet.shutdown()


def test_start_with_eager_subset_defers_the_rest(tmp_path: Path, patched: dict) -> None:
    fleet = Fleet(data_dir=tmp_path)
    fleet.start(
        [_launch(tmp_path, WorkerRole.CHAT), _launch(tmp_path, WorkerRole.EMBED)],
        eager_roles={WorkerRole.EMBED},
    )
    try:
        assert len(fleet.healthy_clients(WorkerRole.EMBED)) == 1  # eager
        assert fleet.healthy_clients(WorkerRole.CHAT) == []  # deferred
    finally:
        fleet.shutdown()


def test_fleet_start_degrades_unready_role_without_raising(
    tmp_path: Path, patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A role that never becomes ready must NOT take down the whole fleet (no
    # raise); it degrades to in-process (no healthy client). Other healthy roles
    # serving is covered by test_fleet_start_reaps_then_serves_healthy_clients.
    monkeypatch.setattr(fleet_mod.time, "sleep", lambda _s: None)
    monkeypatch.setattr(fleet_mod.LlamaServerClient, "health", lambda self: False)
    fleet = Fleet(ready_timeout=0.02, data_dir=tmp_path)
    fleet.start([_launch(tmp_path, WorkerRole.CHAT)])  # does not raise
    try:
        assert fleet.healthy_clients(WorkerRole.CHAT) == []  # degraded -> in-process
    finally:
        fleet.shutdown()


def test_restart_dead_gives_up_after_cap_and_leaves_role_down(
    tmp_path: Path, patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A server that never becomes ready stops being respawned after the cap, so a
    # doomed model (e.g. OOM at launch) doesn't crash-loop forever. wait_ready is
    # stubbed False to isolate the cap logic from its (real) readiness timeout.
    monkeypatch.setattr(fleet_mod, "_RESTART_BACKOFF_S", (0.0,))
    monkeypatch.setattr(fleet_mod.FleetServer, "wait_ready", lambda self, timeout=None: False)
    fleet = Fleet(data_dir=tmp_path)
    server = FleetServer(_launch(tmp_path))
    fleet._servers.append(server)
    server.spawn()
    for _ in range(fleet_mod._MAX_RESTART_ATTEMPTS + 2):
        server._proc._alive = False  # crash before each monitor tick
        fleet._restart_dead()
    assert server.gave_up
    assert server.consecutive_failures == fleet_mod._MAX_RESTART_ATTEMPTS
    assert fleet.healthy_clients(WorkerRole.CHAT) == []  # role stays down (calls error)


def test_restart_resets_failure_count_on_success(tmp_path: Path, patched: dict) -> None:
    server = FleetServer(_launch(tmp_path))
    server.spawn()
    server.consecutive_failures = 3
    server._proc._alive = False
    assert server.restart() is True  # patched health returns True
    assert server.consecutive_failures == 0  # success resets the count


def test_failed_start_detail_returns_stderr_tail(tmp_path: Path, patched: dict) -> None:
    server = FleetServer(_launch(tmp_path))
    server.spawn()  # creates the (empty) stderr log
    server._stderr_log.write_text("ggml_backend_cuda: out of memory")
    assert "out of memory" in server.failed_start_detail()


def test_failed_start_detail_empty_when_no_log(tmp_path: Path, patched: dict) -> None:
    server = FleetServer(_launch(tmp_path))  # never spawned -> no log file
    assert server.failed_start_detail() == ""


def test_stop_removes_stderr_log(tmp_path: Path, patched: dict) -> None:
    server = FleetServer(_launch(tmp_path))
    server.spawn()
    assert server._stderr_log.exists()
    server.stop()
    assert not server._stderr_log.exists()


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


def test_set_listener_attaches_callbacks() -> None:
    fleet = Fleet()
    spawning: list[WorkerRole] = []
    spawned: list[WorkerRole] = []
    fleet.set_listener(on_spawning=spawning.append, on_spawned=spawned.append)
    fleet._notify(fleet._on_spawning, WorkerRole.CHAT)
    fleet._notify(fleet._on_spawned, WorkerRole.CHAT)
    assert spawning == [WorkerRole.CHAT]
    assert spawned == [WorkerRole.CHAT]


def test_notify_swallows_listener_errors() -> None:
    fleet = Fleet()

    def _boom(_role: WorkerRole) -> None:
        raise RuntimeError("listener blew up")

    # Must not propagate: spawn-progress feedback is best-effort.
    fleet._notify(_boom, WorkerRole.CHAT)


def test_notify_skips_when_no_callback() -> None:
    Fleet()._notify(None, WorkerRole.CHAT)  # no callback -> no-op, no error


def test_restart_role_replaces_only_that_role(tmp_path: Path, patched: dict, monkeypatch) -> None:
    monkeypatch.setattr(fleet_mod.time, "sleep", lambda _s: None)
    fleet = Fleet(data_dir=tmp_path)
    fleet.start([_launch(tmp_path, WorkerRole.CHAT), _launch(tmp_path, WorkerRole.EMBED)])
    chat_before = [s for s in fleet._servers if s.role == WorkerRole.CHAT]
    embed_before = [s for s in fleet._servers if s.role == WorkerRole.EMBED]

    fleet.restart_role(WorkerRole.CHAT, [_launch(tmp_path, WorkerRole.CHAT)])

    chat_after = [s for s in fleet._servers if s.role == WorkerRole.CHAT]
    embed_after = [s for s in fleet._servers if s.role == WorkerRole.EMBED]
    assert chat_after and chat_after[0] not in chat_before  # chat respawned
    assert embed_after == embed_before  # embed left running untouched
    fleet.shutdown()


def test_restart_role_fires_spawn_listener(tmp_path: Path, patched: dict, monkeypatch) -> None:
    monkeypatch.setattr(fleet_mod.time, "sleep", lambda _s: None)
    events: list[tuple[str, WorkerRole]] = []
    fleet = Fleet(
        data_dir=tmp_path,
        on_spawning=lambda r: events.append(("up", r)),
        on_spawned=lambda r: events.append(("ready", r)),
    )
    fleet.restart_role(WorkerRole.EMBED, [_launch(tmp_path, WorkerRole.EMBED)])
    assert ("up", WorkerRole.EMBED) in events
    assert ("ready", WorkerRole.EMBED) in events
    fleet.shutdown()


def test_restart_role_empty_launches_just_stops_old(
    tmp_path: Path, patched: dict, monkeypatch
) -> None:
    monkeypatch.setattr(fleet_mod.time, "sleep", lambda _s: None)
    fleet = Fleet(data_dir=tmp_path)
    fleet.start([_launch(tmp_path, WorkerRole.RERANK)])
    fleet.restart_role(WorkerRole.RERANK, [])  # role unconfigured now -> no replacement
    assert [s for s in fleet._servers if s.role == WorkerRole.RERANK] == []
    fleet.shutdown()


def test_restart_role_breaks_when_shutdown_already_requested(
    tmp_path: Path, patched: dict, monkeypatch
) -> None:
    monkeypatch.setattr(fleet_mod.time, "sleep", lambda _s: None)
    fleet = Fleet(data_dir=tmp_path)
    fleet._stop_monitor.set()  # shutdown already in flight
    fleet.restart_role(WorkerRole.CHAT, [_launch(tmp_path, WorkerRole.CHAT)])
    assert fleet._servers == []  # nothing brought up, nothing stranded


def test_restart_role_stops_fresh_servers_if_shutdown_races(
    tmp_path: Path, patched: dict, monkeypatch
) -> None:
    monkeypatch.setattr(fleet_mod.time, "sleep", lambda _s: None)
    fleet = Fleet(data_dir=tmp_path)
    original_bring_up = fleet._bring_up

    def _bring_up_then_signal(server: FleetServer) -> None:
        original_bring_up(server)
        fleet._stop_monitor.set()  # shutdown races in right after the spawn

    monkeypatch.setattr(fleet, "_bring_up", _bring_up_then_signal)
    fleet.restart_role(WorkerRole.CHAT, [_launch(tmp_path, WorkerRole.CHAT)])
    assert fleet._servers == []  # the freshly spawned server was stopped, not retained


class TestReadyTimeoutScaling:
    def test_zero_weights_uses_base(self) -> None:
        assert _ready_timeout_for(300.0, 0) == 300.0

    def test_negative_weights_uses_base(self) -> None:
        assert _ready_timeout_for(300.0, -1) == 300.0

    def test_scales_with_model_size(self) -> None:
        # An 8B Q4 model (~5 GiB) gets the base plus a per-GiB term, comfortably
        # above a slow Mac cold load so wait_ready does not fire prematurely.
        weights = 5 * _GIB
        expected = 300.0 + _READY_TIMEOUT_PER_GIB_S * 5
        assert _ready_timeout_for(300.0, weights) == expected

    def test_base_is_generous_for_cold_loads(self) -> None:
        # Regression guard: 180s fired prematurely on a Mac 8B cold load; the
        # floor must stay well above that even for a small (0-weight) model.
        assert _READY_TIMEOUT_S >= 300.0

    def test_server_ready_timeout_scales_with_launch_weights(self, tmp_path: Path) -> None:
        launch = InstanceLaunch(
            role=WorkerRole.CHAT,
            argv=["/bin/llama-server", "--model", "m.gguf"],
            env_overrides={},
            model="m.gguf",
            port_file=tmp_path / "p.port",
            weights_bytes=4 * _GIB,
        )
        server = FleetServer(launch)
        assert server.weights_bytes == 4 * _GIB
        assert server.ready_timeout == _READY_TIMEOUT_S + _READY_TIMEOUT_PER_GIB_S * 4

    def test_bring_up_passes_scaled_timeout(
        self, tmp_path: Path, patched: dict, monkeypatch
    ) -> None:
        # _bring_up must scale the fleet's configured base by the model size, so a
        # large model is not declared dead while still loading.
        seen: list[float] = []
        monkeypatch.setattr(
            fleet_mod.FleetServer,
            "wait_ready",
            lambda self, timeout=None: seen.append(timeout) or True,
        )
        launch = InstanceLaunch(
            role=WorkerRole.CHAT,
            argv=["/bin/llama-server", "--model", "m.gguf"],
            env_overrides={},
            model="m.gguf",
            port_file=tmp_path / "p.port",
            weights_bytes=2 * _GIB,
        )
        fleet = Fleet(ready_timeout=100.0, data_dir=tmp_path)
        fleet._bring_up(FleetServer(launch))
        assert seen == [100.0 + _READY_TIMEOUT_PER_GIB_S * 2]


def test_bring_up_logs_starting_and_ready(tmp_path: Path, patched: dict, caplog) -> None:
    fleet = Fleet(ready_timeout=1.0, data_dir=tmp_path)
    server = FleetServer(_launch(tmp_path))
    with caplog.at_level("INFO", logger="lilbee.providers.fleet.fleet"):
        fleet._bring_up(server)
    assert server.ready is True
    text = caplog.text.lower()
    assert "starting chat engine" in text
    assert "loading m.gguf" in text
    assert "chat engine ready" in text


def test_fleet_server_exposes_model(tmp_path: Path) -> None:
    assert FleetServer(_launch(tmp_path)).model == "m.gguf"


def test_bring_up_role_breaks_before_spawning_when_stop_requested(tmp_path, monkeypatch) -> None:
    # A shutdown signalled before bring-up starts must abort the launch loop so no
    # new server is spawned during teardown.
    fleet = Fleet(data_dir=tmp_path)
    fleet._pending[WorkerRole.CHAT] = [_launch(tmp_path)]
    fleet._stop_monitor.set()
    brought: list[object] = []
    monkeypatch.setattr(fleet, "_bring_up", lambda server: brought.append(server))

    fleet._bring_up_role(WorkerRole.CHAT)

    assert brought == []  # loop broke before bringing anything up
    assert fleet._servers == []


def test_bring_up_role_stops_fresh_servers_when_stop_races_bringup(tmp_path, monkeypatch) -> None:
    # Shutdown signalled after a server is up but before it is registered: the
    # fresh server is stopped and never added to the live set.
    fleet = Fleet(data_dir=tmp_path)
    fleet._pending[WorkerRole.CHAT] = [_launch(tmp_path)]
    monkeypatch.setattr(fleet, "_bring_up", lambda server: None)

    class _RaceStop:
        """is_set() is False during the launch loop, True at the post-loop check."""

        def __init__(self) -> None:
            self._calls = 0

        def is_set(self) -> bool:
            self._calls += 1
            return self._calls > 1

    fleet._stop_monitor = _RaceStop()
    stopped: list[object] = []
    monkeypatch.setattr(FleetServer, "stop", lambda self: stopped.append(self))

    fleet._bring_up_role(WorkerRole.CHAT)

    assert len(stopped) == 1  # the fresh server was stopped, not registered
    assert fleet._servers == []
