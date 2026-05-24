"""Tests for the multi-GPU fleet supervisor."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from lilbee.providers import base as base_mod
from lilbee.providers.multi_gpu import fleet as fleet_mod
from lilbee.providers.multi_gpu.fleet import (
    Fleet,
    FleetServer,
    InstanceLaunch,
    _child_env,
    pick_free_port,
)
from lilbee.providers.worker.transport import WorkerRole


class FakeProc:
    def __init__(self, *, alive: bool = True, wait_raises: bool = False) -> None:
        self.pid = 4321
        self._alive = alive
        self._wait_raises = wait_raises

    def poll(self) -> int | None:
        return None if self._alive else 0

    def wait(self, timeout: float | None = None) -> int:
        if self._wait_raises:
            raise subprocess.TimeoutExpired("llama-server", timeout)
        self._alive = False
        return 0


def _launch(
    tmp_path: Path, role: WorkerRole = WorkerRole.CHAT, port: int = 42700
) -> InstanceLaunch:
    return InstanceLaunch(
        role=role,
        argv=["/bin/llama-server", "--port", str(port)],
        devices=(0,),
        port=port,
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
    monkeypatch.setattr(fleet_mod.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(fleet_mod.os, "killpg", lambda pgid, sig: killed.append((pgid, sig)))
    monkeypatch.setattr(fleet_mod.LlamaServerClient, "health", lambda self: True)
    monkeypatch.setattr(fleet_mod.LlamaServerClient, "close", lambda self: None)
    return {"killed": killed, "procs": procs}


def test_pick_free_port_returns_int() -> None:
    assert isinstance(pick_free_port(), int)


def test_child_env_pins_devices() -> None:
    env = _child_env((0, 2))
    assert env["CUDA_VISIBLE_DEVICES"] == "0,2"
    assert env["GGML_VK_VISIBLE_DEVICES"] == "0,2"


def test_spawn_writes_port_file_and_creates_client(tmp_path: Path, patched: dict) -> None:
    server = FleetServer(_launch(tmp_path))
    server.spawn()
    assert (tmp_path / "llama-server-chat.port").read_text() == "42700"
    assert server.client is not None
    assert server.is_alive()


def test_wait_ready_true_when_health_ok(tmp_path: Path, patched: dict) -> None:
    server = FleetServer(_launch(tmp_path))
    server.spawn()
    assert server.wait_ready(timeout=1.0) is True


def test_wait_ready_false_when_process_dead(
    tmp_path: Path, patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fleet_mod.subprocess, "Popen", lambda *a, **k: FakeProc(alive=False))
    server = FleetServer(_launch(tmp_path))
    server.spawn()
    assert server.wait_ready(timeout=1.0) is False


def test_wait_ready_false_on_timeout(
    tmp_path: Path, patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fleet_mod.LlamaServerClient, "health", lambda self: False)
    monkeypatch.setattr(fleet_mod.time, "sleep", lambda _s: None)
    server = FleetServer(_launch(tmp_path))
    server.spawn()
    assert server.wait_ready(timeout=0.05) is False


def test_stop_terminates_group_and_cleans_up(tmp_path: Path, patched: dict) -> None:
    import signal

    server = FleetServer(_launch(tmp_path))
    server.spawn()
    server.stop()
    assert (4321, signal.SIGTERM) in patched["killed"]
    assert not (tmp_path / "llama-server-chat.port").exists()


def test_stop_escalates_to_sigkill_on_timeout(
    tmp_path: Path, patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    import signal

    monkeypatch.setattr(fleet_mod.subprocess, "Popen", lambda *a, **k: FakeProc(wait_raises=True))
    server = FleetServer(_launch(tmp_path))
    server.spawn()
    server.stop()
    assert (4321, signal.SIGKILL) in patched["killed"]


def test_fleet_start_groups_clients_by_role(tmp_path: Path, patched: dict) -> None:
    fleet = Fleet()
    by_role = fleet.start(
        [
            _launch(tmp_path, WorkerRole.CHAT, 42700),
            _launch(tmp_path, WorkerRole.CHAT, 42701),
            _launch(tmp_path, WorkerRole.EMBED, 42702),
        ]
    )
    assert len(by_role[WorkerRole.CHAT]) == 2
    assert len(by_role[WorkerRole.EMBED]) == 1


def test_fleet_start_raises_and_tears_down_when_not_ready(
    tmp_path: Path, patched: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fleet_mod.LlamaServerClient, "health", lambda self: False)
    monkeypatch.setattr(fleet_mod.time, "sleep", lambda _s: None)
    fleet = Fleet(ready_timeout=0.05)
    with pytest.raises(base_mod.ProviderError, match="failed to become ready"):
        fleet.start([_launch(tmp_path)])
    # torn down: no servers remain
    assert fleet._servers == []


def test_fleet_shutdown_stops_all(tmp_path: Path, patched: dict) -> None:
    fleet = Fleet()
    fleet.start([_launch(tmp_path, WorkerRole.CHAT, 42700)])
    fleet.shutdown()
    assert fleet._servers == []
