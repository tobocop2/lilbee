"""Tests for the llama-swap process manager."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import swap_manager as sm
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.swap_manager import SwapManager
from lilbee.providers.roles import WorkerRole


class _FakeProc:
    """A stand-in subprocess that records teardown and reports a poll result."""

    def __init__(self, *, poll_result: int | None = None) -> None:
        self.pid = 4321
        self._poll_result = poll_result
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self._poll_result

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


def _fake_response(*, status: int = 200, payload: object = None) -> object:
    class _Resp:
        status_code = status

        def json(self) -> object:
            return payload

    return _Resp()


def _patch_spawn(monkeypatch: pytest.MonkeyPatch, proc: _FakeProc) -> None:
    monkeypatch.setattr(sm, "resolve_llama_swap", lambda: Path("/fake/llama-swap"))
    monkeypatch.setattr(sm.subprocess, "Popen", lambda *a, **k: proc)
    # Isolate lifecycle tests from the real process-tree teardown.
    monkeypatch.setattr(sm, "_stop_process_tree", lambda p: None)


def _patch_http(monkeypatch: pytest.MonkeyPatch, responder) -> None:
    monkeypatch.setattr(sm.httpx, "get", lambda url, timeout=None: responder(url))


def _launch(role: WorkerRole) -> InstanceLaunch:
    return InstanceLaunch(
        role=role,
        argv=["/bin/llama-server"],
        env_overrides={},
        model=f"{role.value}-model",
        port_file=Path(f"/data/{role.value}.port"),
    )


class TestStart:
    def test_writes_config_and_becomes_ready(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        mgr = SwapManager(tmp_path)
        mgr.start([_launch(WorkerRole.CHAT), _launch(WorkerRole.EMBED)])
        # Config was written and the endpoint is live.
        config = json.loads((tmp_path / "llama-swap.json").read_text())
        assert mgr.endpoint().startswith("http://127.0.0.1:")
        # Each member got its own freshly allocated port, distinct from the proxy's.
        member_ports = {
            model_id: int(entry["cmd"].rsplit(" ", 1)[-1])
            for model_id, entry in config["models"].items()
        }
        proxy_port = int(mgr.endpoint().rsplit(":", 1)[-1])
        assert len(set(member_ports.values())) == 2
        assert proxy_port not in member_ports.values()

    def test_raises_when_process_exits_before_ready(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_spawn(monkeypatch, _FakeProc(poll_result=1))  # already exited
        _patch_http(monkeypatch, lambda _url: _fake_response(status=503))
        with pytest.raises(ProviderError, match="exited before it was ready"):
            SwapManager(tmp_path).start([_launch(WorkerRole.CHAT)])

    def test_raises_when_never_healthy_in_time(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))

        def _refuse(_url: str) -> object:
            raise httpx.ConnectError("refused")

        _patch_http(monkeypatch, _refuse)
        monkeypatch.setattr(sm.time, "sleep", lambda _s: None)
        clock = iter([0.0, 10.0, 20.0, 31.0, 31.0])
        monkeypatch.setattr(sm.time, "monotonic", lambda: next(clock))
        with pytest.raises(ProviderError, match="did not start in time"):
            SwapManager(tmp_path).start([_launch(WorkerRole.CHAT)])


class TestEndpoint:
    def test_raises_before_start(self, tmp_path: Path) -> None:
        with pytest.raises(ProviderError, match="not running"):
            SwapManager(tmp_path).endpoint()


class TestRoleReady:
    def _started(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, responder) -> SwapManager:
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_http(monkeypatch, lambda url: _fake_response(status=200))
        mgr = SwapManager(tmp_path)
        mgr.start([_launch(WorkerRole.CHAT)])
        _patch_http(monkeypatch, responder)
        return mgr

    def test_true_when_role_loaded_and_ready(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        running = {"running": [{"model": "chat-0", "state": "ready"}]}
        mgr = self._started(tmp_path, monkeypatch, lambda _url: _fake_response(payload=running))
        assert mgr.role_ready(WorkerRole.CHAT) is True
        assert mgr.role_ready(WorkerRole.EMBED) is False

    def test_true_when_any_replica_ready(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A role is ready as soon as one of its data-parallel replicas is up.
        running = {"running": [{"model": "embed-1", "state": "ready"}]}
        mgr = self._started(tmp_path, monkeypatch, lambda _url: _fake_response(payload=running))
        assert mgr.role_ready(WorkerRole.EMBED) is True

    def test_false_when_role_still_loading(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        running = {"running": [{"model": "chat-0", "state": "starting"}]}
        mgr = self._started(tmp_path, monkeypatch, lambda _url: _fake_response(payload=running))
        assert mgr.role_ready(WorkerRole.CHAT) is False

    def test_false_when_running_probe_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _refuse(_url: str) -> object:
            raise httpx.ConnectError("refused")

        mgr = self._started(tmp_path, monkeypatch, _refuse)
        assert mgr.role_ready(WorkerRole.CHAT) is False


class TestLifecycle:
    def test_shutdown_is_noop_when_not_started(self, tmp_path: Path) -> None:
        SwapManager(tmp_path).shutdown()  # must not raise

    def test_shutdown_terminates_and_clears(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        terminated: list[object] = []
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        monkeypatch.setattr(sm, "_stop_process_tree", lambda p: terminated.append(p))
        mgr = SwapManager(tmp_path)
        mgr.start([_launch(WorkerRole.CHAT)])
        mgr.shutdown()
        assert terminated  # the process tree was torn down
        with pytest.raises(ProviderError):
            mgr.endpoint()  # port cleared after shutdown

    def test_reload_restarts_with_new_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        starts: list[int] = []
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        mgr = SwapManager(tmp_path)
        mgr.start([_launch(WorkerRole.CHAT)])
        monkeypatch.setattr(
            SwapManager, "start", lambda self, launches: starts.append(len(launches))
        )
        mgr.reload([_launch(WorkerRole.CHAT), _launch(WorkerRole.EMBED)])
        assert starts == [2]  # restarted with the new launch set


class _FakeChild:
    """A stand-in psutil.Process that records the signals it receives."""

    def __init__(self, *, running: bool = True) -> None:
        self.running = running
        self.terminated = False
        self.killed = False

    def is_running(self) -> bool:
        return self.running

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


class TestProcessTeardown:
    def test_pick_free_ports_returns_distinct_bound_ports(self) -> None:
        ports = sm._pick_free_ports(3)
        assert len(set(ports)) == 3
        assert all(1024 < port <= 65535 for port in ports)

    def test_terminate_group_posix_sigterm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sm.os, "getpgid", lambda _pid: 99, raising=False)
        signals: list[int] = []
        monkeypatch.setattr(
            sm.os, "killpg", lambda _pgid, signum: signals.append(signum), raising=False
        )
        sm._terminate_group(_FakeProc(poll_result=None))
        assert signals == [sm.signal.SIGTERM]

    def test_terminate_group_escalates_to_sigkill_on_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sm.os, "getpgid", lambda _pid: 99, raising=False)
        signals: list[int] = []
        monkeypatch.setattr(
            sm.os, "killpg", lambda _pgid, signum: signals.append(signum), raising=False
        )

        class _Stuck(_FakeProc):
            def wait(self, timeout: float | None = None) -> int:
                raise subprocess.TimeoutExpired("llama-swap", timeout or 0)

        sm._terminate_group(_Stuck(poll_result=None))
        assert signals == [sm.signal.SIGTERM, sm._SIGKILL]

    def test_stop_process_tree_windows_hard_stops(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sm.sys, "platform", "win32")
        monkeypatch.setattr(sm, "_live_children", lambda _pid: [])
        proc = _FakeProc(poll_result=None)
        sm._stop_process_tree(proc)
        assert proc.terminated is True

    def test_stop_process_tree_posix_terminates_group(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sm.sys, "platform", "linux")
        monkeypatch.setattr(sm, "_live_children", lambda _pid: [])
        groups: list[object] = []
        monkeypatch.setattr(sm, "_terminate_group", lambda p: groups.append(p))
        sm._stop_process_tree(_FakeProc(poll_result=None))
        assert groups

    def test_stop_process_tree_reaps_surviving_children(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # llama-swap puts each server in its own process group, so the group
        # kill misses them; a survivor must be swept or it keeps its port.
        survivor = _FakeChild(running=True)
        monkeypatch.setattr(sm.sys, "platform", "linux")
        monkeypatch.setattr(sm, "_live_children", lambda _pid: [survivor])
        monkeypatch.setattr(sm, "_terminate_group", lambda p: None)
        monkeypatch.setattr(sm.psutil, "wait_procs", lambda procs, timeout: ([], []))
        sm._stop_process_tree(_FakeProc(poll_result=None))
        assert survivor.terminated is True

    def test_reap_survivors_kills_after_grace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stubborn = _FakeChild(running=True)
        monkeypatch.setattr(sm.psutil, "wait_procs", lambda procs, timeout: ([], [stubborn]))
        sm._reap_survivors([stubborn])
        assert stubborn.terminated is True
        assert stubborn.killed is True

    def test_reap_survivors_skips_already_dead(self, monkeypatch: pytest.MonkeyPatch) -> None:
        dead = _FakeChild(running=False)
        monkeypatch.setattr(sm.psutil, "wait_procs", lambda procs, timeout: ([], []))
        sm._reap_survivors([dead])
        assert dead.terminated is False

    def test_live_children_empty_when_process_gone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _gone(_pid: int) -> object:
            raise sm.psutil.NoSuchProcess(_pid)

        monkeypatch.setattr(sm.psutil, "Process", _gone)
        assert sm._live_children(12345) == []

    def test_live_children_finds_spawned_child(self) -> None:
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            assert proc.pid in [child.pid for child in sm._live_children(os.getpid())]
        finally:
            proc.kill()
            proc.wait()

    def test_hard_stop_kills_on_timeout(self) -> None:
        class _Stuck(_FakeProc):
            def wait(self, timeout: float | None = None) -> int:
                raise subprocess.TimeoutExpired("llama-swap", timeout or 0)

        proc = _Stuck(poll_result=None)
        sm._hard_stop(proc)
        assert proc.killed is True
