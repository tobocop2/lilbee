"""Tests for the llama-swap process manager."""

from __future__ import annotations

import itertools
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import swap_manager as sm
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.swap_manager import _UNLOAD_PATH, SwapManager
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

    def test_redirects_llama_swap_stdio_to_a_log_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """llama-swap's stdout/stderr must go to a log file, never an inherited
        terminal: an inherited fd bleeds its HTTP access log onto a TUI/CLI
        parent's screen and corrupts the render."""
        captured: dict[str, object] = {}

        def _capturing_popen(*_args: object, **kwargs: object) -> _FakeProc:
            captured.update(kwargs)
            return _FakeProc(poll_result=None)

        monkeypatch.setattr(sm, "resolve_llama_swap", lambda: Path("/fake/llama-swap"))
        monkeypatch.setattr(sm.subprocess, "Popen", _capturing_popen)
        monkeypatch.setattr(sm, "_stop_process_tree", lambda p: None)
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))

        mgr = SwapManager(tmp_path)
        mgr.start([_launch(WorkerRole.CHAT)])

        log_path = tmp_path / "logs" / "llama-swap.log"
        assert log_path.exists()
        # stdout is the opened log file (its .name is the path), not None
        # (inherited terminal) nor a PIPE; stderr merges into the same file.
        assert getattr(captured["stdout"], "name", None) == str(log_path)
        assert captured["stdout"] is not subprocess.PIPE
        assert captured["stderr"] is subprocess.STDOUT
        # shutdown releases the captured handle.
        mgr.shutdown()
        assert mgr._log_file is None

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
        # sm.time is the global time module, so stray background threads (e.g.
        # Textual timers from earlier TUI tests) also call this fake; an endless
        # rising clock can be neither exhausted nor stalled by extra callers.
        clock = itertools.count(0.0, 10.4)
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

    def test_false_when_shutdown_clears_port_mid_probe(self, tmp_path: Path) -> None:
        # A concurrent shutdown clears _port, so endpoint() raises
        # ProviderError; the read-only probe must report False, not throw.
        mgr = SwapManager(tmp_path)  # never started -> _port is None
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
        self.pid = 999
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

    def test_reap_survivors_waits_for_killed_processes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A TERM-ignoring server gets KILLed; the reap must then wait for it to
        # actually exit (VRAM teardown) before the caller probes free memory.
        stubborn = _FakeChild(running=True)
        waits: list[tuple[list, float]] = []
        results = iter([([], [stubborn]), ([stubborn], [])])

        def _wait_procs(procs: list[object], timeout: float) -> tuple[list, list]:
            waits.append((list(procs), timeout))
            return next(results)

        monkeypatch.setattr(sm.psutil, "wait_procs", _wait_procs)
        sm._reap_survivors([stubborn])
        assert stubborn.killed is True
        assert waits == [
            ([stubborn], sm._ORPHAN_STOP_TIMEOUT_S),
            ([stubborn], sm._KILL_WAIT_TIMEOUT_S),
        ]

    def test_reap_survivors_warns_when_sigkill_is_survived(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        immortal = _FakeChild(running=True)
        monkeypatch.setattr(sm.psutil, "wait_procs", lambda procs, timeout: ([], [immortal]))
        with caplog.at_level(logging.WARNING, logger=sm.__name__):
            sm._reap_survivors([immortal])
        assert "survived SIGKILL" in caplog.text

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


def _own_state_path(tmp_path: Path) -> Path:
    """The state file this process's SwapManager writes for itself."""
    return tmp_path / sm._state_filename(os.getpid())


def _swap_state(*, pid: int = 123, created_at: float | None = None) -> sm._SwapState:
    """A minimal _SwapState for swap-liveness checks."""
    return sm._SwapState(
        pid=pid, pgid=None, owner_pid=None, owner_created_at=None, created_at=created_at
    )


def _write_state(
    tmp_path: Path,
    *,
    pid: int = 7777,
    pgid: int | None = 7777,
    created_at: float | None = None,
    owner_pid: int | None = None,
    owner_created_at: float | None = None,
    member_ports: list[int] | None = None,
    filename: str = "llama-swap.state.json",
) -> Path:
    state_path = tmp_path / filename
    state_path.write_text(
        json.dumps(
            {
                "pid": pid,
                "pgid": pgid,
                "created_at": created_at,
                "owner_pid": owner_pid,
                "owner_created_at": owner_created_at,
                "member_ports": member_ports,
                "name": "llama-swap",
            }
        )
    )
    return state_path


class _FakePsProcess:
    """A stand-in psutil.Process with a settable cmdline and recorded signals."""

    def __init__(
        self,
        pid: int,
        *,
        cmdline: list[str],
        status: str = "running",
        create_time: float = 0.0,
    ) -> None:
        self.pid = pid
        self._cmdline = cmdline
        self._status = status
        self._create_time = create_time
        self.signals: list[int] = []
        self.wait_raises = False

    def cmdline(self) -> list[str]:
        return self._cmdline

    def status(self) -> str:
        return self._status

    def create_time(self) -> float:
        return self._create_time

    def children(self, recursive: bool = False) -> list:
        return []

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)

    def wait(self, timeout: float | None = None) -> int:
        if self.wait_raises:
            raise sm.psutil.TimeoutExpired(timeout or 0, self.pid)
        return 0


def _patch_psutil_process(monkeypatch: pytest.MonkeyPatch, table: dict[int, object]) -> None:
    def _process(pid: int | None = None) -> object:
        if pid is None:  # no-arg form = the current process (used by _write_state)
            return _FakePsProcess(os.getpid(), cmdline=[], create_time=123.0)
        if pid not in table:
            raise sm.psutil.NoSuchProcess(pid)
        return table[pid]

    monkeypatch.setattr(sm.psutil, "Process", _process)


class TestCrossRunReaping:
    def test_start_writes_a_state_file_with_the_swap_pid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        SwapManager(tmp_path).start([_launch(WorkerRole.CHAT)])
        state = json.loads(_own_state_path(tmp_path).read_text())
        assert state["pid"] == 4321
        assert state["name"] == "llama-swap"

    def test_clean_shutdown_removes_the_state_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        mgr = SwapManager(tmp_path)
        mgr.start([_launch(WorkerRole.CHAT)])
        assert _own_state_path(tmp_path).exists()
        mgr.shutdown()
        assert not _own_state_path(tmp_path).exists()

    def test_stale_state_with_dead_pid_cleans_file_without_killing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_state(tmp_path, pid=7777)
        _patch_psutil_process(monkeypatch, {})  # pid 7777 no longer exists
        stopped: list[object] = []
        monkeypatch.setattr(sm, "_stop_stale_swap", lambda state: stopped.append(state))
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        SwapManager(tmp_path).start([_launch(WorkerRole.CHAT)])
        assert stopped == []  # nothing alive to kill
        # the stale file is gone; this owner's own file records the NEW swap
        assert not (tmp_path / "llama-swap.state.json").exists()
        assert json.loads(_own_state_path(tmp_path).read_text())["pid"] == 4321

    def test_stale_state_with_live_llama_swap_kills_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_state(tmp_path, pid=7777)
        stale = _FakePsProcess(7777, cmdline=["/opt/llama-swap", "-config", "x.json"])
        _patch_psutil_process(monkeypatch, {7777: stale})
        stopped: list[sm._SwapState] = []
        monkeypatch.setattr(sm, "_stop_stale_swap", lambda state: stopped.append(state))
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        SwapManager(tmp_path).start([_launch(WorkerRole.CHAT)])
        assert [state.pid for state in stopped] == [7777]

    def test_pid_reuse_with_foreign_cmdline_is_not_killed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The recorded pid is alive but now belongs to another program.
        _write_state(tmp_path, pid=7777)
        impostor = _FakePsProcess(7777, cmdline=["/usr/bin/python3", "train.py"])
        _patch_psutil_process(monkeypatch, {7777: impostor})
        stopped: list[object] = []
        monkeypatch.setattr(sm, "_stop_stale_swap", lambda state: stopped.append(state))
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        SwapManager(tmp_path).start([_launch(WorkerRole.CHAT)])
        assert stopped == []

    def test_corrupt_state_file_is_skipped_not_deleted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An unparseable file may be a sibling's in-flight write; deleting it
        # would erase a LIVE owner's cross-run reap record.
        corrupt = tmp_path / "llama-swap.state.json"
        corrupt.write_text("{not json")
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        SwapManager(tmp_path).start([_launch(WorkerRole.CHAT)])
        assert corrupt.read_text() == "{not json"
        assert json.loads(_own_state_path(tmp_path).read_text())["pid"] == 4321

    def test_torn_state_file_is_left_in_place_by_reap(self, tmp_path: Path) -> None:
        torn = tmp_path / "llama-swap.state.json"
        torn.write_text('{"pid": 77')  # a sibling's write, caught mid-flight
        SwapManager(tmp_path).reap_stale()
        assert torn.read_text() == '{"pid": 77'

    def test_load_state_returns_none_when_absent(self, tmp_path: Path) -> None:
        assert sm._load_state(tmp_path / "missing.json") is None

    def test_is_live_llama_swap_false_for_dead_pid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_psutil_process(monkeypatch, {})
        assert sm._is_live_llama_swap(_swap_state(pid=123)) is False

    def test_is_live_llama_swap_false_for_empty_cmdline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A zombie reports an empty cmdline; never treat it as our process.
        _patch_psutil_process(monkeypatch, {123: _FakePsProcess(123, cmdline=[])})
        assert sm._is_live_llama_swap(_swap_state(pid=123)) is False

    def test_is_live_llama_swap_true_when_swap_create_time_matches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        live = _FakePsProcess(123, cmdline=["/opt/llama-swap"], create_time=42.0)
        _patch_psutil_process(monkeypatch, {123: live})
        assert sm._is_live_llama_swap(_swap_state(pid=123, created_at=42.0)) is True

    def test_swap_pid_reused_by_another_instances_swap_is_not_killed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The recycled pid runs llama-swap, but a DIFFERENT instance's: the
        # create-time mismatch must veto the cmdline match.
        _write_state(tmp_path, pid=7777, created_at=42.0)
        other_swap = _FakePsProcess(7777, cmdline=["/opt/llama-swap"], create_time=5000.0)
        _patch_psutil_process(monkeypatch, {7777: other_swap})
        stopped: list[object] = []
        monkeypatch.setattr(sm, "_stop_stale_swap", lambda state: stopped.append(state))
        SwapManager(tmp_path).reap_stale()
        assert stopped == []

    def test_swap_with_matching_create_time_is_killed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_state(tmp_path, pid=7777, created_at=42.0)
        stale = _FakePsProcess(7777, cmdline=["/opt/llama-swap"], create_time=42.0)
        _patch_psutil_process(monkeypatch, {7777: stale})
        stopped: list[sm._SwapState] = []
        monkeypatch.setattr(sm, "_stop_stale_swap", lambda state: stopped.append(state))
        SwapManager(tmp_path).reap_stale()
        assert [state.pid for state in stopped] == [7777]

    def test_legacy_state_without_swap_create_time_falls_back_to_cmdline(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_state(tmp_path, pid=7777, created_at=None)
        stale = _FakePsProcess(7777, cmdline=["/opt/llama-swap"], create_time=5000.0)
        _patch_psutil_process(monkeypatch, {7777: stale})
        stopped: list[sm._SwapState] = []
        monkeypatch.setattr(sm, "_stop_stale_swap", lambda state: stopped.append(state))
        SwapManager(tmp_path).reap_stale()
        assert [state.pid for state in stopped] == [7777]

    def test_start_records_the_swap_create_time(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_psutil_process(
            monkeypatch, {4321: _FakePsProcess(4321, cmdline=["llama-swap"], create_time=777.0)}
        )
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        SwapManager(tmp_path).start([_launch(WorkerRole.CHAT)])
        assert json.loads(_own_state_path(tmp_path).read_text())["created_at"] == 777.0

    def test_start_records_the_owner_lilbee_pid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        SwapManager(tmp_path).start([_launch(WorkerRole.CHAT)])
        state = json.loads(_own_state_path(tmp_path).read_text())
        assert state["owner_pid"] == os.getpid()
        assert state["owner_created_at"] == pytest.approx(
            sm.psutil.Process(os.getpid()).create_time()
        )

    def test_start_records_the_swap_pgid_on_posix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Platform pinned so the posix pgid lines run on every CI platform.
        monkeypatch.setattr(sm.sys, "platform", "linux")
        monkeypatch.setattr(sm.os, "getpgid", lambda pid: 999, raising=False)
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        SwapManager(tmp_path).start([_launch(WorkerRole.CHAT)])
        assert json.loads(_own_state_path(tmp_path).read_text())["pgid"] == 999

    def test_live_owner_leaves_swap_running_and_state_file_intact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A second lilbee at the same data_dir (e.g. `lilbee sync` beside the
        # running server) must not kill the live owner's healthy swap.
        state_path = _write_state(tmp_path, pid=7777, owner_pid=999)
        original = state_path.read_text()
        owner = _FakePsProcess(999, cmdline=["/usr/bin/python3", "-m", "lilbee"])
        swap = _FakePsProcess(7777, cmdline=["/opt/llama-swap"])
        _patch_psutil_process(monkeypatch, {999: owner, 7777: swap})
        stopped: list[object] = []
        monkeypatch.setattr(sm, "_stop_stale_swap", lambda state: stopped.append(state))
        SwapManager(tmp_path).reap_stale()
        assert stopped == []
        assert state_path.read_text() == original  # the live owner still needs it

    def test_dead_owner_swap_is_reaped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_state(tmp_path, pid=7777, owner_pid=999)
        swap = _FakePsProcess(7777, cmdline=["/opt/llama-swap"])
        _patch_psutil_process(monkeypatch, {7777: swap})  # owner pid 999 is gone
        stopped: list[sm._SwapState] = []
        monkeypatch.setattr(sm, "_stop_stale_swap", lambda state: stopped.append(state))
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        SwapManager(tmp_path).start([_launch(WorkerRole.CHAT)])
        assert [state.pid for state in stopped] == [7777]

    def test_zombie_owner_counts_as_dead(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_state(tmp_path, pid=7777, owner_pid=999)
        zombie = _FakePsProcess(999, cmdline=[], status=sm.psutil.STATUS_ZOMBIE)
        swap = _FakePsProcess(7777, cmdline=["/opt/llama-swap"])
        _patch_psutil_process(monkeypatch, {999: zombie, 7777: swap})
        stopped: list[sm._SwapState] = []
        monkeypatch.setattr(sm, "_stop_stale_swap", lambda state: stopped.append(state))
        SwapManager(tmp_path).reap_stale()
        assert [state.pid for state in stopped] == [7777]

    def test_owner_alive_false_for_missing_owner_pid(self) -> None:
        # Old-format state files carry no owner pid; reap as before.
        assert sm._owner_alive(None, None) is False

    def test_owner_alive_false_on_access_denied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The pid was reused by another user's process; our owner could be read.
        def _denied(pid: int) -> object:
            raise sm.psutil.AccessDenied(pid)

        monkeypatch.setattr(sm.psutil, "Process", _denied)
        assert sm._owner_alive(999, None) is False

    def test_owner_alive_false_when_create_time_differs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A live process at the owner pid born at a different time is pid reuse.
        impostor = _FakePsProcess(999, cmdline=["sleep"], create_time=5000.0)
        _patch_psutil_process(monkeypatch, {999: impostor})
        assert sm._owner_alive(999, 100.0) is False

    def test_owner_alive_true_when_create_time_matches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        owner = _FakePsProcess(999, cmdline=["lilbee"], create_time=100.0)
        _patch_psutil_process(monkeypatch, {999: owner})
        assert sm._owner_alive(999, 100.0) is True

    def test_owner_pid_reused_by_other_process_is_reaped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_state(
            tmp_path,
            pid=7777,
            owner_pid=999,
            owner_created_at=100.0,
            filename=sm._state_filename(999),
        )
        impostor = _FakePsProcess(999, cmdline=["sleep"], create_time=5000.0)
        swap = _FakePsProcess(7777, cmdline=["/opt/llama-swap"])
        _patch_psutil_process(monkeypatch, {999: impostor, 7777: swap})
        stopped: list[sm._SwapState] = []
        monkeypatch.setattr(sm, "_stop_stale_swap", lambda state: stopped.append(state))
        SwapManager(tmp_path).reap_stale()
        assert [state.pid for state in stopped] == [7777]
        assert not (tmp_path / sm._state_filename(999)).exists()

    def test_two_owners_coexist_without_clobbering_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A live owner's per-pid file survives a second instance's start.
        owner_a = _FakePsProcess(999, cmdline=["lilbee"], create_time=100.0)
        _patch_psutil_process(monkeypatch, {999: owner_a})
        a_path = _write_state(
            tmp_path,
            pid=7777,
            owner_pid=999,
            owner_created_at=100.0,
            filename=sm._state_filename(999),
        )
        original = a_path.read_text()
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        mgr_b = SwapManager(tmp_path)
        mgr_b.start([_launch(WorkerRole.CHAT)])
        assert a_path.read_text() == original  # A's record untouched
        assert _own_state_path(tmp_path).exists()  # B wrote its own
        mgr_b.shutdown()
        assert a_path.read_text() == original  # B's shutdown removes only B's file
        assert not _own_state_path(tmp_path).exists()

    def test_dead_owner_per_pid_state_file_is_reaped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state_path = _write_state(
            tmp_path, pid=7777, owner_pid=999, filename=sm._state_filename(999)
        )
        swap = _FakePsProcess(7777, cmdline=["/opt/llama-swap"])
        _patch_psutil_process(monkeypatch, {7777: swap})  # owner pid 999 is gone
        stopped: list[sm._SwapState] = []
        monkeypatch.setattr(sm, "_stop_stale_swap", lambda state: stopped.append(state))
        SwapManager(tmp_path).reap_stale()
        assert [state.pid for state in stopped] == [7777]
        assert not state_path.exists()

    def test_legacy_shared_state_file_is_reaped_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Pre-per-owner format: the single shared file is still scanned and reaped.
        legacy = _write_state(tmp_path, pid=7777, owner_pid=999)
        swap = _FakePsProcess(7777, cmdline=["/opt/llama-swap"])
        _patch_psutil_process(monkeypatch, {7777: swap})
        stopped: list[sm._SwapState] = []
        monkeypatch.setattr(sm, "_stop_stale_swap", lambda state: stopped.append(state))
        SwapManager(tmp_path).reap_stale()
        assert [state.pid for state in stopped] == [7777]
        assert not legacy.exists()
        SwapManager(tmp_path).reap_stale()
        assert len(stopped) == 1  # nothing left to reap a second time


class TestStopStaleSwap:
    def test_terms_the_recorded_process_group(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stale = _FakePsProcess(7777, cmdline=["/opt/llama-swap"])
        _patch_psutil_process(monkeypatch, {7777: stale})
        monkeypatch.setattr(sm.sys, "platform", "linux")
        monkeypatch.setattr(sm, "_live_children", lambda _pid: [])
        signals: list[tuple[int, int]] = []
        monkeypatch.setattr(
            sm.os, "killpg", lambda pgid, sig: signals.append((pgid, sig)), raising=False
        )
        sm._stop_stale_swap(
            sm._SwapState(pid=7777, pgid=8888, owner_pid=None, owner_created_at=None)
        )
        assert signals == [(8888, sm.signal.SIGTERM)]

    def test_escalates_to_sigkill_when_term_is_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stale = _FakePsProcess(7777, cmdline=["/opt/llama-swap"])
        stale.wait_raises = True
        _patch_psutil_process(monkeypatch, {7777: stale})
        monkeypatch.setattr(sm.sys, "platform", "linux")
        monkeypatch.setattr(sm, "_live_children", lambda _pid: [])
        signals: list[int] = []
        monkeypatch.setattr(sm.os, "killpg", lambda _pgid, sig: signals.append(sig), raising=False)
        waits: list[list[object]] = []

        def _wait_procs(procs: list[object], timeout: float) -> tuple[list, list]:
            waits.append(list(procs))
            return ([], [])

        monkeypatch.setattr(sm.psutil, "wait_procs", _wait_procs)
        sm._stop_stale_swap(
            sm._SwapState(pid=7777, pgid=8888, owner_pid=None, owner_created_at=None)
        )
        assert signals == [sm.signal.SIGTERM, sm._SIGKILL]
        # The KILLed swap is awaited so its VRAM is free before the next probe.
        assert [stale] in waits

    def test_signals_the_pid_when_no_pgid_recorded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stale = _FakePsProcess(7777, cmdline=["/opt/llama-swap"])
        _patch_psutil_process(monkeypatch, {7777: stale})
        monkeypatch.setattr(sm, "_live_children", lambda _pid: [])
        sm._stop_stale_swap(
            sm._SwapState(pid=7777, pgid=None, owner_pid=None, owner_created_at=None)
        )
        assert stale.signals == [sm.signal.SIGTERM]

    def test_noop_when_process_died_between_checks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_psutil_process(monkeypatch, {})
        monkeypatch.setattr(sm, "_live_children", lambda _pid: [])
        sm._stop_stale_swap(
            sm._SwapState(pid=7777, pgid=None, owner_pid=None, owner_created_at=None)
        )  # must not raise

    def test_reaps_surviving_servers_of_the_stale_swap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # llama-swap's servers live in their own process groups, so the group
        # kill misses them; the stale reap must sweep them like shutdown does.
        survivor = _FakeChild(running=True)
        stale = _FakePsProcess(7777, cmdline=["/opt/llama-swap"])
        _patch_psutil_process(monkeypatch, {7777: stale})
        monkeypatch.setattr(sm, "_live_children", lambda _pid: [survivor])
        monkeypatch.setattr(sm.psutil, "wait_procs", lambda procs, timeout: ([], []))
        sm._stop_stale_swap(
            sm._SwapState(pid=7777, pgid=None, owner_pid=None, owner_created_at=None)
        )
        assert survivor.terminated is True


class TestAtomicStateWrite:
    def test_write_state_lands_via_replace_with_no_tmp_leftovers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        replaced: list[str] = []
        real_replace = os.replace

        def _spy(src: object, dst: object) -> None:
            replaced.append(str(dst))
            real_replace(src, dst)

        monkeypatch.setattr(sm.os, "replace", _spy)
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        SwapManager(tmp_path).start([_launch(WorkerRole.CHAT)])
        assert replaced == [str(_own_state_path(tmp_path))]
        assert json.loads(_own_state_path(tmp_path).read_text())["pid"] == 4321
        assert [path for path in tmp_path.iterdir() if path.name.endswith(".tmp")] == []

    def test_half_written_tmp_file_is_invisible_to_the_reap_scan(self, tmp_path: Path) -> None:
        # A live writer's in-flight tmp file must never be parsed as a state
        # record nor removed; dead-writer leftovers are TestStaleTmpCleanup's.
        in_flight = tmp_path / f".llama-swap.state.{os.getpid()}.json.tmp"
        in_flight.write_text('{"pid":')
        SwapManager(tmp_path).reap_stale()
        assert in_flight.exists()


_PARENT_RAISES = "<parent-raises>"


class _FakeParentProc:
    """A stand-in psutil.Process parent for the orphan ownership guard."""

    def __init__(self, name: str) -> None:
        self._name = name

    def name(self) -> str:
        return self._name


class _FakeServerProc:
    """A stand-in psutil.Process for the orphan-server sweep."""

    def __init__(
        self,
        pid: int,
        cmdline: list[str],
        *,
        cmdline_raises: bool = False,
        parent_name: str | None = None,
    ) -> None:
        self.pid = pid
        self._cmdline = cmdline
        self._cmdline_raises = cmdline_raises
        self._parent_name = parent_name
        self.terminated = False
        self.killed = False

    def cmdline(self) -> list[str]:
        if self._cmdline_raises:
            raise sm.psutil.NoSuchProcess(self.pid)
        return self._cmdline

    def parent(self) -> _FakeParentProc | None:
        if self._parent_name == _PARENT_RAISES:
            raise sm.psutil.NoSuchProcess(self.pid)
        if self._parent_name is None:
            return None
        return _FakeParentProc(self._parent_name)

    def is_running(self) -> bool:
        return True

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


class TestOrphanServerReaping:
    def test_start_records_member_ports_in_the_state_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        SwapManager(tmp_path).start([_launch(WorkerRole.CHAT), _launch(WorkerRole.EMBED)])
        config = json.loads((tmp_path / "llama-swap.json").read_text())
        expected = sorted(
            int(entry["cmd"].rsplit(" ", 1)[-1]) for entry in config["models"].values()
        )
        assert json.loads(_own_state_path(tmp_path).read_text())["member_ports"] == expected

    def test_dead_swap_live_server_is_killed_by_name_and_port_match(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The servers outlive a SIGKILLed swap in their own process groups; the
        # sweep must stop exactly the ones on the recorded ports.
        _write_state(tmp_path, pid=7777, owner_pid=999, member_ports=[5001, 5002])
        _patch_psutil_process(monkeypatch, {})  # owner and swap are both gone
        orphan = _FakeServerProc(1, ["/opt/llama-server", "-m", "x.gguf", "--port", "5001"])
        recycled = _FakeServerProc(2, ["/usr/bin/python3", "serve.py", "--port", "5002"])
        other_port = _FakeServerProc(3, ["/opt/llama-server", "--port", "9999"])
        no_port = _FakeServerProc(4, ["/opt/llama-server"])
        vanished = _FakeServerProc(5, [], cmdline_raises=True)
        monkeypatch.setattr(
            sm.psutil,
            "process_iter",
            lambda: iter([orphan, recycled, other_port, no_port, vanished]),
        )
        monkeypatch.setattr(sm.psutil, "wait_procs", lambda procs, timeout: (list(procs), []))
        SwapManager(tmp_path).reap_stale()
        assert orphan.terminated is True
        assert recycled.terminated is False  # recycled port, foreign binary
        assert other_port.terminated is False
        assert no_port.terminated is False
        assert not (tmp_path / "llama-swap.state.json").exists()

    def test_server_with_a_live_swap_parent_is_spared(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A current run can reuse a stale record's port; its server still has a
        # live llama-swap parent, so the sweep must not touch it.
        _write_state(tmp_path, pid=7777, owner_pid=999, member_ports=[5001])
        _patch_psutil_process(monkeypatch, {})
        adopted = _FakeServerProc(
            1,
            ["/opt/llama-server", "--port", "5001"],
            parent_name="llama-swap",
        )
        monkeypatch.setattr(sm.psutil, "process_iter", lambda: iter([adopted]))
        monkeypatch.setattr(sm.psutil, "wait_procs", lambda procs, timeout: (list(procs), []))
        SwapManager(tmp_path).reap_stale()
        assert adopted.terminated is False
        assert not (tmp_path / "llama-swap.state.json").exists()

    def test_server_with_a_foreign_parent_is_still_reaped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_state(tmp_path, pid=7777, owner_pid=999, member_ports=[5001, 5002])
        _patch_psutil_process(monkeypatch, {})
        orphan = _FakeServerProc(
            1,
            ["/opt/llama-server", "--port", "5001"],
            parent_name="launchd",
        )
        parent_vanished = _FakeServerProc(
            2,
            ["/opt/llama-server", "--port", "5002"],
            parent_name=_PARENT_RAISES,
        )
        monkeypatch.setattr(sm.psutil, "process_iter", lambda: iter([orphan, parent_vanished]))
        monkeypatch.setattr(sm.psutil, "wait_procs", lambda procs, timeout: (list(procs), []))
        SwapManager(tmp_path).reap_stale()
        assert orphan.terminated is True
        assert parent_vanished.terminated is True

    def test_legacy_state_without_ports_sweeps_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_state(tmp_path, pid=7777, owner_pid=999)
        _patch_psutil_process(monkeypatch, {})

        def _forbidden() -> object:
            raise AssertionError("process_iter must not run without recorded ports")

        monkeypatch.setattr(sm.psutil, "process_iter", _forbidden)
        SwapManager(tmp_path).reap_stale()
        assert not (tmp_path / "llama-swap.state.json").exists()


class TestStaleTmpCleanup:
    def test_dead_writers_tmp_file_is_removed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stale = tmp_path / ".llama-swap.state.424242.json.tmp"
        stale.write_text("{partial")
        monkeypatch.setattr(sm.psutil, "pid_exists", lambda pid: False)
        SwapManager(tmp_path).reap_stale()
        assert not stale.exists()

    def test_live_writers_tmp_file_is_kept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        in_flight = tmp_path / f".llama-swap.state.{os.getpid()}.json.tmp"
        in_flight.write_text("{partial")
        monkeypatch.setattr(sm.psutil, "pid_exists", lambda pid: True)
        SwapManager(tmp_path).reap_stale()
        assert in_flight.exists()

    def test_tmp_file_without_a_pid_is_kept(self, tmp_path: Path) -> None:
        legacy = tmp_path / ".llama-swap.state.json.tmp"
        legacy.write_text("{partial")
        SwapManager(tmp_path).reap_stale()
        assert legacy.exists()


def test_state_owner_pid_parses_state_and_tmp_names() -> None:
    assert sm._state_owner_pid("llama-swap.state.123.json") == 123
    assert sm._state_owner_pid(".llama-swap.state.456.json.tmp") == 456
    assert sm._state_owner_pid("llama-swap.state.json") is None


def test_write_state_is_noop_without_a_process(tmp_path: Path) -> None:
    mgr = SwapManager(tmp_path)
    mgr._write_state()
    assert not _own_state_path(tmp_path).exists()


def test_running_reflects_the_spawned_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
    _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
    mgr = SwapManager(tmp_path)
    assert mgr.running is False
    mgr.start([_launch(WorkerRole.CHAT)])
    assert mgr.running is True
    mgr.shutdown()
    assert mgr.running is False


class TestUnload:
    def test_posts_model_id_to_unload_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mgr = SwapManager(tmp_path)
        mgr._port = 41999
        calls: dict[str, object] = {}

        def fake_post(url: str, json: object, timeout: float) -> object:
            calls["url"] = url
            calls["json"] = json
            return _fake_response(status=200)

        monkeypatch.setattr("lilbee.providers.fleet.swap_manager.httpx.post", fake_post)
        assert mgr.unload("embed-1") is True
        assert calls["url"] == f"http://127.0.0.1:41999{_UNLOAD_PATH}"
        assert calls["json"] == {"model": "embed-1"}

    def test_returns_false_and_never_raises_on_network_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mgr = SwapManager(tmp_path)
        mgr._port = 41999

        def boom(url: str, json: object, timeout: float) -> object:
            raise OSError("connection refused")

        monkeypatch.setattr("lilbee.providers.fleet.swap_manager.httpx.post", boom)
        assert mgr.unload("embed-1") is False

    def test_returns_false_on_http_error_status(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mgr = SwapManager(tmp_path)
        mgr._port = 41999
        monkeypatch.setattr(
            "lilbee.providers.fleet.swap_manager.httpx.post",
            lambda url, json, timeout: _fake_response(status=500),
        )
        assert mgr.unload("embed-1") is False

    def test_returns_false_and_never_raises_before_start(self, tmp_path: Path) -> None:
        # _port is None before start(); unload() must not let endpoint()'s
        # ProviderError escape the never-raise contract.
        mgr = SwapManager(tmp_path)
        assert mgr._port is None
        assert mgr.unload("embed-1") is False


class TestIsLive:
    def test_true_when_proc_alive_and_running_answers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mgr = SwapManager(tmp_path)
        mgr._port = 41999
        mgr._proc = _FakeProc(poll_result=None)  # type: ignore[assignment]
        monkeypatch.setattr(
            "lilbee.providers.fleet.swap_manager.httpx.get",
            lambda url, timeout: _fake_response(status=200),
        )
        assert mgr.is_live() is True

    def test_false_when_proc_is_none(self, tmp_path: Path) -> None:
        mgr = SwapManager(tmp_path)
        assert mgr._proc is None
        assert mgr.is_live() is False

    def test_false_when_proc_has_exited(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mgr = SwapManager(tmp_path)
        mgr._port = 41999
        mgr._proc = _FakeProc(poll_result=1)  # type: ignore[assignment]
        assert mgr.is_live() is False

    def test_false_when_running_probe_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mgr = SwapManager(tmp_path)
        mgr._port = 41999
        mgr._proc = _FakeProc(poll_result=None)  # type: ignore[assignment]

        def boom(url: str, timeout: float) -> object:
            raise OSError("connection refused")

        monkeypatch.setattr("lilbee.providers.fleet.swap_manager.httpx.get", boom)
        assert mgr.is_live() is False

    def test_false_and_no_raise_when_proc_alive_but_port_not_yet_set(self, tmp_path: Path) -> None:
        # Startup-window race: _proc is alive (poll() returns None) but _port is
        # still None. is_live() must return False, not let endpoint()'s
        # ProviderError escape a -> bool method.
        mgr = SwapManager(tmp_path)
        mgr._proc = _FakeProc(poll_result=None)  # type: ignore[assignment]
        assert mgr._port is None
        result = mgr.is_live()
        assert result is False
