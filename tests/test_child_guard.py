"""The plumbing that binds a spawned llama-swap to this process.

The kernel-level behaviour (PR_SET_PDEATHSIG firing on Linux, a job object killing
its members on Windows) can only be verified on those platforms and is covered in
CI, not here. What is verified here is platform dispatch, the process-lifetime
spawner thread, and that every failure path falls back to a plain spawn so a
binding problem never fails a launch.
"""

from __future__ import annotations

import os
import subprocess
import sys
from unittest import mock

import pytest

from lilbee.providers.fleet import child_guard


@pytest.fixture(autouse=True)
def _pin_libc(monkeypatch):
    """Pin the death-signal libc to absent by default.

    ``_libc`` is resolved from the real host at import (a live handle on Linux
    CI, None on macOS/Windows), so a dispatch test that does not set it would
    branch differently per platform. Default it to None; the Linux tests opt in.
    """
    monkeypatch.setattr(child_guard, "_libc", None)


@pytest.fixture(autouse=True)
def _close_module_spawner():
    """Stop any executor the module singleton lazily started, so no worker lingers."""
    yield
    child_guard._spawner.close()


class TestPlatformDispatch:
    def test_linux_spawns_through_the_lifetime_thread_with_the_death_signal(self, monkeypatch):
        # The Linux path is gated on libc having resolved at import, not on the
        # platform string, so a fork never dlopen's; stand libc up for the test.
        monkeypatch.setattr(child_guard, "_libc", mock.MagicMock())
        captured: dict[str, object] = {}

        def _spawn(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return "proc"

        monkeypatch.setattr(child_guard._spawner, "spawn", _spawn)

        result = child_guard.spawn_bound_child(["/x/llama-swap"], stdout=7)

        assert result == "proc"
        assert callable(captured["kwargs"]["preexec_fn"])  # the death-signal preexec
        assert captured["kwargs"]["stdout"] == 7

    def test_keep_warm_spawns_plainly_so_the_engine_outlives_the_process(self, monkeypatch):
        # bind_lifetime=False is the keep_engine_warm path: no death binding on any
        # platform, so a clean exit leaves the warm engine running.
        monkeypatch.setattr(child_guard, "_libc", mock.MagicMock())
        monkeypatch.setattr(child_guard.sys, "platform", "win32")
        assigned: list[int] = []
        monkeypatch.setattr(child_guard, "_assign_to_kill_on_close_job", assigned.append)
        captured: dict[str, object] = {}

        def _spawn(*args, **kwargs):
            captured["kwargs"] = kwargs
            return mock.MagicMock(pid=99)

        monkeypatch.setattr(child_guard._spawner, "spawn", _spawn)

        child_guard.spawn_bound_child(["/x/llama-swap"], bind_lifetime=False)

        assert "preexec_fn" not in captured["kwargs"]
        assert assigned == []  # no job object either

    def test_macos_binds_through_a_death_pipe(self, monkeypatch):
        # No prctl and no job object here, so the portable pipe-EOF watcher is
        # what makes a SIGKILLed lilbee take its engine with it.
        monkeypatch.setattr(child_guard.sys, "platform", "darwin")
        proc = mock.MagicMock(pid=4321)
        watched: list[int] = []
        monkeypatch.setattr(child_guard, "_watch_via_death_pipe", watched.append)
        with mock.patch.object(child_guard.subprocess, "Popen", return_value=proc) as popen:
            result = child_guard.spawn_bound_child(["/x/llama-swap"], stdout=7)

        assert result is proc
        assert watched == [4321]
        assert "preexec_fn" not in popen.call_args.kwargs

    def test_keep_warm_on_macos_gets_no_death_pipe(self, monkeypatch):
        # The whole point of keep_engine_warm is outliving this process, so the
        # portable binding must be skipped exactly like the kernel ones.
        monkeypatch.setattr(child_guard.sys, "platform", "darwin")
        watched: list[int] = []
        monkeypatch.setattr(child_guard, "_watch_via_death_pipe", watched.append)
        with mock.patch.object(child_guard.subprocess, "Popen", return_value=mock.MagicMock()):
            child_guard.spawn_bound_child(["/x/llama-swap"], bind_lifetime=False)

        assert watched == []

    def test_a_failed_death_signal_spawn_falls_back_to_the_death_pipe(self, monkeypatch):
        # Losing prctl must not drop the binding entirely where a pipe still works.
        monkeypatch.setattr(child_guard, "_libc", mock.MagicMock())
        monkeypatch.setattr(child_guard.sys, "platform", "linux")
        proc = mock.MagicMock(pid=77)
        watched: list[int] = []
        monkeypatch.setattr(child_guard, "_watch_via_death_pipe", watched.append)
        calls: list[dict] = []

        def _spawn(*args, **kwargs):
            calls.append(kwargs)
            if "preexec_fn" in kwargs:
                raise OSError("preexec unavailable")
            return proc

        monkeypatch.setattr(child_guard._spawner, "spawn", _spawn)

        assert child_guard.spawn_bound_child(["/x/llama-swap"]) is proc
        assert watched == [77]

    def test_windows_assigns_the_child_to_a_kill_on_close_job(self, monkeypatch):
        monkeypatch.setattr(child_guard.sys, "platform", "win32")
        proc = mock.MagicMock(pid=4321)
        assigned: list[int] = []
        monkeypatch.setattr(child_guard, "_assign_to_kill_on_close_job", assigned.append)
        with mock.patch.object(child_guard.subprocess, "Popen", return_value=proc):
            result = child_guard.spawn_bound_child(["/x/llama-swap"])

        assert result is proc
        assert assigned == [4321]


class _Exited(Exception):
    """Stand-in for os._exit, which never returns, so a test can observe the exit."""


def _raise_exit(code: int) -> None:
    raise _Exited(code)


class TestTheDeathSignalPreexec:
    """The post-fork preexec: bind to the parent's death, but not to a live pid-1 parent."""

    def _arm(self, monkeypatch, *, getppid: int) -> None:
        monkeypatch.setattr(child_guard, "_libc", mock.MagicMock())
        monkeypatch.setattr(child_guard.os, "getppid", lambda: getppid)
        monkeypatch.setattr(child_guard.os, "_exit", _raise_exit)

    def test_absent_libc_yields_no_preexec(self, monkeypatch):
        monkeypatch.setattr(child_guard, "_libc", None)

        assert child_guard._make_pdeathsig_preexec(1234) is None

    def test_a_reparented_child_exits(self, monkeypatch):
        self._arm(monkeypatch, getppid=1)

        preexec = child_guard._make_pdeathsig_preexec(parent_pid=1234)
        with pytest.raises(_Exited) as exc:
            preexec()

        assert exc.value.args[0] == 1
        child_guard._libc.prctl.assert_called_once()

    def test_a_child_whose_parent_is_alive_stays(self, monkeypatch):
        self._arm(monkeypatch, getppid=1234)

        child_guard._make_pdeathsig_preexec(parent_pid=1234)()  # parent alive: must not exit

    def test_a_legitimate_pid_1_parent_is_not_mistaken_for_a_dead_one(self, monkeypatch):
        # lilbee running as pid 1 (a container with no init shim): the child's parent
        # is 1 and alive, so the old getppid()==1 check would wrongly kill the engine.
        self._arm(monkeypatch, getppid=1)

        child_guard._make_pdeathsig_preexec(parent_pid=1)()  # must not exit


class TestFailureNeverFailsASpawn:
    def test_a_windows_job_failure_still_returns_the_child(self, monkeypatch):
        monkeypatch.setattr(child_guard.sys, "platform", "win32")
        proc = mock.MagicMock(pid=1)

        def _boom(_pid):
            raise OSError("no job object here")

        monkeypatch.setattr(child_guard, "_assign_to_kill_on_close_job", _boom)
        with mock.patch.object(child_guard.subprocess, "Popen", return_value=proc):
            result = child_guard.spawn_bound_child(["/x/llama-swap"])

        assert result is proc  # spawned anyway; the next launch reaps it

    def test_a_linux_spawn_error_still_returns_the_child(self, monkeypatch):
        monkeypatch.setattr(child_guard, "_libc", mock.MagicMock())
        proc = mock.MagicMock(pid=55)
        monkeypatch.setattr(child_guard, "_watch_via_death_pipe", lambda _pid: None)

        def _spawn(_argv, **kwargs):
            # The binding attempt carries the death-signal preexec; the retry does not.
            if "preexec_fn" in kwargs:
                raise OSError("preexec rejected")
            return proc

        monkeypatch.setattr(child_guard._spawner, "spawn", _spawn)

        assert child_guard.spawn_bound_child(["/x/llama-swap"]) is proc


@pytest.fixture
def spawner():
    """A spawner whose worker thread is stopped after the test so it does not linger."""
    s = child_guard._LifetimeSpawner()
    yield s
    s.close()


class TestTheLifetimeSpawner:
    def test_it_returns_the_spawned_process(self, spawner):
        with mock.patch.object(child_guard.subprocess, "Popen", return_value="proc") as popen:
            result = spawner.spawn(["/x/llama-swap"], stdout=9)

        assert result == "proc"
        assert popen.call_args.args == (["/x/llama-swap"],)
        assert popen.call_args.kwargs == {"stdout": 9}

    def test_a_spawn_error_reaches_the_caller(self, spawner):
        with (
            mock.patch.object(child_guard.subprocess, "Popen", side_effect=OSError("nope")),
            pytest.raises(OSError, match="nope"),
        ):
            spawner.spawn(["/x/llama-swap"])

    def test_the_worker_is_created_once_and_reused(self, spawner):
        with mock.patch.object(child_guard.subprocess, "Popen", return_value="proc"):
            spawner.spawn(["/x"])
            first = spawner._executor
            spawner.spawn(["/x"])

        assert first is spawner._executor and first is not None

    def test_close_stops_the_worker_and_is_safe_when_never_started(self, spawner):
        spawner.close()  # never started: no-op
        with mock.patch.object(child_guard.subprocess, "Popen", return_value="proc"):
            spawner.spawn(["/x"])
        assert spawner._executor is not None
        spawner.close()
        assert spawner._executor is None


@pytest.mark.skipif(sys.platform == "win32", reason="the death pipe is the POSIX fallback")
class TestTheDeathPipe:
    def test_the_watcher_signals_the_child_when_the_write_end_closes(self, monkeypatch):
        """The real thing: a live sleeper dies once the pipe's write end is closed.

        Closing the write end by hand is exactly what the kernel does when this
        process dies, which is the case that cannot be staged from inside a test.
        """
        held: list[int] = []
        monkeypatch.setattr(child_guard, "_death_pipe_write_fds", held)
        victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            child_guard._watch_via_death_pipe(victim.pid)
            assert held, "no write end was retained"
            assert victim.poll() is None, "victim died before the pipe closed"

            os.close(held[0])

            victim.wait(timeout=15)
        finally:
            if victim.poll() is None:
                victim.kill()
                victim.wait(timeout=10)

    def test_the_write_end_is_held_so_only_this_process_death_closes_it(self, monkeypatch):
        """A dropped reference would close the pipe early and kill a healthy engine."""
        held: list[int] = []
        monkeypatch.setattr(child_guard, "_death_pipe_write_fds", held)
        monkeypatch.setattr(child_guard._spawner, "spawn", lambda *a, **k: mock.MagicMock())

        child_guard._watch_via_death_pipe(1234)

        assert len(held) == 1
        os.fstat(held[0])  # raises if it was closed
        os.close(held[0])

    def test_a_watcher_that_cannot_take_the_read_end_never_signals(self, monkeypatch):
        """A broken binding must degrade to no binding, not to killing the child.

        The first cut dup2'd the read end in a preexec_fn, which subprocess then
        closed (it closes fds *after* preexec), so the watcher's redirect failed
        and it fell straight through to the kill: every bound child died within
        milliseconds of being spawned.
        """
        held: list[int] = []
        monkeypatch.setattr(child_guard, "_death_pipe_write_fds", held)
        victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
        try:
            # pass_fds dropped, so the inherited read end is not there to redirect.
            real_spawn = child_guard._spawner.spawn
            monkeypatch.setattr(
                child_guard._spawner,
                "spawn",
                lambda argv, **kw: real_spawn(argv, **{**kw, "pass_fds": ()}),
            )

            child_guard._watch_via_death_pipe(victim.pid)
            for fd in held:
                os.close(fd)

            with pytest.raises(subprocess.TimeoutExpired):
                victim.wait(timeout=3)
        finally:
            victim.kill()
            victim.wait(timeout=10)

    def test_a_watcher_that_cannot_start_leaks_no_fds(self, monkeypatch):
        """The reap is the fallback, but a leaked fd per failed spawn is not."""
        held: list[int] = []
        monkeypatch.setattr(child_guard, "_death_pipe_write_fds", held)
        monkeypatch.setattr(
            child_guard._spawner, "spawn", mock.MagicMock(side_effect=OSError("no sh"))
        )
        before = len(os.listdir("/dev/fd"))

        child_guard._watch_via_death_pipe(1234)

        assert held == []
        assert len(os.listdir("/dev/fd")) <= before
