"""The plumbing that binds a spawned llama-swap to this process.

The kernel-level behaviour (PR_SET_PDEATHSIG firing on Linux, a job object killing
its members on Windows) can only be verified on those platforms and is covered in
CI, not here. What is verified here is platform dispatch, the process-lifetime
spawner thread, and that every failure path falls back to a plain spawn so a
binding problem never fails a launch.
"""

from __future__ import annotations

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

        result = child_guard.spawn_llama_swap(["/x/llama-swap"], stdout=7)

        assert result == "proc"
        assert captured["kwargs"]["preexec_fn"] is child_guard._set_pdeathsig
        assert captured["kwargs"]["stdout"] == 7

    def test_macos_spawns_plainly_with_no_binding(self, monkeypatch):
        monkeypatch.setattr(child_guard.sys, "platform", "darwin")
        with mock.patch.object(child_guard.subprocess, "Popen", return_value="proc") as popen:
            result = child_guard.spawn_llama_swap(["/x/llama-swap"], stdout=7)

        assert result == "proc"
        # No death-signal preexec, no job object: the reap is the macOS backstop.
        assert "preexec_fn" not in popen.call_args.kwargs

    def test_windows_assigns_the_child_to_a_kill_on_close_job(self, monkeypatch):
        monkeypatch.setattr(child_guard.sys, "platform", "win32")
        proc = mock.MagicMock(pid=4321)
        assigned: list[int] = []
        monkeypatch.setattr(child_guard, "_assign_to_kill_on_close_job", assigned.append)
        with mock.patch.object(child_guard.subprocess, "Popen", return_value=proc):
            result = child_guard.spawn_llama_swap(["/x/llama-swap"])

        assert result is proc
        assert assigned == [4321]


class TestFailureNeverFailsASpawn:
    def test_a_windows_job_failure_still_returns_the_child(self, monkeypatch):
        monkeypatch.setattr(child_guard.sys, "platform", "win32")
        proc = mock.MagicMock(pid=1)

        def _boom(_pid):
            raise OSError("no job object here")

        monkeypatch.setattr(child_guard, "_assign_to_kill_on_close_job", _boom)
        with mock.patch.object(child_guard.subprocess, "Popen", return_value=proc):
            result = child_guard.spawn_llama_swap(["/x/llama-swap"])

        assert result is proc  # spawned anyway; the next launch reaps it

    def test_a_linux_spawn_error_falls_back_to_a_plain_spawn(self, monkeypatch):
        monkeypatch.setattr(child_guard, "_libc", mock.MagicMock())

        def _spawn(_argv, **kwargs):
            # The binding attempt carries the death-signal preexec; the retry does not.
            if "preexec_fn" in kwargs:
                raise OSError("preexec rejected")
            return "plain"

        monkeypatch.setattr(child_guard._spawner, "spawn", _spawn)

        assert child_guard.spawn_llama_swap(["/x/llama-swap"]) == "plain"


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
