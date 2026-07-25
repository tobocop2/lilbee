"""The bounded-reap subprocess runner shared by the device probe and gguf-parser."""

from __future__ import annotations

import os
import subprocess
import sys
from unittest.mock import MagicMock

import pytest

from lilbee.providers.fleet import proc


def test_returns_stdout_and_returncode() -> None:
    out, rc = proc.run_bounded([sys.executable, "-c", "print('hi')"], timeout_s=10, kill_wait_s=2)
    assert out == "hi\n"
    assert rc == 0


def test_default_discards_stderr_but_merge_folds_it() -> None:
    argv = [sys.executable, "-c", "import sys; sys.stderr.write('err'); print('out')"]
    out, _ = proc.run_bounded(argv, timeout_s=10, kill_wait_s=2)
    assert out == "out\n"  # stderr discarded
    merged, _ = proc.run_bounded(argv, timeout_s=10, kill_wait_s=2, merge_stderr=True)
    assert "err" in merged and "out" in merged


@pytest.mark.timeout(10)
def test_a_timed_out_child_is_killed_and_raises() -> None:
    """A sleeping child is killed on timeout, so the call returns instead of hanging.

    The @timeout guard fails the test if the reap were unbounded (the 30s sleep).
    """
    argv = [sys.executable, "-c", "import time; time.sleep(30)"]
    with pytest.raises(subprocess.TimeoutExpired):
        proc.run_bounded(argv, timeout_s=0.5, kill_wait_s=3)


def test_an_unkillable_child_is_abandoned_with_a_warning(monkeypatch, caplog) -> None:
    """If the killed child never exits, run_bounded gives up after kill_wait_s."""
    fake = MagicMock()
    fake.pid = 424242
    # The run times out; the post-kill reap also times out, standing in for a child
    # wedged in uninterruptible I/O that ignores SIGKILL.
    fake.communicate.side_effect = [
        subprocess.TimeoutExpired(cmd="x", timeout=0.1),
        subprocess.TimeoutExpired(cmd="x", timeout=0.1),
    ]
    monkeypatch.setattr(proc.subprocess, "Popen", lambda *a, **k: fake)
    # Force the POSIX group-kill path so the same lines run on every OS. os.killpg
    # and signal.SIGKILL are absent on Windows, so seed them with raising=False.
    monkeypatch.setattr(proc.os, "name", "posix")
    monkeypatch.setattr(proc.os, "killpg", lambda *a: None, raising=False)
    monkeypatch.setattr(proc.signal, "SIGKILL", 9, raising=False)
    with pytest.raises(subprocess.TimeoutExpired), caplog.at_level("WARNING"):
        proc.run_bounded(["stuck"], timeout_s=0.1, kill_wait_s=0.1, label="stuck")
    assert any("abandoned" in r.message for r in caplog.records)


def test_a_ctrl_c_kills_the_child_not_just_a_timeout(monkeypatch) -> None:
    """A KeyboardInterrupt must kill the child, which has its own session.

    The child writes no state file, so no later reap can match it; without this
    path a Ctrl-C during startup strands it holding a device context.
    """
    fake = MagicMock()
    fake.pid = 4242
    fake.communicate.side_effect = KeyboardInterrupt
    monkeypatch.setattr(proc.subprocess, "Popen", lambda *a, **k: fake)
    killed: list[object] = []
    monkeypatch.setattr(proc, "_abandon_group", lambda p, *_a: killed.append(p))
    with pytest.raises(KeyboardInterrupt):
        proc.run_bounded(["probe"], timeout_s=10, kill_wait_s=1)
    assert killed == [fake]


def _fake_spawn(record: dict[str, object]):
    def _spy(argv: list[str], **kwargs: object) -> MagicMock:
        record.update(kwargs)
        fake = MagicMock()
        fake.communicate.return_value = ("", None)
        fake.returncode = 0
        return fake

    return _spy


def test_binding_is_opt_in_so_a_polled_probe_leaks_no_job_handle(monkeypatch) -> None:
    """Default spawns must not be bound.

    The Windows binding leaks a job-object handle per spawn by design. The util
    sampler runs about once a second, so binding it by default would leak a
    handle per sample for the life of the process.
    """
    bound: dict[str, object] = {}
    monkeypatch.setattr(proc, "spawn_bound_child", _fake_spawn(bound))
    plain: dict[str, object] = {}
    monkeypatch.setattr(proc.subprocess, "Popen", _fake_spawn(plain))
    proc.run_bounded(["smi"], timeout_s=10, kill_wait_s=1)
    assert bound == {}  # spawn_bound_child never reached
    assert plain["start_new_session"] is (os.name == "posix")


def test_an_opted_in_child_is_bound_without_a_death_pipe(monkeypatch) -> None:
    """The device probe holds a GPU context, so it binds; a seconds-long child
    needs no death-pipe watcher, which would outlive it."""
    seen: dict[str, object] = {}
    monkeypatch.setattr(proc, "spawn_bound_child", _fake_spawn(seen))
    proc.run_bounded(["probe"], timeout_s=10, kill_wait_s=1, bind_lifetime=True)
    assert seen["death_pipe"] is False
