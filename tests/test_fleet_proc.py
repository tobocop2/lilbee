"""The bounded-reap subprocess runner shared by the device probe and gguf-parser."""

from __future__ import annotations

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
