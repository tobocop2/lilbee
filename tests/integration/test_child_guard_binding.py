"""Real, per-platform verification that child_guard binds a spawned child's lifetime.

The unit tests in tests/test_child_guard.py mock the kernel calls, so they prove the
dispatch and fallbacks but not that the binding fires. These do the real thing on
each OS: a middle process spawns a child through the actual spawn_llama_swap path,
the middle is killed, and the child's fate is checked against the platform contract.

  Linux:   PR_SET_PDEATHSIG -> the child dies with its parent.
  Windows: kill-on-close job object -> the child dies with its parent.
  macOS:   no primitive -> the child survives; the stale-engine reap is the backstop.

Run on CI's real ubuntu/windows/macos integration jobs (make test-integration).
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
import time

import psutil
import pytest

pytestmark = pytest.mark.slow

# The middle process spawns a long-lived child through the real binding path and
# prints its pid. A bare python sleeper is portable and dies on SIGTERM (Linux) or
# a job-close TerminateProcess (Windows).
_CHILD = [sys.executable, "-c", "import time; time.sleep(600)"]
_MIDDLE_SRC = (
    "import sys, time\n"
    "from lilbee.providers.fleet.child_guard import spawn_llama_swap\n"
    f"proc = spawn_llama_swap({_CHILD!r}, start_new_session=(sys.platform != 'win32'))\n"
    "print(proc.pid, flush=True)\n"
    "time.sleep(600)\n"
)


def _spawn_child_via_middle() -> tuple[subprocess.Popen[bytes], int]:
    """Start a middle process that spawns a bound child; return (middle, child pid).

    Kills the middle on any failure to read the pid so a broken spawn never leaks it.
    """
    middle = subprocess.Popen([sys.executable, "-c", _MIDDLE_SRC], stdout=subprocess.PIPE)
    try:
        assert middle.stdout is not None
        return middle, int(middle.stdout.readline().decode().strip())
    except BaseException:
        _kill(middle)
        raise


def _child_gone_within(pid: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not psutil.pid_exists(pid):
            return True
        time.sleep(0.1)
    return not psutil.pid_exists(pid)


def _kill(proc_or_pid: subprocess.Popen[bytes] | int) -> None:
    """Best-effort cleanup so a survived child never leaks out of the test."""
    pid = proc_or_pid.pid if isinstance(proc_or_pid, subprocess.Popen) else proc_or_pid
    with contextlib.suppress(psutil.NoSuchProcess):
        psutil.Process(pid).kill()


@pytest.mark.timeout(60)
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG is Linux-only")
def test_pdeathsig_reaps_the_child_when_the_parent_dies() -> None:
    middle, child_pid = _spawn_child_via_middle()
    try:
        assert psutil.pid_exists(child_pid), "child was not running before the kill"
        middle.kill()  # simulate a lilbee crash
        middle.wait(timeout=10)
        assert _child_gone_within(child_pid, 10), "PR_SET_PDEATHSIG did not reap the child"
    finally:
        _kill(child_pid)
        _kill(middle)


@pytest.mark.timeout(60)
@pytest.mark.skipif(sys.platform != "win32", reason="job objects are Windows-only")
def test_job_object_kills_the_child_when_the_parent_exits() -> None:
    middle, child_pid = _spawn_child_via_middle()
    try:
        assert psutil.pid_exists(child_pid), "child was not running before the kill"
        middle.kill()  # closing the last job handle must terminate the child
        middle.wait(timeout=10)
        assert _child_gone_within(child_pid, 10), "kill-on-close job did not reap the child"
    finally:
        _kill(child_pid)
        _kill(middle)


@pytest.mark.timeout(60)
@pytest.mark.skipif(sys.platform != "darwin", reason="documents the macOS no-binding contract")
def test_macos_child_survives_and_relies_on_the_reap() -> None:
    middle, child_pid = _spawn_child_via_middle()
    try:
        assert psutil.pid_exists(child_pid), "child was not running before the kill"
        middle.kill()
        middle.wait(timeout=10)
        # macOS has no death-binding primitive, so the child outlives its parent;
        # the stale-engine reap on the next launch is what reclaims it.
        assert not _child_gone_within(child_pid, 2), "child unexpectedly died without a binding"
    finally:
        _kill(child_pid)
        _kill(middle)
