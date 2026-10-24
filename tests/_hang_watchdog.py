"""Dump every thread's stack when a test wedges, before any timeout ends it.

Two timers guard each test, both armed by ``LILBEE_TEST_HANG_DUMP_S``.

The early dump is a Python timer at 75% of the test's pytest-timeout. It writes
every thread's stack, the open file descriptors with their targets and the
POSIX locks the process holds, to the dump file and to the real stderr, so the
text reaches the job log while the test is still stuck. pytest-timeout ends a
wedged test at its limit and the CI job can be cancelled soon after, so a dump
that waits for a later limit never lands (the 90 s limit this file once used
sat above the 60 s pytest-timeout and never fired). The early dump does not
stop the test; pytest-timeout still reports it.

The backstop is ``faulthandler.dump_traceback_later``: a C thread that walks
the interpreter's thread states without the GIL. If a native call holds the GIL
(seen on Windows py3.13), no Python timer runs, pytest-timeout cannot fire, and
the worker hangs until the job's ``timeout-minutes``. The backstop dumps the
stacks and ``_exit``\\s the worker so the run fails fast.

Both timers guard a test only from its setup to its teardown, and only in
the process that runs it. Under xdist the controller runs no test, so a run
that stalls outside a test's window, for example while a session fixture
tears down after the last test, hangs until ``timeout-minutes`` with every
dump file empty. The controller therefore re-arms a stall timer on every test
report it receives. When no report arrives for ``_STALL_FACTOR`` times the
backstop, it signals each worker to dump its stacks to its own file, then
dumps its own stacks and ``_exit``\\s, so the run fails with every stack.

Unset or ``0`` is a no-op, so local runs are untouched. Dumps go to
``LILBEE_TEST_HANG_DUMP_DIR/hang-<workerid>.txt`` (default: the pytest rootdir)
for a CI step to print after the test step. The next test's setup truncates
the file, so the copy on stderr is the one that lasts.
"""

from __future__ import annotations

import faulthandler
import os
import signal
import sys
import threading
from pathlib import Path
from typing import TextIO

import psutil
import pytest

_ENV_SECONDS = "LILBEE_TEST_HANG_DUMP_S"
_ENV_DIR = "LILBEE_TEST_HANG_DUMP_DIR"

# The early dump fires at this share of the test's own timeout.
_EARLY_SHARE = 0.75
_FD_SCAN_LIMIT = 1024
# The controller stall limit, as a multiple of the per-test backstop.
_STALL_FACTOR = 4
# The workers dump before the controller exits, so their stacks are on disk.
_STALL_EXIT_GRACE_S = 5.0
# A worker dumps its stacks on this signal. Windows has no SIGUSR1.
_WORKER_DUMP_SIGNAL: signal.Signals | None = getattr(signal, "SIGUSR1", None)
_XDIST_CONTROLLER_PLUGIN = "dsession"

# The open dump-file handle. faulthandler writes to its raw fd from a C thread,
# so it must outlive every test; keep a module reference so it is not closed.
_dump_handle: TextIO | None = None
_timeout_s = 0.0
_early_timer: threading.Timer | None = None
_early_fired = threading.Event()
_stall_s = 0.0
_stall_timer: threading.Timer | None = None


def _worker_id(config: pytest.Config) -> str:
    """xdist worker id (``gw0``), or ``master`` when running single-process."""
    return getattr(config, "workerinput", {}).get("workerid", "master")


def _test_timeout_s(item: pytest.Item) -> float:
    """The pytest-timeout limit for this test, or 0 when it has none."""
    marker = item.get_closest_marker("timeout")
    if marker is not None:
        raw = marker.args[0] if marker.args else marker.kwargs.get("timeout")
    else:
        raw = item.config.getoption("timeout", None) or item.config.getini("timeout")
    try:
        return float(raw or 0)
    except (TypeError, ValueError):
        return 0.0


def _fd_lines() -> list[str]:
    """One line per open descriptor: number and what it points at."""
    lines = []
    for fd in range(_FD_SCAN_LIMIT):
        try:
            os.fstat(fd)
        except OSError:
            continue
        try:
            target = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            target = _fd_path_fallback(fd)
        lines.append(f"  fd {fd}: {target}")
    return lines


def _fd_path_fallback(fd: int) -> str:
    try:
        import fcntl

        return os.fsdecode(fcntl.fcntl(fd, getattr(fcntl, "F_GETPATH", 50), b"\0" * 1024)).split(
            "\0", 1
        )[0]
    except (ImportError, OSError, ValueError):
        return "(path unknown)"


def _lock_lines() -> list[str]:
    """POSIX and flock locks held by this process (Linux; empty elsewhere)."""
    try:
        text = Path("/proc/locks").read_text(encoding="utf-8")
    except OSError:
        return []
    pid = f" {os.getpid()} "
    return [f"  {line}" for line in text.splitlines() if pid in line]


def _write_state(out: TextIO, nodeid: str, delay: float) -> None:
    out.write(f"hang dump: {nodeid} still running after {delay:.0f}s\n")
    out.write("open file descriptors:\n" + "\n".join(_fd_lines()) + "\n")
    out.write("locks held by this process:\n" + "\n".join(_lock_lines() or ["  none"]) + "\n")
    out.write("all thread stacks:\n")
    out.flush()
    faulthandler.dump_traceback(file=out, all_threads=True)
    out.flush()


def _early_dump(config: pytest.Config, nodeid: str, delay: float) -> None:
    handle = _dump_handle
    if handle is None:
        return
    _early_fired.set()
    _write_state(handle, nodeid, delay)
    # Capture would swallow stderr; suspend it so the text reaches the job log.
    capman = config.pluginmanager.getplugin("capturemanager")
    try:
        if capman is None:
            _write_state(sys.stderr, nodeid, delay)
        else:
            with capman.global_and_fixture_disabled():
                _write_state(sys.stderr, nodeid, delay)
    except (OSError, ValueError):
        pass  # The file already holds the dump; a closed stderr must not mask the hang.


def _signal_workers(last_report: str) -> None:
    """Ask every worker to dump its stacks; the controller's children are the workers."""
    handle = _dump_handle
    if handle is None:
        return
    handle.write(f"no test report for {_stall_s:.0f}s after {last_report}; dumping stacks\n")
    handle.flush()
    if _WORKER_DUMP_SIGNAL is None:
        return
    for child in psutil.Process().children():
        try:
            os.kill(child.pid, _WORKER_DUMP_SIGNAL)
        except OSError:
            continue  # The worker exited between the listing and the signal.


def _arm_stall(last_report: str) -> None:
    """Restart the controller's stall timers from now."""
    global _stall_timer
    handle = _dump_handle
    if handle is None:
        return
    if _stall_timer is not None:
        _stall_timer.cancel()
    _stall_timer = threading.Timer(_stall_s, _signal_workers, (last_report,))
    _stall_timer.daemon = True
    _stall_timer.start()
    faulthandler.dump_traceback_later(_stall_s + _STALL_EXIT_GRACE_S, file=handle, exit=True)


def _cancel_stall() -> None:
    global _stall_s, _stall_timer
    if _stall_timer is not None:
        _stall_timer.cancel()
        _stall_timer = None
    if _stall_s > 0:
        faulthandler.cancel_dump_traceback_later()
        _stall_s = 0.0


def pytest_configure(config: pytest.Config) -> None:
    global _dump_handle, _timeout_s
    raw = os.environ.get(_ENV_SECONDS, "").strip()
    _timeout_s = float(raw) if raw else 0.0
    if _timeout_s <= 0:
        return
    dump_dir = Path(os.environ.get(_ENV_DIR) or config.rootpath)
    dump_dir.mkdir(parents=True, exist_ok=True)
    # One file per worker: xdist workers share a rootdir but not a process.
    _dump_handle = (dump_dir / f"hang-{_worker_id(config)}.txt").open("w")
    if hasattr(config, "workerinput") and _WORKER_DUMP_SIGNAL is not None:
        faulthandler.register(_WORKER_DUMP_SIGNAL, file=_dump_handle, all_threads=True)


def pytest_sessionstart(session: pytest.Session) -> None:
    global _stall_s
    if _dump_handle is None or not session.config.pluginmanager.has_plugin(
        _XDIST_CONTROLLER_PLUGIN
    ):
        return
    _stall_s = _timeout_s * _STALL_FACTOR
    _arm_stall("session start")


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if _stall_s > 0:
        _arm_stall(f"{report.nodeid} ({report.when})")


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session: pytest.Session) -> None:
    _cancel_stall()


def pytest_runtest_setup(item: pytest.Item) -> None:
    global _early_timer
    if _dump_handle is None:
        return
    # Record which test armed the timer, so a dump names its trigger even though
    # faulthandler itself only prints stacks. Rewinds each test (no accumulation).
    _dump_handle.seek(0)
    _dump_handle.truncate()
    _dump_handle.write(f"hang watchdog armed for {item.nodeid} ({_timeout_s}s)\n")
    _dump_handle.flush()
    _early_fired.clear()
    early = _test_timeout_s(item) * _EARLY_SHARE
    if 0 < early < _timeout_s:
        _early_timer = threading.Timer(early, _early_dump, (item.config, item.nodeid, early))
        _early_timer.daemon = True
        _early_timer.start()
    # Re-arms (resets) the single C-thread timer for this test. dump_traceback_later
    # dumps every thread by default, which is what we want on a wedged worker.
    faulthandler.dump_traceback_later(_timeout_s, file=_dump_handle, exit=True)


def pytest_runtest_teardown(item: pytest.Item) -> None:
    global _early_timer
    if _dump_handle is None:
        return
    faulthandler.cancel_dump_traceback_later()
    if _early_timer is not None:
        _early_timer.cancel()
        _early_timer = None
    if _early_fired.is_set():
        return  # The test ran long enough to dump; keep the evidence for the CI step.
    # The test finished, so its armed dump is stale; clear it.
    _dump_handle.seek(0)
    _dump_handle.truncate()
    _dump_handle.flush()


def pytest_unconfigure(config: pytest.Config) -> None:
    global _dump_handle
    if _dump_handle is not None:
        _cancel_stall()
        faulthandler.cancel_dump_traceback_later()
        if hasattr(config, "workerinput") and _WORKER_DUMP_SIGNAL is not None:
            faulthandler.unregister(_WORKER_DUMP_SIGNAL)
        _dump_handle.close()
        _dump_handle = None
