"""The bounded exit of a command whose run leaves a pool worker behind.

The run lives on a daemon thread, so the pool workers it starts are daemon threads
too. The interpreter's exit joins them anyway (the thread pools' exit hook), so the
bounded exit for a leftover worker must see them.
"""

from __future__ import annotations

import concurrent.futures
import signal
import sys
import threading
from pathlib import Path

import pytest

from lilbee.app import services as svc_mod
from lilbee.runtime.cancellation import sigint_cancel
from tests import test_cli_second_ctrl_c as base

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal delivery")
_INTERRUPTED = 128 + signal.SIGINT
_HOLD = '    if hold == "stuck":\n        stuck()\n'
# The run gives up on a worker that never returns and ends: no Ctrl+C at all.
_HOLD_ABANDON = (
    '    if hold.startswith("abandon_"):\n'
    "        how = asyncio.to_thread if hold == 'abandon_to_thread' else to_ingest_thread\n"
    "        try:\n"
    "            await asyncio.wait_for(how(stuck), 0.5)\n"
    "        except asyncio.TimeoutError:\n"
    "            pass\n"
    '        mark("ABANDONED")\n'
    "        return\n"
    '    elif hold == "stuck":\n'
    "        stuck()\n"
)


@pytest.fixture
def abandoning_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    assert base._ENTRY.count(_HOLD) == 1
    monkeypatch.setattr(base, "_ENTRY", base._ENTRY.replace(_HOLD, _HOLD_ABANDON))


@posix_only
@pytest.mark.usefixtures("abandoning_entry")
@pytest.mark.parametrize("hold", ["abandon_pool", "abandon_to_thread"])
@pytest.mark.parametrize("command", ["sync", "add"])
def test_a_run_that_leaves_a_stuck_worker_ends_within_the_straggler_bound(
    tmp_path: Path, command: str, hold: str
) -> None:
    run = base._Command(tmp_path, command, hold)
    try:
        run.read_until("ABANDONED")
        status = run.status()
    finally:
        run.end()
    assert status == _INTERRUPTED, run.output
    assert "would not stop" in run.output, run.output


@posix_only
@pytest.mark.parametrize("hold", ["pool", "to_thread"])
@pytest.mark.parametrize("command", ["sync", "rebuild"])
def test_one_ctrl_c_ends_a_run_stuck_on_a_worker_within_the_straggler_bound(
    tmp_path: Path, command: str, hold: str
) -> None:
    run = base._Command(tmp_path, command, hold)
    try:
        run.read_until("READY")
        run.press_ctrl_c()
        status = run.status()
    finally:
        run.end()
    assert status == _INTERRUPTED, run.output


def test_the_straggler_check_sees_a_pool_worker_started_from_a_daemon_thread() -> None:
    release = threading.Event()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="left")
    submitter = threading.Thread(target=lambda: pool.submit(release.wait), daemon=True)
    submitter.start()
    submitter.join(timeout=5)
    try:
        worker = next(t for t in threading.enumerate() if t.name.startswith("left"))
        assert worker.daemon
        assert worker in svc_mod.straggler_threads()
    finally:
        release.set()
        pool.shutdown(wait=True)


def test_the_straggler_check_skips_a_daemon_thread_the_exit_does_not_join() -> None:
    release = threading.Event()
    thread = threading.Thread(target=release.wait, daemon=True)
    thread.start()
    try:
        assert thread not in svc_mod.straggler_threads()
    finally:
        release.set()
        thread.join(timeout=5)


def test_a_thread_started_before_the_scope_is_not_counted(monkeypatch) -> None:
    quits: list[str] = []
    monkeypatch.setattr(svc_mod, "force_quit_from_handler", lambda: quits.append("handler"))
    monkeypatch.setattr(svc_mod, "force_quit", lambda: quits.append("scope"))
    outer: list[int] = []
    original = signal.signal(signal.SIGINT, lambda signum, _frame: outer.append(signum))
    release = threading.Event()
    older = threading.Thread(target=release.wait)
    older.start()
    try:
        with sigint_cancel():
            signal.raise_signal(signal.SIGINT)
        signal.raise_signal(signal.SIGINT)
        assert quits == []
        assert outer == [signal.SIGINT]
    finally:
        signal.signal(signal.SIGINT, original)
        release.set()
        older.join(timeout=5)
