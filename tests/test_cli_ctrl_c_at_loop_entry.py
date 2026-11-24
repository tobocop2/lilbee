"""A second Ctrl+C where a raise loses work: a callback entry, the rollback, a native call.

The second press comes from inside the process at a chosen point, so each case is
deterministic; the real entry point runs as in tests/test_cli_second_ctrl_c.py.
"""

from __future__ import annotations

import signal
import sys
from pathlib import Path

import pytest

from lilbee.app.services import _FORCE_QUIT_NOTE
from tests import test_cli_second_ctrl_c as base

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal delivery")
# A zeroed pthread mutex is a valid unlocked mutex on glibc only.
glibc_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="glibc mutex layout")

_INTERRUPTED = 128 + signal.SIGINT
_MAIN = "from lilbee.runtime.launcher import main"
_HOLD = '    if hold == "stuck":\n        stuck()\n'
_HOLD_WITH_NATIVE = (
    '    if hold == "native":\n        native_stuck()\n    elif hold == "stuck":\n        stuck()\n'
)

# X_HOLD is "<hold>+<press point>": the prelude keeps the hold for held() and the point here.
_PRESS = '''
import asyncio.events
import ctypes
import signal as _signal
import threading as _threading

_hold, _, _press_at = os.environ["X_HOLD"].partition("+")
os.environ["X_HOLD"] = _hold
_presses = []


def press():
    """The second Ctrl+C, sent from inside the process."""
    mark("PRESS 2")
    if _threading.current_thread() is _threading.main_thread():
        _signal.raise_signal(_signal.SIGINT)
    else:
        os.kill(os.getpid(), _signal.SIGINT)


_noted_set = SignalFlag.set


def counted_set(self):
    _noted_set(self)
    _presses.append("set")


SignalFlag.set = counted_set
_real_run = asyncio.events.Handle._run


def _is_run_wakeup(handle):
    owner = getattr(handle._callback, "__self__", None)
    if not isinstance(owner, asyncio.Task) or owner.done():
        return False
    # get_stack shows only the outer coroutine of a suspended task; walk its awaits instead.
    coro = owner.get_coro()
    while coro is not None:
        code = getattr(coro, "cr_code", None)
        if code is not None and code.co_name == "held":
            return True
        coro = getattr(coro, "cr_await", None)
    return False


def run_or_press(self):
    # The first callback of the run's task after the first press: the second lands at its entry.
    if _press_at == "wakeup" and _presses == ["set"] and _is_run_wakeup(self):
        _presses.append("pressed")
        press()
    return _real_run(self)


asyncio.events.Handle._run = run_or_press
_noted_forget = AddRollback.forget_unfinished


def pressing_forget(self):
    if _press_at == "rollback" and _presses == ["set"]:
        _presses.append("pressed")
        press()
    _noted_forget(self)


AddRollback.forget_unfinished = pressing_forget


def native_stuck():
    """Wait in a native call that a signal does not end: a second lock of a held mutex."""
    libc = ctypes.CDLL(None)
    mutex = ctypes.create_string_buffer(64)
    libc.pthread_mutex_lock(mutex)
    mark("READY")
    libc.pthread_mutex_lock(mutex)


'''


def _entry() -> str:
    assert base._ENTRY.count(_HOLD) == 1
    assert base._ENTRY.count(_MAIN) == 1
    return base._ENTRY.replace(_HOLD, _HOLD_WITH_NATIVE).replace(_MAIN, _PRESS + _MAIN)


@pytest.fixture
def pressing_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(base, "_ENTRY", _entry())


def _assert_forced(status: int, output: str) -> None:
    """The forced end: exit 130, after the rollback, the cancel message and the note, in order."""
    assert status == _INTERRUPTED, output
    rollback = output.find("ROLLBACK RAN")
    message = output.lower().find("cancelled")
    note = output.find(_FORCE_QUIT_NOTE)
    assert -1 < rollback < message < note, output


def _status_after_one_press(run: base._Command) -> int:
    """Press Ctrl+C once; the process sends the second itself. Return the exit status."""
    try:
        run.read_until("READY")
        run.press_ctrl_c()
        return run.status()
    finally:
        run.end()


@posix_only
@pytest.mark.usefixtures("pressing_entry")
@pytest.mark.parametrize("command", base._COMMANDS)
def test_a_second_ctrl_c_at_the_entry_of_a_loop_callback_ends_the_command(
    tmp_path: Path, command: str
) -> None:
    run = base._Command(tmp_path, command, "to_thread+wakeup")
    status = _status_after_one_press(run)
    assert "PRESS 2" in run.output, run.output
    _assert_forced(status, run.output)


@posix_only
@pytest.mark.usefixtures("pressing_entry")
@pytest.mark.parametrize("command", base._COMMANDS)
def test_a_second_ctrl_c_during_the_rollback_lets_the_rollback_finish(
    tmp_path: Path, command: str
) -> None:
    run = base._Command(tmp_path, command, "polling+rollback")
    status = _status_after_one_press(run)
    assert "PRESS 2" in run.output, run.output
    _assert_forced(status, run.output)


@posix_only
@glibc_only
@pytest.mark.usefixtures("pressing_entry")
@pytest.mark.parametrize("command", base._COMMANDS)
def test_a_second_ctrl_c_ends_a_command_stuck_in_a_native_call(
    tmp_path: Path, command: str
) -> None:
    run = base._Command(tmp_path, command, "native+none")
    status = base._status_after_two_presses(run)
    _assert_forced(status, run.output)
