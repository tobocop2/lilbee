"""On macOS the parent watch names a process by its start on the mach clock, exactly.

psutil's macOS create time follows the wall clock, so a step moves it; the mach start
(``proc_pid_rusage``, ``ri_proc_start_abstime``) does not. The watch tests go through the
public watch function with the platform set to macOS and the mach read replaced.
"""

from __future__ import annotations

import ctypes
import os
import threading
import time
from collections.abc import Callable

import psutil
import pytest

from lilbee import parent_monitor as pm
from lilbee.parent_monitor import watch_parent_thread

_PARENT = os.getppid()
_PARENT_ABS = 1_000
_OUR_ABS = 5_000
_STEP_S = 7200.0


def _watch(
    monkeypatch: pytest.MonkeyPatch,
    table: dict[int, int | None],
    *,
    after_start: Callable[[], None] | None = None,
) -> bool:
    """True when the watch on the parent fires within 0.4 s of *after_start*.

    *table* maps a pid to its mach start (None: the read fails). psutil's create time
    moves by a +2 h step before *after_start* runs, as it does on a macOS runner.
    """
    create_offset = [0.0]
    stop = threading.Event()
    fired = threading.Event()
    monkeypatch.setattr(pm.sys, "platform", "darwin")
    monkeypatch.setattr(pm, "_macos_start_abstime", table.get)
    monkeypatch.setattr(psutil.Process, "create_time", lambda _self: 100.0 + create_offset[0])
    monkeypatch.setattr(pm.psutil, "pid_exists", lambda _pid: not stop.is_set())
    thread = watch_parent_thread(_PARENT, fired.set, poll_interval_secs=0.01)
    time.sleep(0.1)
    create_offset[0] = _STEP_S
    if after_start is not None:
        after_start()
    gone = fired.wait(timeout=0.4)
    stop.set()
    thread.join(timeout=2.0)
    return gone


def test_a_clock_step_keeps_a_live_parent_live(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _watch(monkeypatch, {_PARENT: _PARENT_ABS, os.getpid(): _OUR_ABS}) is False


def test_another_mach_start_on_the_pid_is_a_reused_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    table: dict[int, int | None] = {_PARENT: _PARENT_ABS, os.getpid(): _OUR_ABS}
    assert _watch(monkeypatch, table, after_start=lambda: table.update({_PARENT: 1_001})) is True


def test_a_failed_read_in_the_loop_reads_the_parent_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    table: dict[int, int | None] = {_PARENT: _PARENT_ABS, os.getpid(): _OUR_ABS}
    assert _watch(monkeypatch, table, after_start=lambda: table.update({_PARENT: None})) is True


def test_a_failed_read_at_watch_start_watches_the_pid_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table: dict[int, int | None] = {_PARENT: None, os.getpid(): _OUR_ABS}
    assert _watch(monkeypatch, table, after_start=lambda: table.update({_PARENT: 1_001})) is False


def test_a_parent_pid_started_after_us_fires_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _watch(monkeypatch, {_PARENT: _OUR_ABS + 1, os.getpid(): _OUR_ABS}) is True


def test_a_failed_read_of_our_own_start_does_not_stop_the_watch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _watch(monkeypatch, {_PARENT: _PARENT_ABS, os.getpid(): None}) is False


def _no_libc(_name: object) -> object:
    raise OSError("no libc")


class _Libc:
    def __init__(self, *, result: int, start: int) -> None:
        self._result = result
        self._start = start

    def proc_pid_rusage(self, _pid, flavor: int, buffer) -> int:
        assert flavor == 0
        buffer[10] = self._start
        return self._result


class TestMachStart:
    def test_reads_ri_proc_start_abstime(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ctypes, "CDLL", lambda _name: _Libc(result=0, start=987_654_321))
        assert pm._macos_start_abstime(42) == 987_654_321

    def test_a_failed_call_reads_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ctypes, "CDLL", lambda _name: _Libc(result=-1, start=7))
        assert pm._macos_start_abstime(42) is None

    @pytest.mark.parametrize(
        "load", [lambda _name: object(), _no_libc], ids=["no-such-call", "libc-does-not-load"]
    )
    def test_an_unusable_libc_reads_none(
        self, monkeypatch: pytest.MonkeyPatch, load: Callable[[object], object]
    ) -> None:
        monkeypatch.setattr(ctypes, "CDLL", load)
        assert pm._macos_start_abstime(42) is None

    def test_the_stamp_pairs_the_boot_session_with_the_start(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pm.sys, "platform", "darwin")
        monkeypatch.setattr(pm, "_macos_boot_session", lambda: "boot-A")
        monkeypatch.setattr(pm, "_macos_start_abstime", lambda _pid: 55)
        assert pm.macos_start_stamp(42) == ("boot-A", 55)

    @pytest.mark.parametrize(
        ("boot", "start"),
        [(None, 55), ("boot-A", None)],
        ids=["no-boot-session", "failed-start-read"],
    )
    def test_a_failed_read_gives_no_stamp(
        self, monkeypatch: pytest.MonkeyPatch, boot: str | None, start: int | None
    ) -> None:
        monkeypatch.setattr(pm.sys, "platform", "darwin")
        monkeypatch.setattr(pm, "_macos_boot_session", lambda: boot)
        monkeypatch.setattr(pm, "_macos_start_abstime", lambda _pid: start)
        assert pm.macos_start_stamp(42) is None

    def test_off_macos_there_is_no_stamp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pm.sys, "platform", "win32")
        assert pm.macos_start_stamp(42) is None
