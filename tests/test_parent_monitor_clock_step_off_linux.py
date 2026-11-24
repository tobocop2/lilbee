"""A wall-clock step off Linux must not read as a dead parent when the boot is the same.

On macOS psutil's create_time moves with a wall-clock step, by the step minus up to two
seconds. The test goes through the public watch function only, with the platform check
forced off Linux and the boot identity fixed.
"""

from __future__ import annotations

import os
import threading
import time
from unittest import mock

import psutil
import pytest

from lilbee import parent_monitor
from lilbee.parent_monitor import watch_parent_thread

_STEP_S = 7200.0
_PARENT_AGE_S = 500.0
_OUR_AGE_S = 100.0
_REAL_TIME = time.time
_REAL_CREATE_TIME_BASE = _REAL_TIME()


def _watch_across_a_step(create_time_lag_s: float, *, boot_after: str) -> bool:
    """True when the watch reports the parent gone after a +2 h step.

    The step moves the wall clock by ``_STEP_S`` and create_time by ``_STEP_S`` minus
    *create_time_lag_s*, as on macOS. *boot_after* is the boot identity read after the step.
    """
    wall_offset = [0.0]
    create_offset = [0.0]
    boot = ["boot-a"]
    stop = threading.Event()
    fired = threading.Event()

    def create_time(self: psutil.Process) -> float:
        age = _PARENT_AGE_S if self.pid != os.getpid() else _OUR_AGE_S
        return _REAL_CREATE_TIME_BASE - age + create_offset[0]

    with (
        mock.patch("lilbee.parent_monitor._is_linux", return_value=False),
        mock.patch("lilbee.parent_monitor.boot_identity", side_effect=lambda: boot[0]),
        mock.patch(
            "lilbee.parent_monitor.time.time", side_effect=lambda: _REAL_TIME() + wall_offset[0]
        ),
        mock.patch.object(psutil.Process, "create_time", create_time),
        mock.patch(
            "lilbee.parent_monitor.psutil.pid_exists", side_effect=lambda pid: not stop.is_set()
        ),
    ):
        thread = watch_parent_thread(os.getppid(), fired.set, poll_interval_secs=0.01)
        time.sleep(0.1)
        wall_offset[0] = _STEP_S
        create_offset[0] = _STEP_S - create_time_lag_s
        boot[0] = boot_after
        gone = fired.wait(timeout=0.4)
        stop.set()
        thread.join(timeout=2.0)
    return gone


@pytest.mark.parametrize(
    ("lag_s", "gone"),
    [(1.9, False), (2.5, False), (4.0, True)],
    ids=["measured-macos-restore-offset", "inside-the-tolerance", "outside-the-tolerance"],
)
def test_monotonic_start_offset_decides_after_a_clock_step(lag_s, gone):
    assert _watch_across_a_step(lag_s, boot_after="boot-a") is gone


def test_a_step_alone_keeps_a_live_parent_live():
    assert _watch_across_a_step(0.0, boot_after="boot-a") is False


def test_a_changed_boot_identity_falls_back_to_the_create_time():
    assert _watch_across_a_step(0.0, boot_after="boot-b") is True


def test_no_boot_identity_records_no_monotonic_start():
    with (
        mock.patch("lilbee.parent_monitor._is_linux", return_value=False),
        mock.patch("lilbee.parent_monitor.boot_identity", return_value=None),
    ):
        assert parent_monitor._mono_stamp(100.0) is None


def test_linux_and_an_unknown_start_record_no_monotonic_start():
    with mock.patch("lilbee.parent_monitor._is_linux", return_value=True):
        assert parent_monitor._mono_stamp(100.0) is None
    with mock.patch("lilbee.parent_monitor._is_linux", return_value=False):
        assert parent_monitor._mono_stamp(None) is None


def test_a_reading_without_a_boot_identity_matches_nothing():
    assert parent_monitor.started_near_mono(100.0, 0.0, None) is False
