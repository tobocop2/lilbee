"""A wall-clock step on Linux must not read as a dead parent.

This file imports only the public watch function, so it also runs against older source.
"""

from __future__ import annotations

import itertools
import os
import sys
import threading
from unittest import mock

import psutil
import pytest

from lilbee.parent_monitor import watch_parent_thread

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="the start ticks come from /proc"
)


def test_a_clock_step_does_not_stop_the_watch_of_a_living_parent():
    """psutil's Linux create_time follows the wall clock (btime); a step moves it on every call.

    The parent here is the real parent of the test process, which stays alive. Each
    create_time call returns a later value, as after repeated clock steps.
    """
    steps = itertools.count(1000, 3)
    stop = threading.Event()
    fired = threading.Event()

    def on_death() -> None:
        fired.set()

    with (
        mock.patch.object(psutil.Process, "create_time", lambda self: float(next(steps))),
        mock.patch(
            "lilbee.parent_monitor.psutil.pid_exists", side_effect=lambda pid: not stop.is_set()
        ),
    ):
        thread = watch_parent_thread(os.getppid(), on_death, poll_interval_secs=0.01)
        assert not fired.wait(timeout=0.3), "a clock step was taken for a reused pid"
        stop.set()
        thread.join(timeout=2.0)
    assert fired.is_set()  # the watcher ends only because the parent was reported gone
