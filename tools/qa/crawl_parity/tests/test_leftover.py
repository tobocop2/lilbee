"""Left behind: a run's own survivors are counted and stopped; nothing else is touched."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import psutil
import pytest
from tools.qa.crawl_parity.leftover import LeftBehind, Tracker

SLEEPER = "import time; time.sleep(120)"
# The parent starts a listening child that outlives it, makes a temp directory, and exits.
LEAKY_PARENT = (
    "import subprocess, sys, tempfile\n"
    'child = \'import socket, time; s = socket.socket(); s.bind(("127.0.0.1", 0)); '
    "s.listen(); time.sleep(120)'\n"
    "subprocess.Popen([sys.executable, '-c', child])\n"
    "tempfile.mkdtemp(prefix='profile-')\n"
)
CLEAN_PARENT = "import tempfile; tempfile.TemporaryDirectory().cleanup()"


@pytest.fixture
def bystander() -> Iterator[subprocess.Popen[bytes]]:
    """A process that runs before, during and after a tracked run and is not part of it."""
    process = subprocess.Popen([sys.executable, "-c", SLEEPER])
    yield process
    process.kill()
    process.wait()


def tracked_run(code: str, temp_root: Path) -> LeftBehind:
    tracker = Tracker(temp_root)
    process = subprocess.Popen([sys.executable, "-c", code], env=tracker.environment())
    tracker.watch(process.pid)
    process.wait()
    return tracker.collect(grace_seconds=1.0)


def test_a_run_that_cleans_up_leaves_nothing(tmp_path: Path) -> None:
    left = tracked_run(CLEAN_PARENT, tmp_path / "tmp")
    assert left.is_empty()
    assert left == LeftBehind()


def test_a_surviving_child_its_socket_and_a_temp_entry_are_reported(tmp_path: Path) -> None:
    left = tracked_run(LEAKY_PARENT, tmp_path / "tmp")
    assert len(left.processes) == 1
    (process,) = left.processes
    assert len(process.listening_ports) == 1 and process.listening_ports[0] > 0
    assert len(left.temp_entries) == 1 and left.temp_entries[0].startswith("profile-")
    assert not left.is_empty()


def test_the_survivor_is_stopped_after_it_is_counted(tmp_path: Path) -> None:
    left = tracked_run(LEAKY_PARENT, tmp_path / "tmp")
    assert not psutil.pid_exists(left.processes[0].pid)


def test_a_process_that_is_not_the_runs_own_is_neither_counted_nor_stopped(
    tmp_path: Path, bystander: subprocess.Popen[bytes]
) -> None:
    left = tracked_run(LEAKY_PARENT, tmp_path / "tmp")
    assert bystander.pid not in {process.pid for process in left.processes}
    assert bystander.poll() is None


def test_a_process_started_during_the_run_by_someone_else_is_not_counted(tmp_path: Path) -> None:
    tracker = Tracker(tmp_path / "tmp")
    tracker.environment()
    stranger = subprocess.Popen([sys.executable, "-c", SLEEPER])
    try:
        left = tracker.collect(grace_seconds=0.2)
        assert left.processes == ()
        assert stranger.poll() is None
    finally:
        stranger.kill()
        stranger.wait()


def test_the_environment_points_every_temp_variable_at_the_runs_root(tmp_path: Path) -> None:
    root = tmp_path / "tmp"
    assert Tracker(root).environment() == {"TMPDIR": str(root), "TEMP": str(root), "TMP": str(root)}
    assert root.is_dir()
