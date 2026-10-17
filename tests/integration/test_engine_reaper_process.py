"""The real reaper process against real user processes, on each OS's own file locks.

The unit tests drive the watch loop in-process. These start the reaper as the
product does (``python -m``, detached) and kill users outright, so the kernel's
release of a dead user's lock is what the reaper has to see.
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from lilbee.providers.fleet.swap_manager import _ensure_engine_reaper
from lilbee.runtime.engine_lock import reaper_running

pytestmark = pytest.mark.slow

_USER_SRC = (
    "import sys, time\n"
    "from pathlib import Path\n"
    "from lilbee.runtime.engine_lock import hold_user_lock\n"
    "hold = hold_user_lock(Path(sys.argv[1]))\n"
    "print('held', flush=True)\n"
    "time.sleep(600)\n"
)


def _start_user(engine_dir: Path) -> subprocess.Popen[str]:
    user = subprocess.Popen(
        [sys.executable, "-c", _USER_SRC, str(engine_dir)],
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert user.stdout is not None
    assert user.stdout.readline().strip() == "held"
    return user


def _until(check, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(0.1)
    return check()


def _reaper_pids(engine_dir: Path) -> list[int]:
    found = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            cmdline = proc.info["cmdline"] or []
            if "lilbee.providers.fleet.engine_reaper" in cmdline and str(engine_dir) in cmdline:
                found.append(proc.info["pid"])
    return found


@pytest.mark.timeout(90)
def test_the_reaper_ends_the_engine_only_after_every_user_is_killed(tmp_path: Path) -> None:
    engine_dir = tmp_path / "engine"
    engine_dir.mkdir()
    record = engine_dir / "llama-swap.state.1.chat.json"
    # A record whose pid is an exited process: the stop removes it and signals nothing.
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait(timeout=30)
    record.write_text(f'{{"pid": {gone.pid}, "member_ports": []}}', encoding="utf-8")
    first, second = _start_user(engine_dir), _start_user(engine_dir)
    try:
        assert _ensure_engine_reaper(engine_dir) is True
        assert _until(lambda: reaper_running(engine_dir), 30), "the reaper never took its lock"
        first.kill()
        first.wait(timeout=10)
        time.sleep(3)
        assert record.exists(), "the engine was stopped while a user still held it"
        second.kill()
        second.wait(timeout=10)
        assert _until(lambda: not record.exists(), 20), "the engine outlived every user"
        assert _until(lambda: not reaper_running(engine_dir), 10), "the reaper did not exit"
    finally:
        for user in (first, second):
            if user.poll() is None:
                user.kill()
        for pid in _reaper_pids(engine_dir):
            with contextlib.suppress(psutil.NoSuchProcess):
                psutil.Process(pid).kill()
