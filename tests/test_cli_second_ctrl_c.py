"""Ctrl+C in sync, add, rebuild, crawl and analyze, sent to the real ``lilbee`` entry point."""

from __future__ import annotations

import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from lilbee.app.services import _FORCE_QUIT_NOTE
from tests._signals import child_preexec
from tests.conftest import clean_env

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal delivery")

# The shell's status for a command that Ctrl+C stops; analyze ends with it too.
_INTERRUPTED = 128 + signal.SIGINT
# A kill at this ceiling ends a child that ignores the presses; its status is then -9.
# It is below pytest's own 60 s, which ends the test worker and reports no status.
_FAILURE_CEILING_S = 45.0

# The real entry point, with the work inside the command replaced by a step that
# opens the store, then waits. A "stuck" step never reaches a poll point, as a
# converter or a fetch caught in a native call does; a "pool" step is stuck on a
# thread of the ingest pool, and a "to_thread" step on a thread of the loop's own
# pool. A "polling" step checks the cancel as the real sync does between files,
# and stops when it is set.
_ENTRY = """\
import asyncio
import os
import sys
import threading

import lilbee.app.analyze as analyze
import lilbee.crawler as crawler
import lilbee.data.ingest as ingest
from lilbee.app.ingest import AddRollback
from lilbee.app.services import get_services
from lilbee.data.offload import to_ingest_thread
from lilbee.data.store.core import Store
from lilbee.runtime.cancellation import SignalFlag


def mark(text):
    os.write(1, f"\\n{text}\\n".encode())


real_set = SignalFlag.set


def noted_set(self):
    real_set(self)
    mark("CANCEL SET")


SignalFlag.set = noted_set
real_close = Store.close


def noted_close(self):
    real_close(self)
    mark("STORE CLOSED")


Store.close = noted_close
real_forget = AddRollback.forget_unfinished


def noted_forget(self):
    real_forget(self)
    mark("ROLLBACK RAN")


AddRollback.forget_unfinished = noted_forget
never = threading.Event()


def stuck():
    mark("READY")
    never.wait()


async def held(cancel):
    get_services().store
    hold = os.environ["X_HOLD"]
    if hold == "stuck":
        stuck()
    elif hold == "pool":
        await to_ingest_thread(stuck)
    elif hold == "to_thread":
        await asyncio.to_thread(stuck)
    mark("READY")
    while not cancel.is_set():
        await asyncio.sleep(0.01)


async def fake_sync(**kwargs):
    await held(kwargs["cancel"])
    raise asyncio.CancelledError


async def fake_crawl(url, **kwargs):
    await held(kwargs["cancel"])
    return []


async def fake_analysis(*args, **kwargs):
    await held(kwargs["cancel"])


ingest.sync = fake_sync
analyze.run_analysis = fake_analysis
crawler.crawler_available = lambda: True
crawler.crawl_and_save = fake_crawl
sys.argv = ["lilbee", *sys.argv[1:]]
from lilbee.runtime.launcher import main

main()
"""


def _argv(command: str, data: Path) -> list[str]:
    """The command line of *command*: the add takes a file, the crawl a URL."""
    if command == "add":
        doc = data.parent / "note.md"
        doc.write_text("# Note\n", encoding="utf-8")
        return ["add", str(doc), "--data-dir", str(data)]
    if command == "crawl":
        return ["add", "https://example.com/", "--crawl", "--data-dir", str(data)]
    return [command, "--data-dir", str(data)]


class _Command:
    """One run of a lilbee command in a fresh interpreter, its output read line by line."""

    def __init__(self, tmp_path: Path, command: str, hold: str) -> None:
        data = tmp_path / "data"
        data.mkdir()
        self.lines: list[str] = []
        self.process = subprocess.Popen(
            [sys.executable, "-u", "-c", _ENTRY, *_argv(command, data)],
            env={**clean_env(data), "HF_HUB_OFFLINE": "1", "X_HOLD": hold},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            preexec_fn=child_preexec(),
        )
        # The kill turns a blocked read of the output into an EOF.
        self.ceiling = threading.Timer(_FAILURE_CEILING_S, self.process.kill)
        self.ceiling.daemon = True
        self.ceiling.start()

    @property
    def output(self) -> str:
        return "".join(self.lines)

    def read_until(self, marker: str) -> None:
        assert self.process.stdout is not None
        for line in iter(self.process.stdout.readline, ""):
            self.lines.append(line)
            if line.strip() == marker:
                return
        raise AssertionError(f"the command ended before {marker}:\n{self.output}")

    def press_ctrl_c(self) -> None:
        """Send one Ctrl+C and wait until the command's handler has set the cancel."""
        self.process.send_signal(signal.SIGINT)
        self.read_until("CANCEL SET")

    def status(self) -> int:
        """Read the rest of the output, then return the exit status."""
        assert self.process.stdout is not None
        self.lines.extend(iter(self.process.stdout.readline, ""))
        status = self.process.wait()
        self.ceiling.cancel()
        return status

    def end(self) -> None:
        self.ceiling.cancel()
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        if self.process.stdout is not None:
            self.process.stdout.close()


_COMMANDS = ["sync", "add", "rebuild", "crawl"]


def _status_after_two_presses(run: _Command) -> int:
    """Press Ctrl+C twice on a step that polls nothing, and return the exit status."""
    try:
        run.read_until("READY")
        run.press_ctrl_c()
        assert run.process.poll() is None, run.output
        run.process.send_signal(signal.SIGINT)
        return run.status()
    finally:
        run.end()


@posix_only
@pytest.mark.parametrize("hold", ["stuck", "pool", "to_thread"])
@pytest.mark.parametrize("command", _COMMANDS)
def test_a_second_ctrl_c_ends_a_command_stuck_between_poll_points(
    tmp_path: Path, command: str, hold: str
) -> None:
    run = _Command(tmp_path, command, hold)
    status = _status_after_two_presses(run)
    assert status == _INTERRUPTED, run.output
    rollback = run.output.find("ROLLBACK RAN")
    message = run.output.lower().find("cancelled")
    note = run.output.find(_FORCE_QUIT_NOTE)
    assert -1 < rollback < message < note, run.output


@posix_only
def test_a_second_ctrl_c_ends_an_analyze_stuck_on_a_pool_thread(tmp_path: Path) -> None:
    run = _Command(tmp_path, "analyze", "to_thread")
    status = _status_after_two_presses(run)
    assert status == _INTERRUPTED, run.output
    assert _FORCE_QUIT_NOTE in run.output, run.output


@posix_only
@pytest.mark.parametrize("command", _COMMANDS)
def test_the_first_ctrl_c_stops_at_the_poll_point_and_closes_the_store(
    tmp_path: Path, command: str
) -> None:
    run = _Command(tmp_path, command, "polling")
    try:
        run.read_until("READY")
        run.press_ctrl_c()
        status = run.status()
    finally:
        run.end()
    assert status == _INTERRUPTED, run.output
    assert "cancelled" in run.output.lower(), run.output
    assert any(line.strip() == "STORE CLOSED" for line in run.lines), run.output
