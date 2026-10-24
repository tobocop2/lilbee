"""Record the browser processes a test's crawl starts, on every OS."""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import psutil

POLL_INTERVAL_S = 0.01
_CHILD_PROCESS_FLAG = "--type="


def _same_file(left: str, right: Path) -> bool:
    return os.path.normcase(os.path.realpath(left)) == os.path.normcase(os.path.realpath(right))


class BrowserLaunches:
    """The command lines of the browser processes started below this process, one per launch."""

    def __init__(self, shell: Path) -> None:
        self._shell = shell
        self._seen: dict[tuple[int, float], list[str]] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, name="browser-launches", daemon=True)

    @property
    def command_lines(self) -> list[list[str]]:
        return list(self._seen.values())

    def _scan(self) -> None:
        for child in psutil.Process().children(recursive=True):
            try:
                key = (child.pid, child.create_time())
                if key in self._seen or not _same_file(child.exe(), self._shell):
                    continue
                command = child.cmdline()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
            # The browser starts renderer, GPU and utility children of the same binary. An exiting
            # process's command line reads empty on Linux, so it cannot be told from a helper.
            if command and not any(arg.startswith(_CHILD_PROCESS_FLAG) for arg in command):
                self._seen[key] = command

    def _poll(self) -> None:
        while not self._stop.is_set():
            self._scan()
            self._stop.wait(POLL_INTERVAL_S)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()


@contextmanager
def record_browser_launches(shell: Path) -> Iterator[BrowserLaunches]:
    """Watch the processes started below this one that run *shell* as a browser, not a helper."""
    launches = BrowserLaunches(shell)
    launches.start()
    try:
        yield launches
    finally:
        launches.stop()
