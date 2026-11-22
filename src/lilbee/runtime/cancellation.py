"""Task cancellation sentinel shared between core library and TUI.

Raised by progress callbacks when the UI signals a long-running
background task should abort. Lives in a neutral module so that
``lilbee.catalog`` (core) can let it propagate without importing TUI code.
"""

from __future__ import annotations

import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol


class TaskCancelledError(Exception):
    """Raised inside a progress callback to abort a long-running task."""


class CancelSignal(Protocol):
    """Anything a long-running run polls to learn it should stop.

    Structural so a run driven from another process can be cancelled by a
    ``multiprocessing`` event as readily as by a ``threading`` one.
    """

    def is_set(self) -> bool: ...


class SignalFlag:
    """A cancel token a signal handler sets with one attribute write and no lock."""

    def __init__(self) -> None:
        self._is_set = False

    def set(self) -> None:
        self._is_set = True

    def is_set(self) -> bool:
        return self._is_set


@contextmanager
def sigint_cancel() -> Iterator[SignalFlag]:
    """Turn Ctrl-C into a token a long run polls, not a mid-write abort.

    A wiki build, a sync, an add, a crawl or an analyze writes as it goes, so the
    default KeyboardInterrupt drops it wherever the interpreter happened to be.
    Setting a token instead lets it stop at a boundary with what it wrote intact.
    The handler stays installed until the scope exits, where the previous handler
    returns on the main thread. A second Ctrl-C raises KeyboardInterrupt, so the
    command unwinds and its rollback and its message run on the main thread; the
    scope's exit then ends the process, because a normal exit joins every worker
    thread, also a stuck one. A further Ctrl-C ends the process from the handler.

    signal.signal only works on the main thread; off it (pytest-xdist workers)
    the token is simply never set and Ctrl-C keeps its default behaviour.
    """
    token = SignalFlag()
    if threading.current_thread() is not threading.main_thread():
        yield token
        return
    from lilbee.app.services import force_quit, force_quit_from_handler

    previous = signal.getsignal(signal.SIGINT)
    unwinding = False

    def _on_sigint(_signum: int, _frame: object) -> None:
        nonlocal unwinding
        if unwinding:
            force_quit_from_handler()
        elif token.is_set():
            unwinding = True
            raise KeyboardInterrupt
        else:
            token.set()

    signal.signal(signal.SIGINT, _on_sigint)
    try:
        yield token
    finally:
        if unwinding:
            force_quit()
        signal.signal(signal.SIGINT, previous)
