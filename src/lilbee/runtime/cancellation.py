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


@contextmanager
def sigint_cancel() -> Iterator[threading.Event]:
    """Turn Ctrl-C into a token the wiki pass polls, not a mid-page abort.

    A build runs for hours and writes pages as it goes, so the default
    KeyboardInterrupt drops it wherever the interpreter happened to be. Setting
    a token instead lets it stop at a source boundary with what it wrote intact.
    The previous handler is restored as soon as it fires, so a second Ctrl-C
    still hard-exits a pass that is not checking the token.

    signal.signal only works on the main thread; off it (pytest-xdist workers)
    the token is simply never set and Ctrl-C keeps its default behaviour.
    """
    token = threading.Event()
    if threading.current_thread() is not threading.main_thread():
        yield token
        return
    previous = signal.getsignal(signal.SIGINT)

    def _on_sigint(_signum: int, _frame: object) -> None:
        signal.signal(signal.SIGINT, previous)
        token.set()

    signal.signal(signal.SIGINT, _on_sigint)
    try:
        yield token
    finally:
        signal.signal(signal.SIGINT, previous)


@contextmanager
def sigint_deferred() -> Iterator[None]:
    """Hold Ctrl-C until the block ends, then deliver it to the handler before the block."""
    with sigint_cancel() as pressed:
        yield
    if pressed.is_set():
        signal.raise_signal(signal.SIGINT)
