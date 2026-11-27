"""Task cancellation sentinel shared between core library and TUI.

Raised by progress callbacks when the UI signals a long-running
background task should abort. Lives in a neutral module so that
``lilbee.catalog`` (core) can let it propagate without importing TUI code.
"""

from __future__ import annotations

import signal
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Protocol

INTERRUPTED_EXIT_STATUS = 128 + signal.SIGINT
"""The status a shell reports for a command that Ctrl+C ended."""


class TaskCancelledError(Exception):
    """Raised inside a progress callback to abort a long-running task."""


class CancelSignal(Protocol):
    """Anything a long-running run polls to learn it should stop.

    Structural so a run driven from another process can be cancelled by a
    ``multiprocessing`` event as readily as by a ``threading`` one.
    """

    def is_set(self) -> bool: ...


class SignalFlag:
    """A cancel token a signal handler sets with one attribute write and no lock.

    A run that the token stops binds its stop here while it runs; the stop only
    hands a cancel to the run's own loop.
    """

    def __init__(self) -> None:
        self._is_set = False
        self._stop_run: Callable[[], object] | None = None
        self._run_ended = False
        self._interrupted = False

    def set(self) -> None:
        self._is_set = True

    def is_set(self) -> bool:
        return self._is_set

    def stop_run(self) -> None:
        """Ask the bound run, if one runs, to stop."""
        stop = self._stop_run
        if stop is not None:
            stop()

    def bind_run(self, stop: Callable[[], object]) -> None:
        """Note that a run started; *stop* asks it to stop and takes no lock."""
        self._stop_run = stop
        self._run_ended = False
        self._interrupted = False

    def interrupt(self) -> None:
        """Interrupt the command: raise KeyboardInterrupt here, or in the wait of the bound run.

        While a run is bound, the main thread can be inside a lock wait of the threading
        module, such as the start of the run's thread. A raise there leaves that lock
        released, so the run's wait raises the interrupt instead. In the wind-down,
        nothing is raised.
        """
        if self.winding_down:
            return
        if self._stop_run is not None:
            self._interrupted = True
            return
        raise KeyboardInterrupt

    @property
    def interrupted(self) -> bool:
        """True once a Ctrl-C interrupted the bound run; the run's wait then raises."""
        return self._interrupted

    def unbind_run(self) -> None:
        """Note that the run ended: what follows is the wind-down of the command."""
        self._stop_run = None
        self._run_ended = True

    @property
    def winding_down(self) -> bool:
        """True once a run ended and no other run has started since."""
        return self._run_ended


@contextmanager
def sigint_cancel() -> Iterator[SignalFlag]:
    """Turn Ctrl-C into a token a long run polls, not a mid-write abort.

    A wiki build, a sync, an add, a crawl or an analyze writes as it goes, so the
    default KeyboardInterrupt drops it wherever the interpreter happened to be.
    Setting a token instead lets it stop at a boundary with what it wrote intact.
    The handler stays installed until the scope exits, where the previous handler
    returns on the main thread. A run bound to the token takes a cancel of its task
    on the first Ctrl-C. A second Ctrl-C before a run raises KeyboardInterrupt, and
    during a run the run's wait raises it, so the command unwinds and its rollback and
    its message run on the main thread. After a run, in the wind-down, it raises
    nothing, so the rollback finishes. Either way the scope's exit then ends the
    process, because a normal exit joins every worker thread, also a stuck one. A
    further Ctrl-C ends the process from the handler.

    A cancelled run can leave a stuck thread behind that the exit joins. While a
    thread that the exit joins and that was born in the scope lives, a Ctrl-C after
    the scope ends the process from the handler; once none lives, the previous
    handler takes the Ctrl-C. The services module's straggler list says which
    threads the exit joins, as it does for the command's bounded exit.

    signal.signal only works on the main thread; off it (pytest-xdist workers)
    the token is simply never set and Ctrl-C keeps its default behaviour.
    """
    token = SignalFlag()
    if threading.current_thread() is not threading.main_thread():
        yield token
        return
    from lilbee.app.services import force_quit, force_quit_from_handler, straggler_threads

    previous = signal.getsignal(signal.SIGINT)
    threads_before = set(straggler_threads())
    left_behind: set[threading.Thread] = set()
    unwinding = False

    def _on_sigint(_signum: int, _frame: object) -> None:
        nonlocal unwinding
        if unwinding:
            force_quit_from_handler()
        elif not token.is_set():
            token.set()
            token.stop_run()
        else:
            unwinding = True
            token.interrupt()

    def _on_sigint_after(signum: int, _frame: object) -> None:
        if any(thread.is_alive() for thread in left_behind):
            force_quit_from_handler()
        signal.signal(signal.SIGINT, previous)
        signal.raise_signal(signum)

    signal.signal(signal.SIGINT, _on_sigint)
    try:
        yield token
    finally:
        if unwinding:
            force_quit()
        if token.is_set():
            left_behind = set(straggler_threads()) - threads_before
        signal.signal(signal.SIGINT, _on_sigint_after if left_behind else previous)
