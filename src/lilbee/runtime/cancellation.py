"""Task cancellation sentinel shared between core library and TUI.

Raised by progress callbacks when the UI signals a long-running
background task should abort. Lives in a neutral module so that
``lilbee.catalog`` (core) can let it propagate without importing TUI code.
"""

from __future__ import annotations

from typing import Protocol


class TaskCancelledError(Exception):
    """Raised inside a progress callback to abort a long-running task."""


class CancelSignal(Protocol):
    """Anything a long-running run polls to learn it should stop.

    Structural so a run driven from another process can be cancelled by a
    ``multiprocessing`` event as readily as by a ``threading`` one.
    """

    def is_set(self) -> bool: ...
