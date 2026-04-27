"""Task cancellation sentinel shared between core library and TUI.

Raised by progress callbacks when the UI signals a long-running
background task should abort. Lives in a neutral module so that
``lilbee.catalog`` (core) can let it propagate without importing TUI code.
"""

from __future__ import annotations


class TaskCancelledError(Exception):
    """Raised inside a progress callback to abort a long-running task."""
