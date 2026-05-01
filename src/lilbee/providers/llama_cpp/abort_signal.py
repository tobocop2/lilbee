"""Process-wide abort flag wired into llama_cpp.Llama's abort_callback."""

from __future__ import annotations

import threading
from typing import Any

_abort = threading.Event()


def request_abort() -> None:
    """Set the process-wide abort flag. Polled by ggml every N tokens."""
    _abort.set()


def clear_abort() -> None:
    """Clear the abort flag so the next inference runs to completion."""
    _abort.clear()


def is_abort_set() -> bool:
    """Return True if request_abort() has been called and not cleared."""
    return _abort.is_set()


def abort_callback(_user_data: Any = None) -> bool:
    """ggml abort_callback: returns True iff request_abort() has fired."""
    return _abort.is_set()
