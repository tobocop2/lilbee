"""What a settings source reports about a value it refuses: a warning, or a variable that stops."""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

log = logging.getLogger(__name__)

_MARK_LENGTH = 12

# A child inherits the marks the startup load logged, so it does not log them again.
SHOWN_ENV_VAR = "LILBEE_LOAD_WARNINGS_SHOWN"
_MARK_SEPARATOR = ","

# Marks of the warnings this process already logged, seeded from the parent's.
_shown: set[str] = {
    mark for mark in os.environ.get(SHOWN_ENV_VAR, "").split(_MARK_SEPARATOR) if mark
}
_shown_lock = threading.Lock()

_collected: ContextVar[list[str] | None] = ContextVar("lilbee_load_warnings", default=None)


class RefusedVariableError(ValueError):
    """The environment sets a LILBEE_* variable lilbee does not run with."""


def _mark(message: str) -> str:
    """A short stable name for *message* that fits an environment variable."""
    return hashlib.sha256(message.encode("utf-8")).hexdigest()[:_MARK_LENGTH]


def warn_on_load(message: str, version: str = "") -> None:
    """Log *message* unless this process already did, and keep it for a collecting load.

    With *version*, the state of the file the message is about, a message that
    was logged for another state of that file is logged again.
    """
    found = _collected.get()
    if found is not None and message not in found:
        found.append(message)
    mark = _mark(message + version)
    with _shown_lock:
        if mark in _shown:
            return
        _shown.add(mark)
    log.warning("%s", message)


def shown_marks() -> str:
    """The marks logged so far, as the value a child inherits in ``SHOWN_ENV_VAR``."""
    with _shown_lock:
        return _MARK_SEPARATOR.join(sorted(_shown))


def forget_shown() -> None:
    """Forget which warnings this process logged, so the next one logs again."""
    with _shown_lock:
        _shown.clear()


def refuse_variable(message: str) -> None:
    """Raise RefusedVariableError, except in a collecting load, whose entry point raises it."""
    if _collected.get() is None:
        raise RefusedVariableError(message)


@contextmanager
def collecting() -> Iterator[list[str]]:
    """Yield the list that receives each load warning; a refused variable does not raise inside."""
    found: list[str] = []
    token = _collected.set(found)
    try:
        yield found
    finally:
        _collected.reset(token)
