"""Warnings about a value a settings source refused: logged once per process tree, kept per load."""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

log = logging.getLogger(__name__)

# Marks of the warnings this process tree already logged; a child inherits it.
SHOWN_ENV_VAR = "LILBEE_LOAD_WARNINGS_SHOWN"
_MARK_SEPARATOR = ","
_MARK_LENGTH = 12

_collected: ContextVar[list[str] | None] = ContextVar("lilbee_load_warnings", default=None)


def _mark(message: str) -> str:
    """A short stable name for *message* that fits an environment variable."""
    return hashlib.sha256(message.encode("utf-8")).hexdigest()[:_MARK_LENGTH]


def warn_on_load(message: str) -> None:
    """Log *message* unless this process tree already did, and keep it for a collecting load."""
    found = _collected.get()
    if found is not None and message not in found:
        found.append(message)
    shown = [mark for mark in os.environ.get(SHOWN_ENV_VAR, "").split(_MARK_SEPARATOR) if mark]
    mark = _mark(message)
    if mark in shown:
        return
    log.warning("%s", message)
    os.environ[SHOWN_ENV_VAR] = _MARK_SEPARATOR.join([*shown, mark])


@contextmanager
def collecting() -> Iterator[list[str]]:
    """Yield the list that receives every load warning reported inside the block."""
    found: list[str] = []
    token = _collected.set(found)
    try:
        yield found
    finally:
        _collected.reset(token)
