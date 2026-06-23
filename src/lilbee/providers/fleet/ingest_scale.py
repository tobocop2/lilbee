"""Refcounted ingest bracket: release the elastic ingest pool when the last ingest ends."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator

from lilbee.providers.base import LLMProvider

_lock = threading.Lock()
_active = 0


def _provider() -> LLMProvider:  # late-bound so tests can patch it and to avoid an import cycle
    from lilbee.app.services import get_services

    return get_services().provider


@contextlib.contextmanager
def ingest_scale() -> Iterator[None]:
    """Bracket an ingest run; release the elastic pool when the last active one exits."""
    global _active
    with _lock:
        _active += 1
    try:
        yield
    finally:
        with _lock:
            _active -= 1
            last = _active == 0
        if last:
            _provider().release_ingest_pool()
