"""Refcounted ingest bracket: release the elastic ingest pool when the last ingest ends."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator

from lilbee.providers.base import LLMProvider


class _IngestScaleCounter:
    """Refcounts active ingests; releases the elastic pool when the last one ends."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = 0

    @contextlib.contextmanager
    def scope(self) -> Iterator[None]:
        with self._lock:
            self._active += 1
        try:
            yield
        finally:
            with self._lock:
                self._active -= 1
                last = self._active == 0
            if last:
                _provider().release_ingest_pool()


_counter = _IngestScaleCounter()


def _provider() -> LLMProvider:  # late-bound so tests can patch it and to avoid an import cycle
    from lilbee.app.services import get_services  # pragma: no cover - patched in tests

    return get_services().provider  # pragma: no cover - patched in tests


def ingest_scale() -> contextlib.AbstractContextManager[None]:
    """Bracket an ingest; release the elastic pool when the last active one exits."""
    return _counter.scope()
