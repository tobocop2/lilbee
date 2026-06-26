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
        # The most recent release thread, exposed so tests can join it deterministically.
        self.last_release_thread: threading.Thread | None = None

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
                self._spawn_release()

    def _spawn_release(self) -> None:
        """Run the pool release off the calling loop so the serving loop never blocks.

        The release fires sequential blocking ``httpx.post`` unloads; on the server
        path the bracket exits on the shared request-serving loop, so running it
        inline would stall concurrent chat and search. Fire-and-forget in a daemon
        thread keeps that loop free; the unloads are best-effort.
        """
        thread = threading.Thread(target=_run_release, name="ingest-release", daemon=True)
        self.last_release_thread = thread
        thread.start()


_counter = _IngestScaleCounter()


def _run_release() -> None:
    """Unload the elastic ingest pool (best-effort)."""
    _provider().release_ingest_pool()


def _provider() -> LLMProvider:  # late-bound so tests can patch it and to avoid an import cycle
    from lilbee.app.services import get_services

    return get_services().provider


def ingest_scale() -> contextlib.AbstractContextManager[None]:
    """Bracket an ingest; release the elastic pool when the last active one exits."""
    return _counter.scope()
