"""Server-side view of the chat model's cold load, for granular launch feedback.

A launcher streams this while a chat model loads so the user sees real progress
(reading weights with a true byte percentage, then the engine load) instead of a
frozen "Warming..." line. The fleet provider drives the tracker through its warm
path; providers without a managed load report nothing and a launcher falls back
to a plain spinner.
"""

from __future__ import annotations

import threading
import time
from enum import StrEnum

from pydantic import BaseModel


class WarmPhase(StrEnum):
    """The stage a chat-model cold load is in."""

    STARTING = "starting"
    """Warm thread has begun; the engine has not been touched yet."""
    READING_WEIGHTS = "reading_weights"
    """Paging the GGUF shards off disk into the page cache; reports a true byte %."""
    LOADING_ENGINE = "loading_engine"
    """The engine is loading the cached weights into VRAM; no byte signal, so
    surfaces show an indeterminate spinner bounded by readiness."""
    READY = "ready"
    """The chat engine is loaded and can serve a first token."""
    ERROR = "error"
    """The load failed; ``error`` carries the user-facing reason."""


class WarmProgress(BaseModel):
    """A snapshot of the chat role's warm state, streamed to a launcher."""

    phase: WarmPhase
    model_ref: str | None = None
    bytes_done: int = 0
    bytes_total: int = 0
    detail: str | None = None
    error: str | None = None
    elapsed_s: float = 0.0

    @property
    def fraction(self) -> float | None:
        """Completion in ``0..1`` while reading weights, ``1`` once ready, else None.

        None signals an indeterminate phase (starting / loading the engine) so a
        renderer shows a spinner instead of a misleading bar.
        """
        if self.phase is WarmPhase.READING_WEIGHTS and self.bytes_total > 0:
            return min(1.0, self.bytes_done / self.bytes_total)
        if self.phase is WarmPhase.READY:
            return 1.0
        return None


class WarmProgressTracker:
    """Thread-safe warm-state holder: the warm thread writes, handlers read.

    The fleet warm-up runs on a daemon thread while the health / SSE handlers
    read concurrently, so every mutation and the snapshot read take the lock.
    ``elapsed_s`` is stamped at read time from the ``begin`` monotonic mark so
    callers always see a live elapsed without the writer ticking a clock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot: WarmProgress | None = None
        self._started_at: float | None = None

    def begin(self, model_ref: str | None) -> None:
        """Mark the start of a cold load; resets elapsed and clears prior state."""
        with self._lock:
            self._started_at = time.monotonic()
            self._snapshot = WarmProgress(phase=WarmPhase.STARTING, model_ref=model_ref)

    def reading(self, bytes_done: int, bytes_total: int, detail: str | None = None) -> None:
        """Report read-phase progress in bytes."""
        self._advance(
            WarmPhase.READING_WEIGHTS,
            bytes_done=bytes_done,
            bytes_total=bytes_total,
            detail=detail,
        )

    def loading_engine(self, detail: str | None = None) -> None:
        """Mark the transition into the indeterminate VRAM-load phase."""
        self._advance(WarmPhase.LOADING_ENGINE, detail=detail)

    def ready(self) -> None:
        """Mark the chat engine ready to serve."""
        self._advance(WarmPhase.READY)

    def fail(self, message: str) -> None:
        """Mark the load as failed with a user-facing reason."""
        self._advance(WarmPhase.ERROR, error=message)

    def snapshot(self) -> WarmProgress | None:
        """Return a copy of the current state with live ``elapsed_s``, or None."""
        with self._lock:
            if self._snapshot is None:
                return None
            elapsed = time.monotonic() - self._started_at if self._started_at is not None else 0.0
            return self._snapshot.model_copy(update={"elapsed_s": elapsed})

    def _advance(
        self,
        phase: WarmPhase,
        *,
        bytes_done: int = 0,
        bytes_total: int = 0,
        detail: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            model_ref = self._snapshot.model_ref if self._snapshot is not None else None
            self._snapshot = WarmProgress(
                phase=phase,
                model_ref=model_ref,
                bytes_done=bytes_done,
                bytes_total=bytes_total,
                detail=detail,
                error=error,
            )
