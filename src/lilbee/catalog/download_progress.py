"""Download progress callback plumbing shared by all surfaces.

The TUI runs under Textual which owns the terminal; tqdm output to
stderr/stdout corrupts the screen. This module provides a tqdm subclass
(``_CallbackProgressBar``) that suppresses terminal output and forwards
progress to a plain ``Callable[[int, int], None]`` callback. The
``_ProgressTracker`` wrapper detects whether progress events actually
fired so the TUI can detect a cache-hit (no progress events) and render
``"already downloaded"`` instead of leaving the bar at 0%.

``make_download_callback`` is the public entry point used by every
surface to convert raw bytes-progress into ``DownloadProgress`` events.
"""

from __future__ import annotations

import io
import threading
import time
from collections.abc import Callable
from typing import Any

from tqdm.auto import tqdm as _base_tqdm

from lilbee.catalog.models import DownloadProgress

ProgressCallback = Callable[[int, int], None]
_BYTES_PER_MB = 1024 * 1024


def make_download_callback(
    on_update: Callable[[DownloadProgress], None],
    *,
    throttle_interval: float = 0.1,
) -> ProgressCallback:
    """Build a download progress callback that converts bytes to human-readable state.
    *on_update(progress: DownloadProgress)* is called at most once per
    ``throttle_interval`` seconds with a float percentage (0.0 to 100.0), a
    ``"<done>/<total> MB"`` detail string, and a cache-hit flag. Both the
    catalog and setup screens use this so byte-to-MB conversion and
    cache-hit detection aren't duplicated.
    """
    last_update_time = 0.0
    seen_partial = False

    def _on_progress(downloaded: int, total: int) -> None:
        nonlocal last_update_time, seen_partial

        if total > 0 and downloaded >= total and not seen_partial:
            on_update(
                DownloadProgress(percent=100.0, detail="already downloaded", is_cache_hit=True)
            )
            return
        seen_partial = True

        now = time.monotonic()
        if now - last_update_time < throttle_interval:
            return
        last_update_time = now

        mb_done = downloaded / _BYTES_PER_MB
        if total > 0:
            pct = min(downloaded * 100.0 / total, 100.0)
            mb_total = total / _BYTES_PER_MB
            on_update(
                DownloadProgress(
                    percent=pct,
                    detail=f"{mb_done:.0f}/{mb_total:.0f} MB",
                    is_cache_hit=False,
                )
            )
        else:
            on_update(DownloadProgress(percent=0.0, detail=f"{mb_done:.0f} MB", is_cache_hit=False))

    return _on_progress


class _CallbackProgressBar(_base_tqdm):
    """tqdm subclass that forwards progress to a plain callback.
    Fully suppresses terminal output by disabling tqdm rendering and redirecting
    its file handle to a devnull sink: prevents ANSI escape sequences from leaking
    into Textual's managed terminal.

    Overrides ``get_lock`` to return a threading lock instead of tqdm's default
    multiprocessing lock. Vanilla tqdm acquires ``self._lock`` even on the
    ``disable=True`` path (std.py:988), and the multiprocessing lock's lazy init
    raises ``ValueError`` when ``sys.stderr.fileno() == -1`` (Textual, Jupyter,
    pytest capture). A thread lock sidesteps that fd handling entirely.
    """

    _lock = threading.RLock()
    _callback: Any = None

    @classmethod
    def get_lock(cls) -> threading.RLock:
        return cls._lock

    def __init__(self, *args: Any, **kwargs: Any):
        kwargs["disable"] = True
        kwargs["file"] = io.StringIO()  # absorb any accidental tqdm output
        super().__init__(*args, **kwargs)
        self._cumulative = 0

    def update(self, n: float = 1) -> bool | None:
        self._cumulative += int(n)
        if self._callback is not None:
            total = self.total if self.total is not None else 0
            self._callback(int(self._cumulative), int(total))
        return None


class _ProgressTracker:
    """Wraps a tqdm_class to detect whether progress updates actually fired."""

    def __init__(self, callback: Any) -> None:
        self.was_used = False
        self._callback = callback

    def make_tqdm_class(self) -> type[_base_tqdm]:
        tracker = self

        class _Cls(_CallbackProgressBar):
            _callback = staticmethod(tracker._callback)

            def update(self, n: float = 1) -> bool | None:
                tracker.was_used = True
                return super().update(n)

        return _Cls
