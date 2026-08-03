"""Download progress callback plumbing shared by all surfaces.

The TUI runs under Textual which owns the terminal; tqdm output to
stderr/stdout corrupts the screen. This module provides a tqdm subclass
(``_CallbackProgressBar``) that suppresses terminal output and tracks
cumulative bytes; the ``_ProgressTracker`` wrapper produces a further subclass
that forwards progress to a plain ``Callable[[int, int], None]`` callback
(aggregating across split shards) and detects whether progress events actually
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
    """tqdm subclass that suppresses terminal output and tracks cumulative bytes.

    ``_ProgressTracker`` produces a further subclass of this that forwards the
    progress to a callback.
    Fully suppresses terminal output by disabling tqdm rendering and redirecting
    its file handle to a devnull sink: prevents ANSI escape sequences from leaking
    into Textual's managed terminal.

    Overrides ``get_lock`` to return a threading lock instead of tqdm's default
    multiprocessing lock. Vanilla tqdm acquires ``self._lock`` even on the
    ``disable=True`` path (std.py:988), and the multiprocessing lock's lazy init
    raises ``ValueError`` when ``sys.stderr.fileno() == -1`` (Textual, Jupyter,
    pytest capture). A thread lock sidesteps that fd handling entirely.

    huggingface_hub reports two byte streams. ``update`` carries bytes written to
    disk; ``update_transfer`` carries bytes off the network. On the xet path the
    disk stream only moves when a buffered block flushes (a handful of events for
    a whole file), while the transfer stream moves continuously. On the HTTP path
    both fire for the same chunk. Tracking them separately and reporting the
    larger keeps xet smooth without double-counting HTTP.
    """

    _lock = threading.RLock()

    @classmethod
    def get_lock(cls) -> threading.RLock:
        return cls._lock

    def __init__(self, *args: Any, **kwargs: Any):
        kwargs["disable"] = True
        kwargs["file"] = io.StringIO()  # absorb any accidental tqdm output
        super().__init__(*args, **kwargs)
        # tqdm's `initial` is the resume offset: bytes already on disk from an
        # interrupted attempt, which huggingface_hub does not re-report. Seeding
        # both counters from it keeps a resumed download's percentage absolute.
        self._written = int(self.n)
        self._transferred = int(self.n)

    @property
    def _cumulative(self) -> int:
        return max(self._written, self._transferred)

    def update(self, n: float = 1) -> bool | None:
        # The base only tracks cumulative bytes and suppresses output; forwarding
        # to the callback (with split-shard aggregation) lives in the
        # _ProgressTracker subclass below.
        self._written += int(n)
        return None

    def update_transfer(self, n: float = 1) -> bool | None:
        """Absorb the network-bytes stream.

        huggingface_hub routes xet transfer progress into this bar only when the
        method exists; otherwise it opens its own tqdm on stderr alongside.
        """
        self._transferred += int(n)
        return None

    def set_transfer_postfix_str(self, s: str = "", refresh: bool = True) -> None:
        """No-op: huggingface_hub sets a transfer rate on bars that take transfer."""


class _ProgressTracker:
    """Wraps a tqdm_class to detect updates and aggregate across split shards.

    For a multi-shard GGUF, each shard gets its own tqdm. Reporting each shard's
    own ``(done, total)`` would show N separate 0->100% cycles against the wrong
    total; instead the tracker carries ``grand_total`` (all shards) and a
    ``completed_base`` (bytes from finished shards), so the callback sees one
    monotonic 0->100% over the whole download. ``grand_total`` of 0 means
    single-file: fall back to the shard's own tqdm total (unchanged behavior).
    """

    def __init__(self, callback: Any, grand_total: int = 0) -> None:
        self.was_used = False
        self._callback = callback
        self.grand_total = grand_total
        self._completed_base = 0

    def shard_done(self, shard_size: int) -> None:
        """Roll a finished shard's bytes into the base for the next shard."""
        self._completed_base += shard_size

    def make_tqdm_class(self) -> type[_base_tqdm]:
        tracker = self

        class _Cls(_CallbackProgressBar):
            def _report(self) -> None:
                tracker.was_used = True
                done = tracker._completed_base + self._cumulative
                shard_total = self.total if self.total is not None else 0
                total = tracker.grand_total or (tracker._completed_base + shard_total)
                tracker._callback(int(done), int(total))

            def update(self, n: float = 1) -> bool | None:
                super().update(n)
                self._report()
                return None

            def update_transfer(self, n: float = 1) -> bool | None:
                super().update_transfer(n)
                self._report()
                return None

        return _Cls
