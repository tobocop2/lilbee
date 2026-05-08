"""Module-level helpers used by ChatScreen: progress callbacks, file cleanup, stream close."""

from __future__ import annotations

import contextlib
import logging
import shutil
import time
from collections.abc import Callable
from typing import Any

from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.widgets.task_bar_controller import ProgressReporter
from lilbee.core.config import cfg
from lilbee.providers.base import ClosableIterator
from lilbee.runtime.progress import (
    BatchProgressEvent,
    BatchStatus,
    DetailedProgressCallback,
    EmbedEvent,
    EventType,
    ExtractEvent,
    FileDoneEvent,
    FileStartEvent,
    ProgressEvent,
    SyncDoneEvent,
)

log = logging.getLogger(__name__)

_ADD_EMBED_THROTTLE_SECONDS = 0.15
"""Throttle EMBED reporter updates to avoid TaskBar update storms.

The embed worker fires one EmbedEvent per sub-batch, which on a fast
laptop can be dozens per second. The Task Center only repaints at 10 Hz
anyway, so we coalesce here at the same cadence.
"""


def _close_stream(stream: Any) -> None:
    """Close a streaming iterator if it satisfies the ClosableIterator protocol."""
    if isinstance(stream, ClosableIterator):
        with contextlib.suppress(Exception):
            stream.close()


def _detail_for_batch_progress(data: BatchProgressEvent, in_flight: list[str]) -> str:
    """Pick the user-facing detail label for a BATCH_PROGRESS tick.

    Per-page rasterization (vision OCR) is the only producer that uses
    BatchStatus.RASTERIZING; it emits an absolute path in data.file
    which never matches the relative source name kept in in_flight, so
    identity-based detection would never fire. Status-based dispatch is
    the reliable discriminator between per-page and per-file ticks.
    """
    if data.status == BatchStatus.RASTERIZING:
        return msg.ADD_PAGE_PROGRESS.format(
            status=data.status.capitalize(), current=data.current, total=data.total
        )
    if in_flight:
        return msg.ADD_SYNCING_FILE.format(file=in_flight[0])
    return msg.ADD_FILE_DONE.format(file=data.file)


def _remove_copied_files(names: list[str]) -> None:
    """Delete files previously copied into documents/ by a /add invocation.

    Called on cancel or failure of the add task so a cancelled file does not
    re-appear on the next sync. Silently tolerates missing entries;
    the user may have removed them concurrently, and the goal is just to
    prevent accidental indexing.
    """
    for name in names:
        target = cfg.documents_dir / name
        try:
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            elif target.exists():
                target.unlink()
        except OSError:
            log.debug("Could not remove copied file %s", target, exc_info=True)


def _build_add_progress_callback(reporter: ProgressReporter) -> DetailedProgressCallback:
    """Build the on_progress callback used by /add.

    Tracks files in flight in start order so the displayed filename pins
    to the oldest unfinished file (the pipeline runs files concurrently;
    without pinning the label flips around the queue). EXTRACT surfaces
    "extracted N pages" once per file so a 44MB scanned PDF doesn't read
    as a hang; EMBED ticks per chunk, throttled to a steady cadence.
    """
    in_flight: list[str] = []
    last_embed_update = 0.0

    def on_progress(event_type: EventType, data: ProgressEvent) -> None:
        nonlocal last_embed_update
        reporter.check_cancelled()
        if event_type == EventType.FILE_START and isinstance(data, FileStartEvent):
            in_flight.append(data.file)
            reporter.update(0, msg.ADD_SYNCING_FILE.format(file=in_flight[0]), indeterminate=True)
        elif event_type == EventType.FILE_DONE and isinstance(data, FileDoneEvent):
            with contextlib.suppress(ValueError):
                in_flight.remove(data.file)
        elif event_type == EventType.BATCH_PROGRESS and isinstance(data, BatchProgressEvent):
            pct = (data.current / data.total * 100.0) if data.total else 0.0
            reporter.update(pct, _detail_for_batch_progress(data, in_flight), indeterminate=False)
        elif event_type == EventType.EXTRACT and isinstance(data, ExtractEvent):
            reporter.update(
                0,
                msg.SYNC_FILE_PROGRESS.format(
                    current=data.page, total=data.total_pages, file=data.file
                ),
                indeterminate=True,
            )
        elif event_type == EventType.EMBED and isinstance(data, EmbedEvent):
            now = time.monotonic()
            if now - last_embed_update < _ADD_EMBED_THROTTLE_SECONDS:
                return
            last_embed_update = now
            pct = int(data.chunk * 100 / data.total_chunks) if data.total_chunks else 0
            reporter.update(pct, msg.SYNC_EMBEDDING.format(file=data.file), indeterminate=False)

    return on_progress


def _build_sync_progress_callback(
    reporter: ProgressReporter,
) -> Callable[[EventType, ProgressEvent], None]:
    """Return the on_progress shim used by ``_do_sync``.

    EXTRACT mirrors the /add path: a 44MB scanned PDF needs a per-page
    tick or the row reads as frozen.
    """
    last_embed_update = 0.0

    def on_progress(event_type: EventType, data: ProgressEvent) -> None:
        nonlocal last_embed_update
        if event_type == EventType.FILE_START and isinstance(data, FileStartEvent):
            pct = int((data.current_file - 1) * 100 / data.total_files)
            status = msg.SYNC_FILE_PROGRESS.format(
                current=data.current_file, total=data.total_files, file=data.file
            )
            reporter.update(pct, status, indeterminate=False)
        elif event_type == EventType.FILE_DONE and isinstance(data, FileDoneEvent):
            reporter.update(0, msg.SYNC_FILE_DONE.format(file=data.file), indeterminate=False)
        elif event_type == EventType.EXTRACT and isinstance(data, ExtractEvent):
            reporter.update(
                0,
                msg.SYNC_FILE_PROGRESS.format(
                    current=data.page, total=data.total_pages, file=data.file
                ),
                indeterminate=True,
            )
        elif event_type == EventType.EMBED and isinstance(data, EmbedEvent):
            now = time.monotonic()
            if now - last_embed_update < _ADD_EMBED_THROTTLE_SECONDS:
                return
            last_embed_update = now
            pct = int(data.chunk * 100 / data.total_chunks) if data.total_chunks else 0
            reporter.update(pct, msg.SYNC_EMBEDDING.format(file=data.file), indeterminate=False)
        elif event_type == EventType.DONE and isinstance(data, SyncDoneEvent):
            total = data.added + data.updated + data.removed
            reporter.update(100, msg.SYNC_STATUS_DONE.format(count=total), indeterminate=False)

    return on_progress
