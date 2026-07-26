"""Module-level helpers used by ChatScreen: progress callbacks, file cleanup, stream close."""

from __future__ import annotations

import contextlib
import logging
import subprocess
import sys
import time
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.widgets.task_bar_controller import ProgressReporter
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


def close_stream(stream: Any) -> None:
    """Close a streaming iterator if it satisfies the ClosableIterator protocol."""
    if isinstance(stream, ClosableIterator):
        with contextlib.suppress(Exception):
            stream.close()


def _opener_argv(platform: str) -> list[str] | None:
    """The platform's open-with-default-app command, or None to use the browser."""
    if platform == "darwin":
        return ["open"]
    if platform.startswith("linux"):
        return ["xdg-open"]
    return None


def open_local_file(href: str) -> None:
    """Open a ``file:`` URL with the OS opener so it lands in the default app
    for its type (an editor for markdown, a viewer for PDF), not a browser
    rendering raw text. Platforms without a known opener fall back to the
    webbrowser module."""
    argv = _opener_argv(sys.platform)
    if argv is None:
        webbrowser.open(href)
        return
    path = url2pathname(urlparse(href).path)
    try:
        subprocess.run([*argv, path], check=False, timeout=10)  # noqa: S603 - fixed opener command; path comes from lilbee's own store
    except (OSError, subprocess.TimeoutExpired):
        log.warning("Could not open source file: %s", path)


def detail_for_batch_progress(data: BatchProgressEvent, in_flight: list[str]) -> str:
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


_PREFERENCE_PREFIX = "pref:"


@dataclass(frozen=True)
class RememberOutcome:
    """A /remember result: the toast message plus the notify severity to use."""

    message: str
    severity: str = "information"


def remember_from_input(raw: str) -> RememberOutcome:
    """Parse, gate, and store a ``/remember`` command; return the toast outcome.

    Pure orchestration so the ``@work`` worker body stays a single call and the
    parse/gate/store path is testable without a running TUI. A leading
    ``pref:`` marks the text as an always-recalled preference; anything else is
    stored as a fact.
    """
    from lilbee.app.memory import MEMORY_DISABLED_HINT, memory_enabled, remember
    from lilbee.app.services import get_services
    from lilbee.data.store import MemoryKind

    if not memory_enabled():
        return RememberOutcome(MEMORY_DISABLED_HINT, "warning")

    text = raw.strip()
    kind = MemoryKind.FACT
    if text[: len(_PREFERENCE_PREFIX)].lower() == _PREFERENCE_PREFIX:
        kind = MemoryKind.PREFERENCE
        text = text[len(_PREFERENCE_PREFIX) :].strip()
    if not text:
        return RememberOutcome(msg.CMD_REMEMBER_USAGE, "warning")

    if not get_services().embedder.embedding_available():
        return RememberOutcome(msg.CMD_REMEMBER_NO_EMBED, "warning")

    remember(text, kind=kind)
    return RememberOutcome(msg.CMD_REMEMBER_SUCCESS.format(kind=kind.value))


def unregister_added_roots(labels: list[str]) -> None:
    """Un-register roots a /add invocation created, for cancel/failure cleanup.

    Called on cancel or failure of the add task so a cancelled source is not
    re-found on the next sync. Only the registry entries this invocation added are
    dropped; the source bytes on disk and files the user owns are never touched.
    """
    from lilbee.app.ingest import unregister_roots

    if labels:
        unregister_roots(labels)


def _throttled_embed_tick(reporter: ProgressReporter) -> Callable[[EmbedEvent], None]:
    """Return the throttled EMBED tick shared by the add/sync/import callbacks."""
    last_update = 0.0

    def _tick(data: EmbedEvent) -> None:
        nonlocal last_update
        now = time.monotonic()
        if now - last_update < _ADD_EMBED_THROTTLE_SECONDS:
            return
        last_update = now
        pct = int(data.chunk * 100 / data.total_chunks) if data.total_chunks else 0
        reporter.update(pct, msg.SYNC_EMBEDDING.format(file=data.file), indeterminate=False)

    return _tick


def build_add_progress_callback(reporter: ProgressReporter) -> DetailedProgressCallback:
    """Build the on_progress callback used by /add.

    Tracks files in flight in start order so the displayed filename pins
    to the oldest unfinished file (the pipeline runs files concurrently;
    without pinning the label flips around the queue). EXTRACT surfaces
    "extracted N pages" once per file so a 44MB scanned PDF doesn't read
    as a hang; EMBED ticks per chunk, throttled to a steady cadence.
    """
    in_flight: list[str] = []
    embed_tick = _throttled_embed_tick(reporter)

    def on_progress(event_type: EventType, data: ProgressEvent) -> None:
        reporter.check_cancelled()
        if event_type == EventType.FILE_START and isinstance(data, FileStartEvent):
            in_flight.append(data.file)
            reporter.update(0, msg.ADD_SYNCING_FILE.format(file=in_flight[0]), indeterminate=True)
        elif event_type == EventType.FILE_DONE and isinstance(data, FileDoneEvent):
            with contextlib.suppress(ValueError):
                in_flight.remove(data.file)
        elif event_type == EventType.BATCH_PROGRESS and isinstance(data, BatchProgressEvent):
            pct = (data.current / data.total * 100.0) if data.total else 0.0
            reporter.update(pct, detail_for_batch_progress(data, in_flight), indeterminate=False)
        elif event_type == EventType.EXTRACT and isinstance(data, ExtractEvent):
            reporter.update(
                0,
                msg.SYNC_FILE_PROGRESS.format(
                    current=data.page, total=data.total_pages, file=data.file
                ),
                indeterminate=True,
            )
        elif event_type == EventType.EMBED and isinstance(data, EmbedEvent):
            embed_tick(data)

    return on_progress


def build_sync_progress_callback(
    reporter: ProgressReporter,
) -> Callable[[EventType, ProgressEvent], None]:
    """Return the on_progress shim used by ``_do_sync``.

    EXTRACT mirrors the /add path: a 44MB scanned PDF needs a per-page
    tick or the row reads as frozen.
    """
    embed_tick = _throttled_embed_tick(reporter)

    def on_progress(event_type: EventType, data: ProgressEvent) -> None:
        # Mirror /add: explicit cancel check on every event so a SYNC task
        # cancelled mid-batch stops at the next progress tick instead of
        # finishing the current file. update() also checks, but events
        # without a reporter.update call (e.g. BATCH_PROGRESS in the
        # ingest_stream path) would otherwise miss the cooperative checkpoint.
        reporter.check_cancelled()
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
            embed_tick(data)
        elif event_type == EventType.DONE and isinstance(data, SyncDoneEvent):
            total = data.added + data.updated + data.removed
            reporter.update(100, msg.SYNC_STATUS_DONE.format(count=total), indeterminate=False)

    return on_progress


def build_import_progress_callback(reporter: ProgressReporter) -> DetailedProgressCallback:
    """Build the on_progress callback used by /import (EMBED events only)."""
    embed_tick = _throttled_embed_tick(reporter)

    def on_progress(event_type: EventType, data: ProgressEvent) -> None:
        reporter.check_cancelled()
        if event_type == EventType.EMBED and isinstance(data, EmbedEvent):
            embed_tick(data)

    return on_progress
