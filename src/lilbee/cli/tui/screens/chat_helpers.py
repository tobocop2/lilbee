"""Module-level helpers used by ChatScreen: progress callbacks, file cleanup, stream close."""

from __future__ import annotations

import contextlib
import ctypes
import errno
import logging
import os
import subprocess
import sys
import time
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse
from urllib.request import url2pathname

from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.task_queue import TaskReport
from lilbee.cli.tui.widgets.task_bar_controller import ProgressReporter
from lilbee.data.ingest.discovery import classify_file, path_within_roots
from lilbee.data.types import CODE_CONTENT_TYPE
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
    OcrStartEvent,
    ProgressEvent,
    SyncDoneEvent,
)

if TYPE_CHECKING:
    from lilbee.data.types import SyncResult

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


class LinkOpenResult(StrEnum):
    """What opening a ``file:`` link did."""

    OPENED = "opened"
    OUTSIDE_ROOTS = "outside_roots"
    NOT_A_DOCUMENT = "not_a_document"
    RESOURCE_FORK = "resource_fork"


LINK_REFUSAL_NOTICES = {
    LinkOpenResult.OUTSIDE_ROOTS: msg.LINK_NOT_OPENED,
    LinkOpenResult.NOT_A_DOCUMENT: msg.LINK_NOT_A_DOCUMENT,
    LinkOpenResult.RESOURCE_FORK: msg.LINK_RESOURCE_FORK,
}

# Source types the ingest accepts whose default handler can run them as a program.
_LAUNCHING_CODE_SUFFIXES = frozenset(
    {
        ".sh",
        ".bash",
        ".zsh",
        ".ksh",
        ".fish",
        ".desktop",
        ".bat",
        ".cmd",
        ".ps1",
        ".psm1",
        ".py",
        ".pyw",
        ".applescript",
    }
)

# A Finder alias or bookmark file starts with this data; it opens its target.
_BOOKMARK_MAGIC = b"book\0\0\0\0mark"

_FINDER_INFO = b"com.apple.FinderInfo"
_FINDER_INFO_SIZE = 32
_FINDER_FLAGS_OFFSET = 8
_IS_ALIAS_FLAG = 0x8000
_RESOURCE_FORK = b"com.apple.ResourceFork"
_ENOATTR_DARWIN = 93
# getxattr(2) errors that mean the file has no such attribute or the volume has none.
_ABSENT_XATTR_ERRNOS = frozenset({_ENOATTR_DARWIN, errno.ENOTSUP})


def _xattr_size(path: Path, name: bytes, buf: ctypes.Array[ctypes.c_char] | None = None) -> int:
    """Size of the extended attribute *name* of *path*, read into *buf* when given.

    0 when the file has no such attribute; raises OSError on any other read error.
    """
    libc = ctypes.CDLL(None, use_errno=True)  # os.getxattr does not exist on macOS
    libc.getxattr.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_int,
    ]
    libc.getxattr.restype = ctypes.c_ssize_t
    size = libc.getxattr(os.fsencode(path), name, buf, 0 if buf is None else len(buf), 0, 0)
    if size >= 0:
        return int(size)
    err = ctypes.get_errno()
    if err in _ABSENT_XATTR_ERRNOS:
        return 0
    raise OSError(err, os.strerror(err), str(path))


def _has_alias_flag(path: Path) -> bool:
    """Whether the Finder flags of *path* mark it as an alias; raises OSError on a read error."""
    buf = ctypes.create_string_buffer(_FINDER_INFO_SIZE)
    _xattr_size(path, _FINDER_INFO, buf)
    flags = int.from_bytes(buf.raw[_FINDER_FLAGS_OFFSET : _FINDER_FLAGS_OFFSET + 2], "big")
    return bool(flags & _IS_ALIAS_FLAG)


def _has_resource_fork(path: Path) -> bool:
    """Whether *path* has a non-empty macOS resource fork; raises OSError on a read error."""
    return _xattr_size(path, _RESOURCE_FORK) > 0


def _is_plain_file(path: Path) -> bool:
    """Whether *path* is a regular file and not an alias; False on a read error."""
    try:
        if not path.is_file():
            return False
        with path.open("rb") as handle:
            if handle.read(len(_BOOKMARK_MAGIC)) == _BOOKMARK_MAGIC:
                return False
        return not (sys.platform == "darwin" and _has_alias_flag(path))
    except OSError:
        return False


def _is_openable_document(path: Path) -> bool:
    """Whether *path* is a plain file of a type the ingest accepts and whose
    default handler shows it instead of running it."""
    if not _is_plain_file(path):
        return False
    content_type = classify_file(path)
    if content_type is None:
        return False
    if content_type != CODE_CONTENT_TYPE:
        return True
    suffix = path.suffix.lower()
    return sys.platform != "win32" and len(suffix) > 1 and suffix not in _LAUNCHING_CODE_SUFFIXES


def _document_refusal(path: Path) -> LinkOpenResult | None:
    """Why the opener refuses *path*, or None when it is a document the opener shows."""
    if not _is_openable_document(path):
        return LinkOpenResult.NOT_A_DOCUMENT
    try:
        forked = sys.platform == "darwin" and _has_resource_fork(path)
    except OSError:
        return LinkOpenResult.NOT_A_DOCUMENT
    return LinkOpenResult.RESOURCE_FORK if forked else None


def open_local_file(href: str) -> LinkOpenResult:
    """Open a ``file:`` URL with the OS opener so it lands in the default app
    for its type (an editor for markdown, a viewer for PDF), not a browser
    rendering raw text. Platforms without a known opener fall back to the
    webbrowser module. Only a document inside the document roots is opened;
    any other path, and any file that is not a plain ingestible document, opens nothing."""
    path = url2pathname(urlparse(href).path)
    try:
        resolved = Path(path).resolve(strict=False)
        inside = "\0" not in path and path_within_roots(resolved)
    except (OSError, ValueError, RuntimeError):
        inside = False
    if not inside:
        log.warning("Refused to open a file outside the document roots: %s", path)
        return LinkOpenResult.OUTSIDE_ROOTS
    refusal = _document_refusal(resolved)
    if refusal is not None:
        log.warning("Refused to open a file (%s): %s", refusal, resolved)
        return refusal
    argv = _opener_argv(sys.platform)
    if argv is None:
        webbrowser.open(resolved.as_uri())
        return LinkOpenResult.OPENED
    try:
        subprocess.run([*argv, str(resolved)], check=False, timeout=10)  # noqa: S603 - fixed opener command; resolved path is a document inside the document roots
    except (OSError, subprocess.TimeoutExpired):
        log.warning("Could not open source file: %s", resolved)
    return LinkOpenResult.OPENED


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
    """Un-register roots a /add invocation created, when its finished sync failed them.

    Only the registry entries this invocation added are dropped, with the skip
    records its sync wrote under them; the source bytes on disk and files the
    user owns are never touched.
    """
    from lilbee.app.ingest import forget_roots

    if labels:
        forget_roots(labels)


def ingest_report(result: SyncResult) -> TaskReport:
    """What *result* did not fully index, for the task row and its detail view."""
    return TaskReport(
        partial=tuple(result.partial),
        failed={name: result.reasons.get(name, "") for name in result.failed},
        skipped={name: result.reasons.get(name, "") for name in result.skipped},
        indexed=len(result.added) + len(result.updated) + len(result.relocated),
    )


def report_ingest(reporter: ProgressReporter, result: SyncResult) -> TaskReport:
    """Attach *result*'s report to the task and show its counts on the row."""
    report = ingest_report(result)
    if report.has_problems:
        reporter.set_report(report)
        reporter.update(100, msg.task_report_summary(report), indeterminate=False)
    return report


def add_indexed_anything(registered: list[str], result: SyncResult) -> bool:
    """Whether any file under this add's registered roots reached the index.

    Sync is global, so its added/updated/relocated lists can name files from
    other sources; only names keyed under a registered label count. A directory
    root keys files as ``label/relpath``; a single-file root keys as ``label``.
    """
    indexed = (*result.added, *result.updated, *result.relocated)
    return any(
        name == root or name.startswith(f"{root}/") for root in registered for name in indexed
    )


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


def _show_tesseract_ocr(reporter: ProgressReporter, data: OcrStartEvent) -> None:
    """Show that Tesseract is OCRing a file, until its first page completes."""
    reporter.update(
        0,
        msg.SYNC_TESSERACT_OCR.format(total=data.total_pages, file=data.file),
        indeterminate=True,
    )


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
        elif event_type == EventType.OCR_START and isinstance(data, OcrStartEvent):
            _show_tesseract_ocr(reporter, data)
        elif event_type == EventType.EMBED and isinstance(data, EmbedEvent):
            embed_tick(data)
        elif event_type == EventType.SYNC_WAIT:
            _show_sync_wait(reporter)

    return on_progress


def _show_sync_wait(reporter: ProgressReporter) -> None:
    """Show on the task bar that this sync waits for another; the sync logs the full notice."""
    reporter.update(0, msg.SYNC_WAITING, indeterminate=True)


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
        elif event_type == EventType.OCR_START and isinstance(data, OcrStartEvent):
            _show_tesseract_ocr(reporter, data)
        elif event_type == EventType.EMBED and isinstance(data, EmbedEvent):
            embed_tick(data)
        elif event_type == EventType.SYNC_WAIT:
            _show_sync_wait(reporter)
        elif event_type == EventType.SYNC_DONE and isinstance(data, SyncDoneEvent):
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
