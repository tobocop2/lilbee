"""Background sync, executor management, and sync status for chat mode."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

from rich.console import Console

from lilbee.cli import theme
from lilbee.ingest import sync
from lilbee.progress import EventType, ExtractEvent, FileStartEvent, ProgressEvent, SyncDoneEvent

if TYPE_CHECKING:
    from lilbee.progress import DetailedProgressCallback


def _format_sync_summary(added: int, updated: int, removed: int, failed: int) -> str | None:
    """Format sync counts into a human-readable summary, or None if nothing changed."""
    counts = {"added": added, "updated": updated, "removed": removed, "failed": failed}
    parts = [f"{n} {label}" for label, n in counts.items() if n]
    return ", ".join(parts) if parts else None


def _sync_progress_printer(con: Console) -> DetailedProgressCallback:
    """Return a callback that prints one-line status for FILE_START and DONE events."""

    def _callback(event_type: EventType, data: ProgressEvent) -> None:
        if event_type == EventType.FILE_START:
            assert isinstance(data, FileStartEvent)
            m = theme.MUTED
            con.print(f"[{m}]Syncing [{data.current_file}/{data.total_files}]: {data.file}[/{m}]")
        elif event_type == EventType.DONE:
            assert isinstance(data, SyncDoneEvent)
            summary = _format_sync_summary(data.added, data.updated, data.removed, data.failed)
            if summary:
                con.print(f"[{theme.MUTED}]Synced: {summary}[/{theme.MUTED}]")

    return _callback


_bg_executor: ThreadPoolExecutor | None = None


def _get_executor() -> ThreadPoolExecutor:
    """Lazy-init a single-worker executor."""
    global _bg_executor
    if _bg_executor is None:
        _bg_executor = ThreadPoolExecutor(max_workers=1)
    return _bg_executor


def shutdown_executor() -> None:
    """Shut down the background executor without blocking.

    Uses wait=False + cancel_futures to avoid blocking the main thread.
    """
    global _bg_executor
    if _bg_executor is None:
        return

    _bg_executor.shutdown(wait=False, cancel_futures=True)
    _bg_executor = None


def _on_sync_done(con: Console, future: Future[object], *, chat_mode: bool = False) -> None:
    """Callback attached to background sync futures — logs errors."""
    exc = future.exception()
    if exc is None:
        return
    if isinstance(exc, asyncio.CancelledError):
        return
    if isinstance(exc, RuntimeError) and "cannot schedule new futures" in str(exc):
        return
    if chat_mode:
        print(f"Background sync error: {exc}")
    else:
        con.print(f"[{theme.ERROR}]Background sync error:[/{theme.ERROR}] {exc}")


class SyncStatus:
    """Thread-safe holder for background sync status text.

    The background sync callback writes here; prompt_toolkit's
    ``bottom_toolbar`` reads it on every render cycle — no cursor
    manipulation, no flickering.
    """

    def __init__(self) -> None:
        self.text: str = ""
        self.pending: int = 0

    def clear(self) -> None:
        self.text = ""


def _chat_sync_callback(status: SyncStatus) -> DetailedProgressCallback:
    """Return a progress callback for chat-mode background sync.

    FILE_START updates *status.text* (rendered by prompt_toolkit's bottom
    toolbar).  On DONE the status is cleared and the summary is printed via
    ``print()`` (goes through StdoutProxy → appears above the prompt).
    """

    status.clear()

    def _callback(event_type: EventType, data: ProgressEvent) -> None:
        queue_suffix = f" (+{status.pending} queued)" if status.pending > 0 else ""
        if event_type == EventType.FILE_START:
            assert isinstance(data, FileStartEvent)
            status.text = (
                f"⟳ Syncing [{data.current_file}/{data.total_files}]: {data.file}{queue_suffix}"
            )
        elif event_type == EventType.EXTRACT:
            assert isinstance(data, ExtractEvent)
            status.text = (
                f"⟳ Vision OCR [{data.page}/{data.total_pages}]: {data.file}{queue_suffix}"
            )
        elif event_type == EventType.DONE:
            status.clear()
            assert isinstance(data, SyncDoneEvent)
            summary = _format_sync_summary(data.added, data.updated, data.removed, data.failed)
            if summary:
                print(f"✓ Synced: {summary}")

    return _callback


def run_sync_background(
    con: Console,
    *,
    force_vision: bool = False,
    chat_mode: bool = False,
    sync_status: SyncStatus | None = None,
) -> Future[object]:
    """Submit sync to a background thread. Returns the Future."""

    status = sync_status or SyncStatus()

    callback = _chat_sync_callback(status) if chat_mode else _sync_progress_printer(con)

    def _run() -> object:
        if chat_mode:
            status.pending -= 1
        return asyncio.run(sync(quiet=True, on_progress=callback, force_vision=force_vision))

    if chat_mode:
        status.pending += 1

    future = _get_executor().submit(_run)
    future.add_done_callback(lambda f: _on_sync_done(con, f, chat_mode=chat_mode))
    return future
