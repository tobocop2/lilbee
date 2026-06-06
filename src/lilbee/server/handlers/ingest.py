"""Sync and add-files handlers (SSE-streamed)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lilbee.app.ingest import copy_files
from lilbee.app.services import get_services
from lilbee.core.config import cfg
from lilbee.core.security import validate_path_within
from lilbee.runtime.ingest_lock import IngestLockRegistry
from lilbee.runtime.progress import SseEvent
from lilbee.server.handlers.sse import SseStream, sse_done, sse_error, sse_event
from lilbee.server.models import AddSummary, SyncSummary

if TYPE_CHECKING:
    from lilbee.app.dataset import ImportSummary
    from lilbee.data.ingest import SyncResult

log = logging.getLogger(__name__)

MAX_ADD_FILES = 100


async def _run_sync_with_sentinel(
    sse: SseStream,
    enable_ocr: bool | None,
    force_rebuild: bool = False,
    retry_skipped: bool = False,
) -> SyncResult:
    """Run ingest.sync() and guarantee the drain sentinel is enqueued."""
    from lilbee.app.ingest import temporary_ocr_config
    from lilbee.data.ingest import sync

    try:
        with temporary_ocr_config(enable_ocr):
            return await sync(
                quiet=True,
                on_progress=sse.callback,
                cancel=sse.cancel,
                force_rebuild=force_rebuild,
                retry_skipped=retry_skipped,
            )
    finally:
        sse.queue.put_nowait(None)


async def sync_stream(
    *, enable_ocr: bool | None = None, force_rebuild: bool = False, retry_skipped: bool = False
) -> AsyncGenerator[str, None]:
    """Trigger sync, yield SSE progress events, then done event.

    When ``force_rebuild`` is true, the underlying sync drops every table and
    re-ingests from ``cfg.documents_dir`` (the REST equivalent of ``lilbee rebuild``).
    When ``retry_skipped`` is true, it clears the failed-file markers so files
    that were skipped on a previous sync get another attempt, without dropping
    the store.
    """
    sse = SseStream()
    task = asyncio.create_task(
        _run_sync_with_sentinel(sse, enable_ocr, force_rebuild, retry_skipped)
    )
    async for event in sse.drain(task, "Sync stream"):
        yield event
    if not sse.cancel.is_set() and task.done() and not task.cancelled():
        exc = task.exception()
        if exc is not None:
            yield sse_error(str(exc))
            return
        yield sse_done(task.result().model_dump())


async def _run_add(
    paths: list[str],
    force: bool,
    enable_ocr: bool | None,
    ocr_timeout: float | None,
    sse: SseStream,
) -> AddSummary:
    """Copy files and sync, returning the summary for the final done event."""
    from lilbee.app.ingest import temporary_ocr_config
    from lilbee.data.ingest import sync

    try:
        errors: list[str] = []
        valid: list[Path] = []
        for p_str in paths:
            p = Path(p_str)
            if not p.exists():
                errors.append(p_str)
            else:
                valid.append(p)

        copy_result = copy_files(valid, force=force)

        if sse.cancel.is_set():
            return AddSummary(copied=copy_result.copied, skipped=copy_result.skipped, errors=errors)

        with temporary_ocr_config(enable_ocr, ocr_timeout):
            sync_result = await sync(quiet=True, on_progress=sse.callback, cancel=sse.cancel)

        return AddSummary(
            copied=copy_result.copied,
            skipped=copy_result.skipped,
            errors=errors,
            sync=SyncSummary(**sync_result.model_dump()),
        )
    finally:
        sse.queue.put_nowait(None)


def validate_add_paths(
    data: dict[str, Any],
) -> tuple[list[str], bool, bool | None, float | None]:
    """Validate add-files input. Raises ValueError on bad input."""
    paths = data.get("paths")
    if not isinstance(paths, list) or not paths:
        raise ValueError("'paths' must be a non-empty list of strings")
    if len(paths) > MAX_ADD_FILES:
        raise ValueError(f"Too many files: {len(paths)} exceeds limit of {MAX_ADD_FILES}")

    for p_str in paths:
        validate_path_within(cfg.documents_dir / Path(p_str).name, cfg.documents_dir)

    force = bool(data.get("force", False))
    enable_ocr, ocr_timeout = _parse_ocr_params(data)
    return paths, force, enable_ocr, ocr_timeout


def _parse_ocr_params(data: dict[str, Any]) -> tuple[bool | None, float | None]:
    """Extract and coerce OCR parameters from a request dict."""
    enable_ocr = data.get("enable_ocr")
    ocr_timeout = data.get("ocr_timeout")
    if enable_ocr is not None:
        enable_ocr = bool(enable_ocr)
    if ocr_timeout is not None:
        ocr_timeout = float(ocr_timeout)
    return enable_ocr, ocr_timeout


async def add_files_stream(data: dict[str, Any]) -> AsyncGenerator[str, None]:
    """Copy files, sync, and yield SSE progress events.

    Contended sources emit ``already_ingesting`` and the stream closes
    without a ``done`` event, signalling the client to wait rather than retry.
    """
    paths = data.get("paths", [])
    force = bool(data.get("force", False))
    enable_ocr, ocr_timeout = _parse_ocr_params(data)

    registry = get_services().ingest_lock_registry
    acquired, busy = await registry.acquire(paths)
    try:
        for name in busy:
            log.info("Rejecting /api/add for %s: already ingesting", name)
            yield sse_event(SseEvent.ALREADY_INGESTING, {"source": name})

        if not acquired:
            return

        sse = SseStream()
        task = asyncio.create_task(_run_add(paths, force, enable_ocr, ocr_timeout, sse))
        try:
            async for event in sse.drain(task, "Add files stream"):
                yield event
            if not sse.cancel.is_set() and task.done() and not task.cancelled():
                exc = task.exception()
                if exc is not None:
                    yield sse_error(str(exc))
                    return
                summary = task.result()
                yield sse_done(summary.model_dump())
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
    finally:
        IngestLockRegistry.release(acquired)


async def _run_import_with_sentinel(sse: SseStream, data: bytes, fmt: str) -> ImportSummary:
    """Run the dataset import and guarantee the drain sentinel is enqueued."""
    from lilbee.app.dataset import import_from_bytes

    try:
        return await import_from_bytes(data, fmt, on_progress=sse.callback)
    finally:
        sse.queue.put_nowait(None)


async def import_stream(data: bytes, fmt: str) -> AsyncGenerator[str, None]:
    """Import a dataset, yield SSE embed-progress events, then a done event."""
    sse = SseStream()
    task = asyncio.create_task(_run_import_with_sentinel(sse, data, fmt))
    async for event in sse.drain(task, "Import stream"):
        yield event
    if not sse.cancel.is_set() and task.done() and not task.cancelled():
        exc = task.exception()
        if exc is not None:
            yield sse_error(str(exc))
            return
        yield sse_done(task.result().model_dump())
