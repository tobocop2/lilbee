"""Sync and add-files handlers (SSE-streamed)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from collections.abc import AsyncGenerator, Callable, Coroutine
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from lilbee.app.ingest import copy_files
from lilbee.app.services import get_services
from lilbee.core.config import cfg
from lilbee.core.security import validate_path_within
from lilbee.runtime.progress import SseEvent
from lilbee.server.handlers.sse import SseStream, sse_done, sse_error, sse_event
from lilbee.server.models import AddSummary, SyncSummary

if TYPE_CHECKING:
    from lilbee.app.dataset import ImportSummary
    from lilbee.data.ingest import SyncResult

log = logging.getLogger(__name__)

# Payload carried alongside each source key through _ingest_stream: a server
# path (str) for /api/add, or an (filename, content) pair for /api/add/upload.
_Payload = TypeVar("_Payload")


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
    # No file-count cap: the resource guard is the app's size-based
    # request_max_body_size; a count limit only breaks the point-lilbee-at-
    # your-codebase use case (hundreds of small files).

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


async def _ingest_stream(
    items: list[tuple[str, _Payload]],
    run: Callable[[list[_Payload], SseStream], Coroutine[Any, Any, AddSummary]],
    label: str,
) -> AsyncGenerator[str, None]:
    """Lock per source, run ``run`` over the acquired subset, and stream SSE.

    Shared by /api/add (server paths) and /api/add/upload (uploaded content).
    Each item is ``(lock_key, payload)``: the key is the source identifier used
    for the per-source ingest lock; the payload is what ``run`` receives for the
    subset whose lock was acquired. Contended sources emit ``already_ingesting``
    and the stream closes without a ``done`` event, signalling the client to wait
    rather than retry.
    """
    registry = get_services().ingest_lock_registry
    acquired, busy = await registry.acquire([key for key, _payload in items])
    try:
        for name in busy:
            log.info("Rejecting %s for %s: already ingesting", label, name)
            yield sse_event(SseEvent.ALREADY_INGESTING, {"source": name})

        if not acquired:
            return

        acquired_names = {name for name, _lock in acquired}
        locked = [
            payload
            for key, payload in items
            if registry.canonical_source_name(key) in acquired_names
        ]
        sse = SseStream()
        task = asyncio.create_task(run(locked, sse))
        try:
            async for event in sse.drain(task, label):
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
        registry.release(acquired)


async def add_files_stream(
    paths: list[str],
    *,
    force: bool = False,
    enable_ocr: bool | None = None,
    ocr_timeout: float | None = None,
) -> AsyncGenerator[str, None]:
    """Copy server-side files, sync, and yield SSE progress events.

    Takes the already-validated/parsed values from ``validate_add_paths`` so the
    request dict is decoded once.
    """
    async for event in _ingest_stream(
        [(p, p) for p in paths],
        lambda locked, sse: _run_add(locked, force, enable_ocr, ocr_timeout, sse),
        "Add files stream",
    ):
        yield event


def _clean_upload_name(name: str) -> str:
    """Normalize one upload filename to a safe relative path inside the corpus.

    Relative paths are preserved (a source tree keeps its layout instead of
    colliding on basenames); absolute paths, drive letters, and ``..`` segments
    are rejected. Raises ValueError on bad input.
    """
    normalized = name.replace("\\", "/")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        raise ValueError(f"upload filename must be relative: {name!r}")
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if not parts:
        raise ValueError(f"invalid upload filename: {name!r}")
    if ".." in parts:
        raise ValueError(f"upload filename may not contain '..': {name!r}")
    relative = "/".join(parts)
    validate_path_within(cfg.documents_dir / relative, cfg.documents_dir)
    return relative


def validate_uploads(files: list[tuple[str, bytes]]) -> list[tuple[str, bytes]]:
    """Validate uploaded (filename, content) pairs. Raises ValueError on bad input.

    Filenames keep their relative path, validated to stay inside
    ``cfg.documents_dir``, so an uploaded source tree preserves its layout
    instead of colliding on basenames. There is no file-count cap; the
    resource guard is the app's size-based request_max_body_size.
    """
    if not files:
        raise ValueError("no files uploaded")
    return [(_clean_upload_name(name), content) for name, content in files]


async def _run_upload(files: list[tuple[str, bytes]], sse: SseStream) -> AddSummary:
    """Write uploaded file bytes into ``cfg.documents_dir``, then sync.

    The upload equivalent of :func:`_run_add`: instead of copying from a
    server-readable path, the client's content is written straight into the
    documents dir and the same ingest pipeline runs. This is what lets an
    external-mode client (a remote lilbee / GPU box) ingest files that live only
    on the client. Unchanged content is a no-op re-embed inside ``sync`` (it
    hashes each source), so there is no separate force flag.
    """
    from lilbee.app.ingest import temporary_ocr_config
    from lilbee.data.ingest import sync

    try:
        cfg.documents_dir.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        for name, content in files:
            dest = cfg.documents_dir / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)
            written.append(name)
        with temporary_ocr_config(None):
            sync_result = await sync(quiet=True, on_progress=sse.callback, cancel=sse.cancel)
        return AddSummary(
            copied=written,
            skipped=[],
            errors=[],
            sync=SyncSummary(**sync_result.model_dump()),
        )
    finally:
        sse.queue.put_nowait(None)


async def add_uploads_stream(files: list[tuple[str, bytes]]) -> AsyncGenerator[str, None]:
    """Ingest uploaded file content, yielding the same SSE progress as add_files_stream.

    Locks per source name (the validated relative path) so an upload never
    races an in-flight add of the same source.
    """
    async for event in _ingest_stream(
        [(name, (name, content)) for name, content in files],
        lambda locked, sse: _run_upload(locked, sse),
        "Add uploads stream",
    ):
        yield event


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
