"""Sync and add-files handlers (SSE-streamed)."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
import re
from collections.abc import AsyncGenerator, Callable, Coroutine
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from lilbee.app.ingest import register_sources
from lilbee.app.services import get_services
from lilbee.core.config import cfg
from lilbee.core.security import validate_path_within
from lilbee.data.ingest.discovery import excluded_extension_reasons
from lilbee.runtime.ingest_lock import IngestLockRegistry
from lilbee.runtime.progress import SseEvent
from lilbee.server.handlers.sse import SseStream, sse_event
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
    prune_ignored: bool = False,
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
                prune_ignored=prune_ignored,
            )
    finally:
        sse.queue.put_nowait(None)


async def sync_stream(
    *,
    enable_ocr: bool | None = None,
    force_rebuild: bool = False,
    retry_skipped: bool = False,
    prune_ignored: bool = False,
) -> AsyncGenerator[str, None]:
    """Trigger sync, yield SSE progress events, then done event.

    When ``force_rebuild`` is true, the underlying sync drops every table and
    re-ingests from ``cfg.documents_dir`` (the REST equivalent of ``lilbee rebuild``).
    When ``retry_skipped`` is true, it clears the failed-file markers so files
    that were skipped on a previous sync get another attempt, without dropping
    the store. When ``prune_ignored`` is true, it also drops sources a
    ``.lilbeeignore`` now excludes.
    """
    sse = SseStream()
    task = asyncio.create_task(
        _run_sync_with_sentinel(sse, enable_ocr, force_rebuild, retry_skipped, prune_ignored)
    )
    async for event in sse.drain(task, "Sync stream"):
        yield event
    frame = sse.terminal_frame(task, lambda result: result.model_dump())
    if frame is not None:
        yield frame


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

        reg_result = register_sources(valid, force=force)

        errors.extend(reg_result.refused)
        if sse.cancel.is_set():
            return AddSummary(
                copied=reg_result.registered,
                skipped=reg_result.skipped,
                tracked=reg_result.tracked,
                errors=errors,
            )

        if not reg_result.registered and not reg_result.skipped and not reg_result.tracked:
            # Nothing reached the corpus, and sync() is a whole-vault pass
            # holding the ingest lock. A *tracked* or *skipped* file is not this
            # case: it is already in the corpus but may never have been indexed,
            # and a tracked one may have just had its skip marker cleared.
            return AddSummary(copied=[], skipped=[], errors=errors)

        with temporary_ocr_config(enable_ocr, ocr_timeout):
            sync_result = await sync(quiet=True, on_progress=sse.callback, cancel=sse.cancel)

        return AddSummary(
            copied=reg_result.registered,
            skipped=reg_result.skipped,
            tracked=reg_result.tracked,
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
        # The basename becomes the source root's label (its key prefix). It must
        # be a clean single segment: Path(x).name cannot traverse, so this only
        # rejects an empty name ("/", "a/") that would name no root.
        name = Path(p_str).name
        if not name:
            raise ValueError(f"{p_str!r} does not name a file")
        validate_path_within(cfg.documents_dir / name, cfg.documents_dir)

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
    subset whose lock was acquired. Contended sources emit ``already_ingesting``.
    When every source is contended the stream closes with no ``done`` event,
    signalling the client to wait rather than retry. When only some are, the
    acquired subset still runs and the ``done`` summary names the contended ones
    in ``already_ingesting``, so a partial batch never reads as a full success.
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
        locked = [payload for key, payload in items if key in acquired_names]
        sse = SseStream()
        task = asyncio.create_task(run(locked, sse))
        try:
            async for event in sse.drain(task, label):
                yield event
            # already_ingesting names the sources this run never attempted, so
            # a client reading only the terminal event still sees a partial batch.
            frame = sse.terminal_frame(
                task, lambda s: s.model_copy(update={"already_ingesting": list(busy)}).model_dump()
            )
            if frame is not None:
                yield frame
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
        [(IngestLockRegistry.canonical_source_name(p), p) for p in paths],
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
    reason = excluded_extension_reasons().get(Path(relative).suffix.lower())
    if reason is not None:
        raise ValueError(f"{name!r}: {reason}")
    validate_path_within(cfg.documents_dir / relative, cfg.documents_dir)
    return relative


def validate_upload_names(names: list[str | None]) -> list[str]:
    """Validate uploaded filenames, returning the cleaned relative paths.

    Names only, deliberately: the route validates before reading any part's
    bytes, so a request that will be rejected never costs a full copy of the
    payload in the server's own memory. Filenames keep their relative path,
    validated to stay inside ``cfg.documents_dir``, so an uploaded source tree
    preserves its layout instead of colliding on basenames. There is no
    file-count cap; the resource guard is the app's request_max_body_size.

    Raises ValueError on bad input.
    """
    if not names:
        raise ValueError("no files uploaded")
    # A multipart part is allowed to carry no filename at all; that is not a
    # file upload, and the cleaner's message should name it as missing rather
    # than crash on None.
    return [_clean_upload_name(name if name is not None else "") for name in names]


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
            if not _move_same_content(name, content, dest):
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


def _move_same_content(name: str, content: bytes, dest: Path) -> bool:
    """Move the one indexed file holding *content* to *dest*, so sync repoints its key."""
    digest = hashlib.sha256(content).hexdigest()
    matches = [
        s["filename"]
        for s in get_services().store.get_sources()
        if s["file_hash"] == digest and s["filename"] != name
    ]
    if len(matches) != 1:
        return False
    old_path = cfg.documents_dir / matches[0]
    if not old_path.is_file():
        return False
    os.replace(old_path, dest)
    log.info("Moved %s to %s: the same content arrived under a new name", matches[0], name)
    return True


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
    frame = sse.terminal_frame(task, lambda result: result.model_dump())
    if frame is not None:
        yield frame
