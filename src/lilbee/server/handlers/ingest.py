"""Sync and add-files handlers (SSE-streamed)."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import logging
import os
import re
import weakref
from collections.abc import AsyncGenerator, Callable, Coroutine
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar

from lilbee.app.ingest import (
    AddRollback,
    RegisterResult,
    indexed_stamps,
    leave_as_cancel,
    register_sources,
    unfinished_since,
)
from lilbee.app.services import get_services
from lilbee.core.config import cfg, validate_ocr_timeout
from lilbee.core.config.enums import OcrMode
from lilbee.core.security import validate_path_within
from lilbee.data.ingest.discovery import excluded_extension_reasons
from lilbee.runtime.cancellation import CancelSignal
from lilbee.runtime.ingest_lock import IngestLockRegistry
from lilbee.runtime.lock import refuse_where_a_sync_cannot_lock
from lilbee.runtime.progress import SseEvent
from lilbee.server.handlers.sse import SseStream, sse_event
from lilbee.server.models import AddSummary, SyncSummary

if TYPE_CHECKING:
    from lilbee.app.dataset import ImportSummary
    from lilbee.data.ingest import SyncResult

log = logging.getLogger(__name__)
# What a sync of this server is called in the notice of a sync that waits for it.
_SYNC_STARTER = "lilbee serve"

# Payload carried alongside each source key through _ingest_stream: a server
# path (str) for /api/add, or an (filename, content) pair for /api/add/upload.
_Payload = TypeVar("_Payload")
_Result = TypeVar("_Result")


async def _to_thread_unbroken(work: Callable[[], _Result]) -> _Result:
    """Run blocking *work* in a thread to its end; a cancel that arrives meanwhile lands after it.

    The caller is cancelled again when *work* has returned, so its next await raises.
    """
    running = asyncio.ensure_future(asyncio.to_thread(work))
    cancelled = False
    while not running.done():
        try:
            await asyncio.shield(running)
        except asyncio.CancelledError:
            cancelled = True
    task = asyncio.current_task()
    if cancelled and task is not None:
        task.cancel()
    return running.result()


async def _run_sync_with_sentinel(
    sse: SseStream,
    ocr: OcrMode | None,
    ocr_timeout: float | None = None,
    force_rebuild: bool = False,
    retry_skipped: bool = False,
    prune_ignored: bool = False,
) -> SyncResult:
    """Run ingest.sync() and guarantee the drain sentinel is enqueued."""
    from lilbee.app.ingest import temporary_ocr_config
    from lilbee.data.ingest import sync

    try:
        with temporary_ocr_config(ocr, ocr_timeout):
            return await sync(
                quiet=True,
                on_progress=sse.callback,
                cancel=sse.cancel,
                force_rebuild=force_rebuild,
                retry_skipped=retry_skipped,
                prune_ignored=prune_ignored,
                started_by=_SYNC_STARTER,
            )
    finally:
        sse.queue.put_nowait(None)


async def sync_stream(
    *,
    ocr: OcrMode | None = None,
    ocr_timeout: float | None = None,
    force_rebuild: bool = False,
    retry_skipped: bool = False,
    prune_ignored: bool = False,
    done_payload: Callable[[SyncResult], dict[str, Any]] = lambda result: result.model_dump(),
) -> AsyncGenerator[str, None]:
    """Trigger sync, yield SSE progress events, then done event.

    When ``force_rebuild`` is true, the underlying sync re-ingests every file from
    ``cfg.documents_dir`` over the index (the REST equivalent of ``lilbee rebuild``).
    When ``retry_skipped`` is true, it releases the records of failed files and
    reads partly read files again, so both get another attempt without dropping
    the store. When ``prune_ignored`` is true, it also drops sources a
    ``.lilbeeignore`` now excludes. ``done_payload`` builds the done event's data
    from the sync's result.
    """
    sse = SseStream()
    task = asyncio.create_task(
        _run_sync_with_sentinel(sse, ocr, ocr_timeout, force_rebuild, retry_skipped, prune_ignored)
    )
    async for event in sse.drain(task, "Sync stream"):
        yield event
    frame = sse.terminal_frame(task, done_payload)
    if frame is not None:
        yield frame


def _not_by_user() -> bool:
    """The user-cancel test given to the shared rollback, which then only names the cancel.

    The server runs the rollback itself, in a thread: it reads the store and
    writes the config, which must not block the event loop.
    """
    return False


def _register(
    paths: list[Path], force: bool, rollback: AddRollback, cancel: CancelSignal
) -> RegisterResult:
    """Register *paths*, then read the stamps *rollback* needs to undo it, in one thread."""
    registration = register_sources(paths, force=force)
    rollback.registered(registration, cancel)
    return registration


async def _rolled_back(
    rollback: AddRollback, errors: list[str], registration: RegisterResult | None = None
) -> AddSummary:
    """Undo what a user-cancelled add did not finish; return the summary that names it.

    The undo runs to its end before the add lets go of its ingest locks.
    """
    await _to_thread_unbroken(rollback.forget_unfinished)
    if registration is None:
        registration = RegisterResult()
    return AddSummary(
        copied=[label for label in registration.registered if label not in rollback.not_added],
        name_taken=registration.name_taken,
        overlapping=registration.overlapping,
        absorbed=[label for labels in rollback.absorbed_into.values() for label in labels],
        tracked=registration.tracked,
        errors=errors if rollback.error is None else [*errors, rollback.error],
        cancelled=True,
        not_added=rollback.not_added,
    )


async def _run_add(
    paths: list[str],
    force: bool,
    ocr: OcrMode | None,
    ocr_timeout: float | None,
    sse: SseStream,
) -> AddSummary:
    """Register the paths and sync, returning the summary for the final done event.

    A cancel by the user (``POST /api/add/cancel``) un-registers each root with
    no file indexed since the sync started, as Ctrl+C on ``lilbee add`` does.
    """
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

        # A stream that stops while the registration runs stops the sync, not the add.
        stopped = sse.cancel.is_set()
        rollback = AddRollback(paths=valid, at_sync=False)
        try:
            with leave_as_cancel(rollback, sse.cancel, _not_by_user):
                reg_result = await _to_thread_unbroken(
                    functools.partial(_register, valid, force, rollback, sse.cancel)
                )
        except asyncio.CancelledError:
            if not sse.stopped_by_user:
                raise
            return await _rolled_back(rollback, errors)

        errors.extend(reg_result.refused)
        if sse.stopped_by_user:
            return await _rolled_back(rollback, errors, reg_result)
        if stopped:
            return AddSummary(
                copied=reg_result.registered,
                name_taken=reg_result.name_taken,
                overlapping=reg_result.overlapping,
                absorbed=reg_result.absorbed,
                tracked=reg_result.tracked,
                errors=errors,
            )

        if not reg_result.reached_corpus:
            return AddSummary(copied=[], name_taken=reg_result.name_taken, errors=errors)

        try:
            with (
                leave_as_cancel(rollback, sse.cancel, _not_by_user),
                temporary_ocr_config(ocr, ocr_timeout),
            ):
                sync_result = await sse.stoppable(
                    sync(
                        quiet=True,
                        on_progress=sse.callback,
                        cancel=sse.cancel,
                        started_by=_SYNC_STARTER,
                    )
                )
        except asyncio.CancelledError:
            if not sse.stopped_by_user:
                raise
            return await _rolled_back(rollback, errors, reg_result)

        return AddSummary(
            copied=reg_result.registered,
            name_taken=reg_result.name_taken,
            overlapping=reg_result.overlapping,
            absorbed=reg_result.absorbed,
            tracked=reg_result.tracked,
            errors=errors,
            sync=SyncSummary(**sync_result.model_dump()),
        )
    finally:
        sse.queue.put_nowait(None)


def validate_add_paths(
    data: dict[str, Any],
) -> tuple[list[str], bool, OcrMode | None, float | None]:
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
            raise ValueError(f"'{p_str}' does not name a file")
        validate_path_within(cfg.documents_dir / name, cfg.documents_dir)

    force = bool(data.get("force", False))
    ocr, ocr_timeout = _parse_ocr_params(data)
    return paths, force, ocr, ocr_timeout


def parse_ocr_mode(value: Any) -> OcrMode | None:
    """Decode a request's ``ocr`` value; raises ValueError naming the accepted modes."""
    if value is None:
        return None
    try:
        return OcrMode(value)
    except ValueError as exc:
        accepted = ", ".join(mode.value for mode in OcrMode)
        raise ValueError(f"ocr must be one of {accepted}; got {value!r}") from exc


def _parse_ocr_params(data: dict[str, Any]) -> tuple[OcrMode | None, float | None]:
    """Extract, decode, and validate OCR parameters from a request dict.

    Raises ValueError on an unknown ocr mode, or a non-numeric or
    out-of-bound ocr_timeout.
    """
    ocr_timeout = data.get("ocr_timeout")
    if ocr_timeout is not None:
        ocr_timeout = float(ocr_timeout)
    validate_ocr_timeout(ocr_timeout)
    return parse_ocr_mode(data.get("ocr")), ocr_timeout


def _ingest_stream(
    items: list[tuple[str, _Payload]],
    run: Callable[[list[_Payload], SseStream], Coroutine[Any, Any, AddSummary]],
    label: str,
    add_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """Reserve *add_id* now, and return the stream of :func:`_locked_stream`.

    The id is held from this call, before the route answers, so a cancel that
    arrives before the run starts still reaches it: the run starts stopped and
    ends with its ``done`` event. Raises
    :class:`~lilbee.runtime.add_registry.AddIdRunningError` when another add
    holds *add_id*.
    """
    sse = SseStream()
    release = get_services().add_registry.reserve(add_id, sse.stop_by_user)
    stream = _locked_stream(items, run, label, sse, release)
    # A body that is never iterated never runs its finally, so the id goes with the stream.
    weakref.finalize(stream, release)
    return stream


async def _locked_stream(
    items: list[tuple[str, _Payload]],
    run: Callable[[list[_Payload], SseStream], Coroutine[Any, Any, AddSummary]],
    label: str,
    sse: SseStream,
    release: Callable[[], None],
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

    The ``already_ingesting`` events of a partial batch go out through the
    run's own queue, so the run has taken its first step before the client
    reads any event. A client that closes the stream at ``already_ingesting``
    then stops the acquired subset as any close does, after the run's first
    uninterruptible step (the registration, or the file writes of an upload).

    *release* lets go of the run's ``add_id``. It runs when the run has ended,
    before the terminal frame, so a cancel after the ``done`` event answers 404.
    """
    registry = get_services().ingest_lock_registry
    acquired: list[tuple[str, asyncio.Lock]] = []
    try:
        acquired, busy = await registry.acquire([key for key, _payload in items])
        rejections = [_already_ingesting(label, name) for name in busy]
        if not acquired:
            for rejection in rejections:
                yield rejection
            return

        acquired_names = {name for name, _lock in acquired}
        locked = [payload for key, payload in items if key in acquired_names]
        for rejection in rejections:
            sse.queue.put_nowait(rejection)
        task = asyncio.create_task(run(locked, sse))
        try:
            async for event in sse.drain(task, label):
                yield event
            release()
            # already_ingesting names the sources this run never attempted, so
            # a client reading only the terminal event still sees a partial batch.
            frame = sse.terminal_frame(
                task,
                lambda s: s.model_copy(update={"already_ingesting": list(busy)}).model_dump(),
            )
            if frame is not None:
                yield frame
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
    finally:
        release()
        registry.release(acquired)


def _already_ingesting(label: str, name: str) -> str:
    """The event that rejects *name*, whose ingest lock another run holds."""
    log.info("Rejecting %s for %s: already ingesting", label, name)
    return sse_event(SseEvent.ALREADY_INGESTING, {"source": name})


def add_files_stream(
    paths: list[str],
    *,
    force: bool = False,
    ocr: OcrMode | None = None,
    ocr_timeout: float | None = None,
    add_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """Register server-side files, sync, and yield SSE progress events.

    Takes the already-validated/parsed values from ``validate_add_paths`` so the
    request dict is decoded once. *add_id* names the add for ``cancel_add``; it
    is reserved by this call, which raises ``AddIdRunningError`` when it is taken.
    """
    return _ingest_stream(
        [(IngestLockRegistry.canonical_source_name(p), p) for p in paths],
        lambda locked, sse: _run_add(locked, force, ocr, ocr_timeout, sse),
        "Add files stream",
        add_id,
    )


def _clean_upload_name(name: str) -> str:
    """Normalize one upload filename to a safe relative path inside the corpus.

    Relative paths are preserved (a source tree keeps its layout instead of
    colliding on basenames); absolute paths, drive letters, and ``..`` segments
    are rejected. Raises ValueError on bad input.
    """
    normalized = name.replace("\\", "/")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        raise ValueError(f"upload filename must be relative: '{name}'")
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if not parts:
        raise ValueError(f"invalid upload filename: '{name}'")
    if ".." in parts:
        raise ValueError(f"upload filename may not contain '..': '{name}'")
    relative = "/".join(parts)
    reason = excluded_extension_reasons().get(Path(relative).suffix.lower())
    if reason is not None:
        raise ValueError(f"'{name}': {reason}")
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


async def _run_upload(
    files: list[tuple[str, bytes]],
    sse: SseStream,
    ocr: OcrMode | None = None,
    ocr_timeout: float | None = None,
) -> AddSummary:
    """Write uploaded file bytes into ``cfg.documents_dir``, then sync.

    The upload equivalent of :func:`_run_add`: instead of copying from a
    server-readable path, the client's content is written straight into the
    documents dir and the same ingest pipeline runs. This is what lets an
    external-mode client (a remote lilbee / GPU box) ingest files that live only
    on the client. Unchanged content is a no-op re-embed inside ``sync`` (it
    hashes each source), so there is no separate force flag. A data root that
    the sync cannot lock refuses the upload before any file is written. A
    stream that stops during the writes stops the sync, not the writes: an
    uploaded file exists only in the request, so a dropped stream must not drop
    it. A cancel by the user (``POST /api/add/cancel``) deletes each file this
    upload created that the sync did not index; a file it overwrote or moved stays.
    """
    from lilbee.app.ingest import temporary_ocr_config
    from lilbee.data.ingest import sync

    try:
        written = await _to_thread_unbroken(functools.partial(_write_uploads, files))
        rollback = AddRollback()
        try:
            with (
                leave_as_cancel(rollback, sse.cancel, _not_by_user),
                temporary_ocr_config(ocr, ocr_timeout),
            ):
                sync_result = await sse.stoppable(
                    sync(
                        quiet=True,
                        on_progress=sse.callback,
                        cancel=sse.cancel,
                        started_by=_SYNC_STARTER,
                    )
                )
        except asyncio.CancelledError:
            if not sse.stopped_by_user:
                raise
            not_added = await _to_thread_unbroken(
                functools.partial(_delete_unfinished_uploads, written)
            )
            return AddSummary(
                copied=[name for name in written.names if name not in not_added],
                errors=[] if rollback.error is None else [rollback.error],
                cancelled=True,
                not_added=not_added,
            )
        return AddSummary(
            copied=written.names,
            errors=[],
            sync=SyncSummary(**sync_result.model_dump()),
        )
    finally:
        sse.queue.put_nowait(None)


def _delete_unfinished_uploads(written: _Written) -> list[str]:
    """Delete each file the upload created that the sync did not index; return their names."""
    unfinished = unfinished_since(written.created, written.before)
    for name in unfinished:
        (cfg.documents_dir / name).unlink(missing_ok=True)
    return unfinished


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


def _place_upload(name: str, content: bytes, dest: Path) -> bool:
    """Put an uploaded file at *dest*: a move of the same indexed content, else a write.

    Returns whether the upload created the file: a write where no file was.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    # lexists: a dangling symlink at *dest* is the user's, so the upload did not create it.
    existed = os.path.lexists(dest)
    if _move_same_content(name, content, dest):
        return False
    dest.write_bytes(content)
    return not existed


class _Written(NamedTuple):
    """What an upload wrote, and what its cancel by the user needs to undo it."""

    names: list[str]
    created: list[str]
    """The names of the files the upload created."""
    before: dict[str, str]
    """The index stamps of *created* before the sync."""


def _write_uploads(files: list[tuple[str, bytes]]) -> _Written:
    """Write every uploaded file into ``cfg.documents_dir``; return what was written."""
    refuse_where_a_sync_cannot_lock(cfg.data_root)
    cfg.documents_dir.mkdir(parents=True, exist_ok=True)
    created = [
        name for name, content in files if _place_upload(name, content, cfg.documents_dir / name)
    ]
    return _Written([name for name, _content in files], created, indexed_stamps(created))


def add_uploads_stream(
    files: list[tuple[str, bytes]],
    *,
    ocr: OcrMode | None = None,
    ocr_timeout: float | None = None,
    add_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """Ingest uploaded file content, yielding the same SSE progress as add_files_stream.

    Locks per source name (the validated relative path) so an upload never
    races an in-flight add of the same source. *add_id* names the upload for
    ``cancel_add``; it is reserved by this call, which raises
    ``AddIdRunningError`` when it is taken.
    """
    return _ingest_stream(
        [(name, (name, content)) for name, content in files],
        lambda locked, sse: _run_upload(locked, sse, ocr, ocr_timeout),
        "Add uploads stream",
        add_id,
    )


def cancel_add(add_id: str) -> bool:
    """Stop the add with *add_id* as the user's cancel; False when no add with it runs.

    The add removes what it did not finish, and its stream ends with a ``done``
    event that carries ``cancelled`` and ``not_added``.
    """
    return get_services().add_registry.cancel(add_id)


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
