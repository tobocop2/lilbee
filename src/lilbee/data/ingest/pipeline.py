"""Top-level sync orchestration: discovery, dispatch, batching, post-sync hooks."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from pathlib import Path
from typing import Any, cast

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from lilbee.app.services import get_services
from lilbee.core.config import cfg
from lilbee.data.ingest.code import ingest_code_sync
from lilbee.data.ingest.discovery import classify_file, discover_files, file_hash
from lilbee.data.ingest.extract import ingest_document, ingest_markdown
from lilbee.data.ingest.skip_marker import (
    clear_skip_markers,
    load_skip_markers,
    write_skip_markers,
)
from lilbee.data.ingest.types import ChunkRecord, FileToProcess, SyncResult, _IngestResult
from lilbee.data.store import ChunkWrite
from lilbee.runtime.asyncio_loop import is_executor_shutdown
from lilbee.runtime.cancellation import TaskCancelledError
from lilbee.runtime.cpu import cpu_quota
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
    noop_callback,
)

log = logging.getLogger(__name__)

# Limit concurrent ingestion. Sourced from cpu_quota() so worker storms
# can't starve the TUI's asyncio main thread on macOS.
_MAX_CONCURRENT = cpu_quota()


async def _rebuild_concept_clusters() -> None:
    """Re-run Leiden clustering after sync. No-op if disabled."""
    if not cfg.concept_graph:
        return
    from lilbee.retrieval.concepts import concepts_available

    if not concepts_available():
        return
    try:
        cg = get_services().concepts
        if not cg.get_graph():
            return
        await asyncio.to_thread(cg.rebuild_clusters)
    except Exception:
        log.warning("Concept cluster rebuild failed", exc_info=True)


async def _index_concepts(records: list[ChunkRecord], source_name: str) -> None:
    """Extract and index concepts for ingested chunks. No-op if disabled."""
    if not cfg.concept_graph or not records:
        return
    from lilbee.retrieval.concepts import concepts_available

    if not concepts_available():
        return
    try:
        cg = get_services().concepts
        texts = [r["chunk"] for r in records]
        concept_lists = await asyncio.to_thread(cg.extract_concepts_batch, texts)
        chunk_ids = [(source_name, r["chunk_index"]) for r in records]
        await asyncio.to_thread(cg.build_from_chunks, chunk_ids, concept_lists)
    except Exception:
        log.warning("Concept indexing failed for %s", source_name, exc_info=True)


async def _produce_records(
    path: Path,
    source_name: str,
    content_type: str,
    *,
    quiet: bool = False,
    on_progress: DetailedProgressCallback = noop_callback,
) -> list[ChunkRecord]:
    """Extract, chunk, and embed a single file into store-ready records.

    The LanceDB write is deferred: records are returned to the caller and written
    in a batched flush (see :func:`_flush_writes`), so bulk ingest pays one
    write-lock acquisition per batch instead of one per file. Concept indexing
    runs here because it reads the in-memory records, not the store.
    """
    records: list[ChunkRecord]
    if content_type == "code":
        records = await asyncio.to_thread(ingest_code_sync, path, source_name, on_progress)
    elif path.suffix.lower() == ".md":
        records = await ingest_markdown(path, source_name, on_progress)
    else:
        records = await ingest_document(
            path,
            source_name,
            content_type,
            quiet=quiet,
            on_progress=on_progress,
        )

    await _index_concepts(records, source_name)
    return records


def _plan_file_changes(
    disk_files: dict[str, Path],
    existing_sources: dict[str, str],
    cancel: threading.Event | None,
    skip_markers: dict[str, str] | None = None,
) -> tuple[list[FileToProcess], list[str], list[str], int]:
    """Diff disk against the store. Returns (to_process, added, updated, unchanged_count).

    A file whose current hash matches a marker in ``skip_markers`` (set by a
    prior failed attempt) is treated as unchanged so we don't retry every
    sync. Edit the file or run ``/sync --force-rebuild`` to clear the marker
    and try again.
    """
    skip_markers = skip_markers or {}
    files_to_process: list[FileToProcess] = []
    added: list[str] = []
    updated: list[str] = []
    unchanged = 0
    for name, path in sorted(disk_files.items()):
        if cancel and cancel.is_set():
            break
        content_type = classify_file(path)
        if content_type is None:
            raise ValueError(f"Unsupported file slipped through discovery: {name}")
        old_hash = existing_sources.get(name)
        current_hash = file_hash(path)
        if old_hash == current_hash:
            unchanged += 1
            continue
        if skip_markers.get(name) == current_hash:
            # Failed last sync at this exact hash; skip the retry.
            unchanged += 1
            continue
        # needs_cleanup=True unconditionally: delete_by_source is idempotent,
        # and this closes the race where a prior ingest wrote chunks but died
        # before upsert_source, leaving orphaned chunks that would duplicate.
        files_to_process.append(
            FileToProcess(name, path, content_type, current_hash, needs_cleanup=True)
        )
        if old_hash is not None:
            updated.append(name)
        else:
            added.append(name)
    return files_to_process, added, updated, unchanged


def detect_pending() -> int:
    """Count files in documents/ that are out of sync with the store.

    Cheap operation: filesystem walk + SHA-256 hashing + a single
    sources-table read. No embedding, no writes. Returns the total of
    added + updated + removed, which is what the TaskBar hint surfaces.
    Reuses ``_plan_file_changes`` so the diff logic stays single-sourced.
    Honors skip markers: a file that failed last time at this hash does
    not show up as pending.
    """
    if not cfg.documents_dir.exists():
        return 0
    disk_files = discover_files()
    existing_sources = {s["filename"]: s["file_hash"] for s in get_services().store.get_sources()}
    removed = sum(1 for name in existing_sources if name not in disk_files)
    skip_markers = load_skip_markers(cfg.data_root)
    files_to_process, _, _, _ = _plan_file_changes(
        disk_files, existing_sources, cancel=None, skip_markers=skip_markers
    )
    return len(files_to_process) + removed


def _load_pruned_skip_markers(disk_files: dict[str, Path], *, clear_first: bool) -> dict[str, str]:
    """Read the skip-marker file (optionally clearing it first) and drop entries
    for files no longer on disk, so the marker set tracks the current corpus."""
    if clear_first:
        # Clearing the markers makes the diff re-include the skipped files.
        clear_skip_markers(cfg.data_root)
    markers = load_skip_markers(cfg.data_root)
    if not markers:
        return markers
    return {name: fhash for name, fhash in markers.items() if name in disk_files}


def _persist_skip_markers(
    markers: dict[str, str],
    pending_hashes: dict[str, str],
    *,
    succeeded: list[str],
    failed: list[str],
) -> None:
    """Mark files that produced no chunks so the next sync skips them, clear the
    markers for files that ingested cleanly, then write the file back."""
    for name in succeeded:
        markers.pop(name, None)
    for name in failed:
        fhash = pending_hashes.get(name)
        if fhash:
            markers[name] = fhash
    write_skip_markers(cfg.data_root, markers)


async def sync(
    force_rebuild: bool = False,
    quiet: bool = False,
    *,
    on_progress: DetailedProgressCallback = noop_callback,
    cancel: threading.Event | None = None,
    retry_skipped: bool = False,
) -> SyncResult:
    """Sync documents/ with the vector store.
    Returns a SyncResult with the added/updated/removed/unchanged/failed/skipped lists.
    When *quiet* is True, the Rich progress bar is suppressed (for JSON output).
    When *cancel* is set, processing stops between files without data loss.
    When *retry_skipped* (or *force_rebuild*) is set, the failed-file skip
    markers are cleared so this sync attempts every file.
    """
    _store = get_services().store

    if force_rebuild:
        _store.drop_all()

    cfg.documents_dir.mkdir(parents=True, exist_ok=True)

    disk_files = discover_files()
    existing_sources = {s["filename"]: s["file_hash"] for s in _store.get_sources()}
    skip_markers = _load_pruned_skip_markers(disk_files, clear_first=force_rebuild or retry_skipped)

    removed: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []

    # Find files to remove (in DB but not on disk)
    to_remove = [name for name in existing_sources if name not in disk_files]
    if to_remove:
        _store.remove_documents(to_remove)
        removed.extend(to_remove)

    files_to_process, added, updated, unchanged = _plan_file_changes(
        disk_files, existing_sources, cancel, skip_markers=skip_markers
    )
    # Track skip markers for files processed this run, keyed by name → hash.
    pending_hashes = {entry.name: entry.file_hash for entry in files_to_process}

    # Snapshot the cumulative truncation counter so the delta over this sync can
    # surface "N chunks truncated" instead of being lost in per-chunk debug logs.
    truncated_before = get_services().embedder.truncated_total

    # Ingest files (with optional progress bar)
    if files_to_process:
        get_services().embedder.validate_model()
        await ingest_batch(
            files_to_process,
            added,
            updated,
            failed,
            skipped,
            quiet=quiet,
            on_progress=on_progress,
            cancel=cancel,
        )

    _persist_skip_markers(
        skip_markers, pending_hashes, succeeded=added + updated, failed=failed + skipped
    )

    if files_to_process or removed:
        _store.ensure_fts_index()
        await _rebuild_concept_clusters()
        # circular: lilbee.wiki imports lilbee.data.ingest.file_hash, so the
        # post-ingest hook stays function-local at this boundary.
        from lilbee.wiki.ingest import incremental_update

        await incremental_update(set(added) | set(updated) | set(removed))

    result = SyncResult(
        added=added,
        updated=updated,
        removed=removed,
        unchanged=unchanged,
        failed=failed,
        skipped=skipped,
        truncated=get_services().embedder.truncated_total - truncated_before,
    )
    on_progress(
        EventType.DONE,
        SyncDoneEvent(
            added=len(result.added),
            updated=len(result.updated),
            removed=len(result.removed),
            failed=len(result.failed),
            skipped=len(result.skipped),
        ),
    )
    return result


def _phase_progress_callback(
    progress: Progress, ptask: Any, chain: DetailedProgressCallback
) -> DetailedProgressCallback:
    """Wrap *chain*, updating the bar's description on per-page / per-chunk events.

    EXTRACT (vision OCR page i/N) and EMBED (chunk i/N) events would otherwise
    leave the bar frozen between file completions; surfacing them on the spinner
    description keeps a single large file's row visibly moving. All events still
    forward to *chain* so the caller's own callback (TUI / JSON) is unaffected.
    """

    def _callback(event_type: EventType, data: ProgressEvent) -> None:
        if event_type is EventType.EXTRACT and isinstance(data, ExtractEvent):
            progress.update(
                ptask, description=f"OCR {data.file} (page {data.page}/{data.total_pages})"
            )
        elif event_type is EventType.EMBED and isinstance(data, EmbedEvent):
            progress.update(
                ptask, description=f"Embedding {data.file} ({data.chunk}/{data.total_chunks})"
            )
        chain(event_type, data)

    return _callback


async def ingest_batch(
    files_to_process: list[FileToProcess],
    added: list[str],
    updated: list[str],
    failed: list[str],
    skipped: list[str],
    *,
    quiet: bool = False,
    on_progress: DetailedProgressCallback = noop_callback,
    cancel: threading.Event | None = None,
) -> None:
    """Ingest a batch of files, optionally showing a Rich progress bar.
    When *needs_cleanup* is True, old chunks are deleted immediately before
    ingesting new ones so the two operations are atomic per file.
    When *cancel* is set, pending files raise CancelledError before starting.
    """
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT)
    total_files = len(files_to_process)

    async def _process_one(
        name: str,
        path: Path,
        content_type: str,
        fhash: str,
        needs_cleanup: bool,
        file_index: int,
    ) -> _IngestResult:
        async with semaphore:
            if cancel and cancel.is_set():
                raise asyncio.CancelledError

            try:
                on_progress(
                    EventType.FILE_START,
                    FileStartEvent(file=name, total_files=total_files, current_file=file_index),
                )
            except TaskCancelledError as exc:
                # FILE_START itself can raise the cooperative cancel signal;
                # normalize so _collect_results can drain siblings cleanly.
                raise asyncio.CancelledError from exc
            try:
                # The source's old chunks are deleted in the same locked
                # transaction as the new write (see _flush_writes), so cleanup is
                # carried on the result rather than run eagerly here.
                records = await _produce_records(
                    path,
                    name,
                    content_type,
                    quiet=quiet,
                    on_progress=on_progress,
                )
                on_progress(
                    EventType.FILE_DONE,
                    FileDoneEvent(file=name, status="ok", chunks=len(records)),
                )
                return _IngestResult(
                    name,
                    path,
                    len(records),
                    error=None,
                    file_hash=fhash,
                    records=records,
                    needs_cleanup=needs_cleanup,
                )
            except (asyncio.CancelledError, TaskCancelledError) as exc:
                # TaskCancelledError is the TUI's cooperative cancel signal raised
                # by reporter.check_cancelled() inside on_progress; treat it as
                # asyncio cancellation so _collect_results can drain siblings
                # cleanly instead of orphaning their pending exceptions.
                raise asyncio.CancelledError from exc
            except Exception as exc:
                # During shutdown, worker pools raise RuntimeError from
                # submit(). Prefer to treat these as cancellation rather than
                # as ingest failures. Detect via the cancel flag (source of
                # truth) or the executor's well-known shutdown message as a
                # fallback when cancel was set after the submit race.
                if (cancel and cancel.is_set()) or is_executor_shutdown(exc):
                    raise asyncio.CancelledError from exc
                # Suppress TaskCancelledError on the FILE_DONE notice: the user
                # already cancelled, and re-raising here would leak past
                # _process_one and strand sibling tasks awaiting in
                # _collect_results.
                with contextlib.suppress(TaskCancelledError):
                    on_progress(
                        EventType.FILE_DONE,
                        FileDoneEvent(file=name, status="error", chunks=0),
                    )
                return _IngestResult(name, path, 0, error=exc)

    if quiet:
        tasks = [
            asyncio.ensure_future(_process_one(name, path, ct, fh, cleanup, idx))
            for idx, (name, path, ct, fh, cleanup) in enumerate(files_to_process, 1)
        ]
        await _collect_results(tasks, added, updated, failed, skipped, on_progress=on_progress)
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            ptask = progress.add_task("Ingesting documents...", total=total_files)
            # The bar advances once per file (in _collect_results), so a single
            # multi-page scanned PDF would freeze at "0/1" through its whole
            # OCR + embed phase. Drive the spinner's description off the same
            # EXTRACT (OCR page i/N) and EMBED (chunk i/N) events the TUI uses
            # so the row visibly moves while one file is being worked.
            phase_progress = _phase_progress_callback(progress, ptask, on_progress)
            tasks = [
                asyncio.ensure_future(_process_one(name, path, ct, fh, cleanup, idx))
                for idx, (name, path, ct, fh, cleanup) in enumerate(files_to_process, 1)
            ]
            await _collect_results(
                tasks,
                added,
                updated,
                failed,
                skipped,
                on_progress=phase_progress,
                progress=progress,
                ptask=ptask,
            )


# Accumulate roughly this many chunks across documents before one batched
# LanceDB write. Bounds buffered-vector memory while amortizing the write lock
# and per-transaction overhead over many documents instead of one write per file.
_WRITE_FLUSH_CHUNKS = 2000


async def _collect_results(
    tasks: list[asyncio.Task[_IngestResult]],
    added: list[str],
    updated: list[str],
    failed: list[str],
    skipped: list[str],
    *,
    on_progress: DetailedProgressCallback = noop_callback,
    progress: Progress | None = None,
    ptask: Any = None,
) -> None:
    """Collect task results, batching successful writes and updating a progress bar.

    Successful files are buffered and flushed to LanceDB in batches (one locked
    transaction per batch) rather than one write per file. The buffer is flushed
    on the way out too -- even on cancel -- so completed-but-unwritten work is
    persisted. On exception (typically asyncio.CancelledError from a user cancel),
    cancel every sibling task and await them with ``return_exceptions=True`` so
    their pending CancelledErrors don't surface as
    "Task exception was never retrieved" warnings.
    """
    buffer: list[_IngestResult] = []
    buffered_chunks = 0
    try:
        for completed_count, fut in enumerate(asyncio.as_completed(tasks), 1):
            result = await fut
            status = _classify_result(result, added, updated, failed, skipped)
            if status is BatchStatus.INGESTED:
                buffer.append(result)
                buffered_chunks += result.chunk_count
                if buffered_chunks >= _WRITE_FLUSH_CHUNKS:
                    await asyncio.to_thread(_flush_writes, buffer, added, updated, failed)
                    buffered_chunks = 0
            if progress is not None and ptask is not None:
                desc = (
                    f"Ingested {result.name}" if result.error is None else f"Failed {result.name}"
                )
                progress.update(ptask, description=desc)
                progress.advance(ptask)
            with contextlib.suppress(TaskCancelledError):
                on_progress(
                    EventType.BATCH_PROGRESS,
                    BatchProgressEvent(
                        file=result.name,
                        status=status,
                        current=completed_count,
                        total=len(tasks),
                    ),
                )
    finally:
        await asyncio.to_thread(_flush_writes, buffer, added, updated, failed)
        pending = [t for t in tasks if not t.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


def _discard_from_list(lst: list[str], value: str) -> None:
    """Remove *value* from *lst* if present."""
    with contextlib.suppress(ValueError):
        lst.remove(value)


def _classify_result(
    result: _IngestResult,
    added: list[str],
    updated: list[str],
    failed: list[str],
    skipped: list[str],
) -> BatchStatus:
    """Record a completed file's outcome and return its batch status.

    Failures and zero-chunk files are tracked here; a successful file is reported
    as ``INGESTED`` and its chunks are persisted by the batched flush, so it stays
    in ``added`` / ``updated`` until then.
    """
    if result.error is not None:
        # Log the error message without the traceback: ingest failures are
        # already surfaced to callers via SyncResult.failed, and the raw
        # traceback from log.exception bleeds into the TUI chat pane via the
        # stderr bridge. Full stack traces stay reachable by
        # lowering LILBEE_LOG_LEVEL to DEBUG.
        log.warning("Failed to ingest %s: %s", result.name, result.error)
        log.debug("Traceback for failed ingest of %s", result.name, exc_info=result.error)
        _discard_from_list(added, result.name)
        _discard_from_list(updated, result.name)
        failed.append(result.name)
        return BatchStatus.FAILED
    if result.chunk_count == 0:
        # No chunks produced (e.g. scanned PDF without vision model, or
        # vision OCR returned no text). Don't record as a source so it
        # gets retried on next sync, and surface as skipped so the user
        # knows the file did not actually land in the store.
        _discard_from_list(added, result.name)
        _discard_from_list(updated, result.name)
        skipped.append(result.name)
        return BatchStatus.SKIPPED
    return BatchStatus.INGESTED


def _flush_writes(
    buffer: list[_IngestResult],
    added: list[str],
    updated: list[str],
    failed: list[str],
) -> None:
    """Write the buffered documents in one transaction; track a write failure.

    Each buffered file's chunks, its cleanup delete, and its source upsert land
    together in ``Store.write_chunks_batch`` under a single write lock. If the
    batch write fails, every file in it is moved to ``failed`` since its chunks
    did not persist. The buffer is cleared either way.
    """
    if not buffer:
        return
    items = [
        ChunkWrite(
            source=r.name,
            file_hash=r.file_hash or file_hash(r.path),
            records=cast(list[dict], r.records or []),
            needs_cleanup=r.needs_cleanup,
        )
        for r in buffer
    ]
    try:
        get_services().store.write_chunks_batch(items)
    except Exception as exc:
        for r in buffer:
            log.warning("Failed to write %s: %s", r.name, exc)
            _discard_from_list(added, r.name)
            _discard_from_list(updated, r.name)
            if r.name not in failed:
                failed.append(r.name)
    buffer.clear()
