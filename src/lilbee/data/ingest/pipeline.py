"""Top-level sync orchestration: discovery, dispatch, batching, post-sync hooks."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from collections.abc import Callable, Coroutine, Iterable, Iterator
from dataclasses import dataclass
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
    write_skip_reasons,
)
from lilbee.data.ingest.types import (
    ChunkRecord,
    FileChangePlan,
    FileToProcess,
    SyncResult,
    _IngestResult,
)
from lilbee.data.store import (
    SOURCE_STAT_UNKNOWN,
    ChunkWrite,
    ConceptRecords,
    PageTextRecord,
    SourceRecord,
    SourceStat,
    SourceStatBackfill,
    SourceType,
    source_stat,
)
from lilbee.runtime.asyncio_loop import is_executor_shutdown
from lilbee.runtime.cancellation import TaskCancelledError
from lilbee.runtime.cpu import cpu_quota
from lilbee.runtime.lock import LockTimeoutError
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


def _max_concurrent() -> int:
    """Files allowed in their compute phase at once.

    ``cpu_quota()`` (cpu_count // 2) keeps worker storms from starving the TUI's asyncio
    main thread. But with a data-parallel fleet (N vision/embed servers, one per GPU), a
    few-core box would cap file concurrency below the GPU count and starve the extra
    cards, so the bottleneck role's total slots set the floor: vision OCR (replicas x
    per-server pages) when a vision model is configured, else the embed replicas.
    """
    from lilbee.core.config import cfg

    # Only a multi-replica (multi-GPU) fleet scales above cpu_quota; with one replica per
    # role the cpu_quota cap is untouched, so single-GPU/CPU hosts and the macOS TUI see
    # exactly the previous behavior.
    vision_slots = (
        cfg.vision_replicas * cfg.vision_ocr_concurrency
        if cfg.vision_model and cfg.vision_replicas > 1
        else 0
    )
    embed_slots = cfg.embed_replicas if cfg.embed_replicas > 1 else 0
    return max(cpu_quota(), vision_slots, embed_slots)


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


async def _build_concept_records(
    records: list[ChunkRecord], source_name: str
) -> ConceptRecords | None:
    """Extract concepts for ingested chunks and build their table rows. None if disabled.

    Pure record building, no store access: the rows are buffered on the file's
    ingest result and written once per flush (see :func:`_flush_concept_records`),
    so a large sync pays one concept-table write per flush, not per file.
    """
    if not cfg.concept_graph or not records:
        return None
    from lilbee.retrieval.concepts import concepts_available

    if not concepts_available():
        return None
    try:
        cg = get_services().concepts
        texts = [r["chunk"] for r in records]
        concept_lists = await asyncio.to_thread(cg.extract_concepts_batch, texts)
        chunk_ids = [(source_name, r["chunk_index"]) for r in records]
        return await asyncio.to_thread(cg.build_concept_records, chunk_ids, concept_lists)
    except Exception:
        log.warning("Concept extraction failed for %s", source_name, exc_info=True)
        return None


async def _produce_records(
    path: Path,
    source_name: str,
    content_type: str,
    *,
    quiet: bool = False,
    on_progress: DetailedProgressCallback = noop_callback,
    page_texts_out: list[PageTextRecord] | None = None,
) -> list[ChunkRecord]:
    """Extract, chunk, and embed a single file into store-ready records.

    The LanceDB write is deferred: records are returned to the caller and written
    in a batched flush (see :func:`_flush_writes`), so bulk ingest pays one
    write-lock acquisition per batch instead of one per file. The per-page text
    dataset rows land in ``page_texts_out`` and are written by the same flush.
    """
    records: list[ChunkRecord]
    page_texts: list[PageTextRecord] = page_texts_out if page_texts_out is not None else []
    if content_type == "code":
        records = await asyncio.to_thread(ingest_code_sync, path, source_name, on_progress)
    elif path.suffix.lower() == ".md":
        records = await ingest_markdown(path, source_name, on_progress, page_texts_out=page_texts)
    else:
        records = await ingest_document(
            path,
            source_name,
            content_type,
            quiet=quiet,
            on_progress=on_progress,
            page_texts_out=page_texts,
        )

    return records


def _disk_stat(path: Path) -> SourceStat | None:
    """Current size/mtime of *path* stamped with now, or None when it cannot be stat'd."""
    try:
        st = path.stat()
    except OSError:
        return None
    return SourceStat(st.st_size, st.st_mtime_ns, time.time_ns())


def _stat_unchanged(stored: SourceStat, current: SourceStat) -> bool:
    """Whether the stored stat proves the file unchanged without hashing it.

    Git-style racily-clean guard: a matching (size, mtime) only counts when the
    mtime is strictly older than the time the stat was recorded; a same-size
    edit landing in the same mtime tick is otherwise missed forever.
    """
    if (stored.size_bytes, stored.mtime_ns) != (current.size_bytes, current.mtime_ns):
        return False
    # Unknown capture or mtime >= capture hashes anyway; clock skew can only widen
    # hashing, never widen skipping past the pre-existing same-tick window.
    return stored.captured_ns != SOURCE_STAT_UNKNOWN and current.mtime_ns < stored.captured_ns


@dataclass(frozen=True)
class _FileChangeVerdict:
    """One file's sync verdict: process it, or unchanged (optionally backfilling its stat)."""

    to_process: FileToProcess | None = None
    backfill: SourceStatBackfill | None = None
    is_update: bool = False


def _classify_file_change(
    name: str,
    path: Path,
    record: SourceRecord | None,
    skip_markers: dict[str, str],
) -> _FileChangeVerdict:
    """Decide one file's verdict: stat-unchanged, hash-unchanged, skip-marked, or process."""
    content_type = classify_file(path)
    if content_type is None:
        raise ValueError(f"Unsupported file slipped through discovery: {name}")
    stored_stat = source_stat(record) if record is not None else None
    current_stat = _disk_stat(path)
    if (
        record is not None
        and stored_stat is not None
        and current_stat is not None
        and _stat_unchanged(stored_stat, current_stat)
    ):
        return _FileChangeVerdict()
    old_hash = record["file_hash"] if record is not None else None
    current_hash = file_hash(path)
    if old_hash == current_hash:
        # Content verified unchanged; persist the stat pair so the next
        # sync skips the hash entirely.
        backfill = (
            SourceStatBackfill(record, current_stat)
            if record is not None and current_stat is not None
            else None
        )
        return _FileChangeVerdict(backfill=backfill)
    if skip_markers.get(name) == current_hash:
        # Failed last sync at this exact hash; skip the retry.
        return _FileChangeVerdict()
    # needs_cleanup=True unconditionally: delete_by_source is idempotent,
    # and this closes the race where a prior ingest wrote chunks but died
    # before upsert_source, leaving orphaned chunks that would duplicate.
    return _FileChangeVerdict(
        to_process=FileToProcess(
            name, path, content_type, current_hash, needs_cleanup=True, stat=current_stat
        ),
        is_update=old_hash is not None,
    )


def _plan_file_changes(
    disk_files: dict[str, Path],
    existing_sources: dict[str, SourceRecord],
    cancel: threading.Event | None,
    skip_markers: dict[str, str] | None = None,
) -> FileChangePlan:
    """Diff disk against the store, hashing only files whose size/mtime drifted.

    A tracked file whose stored (size, mtime) matches the disk stat, and whose
    mtime predates the stat capture (see :func:`_stat_unchanged`), is unchanged
    without reading its bytes; everything else is SHA-256 hashed. A file whose
    current hash matches a marker in ``skip_markers`` (set by a prior failed
    attempt) is treated as unchanged so we don't retry every sync. Edit the file
    or run ``/sync --force-rebuild`` to clear the marker and try again.
    """
    skip_markers = skip_markers or {}
    files_to_process: list[FileToProcess] = []
    added: dict[str, None] = {}
    updated: dict[str, None] = {}
    stat_backfills: list[SourceStatBackfill] = []
    unchanged = 0
    for name, path in sorted(disk_files.items()):
        if cancel and cancel.is_set():
            break
        verdict = _classify_file_change(name, path, existing_sources.get(name), skip_markers)
        if verdict.to_process is None:
            unchanged += 1
            if verdict.backfill is not None:
                stat_backfills.append(verdict.backfill)
            continue
        files_to_process.append(verdict.to_process)
        if verdict.is_update:
            updated[name] = None
        else:
            added[name] = None
    return FileChangePlan(files_to_process, added, updated, unchanged, stat_backfills)


def _removable_sources(sources: list[SourceRecord], disk_files: dict[str, Path]) -> list[str]:
    """Document sources whose backing file is gone.

    Imported sources are detached (no file under documents/), so a missing
    disk file must not mark them for removal.
    """
    return [
        s["filename"]
        for s in sources
        if s["filename"] not in disk_files and s["source_type"] != SourceType.IMPORTED
    ]


def detect_pending() -> int:
    """Count files in documents/ that are out of sync with the store.

    Cheap operation: filesystem walk + stat-gated SHA-256 hashing + a single
    sources-table read. No embedding, no writes. Returns the total of
    added + updated + removed, which is what the TaskBar hint surfaces.
    Reuses ``_plan_file_changes`` so the diff logic stays single-sourced.
    Honors skip markers: a file that failed last time at this hash does
    not show up as pending. Blocking: callers on the event loop run it via
    ``asyncio.to_thread``.
    """
    if not cfg.documents_dir.exists():
        return 0
    disk_files = discover_files()
    sources = get_services().store.get_sources()
    existing_sources = {s["filename"]: s for s in sources}
    removed = len(_removable_sources(sources, disk_files))
    skip_markers = load_skip_markers(cfg.data_root)
    plan = _plan_file_changes(disk_files, existing_sources, cancel=None, skip_markers=skip_markers)
    return len(plan.files_to_process) + removed


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
    succeeded: Iterable[str],
    failed: Iterable[str],
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
        # drop_all preserves the memories table, so refresh its vectors under the
        # (possibly changed) embedding model. No-op when empty or no embedder.
        _embedder = get_services().embedder
        if _embedder.embedding_available():
            _store.rebuild_memory_embeddings(lambda texts: _embedder.embed_batch(texts))

    cfg.documents_dir.mkdir(parents=True, exist_ok=True)

    disk_files = discover_files()
    sources = _store.get_sources()
    existing_sources = {s["filename"]: s for s in sources}
    skip_markers = _load_pruned_skip_markers(disk_files, clear_first=force_rebuild or retry_skipped)

    removed: list[str] = []
    failed: dict[str, None] = {}
    skipped: dict[str, None] = {}
    reasons: dict[str, str] = {}  # filename → why it was skipped/failed (for reporting)
    flush_failed: set[str] = set()

    # Find files to remove (document sources whose file is gone; imports are kept)
    to_remove = _removable_sources(sources, disk_files)
    if to_remove:
        _store.remove_documents(to_remove)
        removed.extend(to_remove)

    # The planning pass stats (and where needed hashes) every file on disk;
    # off the event loop so a large corpus doesn't freeze the TUI.
    plan = await asyncio.to_thread(
        _plan_file_changes, disk_files, existing_sources, cancel, skip_markers
    )
    files_to_process, added, updated = plan.files_to_process, plan.added, plan.updated
    if plan.stat_backfills:
        await asyncio.to_thread(_store.update_source_stats, plan.stat_backfills)
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
            flush_failed=flush_failed,
            reasons=reasons,
        )

    # A flush failure is a transient store-side problem, not a verdict on the
    # file: leaving it unmarked re-plans it next sync instead of skipping it.
    marker_failed = [name for name in (*failed, *skipped) if name not in flush_failed]
    _persist_skip_markers(
        skip_markers, pending_hashes, succeeded=[*added, *updated], failed=marker_failed
    )
    # Persist the human-readable reason for each skip-marked file (informational;
    # the hash markers above drive the resume logic). Only marker_failed files,
    # so a transient flush failure doesn't leave a stale reason behind.
    write_skip_reasons(cfg.data_root, {n: reasons[n] for n in marker_failed if n in reasons})

    if files_to_process or removed:
        _store.ensure_fts_index()
        _store.ensure_vector_index()
        _store.optimize_sources()
        await _rebuild_concept_clusters()
        # circular: lilbee.wiki imports lilbee.data.ingest.file_hash, so the
        # post-ingest hook stays function-local at this boundary.
        from lilbee.wiki.ingest import incremental_update

        await incremental_update(set(added) | set(updated) | set(removed))

    result = SyncResult(
        added=list(added),
        updated=list(updated),
        removed=removed,
        unchanged=plan.unchanged,
        failed=list(failed),
        skipped=list(skipped),
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


# In-flight task cap, as a multiple of _max_concurrent(): enough queued tasks to
# keep every compute slot fed, without materializing one task object per file.
_TASK_WINDOW_MULTIPLIER = 2


async def ingest_batch(
    files_to_process: list[FileToProcess],
    added: dict[str, None],
    updated: dict[str, None],
    failed: dict[str, None],
    skipped: dict[str, None],
    *,
    quiet: bool = False,
    on_progress: DetailedProgressCallback = noop_callback,
    cancel: threading.Event | None = None,
    flush_failed: set[str] | None = None,
    reasons: dict[str, str] | None = None,
) -> None:
    """Ingest a batch of files, optionally showing a Rich progress bar.
    When *needs_cleanup* is True, old chunks are deleted immediately before
    ingesting new ones so the two operations are atomic per file.
    When *cancel* is set, pending files raise CancelledError before starting.
    """
    semaphore = asyncio.Semaphore(_max_concurrent())
    window = _max_concurrent() * _TASK_WINDOW_MULTIPLIER
    total_files = len(files_to_process)

    async def _process_one(entry: FileToProcess, file_index: int) -> _IngestResult:
        name = entry.name
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
                page_texts: list[PageTextRecord] = []
                records = await _produce_records(
                    entry.path,
                    name,
                    entry.content_type,
                    quiet=quiet,
                    on_progress=on_progress,
                    page_texts_out=page_texts,
                )
                concept_records = await _build_concept_records(records, name)
                on_progress(
                    EventType.FILE_DONE,
                    FileDoneEvent(file=name, status="ok", chunks=len(records)),
                )
                return _IngestResult(
                    name,
                    entry.path,
                    len(records),
                    error=None,
                    file_hash=entry.file_hash,
                    records=records,
                    needs_cleanup=entry.needs_cleanup,
                    page_texts=page_texts,
                    stat=entry.stat,
                    concept_records=concept_records,
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
                return _IngestResult(name, entry.path, 0, error=exc)

    pending = (_process_one(entry, idx) for idx, entry in enumerate(files_to_process, 1))
    if quiet:
        await _collect_results(
            pending,
            total_files,
            added,
            updated,
            failed,
            skipped,
            window=window,
            on_progress=on_progress,
            flush_failed=flush_failed,
            reasons=reasons,
        )
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
            await _collect_results(
                pending,
                total_files,
                added,
                updated,
                failed,
                skipped,
                window=window,
                on_progress=phase_progress,
                progress=progress,
                ptask=ptask,
                flush_failed=flush_failed,
                reasons=reasons,
            )


# Accumulate roughly this many chunks across documents before one batched
# LanceDB write. Bounds buffered-vector memory while amortizing the write lock
# and per-transaction overhead over many documents instead of one write per file.
_WRITE_FLUSH_CHUNKS = 2000


def _refill_window(
    in_flight: set[asyncio.Task[_IngestResult]],
    pending: Iterator[Coroutine[Any, Any, _IngestResult]],
    window: int,
) -> None:
    """Top up the in-flight task set from *pending*, capped at *window* tasks."""
    while len(in_flight) < window:
        coro = next(pending, None)
        if coro is None:
            return
        in_flight.add(asyncio.ensure_future(coro))


async def _collect_results(
    pending: Iterator[Coroutine[Any, Any, _IngestResult]],
    total: int,
    added: dict[str, None],
    updated: dict[str, None],
    failed: dict[str, None],
    skipped: dict[str, None],
    *,
    window: int,
    on_progress: DetailedProgressCallback = noop_callback,
    progress: Progress | None = None,
    ptask: Any = None,
    flush_failed: set[str] | None = None,
    reasons: dict[str, str] | None = None,
) -> None:
    """Run *pending* through a bounded task window, batching writes and progress.

    At most *window* tasks exist at once: results are consumed as they complete
    and the window is refilled from the iterator, so memory stays flat however
    many files a sync covers. Successful files are buffered and flushed to
    LanceDB in batches (one locked transaction per batch) rather than one write
    per file. The buffer is flushed on the way out too -- even on cancel -- so
    completed-but-unwritten work is persisted. On exception (typically
    asyncio.CancelledError from a user cancel), cancel every in-flight sibling
    and await them with ``return_exceptions=True`` so their pending
    CancelledErrors don't surface as "Task exception was never retrieved".
    """
    buffer: list[_IngestResult] = []
    buffered_chunks = 0
    completed_count = 0
    to_purge: list[str] = []
    in_flight: set[asyncio.Task[_IngestResult]] = set()
    try:
        _refill_window(in_flight, pending, window)
        while in_flight:
            done, still_running = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
            in_flight = set(still_running)
            for fut in done:
                result = fut.result()
                completed_count += 1
                status = _classify_result(result, added, updated, failed, skipped, reasons)
                if status is BatchStatus.INGESTED:
                    buffered_chunks = await _buffer_and_maybe_flush(
                        result, buffer, buffered_chunks, added, updated, failed, flush_failed
                    )
                elif status is BatchStatus.SKIPPED and result.needs_cleanup:
                    # An already-indexed file edited to yield zero chunks and zero
                    # page texts is never buffered, so the flush's cleanup never
                    # runs; purge its stale chunks/source row so it stops surfacing.
                    to_purge.append(result.name)
                _report_file_progress(
                    result, status, completed_count, total, on_progress, progress, ptask
                )
            _refill_window(in_flight, pending, window)
    finally:
        # The inner finally guarantees the sibling cancel even if the flush
        # itself raises (e.g. a cancellation landing on the to_thread await).
        try:
            await asyncio.to_thread(_flush_writes, buffer, added, updated, failed, flush_failed)
            await asyncio.to_thread(_purge_emptied_sources, to_purge)
        finally:
            still_pending = [t for t in in_flight if not t.done()]
            for task in still_pending:
                task.cancel()
            if still_pending:
                await asyncio.gather(*still_pending, return_exceptions=True)


async def _buffer_and_maybe_flush(
    result: _IngestResult,
    buffer: list[_IngestResult],
    buffered_chunks: int,
    added: dict[str, None],
    updated: dict[str, None],
    failed: dict[str, None],
    flush_failed: set[str] | None,
) -> int:
    """Buffer one ingested file, flushing at the chunk threshold; returns the new count."""
    buffer.append(result)
    # Zero-chunk files count one unit so the buffer stays bounded.
    buffered_chunks += max(result.chunk_count, 1)
    if buffered_chunks >= _WRITE_FLUSH_CHUNKS:
        await asyncio.to_thread(_flush_writes, buffer, added, updated, failed, flush_failed)
        buffered_chunks = 0
    return buffered_chunks


def _report_file_progress(
    result: _IngestResult,
    status: BatchStatus,
    completed_count: int,
    total: int,
    on_progress: DetailedProgressCallback,
    progress: Progress | None,
    ptask: Any,
) -> None:
    """Advance the Rich bar (when present) and emit one BATCH_PROGRESS event."""
    if progress is not None and ptask is not None:
        desc = f"Ingested {result.name}" if result.error is None else f"Failed {result.name}"
        progress.update(ptask, description=desc)
        progress.advance(ptask)
    with contextlib.suppress(TaskCancelledError):
        on_progress(
            EventType.BATCH_PROGRESS,
            BatchProgressEvent(
                file=result.name,
                status=status,
                current=completed_count,
                total=total,
            ),
        )


def _classify_result(
    result: _IngestResult,
    added: dict[str, None],
    updated: dict[str, None],
    failed: dict[str, None],
    skipped: dict[str, None],
    reasons: dict[str, str] | None = None,
) -> BatchStatus:
    """Record a completed file's outcome and return its batch status.

    Failures and zero-chunk files are tracked here; a successful file is reported
    as ``INGESTED`` and its chunks are persisted by the batched flush, so it stays
    in ``added`` / ``updated`` until then. When *reasons* is given, the
    human-readable cause is recorded there (filename → reason) for reporting.
    """
    if result.error is not None:
        # A traceback here would bleed into the TUI chat pane; the full trace stays at DEBUG.
        log.warning("Failed to ingest %s: %s", result.name, result.error)
        log.debug("Traceback for failed ingest of %s", result.name, exc_info=result.error)
        added.pop(result.name, None)
        updated.pop(result.name, None)
        failed[result.name] = None
        if reasons is not None:
            reasons[result.name] = f"{type(result.error).__name__}: {result.error}"
        return BatchStatus.FAILED
    if result.chunk_count == 0 and not result.page_texts:
        # Nothing extracted: no source row, so the file is retried next sync; a
        # zero-chunk file WITH page texts stays ingested so it stops replanning.
        added.pop(result.name, None)
        updated.pop(result.name, None)
        skipped[result.name] = None
        if reasons is not None:
            reasons[result.name] = "no text extracted (0 chunks)"
        return BatchStatus.SKIPPED
    return BatchStatus.INGESTED


# Back off briefly before the single flush retry: the usual contender is a
# search-triggered FTS optimize holding the store lock past its 30s timeout.
_FLUSH_RETRY_DELAY_SECONDS = 2.0


def _retry_after_lock_timeout(write: Callable[[], object]) -> None:
    """Run one store write, retrying once after a lock timeout."""
    try:
        write()
    except LockTimeoutError:
        log.warning(
            "Store write lock busy; retrying batch flush in %.0fs", _FLUSH_RETRY_DELAY_SECONDS
        )
        time.sleep(_FLUSH_RETRY_DELAY_SECONDS)
        write()


def _flush_batch(buffer: list[_IngestResult]) -> None:
    """Persist one flush unit in a single locked ``write_chunks_batch`` transaction.

    Page texts travel inside each :class:`ChunkWrite` so the store writes them
    after the cleanup delete (which clears the source's old page-text rows) and
    before the source row: a page-text failure leaves the row stale and the file
    replans next sync instead of losing its pages forever behind the stat
    short-circuit. The write retries once on a lock timeout.
    """
    store = get_services().store
    items = [
        ChunkWrite(
            source=r.name,
            file_hash=r.file_hash or file_hash(r.path),
            records=cast(list[dict], r.records or []),
            needs_cleanup=r.needs_cleanup,
            stat=r.stat,
            page_texts=cast(list[dict], r.page_texts or []),
        )
        for r in buffer
    ]
    _retry_after_lock_timeout(lambda: store.write_chunks_batch(items))
    _flush_concept_records(buffer)


def _flush_concept_records(buffer: list[_IngestResult]) -> None:
    """Write the flush unit's buffered concept rows in one batched pass.

    Runs after the chunk write so a failed flush (files moved to ``failed``
    and replanned) never lands concept rows for unwritten chunks. A concept
    write failure is logged and never fails the files, matching the
    per-file extraction failure semantics.
    """
    batches = [r.concept_records for r in buffer if r.concept_records is not None]
    if not batches:
        return
    try:
        get_services().concepts.write_concept_records(ConceptRecords.merged(batches))
    except Exception:
        log.warning("Concept indexing failed for %d-file batch", len(batches), exc_info=True)


def _purge_emptied_sources(names: list[str]) -> None:
    """Remove the prior index entry for files that now extract to nothing.

    An already-indexed file edited to yield zero chunks and zero page texts is
    classified SKIPPED and never buffered, so the batched cleanup delete never
    runs and its old chunks and source row would linger in search results. Full
    removal here keeps the index consistent; ``remove_documents`` is a no-op for
    never-indexed (brand-new empty) files, so unindexed inputs cost nothing.
    """
    if not names:
        return
    get_services().store.remove_documents(names)


def _flush_writes(
    buffer: list[_IngestResult],
    added: dict[str, None],
    updated: dict[str, None],
    failed: dict[str, None],
    flush_failed: set[str] | None = None,
) -> None:
    """Flush the buffered documents to the store; track a write failure.

    Each buffered file's page texts, chunks, cleanup delete, and source upsert
    are written by :func:`_flush_batch`. If that fails, every file in the batch
    is moved to ``failed`` since its source row did not land, and recorded in
    *flush_failed* so the caller replans them next sync instead of skip-marking
    them; the exception never escapes, so the caller's sibling-cancel and
    skip-marker path always runs. The buffer is cleared either way.
    """
    if not buffer:
        return
    try:
        _flush_batch(buffer)
    except Exception as exc:
        for r in buffer:
            log.warning("Failed to write %s: %s", r.name, exc)
            added.pop(r.name, None)
            updated.pop(r.name, None)
            failed[r.name] = None
            if flush_failed is not None:
                flush_failed.add(r.name)
    finally:
        buffer.clear()
