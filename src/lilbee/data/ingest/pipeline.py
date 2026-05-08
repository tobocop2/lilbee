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

from lilbee.core.config import cfg
from lilbee.core.services import get_services
from lilbee.data.ingest.code import ingest_code_sync
from lilbee.data.ingest.discovery import classify_file, discover_files, file_hash
from lilbee.data.ingest.extract import ingest_document, ingest_markdown
from lilbee.data.ingest.types import ChunkRecord, FileToProcess, SyncResult, _IngestResult
from lilbee.runtime.asyncio_loop import is_executor_shutdown
from lilbee.runtime.cpu import cpu_quota
from lilbee.runtime.progress import (
    BatchProgressEvent,
    BatchStatus,
    DetailedProgressCallback,
    EventType,
    FileDoneEvent,
    FileStartEvent,
    SyncDoneEvent,
    noop_callback,
    shared_progress,
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


async def _incremental_wiki_update(changed_sources: set[str]) -> None:
    """Regenerate only the wiki pages touched by *changed_sources*.

    Runs after a successful sync. Builds a fresh ``ExtractedEntity``
    set from the current corpus, keeps the records that either have no
    page on disk yet or whose chunk trail includes one of the changed
    sources, and regenerates just those. Above
    ``cfg.wiki_ingest_update_cap`` touched pages the auto-update
    bails out and logs a manual-update hint instead.
    """
    if not cfg.wiki or not changed_sources:
        return
    # circular: the wiki layer imports lilbee.data.ingest.file_hash, so these
    # stay function-local to break the cycle at the hook-entry boundary.
    from lilbee.data.store import SearchChunk
    from lilbee.wiki import append_wiki_log, build_wiki, update_wiki_index
    from lilbee.wiki.entity_extractor import EntityKind, get_entity_extractor
    from lilbee.wiki.shared import (
        WikiLogAction,
        WikiSubdir,
    )

    svc = get_services()
    extractor = get_entity_extractor(cfg.wiki_entity_mode, svc.provider, cfg)

    chunks: list[SearchChunk] = []
    for record in svc.store.get_sources():
        chunks.extend(svc.store.get_chunks_by_source(record["filename"]))
    entities = await asyncio.to_thread(extractor.extract, chunks)

    wiki_root = cfg.data_root / cfg.wiki_dir
    touched = []
    for entity in entities:
        # The extractor emits only ENTITY kind; CONCEPT is reserved
        # for LLM-curated pages produced inside the batched call and is
        # intentionally not considered here. Keeping the dispatch
        # neutral guards against a future extractor that re-introduces
        # CONCEPT.
        subdir = WikiSubdir.CONCEPTS if entity.kind is EntityKind.CONCEPT else WikiSubdir.ENTITIES
        page_path = wiki_root / subdir / f"{entity.slug}.md"
        if not page_path.exists():
            touched.append(entity)
            continue
        if any(ref.source in changed_sources for ref in entity.chunk_refs):
            touched.append(entity)

    if not touched:
        return

    if len(touched) > cfg.wiki_ingest_update_cap:
        # warning, not info: the default LILBEE_LOG_LEVEL is WARNING, so
        # log.info would silently drop the manual-update hint and the user
        # would see no signal at all during `lilbee sync` when the cap trips.
        log.warning(
            "Wiki auto-update skipped: %d pages touched (cap %d). "
            "Run 'lilbee wiki update' to refresh.",
            len(touched),
            cfg.wiki_ingest_update_cap,
        )
        append_wiki_log(
            WikiLogAction.INGEST,
            f"skipped: {len(touched)} pages exceeds cap {cfg.wiki_ingest_update_cap}",
        )
        return

    # extract_concepts=False so an incremental sync does not churn
    # concept slugs. Concept curation is a deliberate, user-invoked
    # refresh (full `lilbee wiki build`).
    pages = await asyncio.to_thread(
        build_wiki, touched, svc.provider, svc.store, cfg, extract_concepts=False
    )
    update_wiki_index()
    append_wiki_log(
        WikiLogAction.INGEST,
        f"{len(pages)} pages regenerated for {', '.join(sorted(changed_sources))}",
    )


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


async def _ingest_file(
    path: Path,
    source_name: str,
    content_type: str,
    *,
    quiet: bool = False,
    on_progress: DetailedProgressCallback = noop_callback,
) -> int:
    """Ingest a single file. Returns chunk count."""
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

    store = get_services().store
    chunk_count = await asyncio.to_thread(store.add_chunks, cast(list[dict], records))
    await _index_concepts(records, source_name)
    return chunk_count


def _plan_file_changes(
    disk_files: dict[str, Path],
    existing_sources: dict[str, str],
    cancel: threading.Event | None,
) -> tuple[list[FileToProcess], list[str], list[str], int]:
    """Diff disk against the store. Returns (to_process, added, updated, unchanged_count)."""
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
    """
    if not cfg.documents_dir.exists():
        return 0
    disk_files = discover_files()
    existing_sources = {s["filename"]: s["file_hash"] for s in get_services().store.get_sources()}
    removed = sum(1 for name in existing_sources if name not in disk_files)
    files_to_process, _, _, _ = _plan_file_changes(disk_files, existing_sources, cancel=None)
    return len(files_to_process) + removed


async def sync(
    force_rebuild: bool = False,
    quiet: bool = False,
    *,
    on_progress: DetailedProgressCallback = noop_callback,
    cancel: threading.Event | None = None,
) -> SyncResult:
    """Sync documents/ with the vector store.
    Returns summary dict with keys: added, updated, removed, unchanged, failed.
    When *quiet* is True, the Rich progress bar is suppressed (for JSON output).
    When *cancel* is set, processing stops between files without data loss.
    """
    _store = get_services().store

    if force_rebuild:
        _store.drop_all()

    cfg.documents_dir.mkdir(parents=True, exist_ok=True)

    disk_files = discover_files()
    existing_sources = {s["filename"]: s["file_hash"] for s in _store.get_sources()}

    removed: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []

    # Find files to remove (in DB but not on disk)
    for name in existing_sources:
        if name not in disk_files:
            _store.delete_by_source(name)
            _store.delete_source(name)
            removed.append(name)

    files_to_process, added, updated, unchanged = _plan_file_changes(
        disk_files, existing_sources, cancel
    )

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

    if files_to_process or removed:
        _store.ensure_fts_index()
        await _rebuild_concept_clusters()
        await _incremental_wiki_update(set(added) | set(updated) | set(removed))

    result = SyncResult(
        added=added,
        updated=updated,
        removed=removed,
        unchanged=unchanged,
        failed=failed,
        skipped=skipped,
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

            on_progress(
                EventType.FILE_START,
                FileStartEvent(file=name, total_files=total_files, current_file=file_index),
            )
            try:
                if needs_cleanup:
                    get_services().store.delete_by_source(name)
                chunk_count = await _ingest_file(
                    path,
                    name,
                    content_type,
                    quiet=quiet,
                    on_progress=on_progress,
                )
                on_progress(
                    EventType.FILE_DONE,
                    FileDoneEvent(file=name, status="ok", chunks=chunk_count),
                )
                return _IngestResult(name, path, chunk_count, error=None, file_hash=fhash)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # During shutdown, worker pools raise RuntimeError from
                # submit(). Prefer to treat these as cancellation rather than
                # as ingest failures. Detect via the cancel flag (source of
                # truth) or the executor's well-known shutdown message as a
                # fallback when cancel was set after the submit race.
                if (cancel and cancel.is_set()) or is_executor_shutdown(exc):
                    raise asyncio.CancelledError from exc
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
            token = shared_progress.set((progress, ptask))
            try:
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
                    on_progress=on_progress,
                    progress=progress,
                    ptask=ptask,
                )
            finally:
                shared_progress.reset(token)


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
    """Collect task results, optionally updating a Rich progress bar."""
    for completed_count, fut in enumerate(asyncio.as_completed(tasks), 1):
        result = await fut
        _apply_result(result, added, updated, failed, skipped)
        if progress is not None and ptask is not None:
            desc = f"Ingested {result.name}" if result.error is None else f"Failed {result.name}"
            progress.update(ptask, description=desc)
            progress.advance(ptask)
        if result.error is not None:
            progress_status = BatchStatus.FAILED
        elif result.chunk_count == 0:
            progress_status = BatchStatus.SKIPPED
        else:
            progress_status = BatchStatus.INGESTED
        on_progress(
            EventType.BATCH_PROGRESS,
            BatchProgressEvent(
                file=result.name,
                status=progress_status,
                current=completed_count,
                total=len(tasks),
            ),
        )


def _discard_from_list(lst: list[str], value: str) -> None:
    """Remove *value* from *lst* if present."""
    with contextlib.suppress(ValueError):
        lst.remove(value)


def _apply_result(
    result: _IngestResult,
    added: list[str],
    updated: list[str],
    failed: list[str],
    skipped: list[str],
) -> None:
    """Record an ingestion result: update store on success, track failure."""
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
        return
    if result.chunk_count == 0:
        # No chunks produced (e.g. scanned PDF without vision model, or
        # vision OCR returned no text). Don't record as a source so it
        # gets retried on next sync, and surface as skipped so the user
        # knows the file did not actually land in the store.
        _discard_from_list(added, result.name)
        _discard_from_list(updated, result.name)
        skipped.append(result.name)
        return

    fhash = result.file_hash or file_hash(result.path)
    get_services().store.upsert_source(result.name, fhash, result.chunk_count)
