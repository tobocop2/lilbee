"""Top-level sync orchestration: discovery, dispatch, batching, post-sync hooks."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
import time
from collections import deque
from collections.abc import AsyncGenerator, Callable, Coroutine, Iterable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import count
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
from lilbee.core.config import Config, active_config
from lilbee.data.extract.chunk import ChunkLimitError
from lilbee.data.extract.document import (
    extract_batching,
    ingest_document,
    ingest_markdown,
    warn_if_table_model_ignored,
)
from lilbee.data.extract.trace import configure_from_env as configure_trace_from_env
from lilbee.data.ingest.adaptive import (
    AdaptiveController,
    ResizableGate,
    enumerate_fleet_devices,
    make_signal_sampler,
    profile_for,
    resolve_mode,
)
from lilbee.data.ingest.code import ingest_code_sync
from lilbee.data.ingest.discovery import (
    ExclusionReason,
    classify_file,
    discover_corpus,
    discover_files,
    file_hash,
    resolve_source_root,
)
from lilbee.data.ingest.errors import error_reason
from lilbee.data.ingest.fanout import (
    WORKER_LOG_NAME,
    ShardDone,
    ShardOptions,
    ShardSpec,
    aggregate_results,
    plan_fanout,
    run_workers,
)
from lilbee.data.ingest.ignore import IgnoreRules
from lilbee.data.ingest.skip_marker import (
    clear_skip_markers,
    describe_skips,
    load_skip_markers,
    load_skip_reasons,
    write_skip_markers,
    write_skip_reasons,
)
from lilbee.data.offload import (
    embed_inflight_target,
    max_workers,
    to_executor,
    to_ingest_thread,
)
from lilbee.data.store import (
    SOURCE_STAT_UNKNOWN,
    ChunkWrite,
    ConceptRecords,
    PageTextRecord,
    SourceMeta,
    SourceRecord,
    SourceStat,
    SourceStatBackfill,
    SourceType,
    Store,
    source_stat,
)
from lilbee.data.title import derive_title
from lilbee.data.types import (
    ChunkRecord,
    FileChangePlan,
    FileToProcess,
    ShardId,
    SyncResult,
    _IngestResult,
)
from lilbee.runtime.asyncio_loop import is_executor_shutdown
from lilbee.runtime.cancellation import CancelSignal, TaskCancelledError
from lilbee.runtime.cpu import available_cpu_count, cpu_quota
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
    main thread, and is the cap for text/code ingest. Vision OCR is different: every file
    in compute holds a continuous-batching slot on a vision server, so an OCR run is bounded
    by the vision slot capacity. Once the fleet is up that capacity is the servers' real
    fitted ``--parallel`` slots (a memory-constrained card fits fewer than requested); until
    then it is estimated from ``replicas x per-server pages``. Sizing to real capacity keeps
    the OCR queue shallow instead of piling pages behind the dispatcher with their deadlines
    ticking, and still keeps every slot fed.
    """
    from lilbee.providers.fleet.replicas import gpu_device_count, resolve_replica_count
    from lilbee.providers.roles import WorkerRole

    config = active_config()
    if config.vision_model:
        fitted = get_services().provider.vision_slot_capacity()
        if fitted is not None:
            return fitted
        replicas = resolve_replica_count(WorkerRole.VISION, gpu_device_count())
        return max(1, replicas * config.vision_ocr_concurrency)
    if config.ingest_max_inflight > 0:
        return config.ingest_max_inflight  # explicit override
    # Auto: keep every embed replica fed. The CPU-bound quota alone leaves a
    # many-core multi-GPU box starved (~4 files/card), so scale admission with
    # the detected fleet size -- no manual cap needed.
    return max(cpu_quota(), embed_inflight_target())


async def _rebuild_concept_clusters() -> None:
    """Re-run Leiden clustering after sync. No-op if disabled."""
    if not active_config().concept_graph:
        return
    from lilbee.retrieval.concepts import concepts_available

    if not concepts_available():
        return
    try:
        cg = get_services().concepts
        if not cg.get_graph():
            return
        await to_ingest_thread(cg.rebuild_clusters)
    except Exception:
        log.warning("Concept cluster rebuild failed", exc_info=True)


async def build_entity_records(records: list[ChunkRecord], source_name: str) -> list[dict] | None:
    """Extract typed entities for ingested chunks. None when the mode is off.

    Gated twice: the ``entity_extraction`` config flag, and a schema already
    induced into the index; absent either, syncs cost nothing. Extraction
    failures degrade to no rows for the file, mirroring concept extraction.
    """
    config = active_config()
    if not config.entity_extraction or not records:
        return None
    from lilbee.retrieval.entities import ExtractorKind, extract_entities, load_schema

    schema = load_schema(get_services().store)
    if schema is None:
        return None
    nlp = None
    if any(t.kind is ExtractorKind.SPACY for t in schema.types):
        from lilbee.retrieval.concepts import concepts_available
        from lilbee.retrieval.concepts.nlp import load_spacy_pipeline

        if concepts_available():
            try:
                nlp = load_spacy_pipeline()
            except ImportError:
                log.warning("spaCy model unavailable; spacy-kind entity types skipped")
    provider = None
    if any(t.kind is ExtractorKind.LLM for t in schema.types):
        provider = get_services().provider
    try:
        return await to_ingest_thread(
            extract_entities,
            cast("list[Mapping[str, Any]]", records),
            schema,
            provider=provider,
            nlp=nlp,
        )
    except Exception:
        log.warning("Entity extraction failed for %s", source_name, exc_info=True)
        return None


async def build_concept_records(
    records: list[ChunkRecord], source_name: str
) -> ConceptRecords | None:
    """Extract concepts for ingested chunks and build their table rows. None if disabled.

    Pure record building, no store access: the rows are buffered on the file's
    ingest result and written once per flush (see :func:`_flush_concept_records`),
    so a large sync pays one concept-table write per flush, not per file.
    """
    if not active_config().concept_graph or not records:
        return None
    from lilbee.retrieval.concepts import concepts_available

    if not concepts_available():
        return None
    try:
        cg = get_services().concepts
        texts = [r["chunk"] for r in records]
        concept_lists = await to_ingest_thread(cg.extract_concepts_batch, texts)
        chunk_ids = [(source_name, r["chunk_index"]) for r in records]
        return await to_ingest_thread(cg.build_concept_records, chunk_ids, concept_lists)
    except Exception:
        log.warning("Concept extraction failed for %s", source_name, exc_info=True)
        return None


async def produce_records(
    path: Path,
    source_name: str,
    content_type: str,
    *,
    quiet: bool = False,
    on_progress: DetailedProgressCallback = noop_callback,
    page_texts_out: list[PageTextRecord] | None = None,
) -> tuple[list[ChunkRecord], SourceMeta]:
    """Extract, chunk, and embed a single file into (records, source metadata).

    The LanceDB write is deferred: records are returned to the caller and written
    in a batched flush (see :func:`_flush_writes`), so bulk ingest pays one
    write-lock acquisition per batch instead of one per file. The per-page text
    dataset rows land in ``page_texts_out`` and are written by the same flush.
    The returned metadata (extraction-provided when available, stem-derived title
    otherwise) stamps every record's ``title`` and updates the source row.
    """
    records: list[ChunkRecord]
    page_texts: list[PageTextRecord] = page_texts_out if page_texts_out is not None else []
    if content_type == "code":
        records = await to_ingest_thread(ingest_code_sync, path, source_name, on_progress)
        meta = SourceMeta(title=derive_title(source_name))
    elif path.suffix.lower() == ".md":
        records, meta = await ingest_markdown(
            path, source_name, on_progress, page_texts_out=page_texts
        )
    else:
        records, meta = await ingest_document(
            path,
            source_name,
            content_type,
            quiet=quiet,
            on_progress=on_progress,
            page_texts_out=page_texts,
        )

    for record in records:
        # NULL (not "") for an absent title, so chunk rows match the migration
        # and the _sources table, which both persist absence as NULL.
        record["title"] = meta.title or None
    return records, meta


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
    """One file's sync verdict: process it, hold it out on its skip marker, or unchanged.

    A held file is not in the index, so it is never counted as unchanged.
    """

    to_process: FileToProcess | None = None
    backfill: SourceStatBackfill | None = None
    is_update: bool = False
    held: bool = False


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
        return _FileChangeVerdict(held=True)
    # needs_cleanup=True unconditionally: delete_by_source is idempotent,
    # and this closes the race where a prior ingest wrote chunks but died
    # before upsert_source, leaving orphaned chunks that would duplicate.
    return _FileChangeVerdict(
        to_process=FileToProcess(
            name, path, content_type, current_hash, needs_cleanup=True, stat=current_stat
        ),
        is_update=old_hash is not None,
    )


def _plan_workers() -> int:
    """Worker count for the parallel planning pass: config override, else auto.

    ``config.ingest_workers`` (also set per run by ``add --max-cpus``) wins when
    positive; otherwise size to the container-aware CPU budget so a big corpus
    hashes on every core the pod actually has, not the host's vCPU count.
    """
    configured = active_config().ingest_workers
    return configured if configured > 0 else available_cpu_count()


# How often the plan pass logs progress. The pass can run for tens of minutes on
# a multi-million-file corpus while the Rich bar renders nothing without a TTY
# and stdout is block-buffered when piped; a periodic line (which logging flushes
# per record) keeps a headless run observable instead of looking hung.
_PLAN_LOG_INTERVAL_S = 10.0


class _PlanProgress:
    """Periodic progress for the plan/hash pass, with rate and ETA.

    Emitted at warning level, not info: the default LILBEE_LOG_LEVEL is WARNING,
    so an info line would be filtered before any handler and a headless
    ``lilbee sync`` would show nothing during the plan pass and still look hung.
    """

    def __init__(self, total: int) -> None:
        self._total = total
        self._done = 0
        self._started = time.monotonic()
        self._last = self._started

    def tick(self) -> None:
        self._done += 1
        now = time.monotonic()
        if now - self._last < _PLAN_LOG_INTERVAL_S:
            return
        self._last = now
        elapsed = now - self._started
        rate = self._done / elapsed if elapsed > 0 else 0.0
        remaining = (self._total - self._done) / rate if rate > 0 else 0.0
        log.warning(
            "Planning: examined %d/%d files (%.0f%%, %.0f files/s, ~%.0fs left)",
            self._done,
            self._total,
            100.0 * self._done / self._total,
            rate,
            remaining,
        )


class _StreamStop:
    """Stop signal for a streamed plan: the caller's cancel, or the stream closing.

    Closing the stream has to reach the plan batch in flight, not just the next one.
    Build-vs-buy: the hashers run in a thread pool, so the flag must be a
    thread-visible ``threading.Event``; ``anyio.CancelScope`` is async-only.
    """

    def __init__(self, cancel: CancelSignal | None) -> None:
        self._cancel = cancel
        self._closed = threading.Event()

    def close(self) -> None:
        self._closed.set()

    def is_set(self) -> bool:
        return self._closed.is_set() or (self._cancel is not None and self._cancel.is_set())


def _classify_pooled(
    pool: ThreadPoolExecutor,
    items: list[tuple[str, Path]],
    classify: Callable[[str, Path], _FileChangeVerdict],
    cancel: CancelSignal | None,
    progress: _PlanProgress,
) -> dict[str, _FileChangeVerdict]:
    """Fan *items* across *pool*, returning name -> verdict for what completed."""
    verdicts: dict[str, _FileChangeVerdict] = {}
    futures: dict[Future[_FileChangeVerdict], str] = {}
    for name, path in items:
        if cancel and cancel.is_set():
            break
        futures[pool.submit(classify, name, path)] = name
    for future in as_completed(futures):
        if cancel and cancel.is_set():
            # Drop queued-but-unstarted work; running tasks drain on their own.
            for pending in futures:
                pending.cancel()
            break
        verdicts[futures[future]] = future.result()
        progress.tick()
    return verdicts


def _classify_changes(
    items: list[tuple[str, Path]],
    existing_sources: dict[str, SourceRecord],
    skip_markers: dict[str, str],
    cancel: CancelSignal | None,
    *,
    progress: _PlanProgress | None = None,
    pool: ThreadPoolExecutor | None = None,
) -> dict[str, _FileChangeVerdict]:
    """Classify each file (stat + hash) by name, fanning across a thread pool.

    Independent per file and side-effect-free, so pooled classification matches a
    serial pass; ``hashlib`` releases the GIL during digest, giving real speedup
    on a large corpus. Returns name -> verdict. A set ``cancel`` stops promptly:
    submission halts and queued-but-unstarted work is cancelled, so a mid-pass
    cancel over a huge corpus does not hash every remaining file. A *pool* and
    *progress* passed in are shared across the batches of a streamed plan, so the
    ETA covers the whole corpus and the workers are spun up once.
    """

    def _classify(name: str, path: Path) -> _FileChangeVerdict:
        return _classify_file_change(name, path, existing_sources.get(name), skip_markers)

    total = len(items)
    progress = progress or _PlanProgress(total)
    if pool is not None:
        return _classify_pooled(pool, items, _classify, cancel, progress)
    workers = _plan_workers()
    if workers <= 1 or total <= 1:
        verdicts: dict[str, _FileChangeVerdict] = {}
        for name, path in items:
            if cancel and cancel.is_set():
                break
            verdicts[name] = _classify(name, path)
            progress.tick()
        return verdicts
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="lilbee-plan") as owned:
        return _classify_pooled(owned, items, _classify, cancel, progress)


def _plan_file_changes(
    disk_files: dict[str, Path],
    existing_sources: dict[str, SourceRecord],
    cancel: CancelSignal | None,
    skip_markers: dict[str, str] | None = None,
) -> FileChangePlan:
    """Diff every disk file against the store in one pass (see :func:`_plan_items`)."""
    return _plan_items(sorted(disk_files.items()), existing_sources, cancel, skip_markers or {})


def _plan_items(
    items: list[tuple[str, Path]],
    existing_sources: dict[str, SourceRecord],
    cancel: CancelSignal | None,
    skip_markers: dict[str, str],
    *,
    progress: _PlanProgress | None = None,
    pool: ThreadPoolExecutor | None = None,
) -> FileChangePlan:
    """Diff *items* (sorted name -> path) against the store, hashing only drifted files.

    A tracked file whose stored (size, mtime) matches the disk stat, and whose
    mtime predates the stat capture (see :func:`_stat_unchanged`), is unchanged
    without reading its bytes; everything else is SHA-256 hashed. A file whose
    current hash matches a marker in ``skip_markers`` (set by a prior failed
    attempt) is held out rather than retried every sync, and is reported as held
    out rather than counted unchanged. Edit the file or run
    ``/sync --force-rebuild`` to clear the marker and try again.

    Classification fans across a thread pool (see :func:`_classify_changes`); the
    plan is assembled from the results in the original sorted order, so a partial
    or reordered completion never yields a wrong or reordered plan -- only a
    shorter one when cancelled mid-pass.
    """
    verdicts = _classify_changes(
        items, existing_sources, skip_markers, cancel, progress=progress, pool=pool
    )

    files_to_process: list[FileToProcess] = []
    added: dict[str, None] = {}
    updated: dict[str, None] = {}
    stat_backfills: list[SourceStatBackfill] = []
    held_out: list[str] = []
    unchanged = 0
    # Assemble in the original sorted order so a partial (cancelled) or reordered
    # completion never changes the plan the serial pass would have produced.
    for name, _path in items:
        verdict = verdicts.get(name)
        if verdict is None:
            continue  # cancelled before this file was classified
        if verdict.to_process is None:
            if verdict.held:
                held_out.append(name)
            else:
                unchanged += 1
            if verdict.backfill is not None:
                stat_backfills.append(verdict.backfill)
            continue
        files_to_process.append(verdict.to_process)
        if verdict.is_update:
            updated[name] = None
        else:
            added[name] = None
    return FileChangePlan(files_to_process, added, updated, unchanged, stat_backfills, held_out)


@dataclass(frozen=True)
class _Move:
    """One relocated source: its old key, its new key, and the new file's stat."""

    old: str
    new: str
    stat: SourceStat | None


class _MovePool:
    """Absent sources indexed by content hash, consumed as moves are paired.

    Built once per sync and drained across the streamed plan's batches, so a file
    that moved matches exactly the one old key it would have matched in a
    single-pass plan however the corpus is sharded.
    """

    def __init__(self, absent: list[str], existing_sources: dict[str, SourceRecord]) -> None:
        by_hash: dict[str, list[str]] = {}
        for name in absent:
            record = existing_sources.get(name)
            if record is not None:
                by_hash.setdefault(record["file_hash"], []).append(name)
        for candidates in by_hash.values():
            candidates.sort()
        self._by_hash = by_hash

    def take(self, file_hash: str) -> str | None:
        """The next absent source with this content hash, or None."""
        matches = self._by_hash.get(file_hash)
        return matches.pop(0) if matches else None


def _detect_moves(
    files_to_process: list[FileToProcess],
    added: dict[str, None],
    pool: _MovePool,
) -> list[_Move]:
    """Pair brand-new files with absent sources of the same content hash.

    Only additions (files with a new name) can be moves; an update keeps its name.
    When several absent sources share a hash, pairing is deterministic (sorted)
    and one-to-one, so a duplicated file that moved matches exactly one old key
    and any leftovers stay indexed under their old key.
    """
    moves: list[_Move] = []
    for entry in files_to_process:
        if entry.name not in added:
            continue
        old = pool.take(entry.file_hash)
        if old is not None:
            moves.append(_Move(old, entry.name, entry.stat))
    return moves


def _apply_moves(
    moves: list[_Move],
    files_to_process: list[FileToProcess],
    added: dict[str, None],
) -> tuple[list[FileToProcess], list[str]]:
    """Fold detected moves out of the add set after they were relocated.

    Drops each moved file from the ingest list and the added set: its chunks were
    re-keyed onto the new source name, not rebuilt. Returns the trimmed
    ``(files_to_process, relocated)``.
    """
    moved_new = {m.new for m in moves}
    for name in moved_new:
        added.pop(name, None)
    remaining = [e for e in files_to_process if e.name not in moved_new]
    return remaining, sorted(moved_new)


def _absent_sources(sources: list[SourceRecord], disk_files: dict[str, Path]) -> list[str]:
    """Document sources whose backing file is not on disk this sync.

    A vanished file is never removed on its own (its chunks stay searchable, a
    dead path-link discovered at open time); this set exists only to pair a
    reappeared identical file to its old key in move detection. Imported sources
    have no backing file, so they are excluded.
    """
    return [
        s["filename"]
        for s in sources
        if s["filename"] not in disk_files and s["source_type"] != SourceType.IMPORTED
    ]


# A plan batch is only done when its slowest hash is, so the first one is small (work
# reaches the fleet within a second) and later ones amortize that barrier.
_PLAN_SHARD_MIN_FILES = 256
_PLAN_SHARD_MAX_FILES = 8192


def _plan_batch_bounds(total: int) -> Iterator[tuple[int, int]]:
    """(start, stop) slices covering *total* files, doubling up to the cap.

    Build-vs-buy: ``itertools.batched`` is the stock slicer but is 3.12+ (floor is
    3.11) and fixed-size, so it cannot ramp the batch size.
    """
    start = 0
    size = _PLAN_SHARD_MIN_FILES
    while start < total:
        stop = min(start + size, total)
        yield start, stop
        start = stop
        size = min(size * 2, _PLAN_SHARD_MAX_FILES)


@dataclass
class _StreamedPlan:
    """Bookkeeping a streamed plan accumulates across its batches.

    ``added`` and ``updated`` are the dicts the ingest pass mutates as files land.
    """

    added: dict[str, None] = field(default_factory=dict)
    updated: dict[str, None] = field(default_factory=dict)
    # Processed files' content hashes, for the skip markers written after the run.
    pending_hashes: dict[str, str] = field(default_factory=dict)
    relocated: list[str] = field(default_factory=list)
    # Old keys of relocated sources. The wiki index is keyed by source name, so
    # without these a move leaves the old name in it forever: its mentions
    # double-count and its dead chunk refs occupy the per-subject cap.
    relocated_from: list[str] = field(default_factory=list)
    unchanged: int = 0
    # Files a skip marker held out of this run, in plan order (ordered set).
    held_out: dict[str, None] = field(default_factory=dict)
    planned: int = 0
    # Files this pass's slice holds, from the discovery walk. Fixed before the
    # first batch is planned, so it is what progress is measured against: the
    # plan's own running totals grow as batches land and cannot say how much of
    # the corpus is left. Summed across a fan-out's workers it is the corpus.
    corpus_total: int = 0

    @property
    def resolved(self) -> int:
        """Files the plan disposed of without ingest, as it disposes of them.

        Unchanged files, skip-marker held-out files and repointed moves are done
        as far as the corpus is concerned, and nothing downstream ingests them,
        so an incremental sync would otherwise show a handful of changed files
        against the whole corpus. Files a cancelled plan never classified are
        absent from every count and so are not claimed as done.
        """
        return self.unchanged + len(self.held_out) + len(self.relocated)


async def _absorb_plan_batch(
    plan: FileChangePlan, state: _StreamedPlan, moves: _MovePool
) -> list[FileToProcess]:
    """Fold one batch's plan into *state* and return the files it queues for ingest.

    Relocations and stat backfills are written per batch, so they contend with the
    batch flushes and take the same one-shot lock retry.
    """
    store = get_services().store
    entries = plan.files_to_process
    state.unchanged += plan.unchanged
    state.held_out.update(dict.fromkeys(plan.held_out))
    state.added.update(plan.added)
    state.updated.update(plan.updated)

    detected = _detect_moves(entries, plan.added, moves)
    if detected:
        relocations = [(m.old, m.new, m.stat) for m in detected]
        await to_ingest_thread(
            _retry_after_lock_timeout, lambda: store.relocate_sources(relocations)
        )
        entries, relocated = _apply_moves(detected, entries, plan.added)
        state.relocated_from.extend(m.old for m in detected)
        for name in relocated:
            state.added.pop(name, None)
        state.relocated.extend(relocated)
    if plan.stat_backfills:
        backfills = plan.stat_backfills
        await to_ingest_thread(
            _retry_after_lock_timeout, lambda: store.update_source_stats(backfills)
        )

    state.pending_hashes.update((entry.name, entry.file_hash) for entry in entries)
    state.planned += len(entries)
    return entries


async def _plan_batches(
    disk_files: dict[str, Path],
    existing_sources: dict[str, SourceRecord],
    skip_markers: dict[str, str],
    absent: list[str],
    state: _StreamedPlan,
    cancel: CancelSignal | None,
) -> AsyncGenerator[list[FileToProcess]]:
    """Plan the corpus batch by batch, yielding each batch's files to ingest.

    Shards are contiguous slices of one sorted item list consumed in order, so
    this delivers the single-pass plan of :func:`_plan_items` in pieces. The next
    batch is planned while the current one ingests, on a dedicated thread: the
    shared ingest pool is saturated by extraction and would stall the stream it
    feeds. Empty batches are not yielded, so the first yield means there is work.
    """
    items = sorted(disk_files.items())
    moves = _MovePool(absent, existing_sources)
    progress = _PlanProgress(len(items))
    stop = _StreamStop(cancel)
    workers = _plan_workers()
    hashers = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="lilbee-plan")
    driver = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lilbee-plan-driver")

    def _plan(lo: int, hi: int) -> FileChangePlan:
        return _plan_items(
            items[lo:hi],
            existing_sources,
            stop,
            skip_markers,
            progress=progress,
            pool=hashers if workers > 1 else None,
        )

    bounds = list(_plan_batch_bounds(len(items)))
    try:
        ahead = asyncio.ensure_future(to_executor(driver, _plan, *bounds[0])) if bounds else None
        for index, _bound in enumerate(bounds):
            if ahead is None:
                break
            plan = await ahead
            ahead = (
                asyncio.ensure_future(to_executor(driver, _plan, *bounds[index + 1]))
                if index + 1 < len(bounds) and not stop.is_set()
                else None
            )
            entries = await _absorb_plan_batch(plan, state, moves)
            if entries:
                yield entries
    finally:
        stop.close()
        if ahead is not None:
            ahead.cancel()
            # Retrieve the outcome: a prefetch that had already failed would
            # otherwise surface as an unretrieved task exception.
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await ahead
        # No wait: the running batch notices the stop and exits on its own, and a
        # generator being closed must not block the event loop on a hash in flight.
        hashers.shutdown(wait=False)
        driver.shutdown(wait=False)


def detect_pending() -> int:
    """Count files in documents/ that are out of sync with the store.

    Cheap operation: filesystem walk + stat-gated SHA-256 hashing + a single
    sources-table read. No embedding, no writes. Returns the count of files that
    would be ingested (added + updated), which is what the TaskBar hint surfaces.
    A vanished file is not pending work: sync leaves it indexed, so it is not
    counted. Reuses ``_plan_file_changes`` so the diff logic stays single-sourced.
    Honors skip markers: a file that failed last time at this hash does
    not show up as pending. Blocking: callers on the event loop run it via
    ``asyncio.to_thread``.
    """
    config = active_config()
    disk_files = discover_files()
    if not disk_files:
        return 0
    existing_sources = {s["filename"]: s for s in get_services().store.get_sources()}
    skip_markers = load_skip_markers(config.data_root)
    plan = _plan_file_changes(disk_files, existing_sources, cancel=None, skip_markers=skip_markers)
    return len(plan.files_to_process)


# Refused files named in the log line before it falls back to a count.
_EXCLUDED_LOG_SAMPLE = 5


def _log_excluded(excluded: dict[str, ExclusionReason]) -> None:
    """Warn once per exclusion reason, naming at most ``_EXCLUDED_LOG_SAMPLE`` files."""
    by_reason: dict[ExclusionReason, list[str]] = {}
    for name, why in excluded.items():
        by_reason.setdefault(why, []).append(name)
    for why, names in by_reason.items():
        shown = sorted(names)[:_EXCLUDED_LOG_SAMPLE]
        more = f" and {len(names) - len(shown)} more" if len(names) > len(shown) else ""
        log.warning("Skipped %d file(s), %s: %s%s", len(names), why.value, ", ".join(shown), more)


def _load_sync_skip_markers(*, clear_first: bool) -> dict[str, str]:
    """Read the skip-marker file, optionally clearing it first.

    Entries are kept whether or not this pass discovered the file. A marker is
    what holds a removed or unextractable file out of the next sync, so a pass
    that cannot see the file -- its root is unmounted or moved, or this worker
    owns only a shard of the corpus -- must not erase the record and re-offer
    the file the moment it comes back. A marker is dropped when the file
    ingests cleanly, or by ``retry-skipped`` / ``rebuild``.
    """
    data_root = active_config().data_root
    if clear_first:
        # Clearing the markers makes the diff re-include the skipped files.
        clear_skip_markers(data_root)
    return load_skip_markers(data_root)


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
    write_skip_markers(active_config().data_root, markers)


def _persist_skip_reasons(markers: dict[str, str], reasons: dict[str, str]) -> None:
    """Write the reasons sidecar so it explains exactly the markers that survive.

    Merged onto what is already recorded, not replaced: the reasons for files
    this sync never touched (a removal, an earlier failure) explain markers that
    are still in force, and dropping them would leave the user with a file held
    out of every sync and nothing saying why. Reasons whose marker is gone are
    dropped in the same step, so the two sidecars cannot drift apart.
    """
    data_root = active_config().data_root
    merged = load_skip_reasons(data_root) | reasons
    write_skip_reasons(data_root, {name: why for name, why in merged.items() if name in markers})


def _force_rebuild_store(store: Any) -> None:
    """Drop the store and re-embed the preserved memories table (blocking).

    Run off the event loop by ``sync``. ``drop_all`` keeps the memories table, so
    its vectors are refreshed under the (possibly changed) embedding model;
    a no-op when empty or no embedder.
    """
    store.drop_all()
    embedder = get_services().embedder
    if embedder.embedding_available():
        store.rebuild_memory_embeddings(lambda texts: embedder.embed_batch(texts))


def _reconcile_missing(
    disk_files: dict[str, Path],
    sources: list[SourceRecord],
    failed: Iterable[str],
    skipped: Iterable[str],
    held: Iterable[str],
) -> list[str]:
    """On-disk document files absent from the store that no mechanism accounts for.

    A file discovery found and classified that ended up in neither the sources table
    nor any of the accounting sets was dropped with no signal -- the silent
    data-loss case (a scanned PDF that never made it into the index yet reported no
    error). Everything legitimately not indexed is excluded: a failed extraction is in
    ``failed``, a zero-text file this run attempted is in ``skipped``, a file this run
    held out on its skip marker is in ``held``, and an unsupported type was never
    returned by discovery in the first place. ``held`` is the run's held-out set, not
    the marker file: a stale marker (file edited since it failed) must not hide a drop.
    """
    accounted = {s["filename"] for s in sources} | set(failed) | set(skipped) | set(held)
    return sorted(name for name in disk_files if name not in accounted)


def _ignored_sources(sources: list[SourceRecord], rules: IgnoreRules) -> list[str]:
    """Indexed sources a ``.lilbeeignore`` now excludes.

    Asked of the patterns rather than of discovery, which cannot tell an excluded
    file from a lost one -- both simply stop being yielded, and the two need
    opposite handling. Reading every source rather than only the undiscovered
    ones keeps the answer independent of which slice of the corpus this pass
    walked. Imported sources have no file to match a pattern against.
    """
    ignored = []
    for source in sources:
        if source["source_type"] == SourceType.IMPORTED:
            continue
        resolved = resolve_source_root(source["filename"])
        if resolved is not None and rules.excludes_path(resolved[1], base=resolved[0]):
            ignored.append(source["filename"])
    return ignored


def _forget_ignored(sources: list[SourceRecord], rules: IgnoreRules) -> list[str]:
    """Drop sources the patterns now exclude from the index. Returns what went.

    A corpus-wide pass: it reads every source, so one worker of a fan-out must
    not run it. No skip marker is written either, because the ignore file is
    itself the durable statement -- a marker could only outlive it and hold a
    source out after its pattern was deleted. Deleting the pattern brings the
    source back on the next sync.
    """
    names = _ignored_sources(sources, rules)
    if not names:
        return []
    from lilbee.app.ingest import forget_removed_from_wiki_index

    removed = list(get_services().store.remove_documents(names).removed)
    forget_removed_from_wiki_index(removed)
    return removed


def _forget_refused(
    excluded: Mapping[str, ExclusionReason], existing: Mapping[str, SourceRecord]
) -> list[str]:
    """Drop indexed sources that discovery now refuses. Returns what went."""
    names = [name for name in excluded if name in existing]
    if not names:
        return []
    from lilbee.app.ingest import forget_removed_from_wiki_index

    removed = list(get_services().store.remove_documents(names).removed)
    forget_removed_from_wiki_index(removed)
    return removed


def _require_embedding_model() -> None:
    """Refuse ingest without an embedding model.

    Ingest has no degraded mode: without one, every chunk fails to embed after
    the run has already paid the parse and OCR cost. Search and chat fall back
    to keyword via embedding_available() and carry on.
    """
    if not get_services().embedder.validate_model():
        ref = active_config().embedding_model
        detail = (
            f"{ref!r} is not available: pull it, or set a different embedding_model."
            if ref
            else "None is configured: pull one and set embedding_model."
        )
        raise RuntimeError(f"Ingest needs an embedding model. {detail}")


async def _run_post_ingest_passes(
    store: Any,
    *,
    indexed_anything: bool,
    touched: set[str],
    cancel: CancelSignal | None,
) -> None:
    """Index maintenance, concept clusters, the wiki hook, and the entity lifecycle.

    The index and wiki passes only run when this sync indexed something. The
    entity lifecycle runs every sync (a cheap no-op when off or already current)
    so turning the setting on takes effect without a separate operation.
    """
    if indexed_anything:
        store.ensure_fts_index()
        store.ensure_scalar_indexes()
        store.ensure_vector_index()
        store.optimize_sources()
        await _rebuild_concept_clusters()
        await _update_wiki(touched, active_config())

    from lilbee.retrieval.entities.lifecycle import ensure_entities

    await to_ingest_thread(ensure_entities, cancel)


async def _update_wiki(changed_sources: set[str], config: Config) -> None:
    """Refresh the wiki index, and regenerate pages when auto-update is on.

    The index refresh runs whenever the wiki is enabled, because it spends no
    LLM call and it is what lets the browse tree list a page the moment its
    document lands. Regeneration is the expensive half and stays behind
    wiki_auto_update.

    Best effort: the ingest itself already succeeded and `lilbee wiki update`
    re-runs the regeneration, so a wiki failure must not skip the post-ingest
    entity pass or the reconciliation guard.
    """
    if not config.wiki:
        return
    # circular: lilbee.wiki imports lilbee.data.ingest.file_hash, so the
    # post-ingest hook stays function-local at this boundary.
    from lilbee.wiki.ingest import incremental_update
    from lilbee.wiki.stubs import refresh_stub_index

    try:
        await to_ingest_thread(
            refresh_stub_index, get_services().store, config, sources=changed_sources
        )
    except Exception:
        log.warning("Wiki index refresh failed after sync", exc_info=True)

    if not config.wiki_auto_update:
        return
    try:
        await incremental_update(changed_sources, config)
    except Exception:
        log.warning("Wiki auto-update failed after sync", exc_info=True)


def _worker_failure_message(failures: list[ShardDone], specs: list[ShardSpec]) -> str:
    """What a failed fan-out reports; its workers' shards are kept for the re-run."""
    detail = "; ".join(
        f"worker {failure.index} ({specs[failure.index].config.data_root / WORKER_LOG_NAME}): "
        f"{failure.error}"
        for failure in failures
    )
    return (
        f"{len(failures)} ingest worker(s) failed, so the index was not updated: {detail}. "
        "Their work is kept: re-running sync continues from where they stopped."
    )


def _merge_worker_shards(store: Store, specs: list[ShardSpec], touched: set[str]) -> None:
    """Fold every worker's shard into this index.

    A store with no chunks of its own takes the shards whole; one that already
    holds a corpus takes only the sources this run touched, so a re-sync replaces
    those rows instead of appending a second copy of everything.
    """
    from lilbee.data.store.shard_merge import merge_shards

    scope = touched if store.has_chunks() else None
    merge_shards(store, [spec.config.lancedb_dir for spec in specs], sources=scope)


async def _sync_across_workers(
    specs: list[ShardSpec],
    store: Store,
    *,
    prune_ignored: bool = False,
    options: ShardOptions,
    quiet: bool,
    on_progress: DetailedProgressCallback,
    cancel: CancelSignal | None,
) -> SyncResult:
    """Ingest on one worker per GPU, then fold their shards into the one index."""
    verdicts = await run_workers(
        specs, options=options, quiet=quiet, on_progress=on_progress, cancel=cancel
    )
    if cancel is not None and cancel.is_set():
        raise asyncio.CancelledError
    if failures := [verdict for verdict in verdicts if verdict.error is not None]:
        raise RuntimeError(_worker_failure_message(failures, specs))
    result = aggregate_results(verdicts)
    touched = set(result.added) | set(result.updated) | set(result.relocated)
    await to_ingest_thread(_merge_worker_shards, store, specs, touched)
    # No worker sees the whole corpus, so each one leaves this pass to the parent.
    if prune_ignored:
        result.removed = await to_ingest_thread(
            _forget_ignored, store.get_sources(), IgnoreRules.for_corpus()
        )
    await _run_post_ingest_passes(
        store, indexed_anything=bool(touched), touched=touched, cancel=cancel
    )
    on_progress(
        EventType.DONE,
        SyncDoneEvent(
            added=len(result.added),
            updated=len(result.updated),
            removed=len(result.removed),
            failed=len(result.failed),
            skipped=len(result.skipped),
            relocated=len(result.relocated),
        ),
    )
    return result


async def sync(
    force_rebuild: bool = False,
    quiet: bool = False,
    *,
    on_progress: DetailedProgressCallback = noop_callback,
    cancel: CancelSignal | None = None,
    retry_skipped: bool = False,
    prune_ignored: bool = False,
    shard: ShardId | None = None,
) -> SyncResult:
    """Sync documents/ with the vector store.
    Returns a SyncResult with the added/updated/removed/unchanged/failed/skipped lists.
    When *quiet* is True, the Rich progress bar is suppressed (for JSON output).
    When *cancel* is set mid-run, planning and processing stop between files
    without data loss (completed work is flushed) and CancelledError is raised;
    a cancel already set on entry returns an empty result instead.
    When *retry_skipped* (or *force_rebuild*) is set, the failed-file skip
    markers are cleared so this sync attempts every file.
    When *prune_ignored* is set, sources a ``.lilbeeignore`` now excludes are
    dropped from the index. Off by default: the patterns govern what sync takes
    in, and removing what a past sync already indexed is the caller's decision.
    A *shard* runs this sync as one worker of a multi-GPU fan-out: it sees only
    its slice of the corpus and leaves the corpus-wide passes to the parent.
    """
    config = active_config()
    _store = get_services().store

    if force_rebuild:
        # drop_all + memory re-embedding are heavy blocking store work; run them
        # off the event loop so a rebuild doesn't stall other admitted requests.
        await to_ingest_thread(_force_rebuild_store, _store)

    config.documents_dir.mkdir(parents=True, exist_ok=True)

    if shard is None and (specs := plan_fanout()):
        return await _sync_across_workers(
            specs,
            _store,
            prune_ignored=prune_ignored,
            options=ShardOptions(
                parent_pid=os.getpid(),
                force_rebuild=force_rebuild,
                retry_skipped=retry_skipped,
            ),
            quiet=quiet,
            on_progress=on_progress,
            cancel=cancel,
        )

    rules = IgnoreRules.for_corpus()
    scan = discover_corpus(shard, rules)
    disk_files = scan.files
    sources = _store.get_sources()
    existing_sources = {s["filename"]: s for s in sources}
    skip_markers = _load_sync_skip_markers(clear_first=force_rebuild or retry_skipped)

    failed: dict[str, None] = {}
    # Refused formats start the run skipped; they get no skip marker (no planned hash).
    skipped: dict[str, None] = dict.fromkeys(scan.excluded)
    # filename → why it was skipped/failed (for reporting)
    reasons: dict[str, str] = {name: why.value for name, why in scan.excluded.items()}
    flush_failed: set[str] = set()
    _log_excluded(scan.excluded)

    # Opt-in, and corpus-wide: a worker sees one slice but the whole sources
    # table, so it leaves this pass to the parent rather than racing its siblings.
    ignored = _forget_ignored(sources, rules) if prune_ignored and shard is None else []
    # Shard-safe: a worker removes only keys from its own slice.
    refused = _forget_refused(scan.excluded, existing_sources)

    # Sources whose backing file is not on disk this pass. A vanished file is NOT
    # removed: it stays indexed and searchable, a dead path-link the user
    # discovers only when they try to open it, and the set pairs a reappeared
    # identical file to its old key below. What was just removed leaves the set,
    # where it could otherwise capture a real move.
    gone = set(ignored) | set(refused)
    absent = [name for name in _absent_sources(sources, disk_files) if name not in gone]

    # The planning pass stats (and where needed hashes) every file on disk, batch by
    # batch off the event loop and overlapped with ingest. A brand-new file whose
    # content hash matches an absent source is folded in per batch as a move, not an
    # add: repointed in place so its chunks and embeddings are reused, not rebuilt.
    state = _StreamedPlan(corpus_total=len(disk_files))
    added, updated, pending_hashes = state.added, state.updated, state.pending_hashes
    plan_batches = _plan_batches(disk_files, existing_sources, skip_markers, absent, state, cancel)

    # Snapshot the cumulative truncation counter so the delta over this sync can
    # surface "N chunks truncated" instead of being lost in per-chunk debug logs.
    truncated_before = get_services().embedder.truncated_total

    # Ingest files (with optional progress bar). Only non-empty batches are yielded,
    # so the first one arriving is what proves there is work to do.
    try:
        first = await anext(plan_batches, None)
        if first is not None:
            # Hold the embed fleet resident for the whole batch: an unevenly loaded
            # replica must not idle-unload and reload cold mid-run (which snowballs
            # into a fleet collapse). The ContextVar propagates into the ingest
            # thread pool, where the fleet actually spawns on the first embed.
            from lilbee.providers.fleet.ingest_warmth import keep_fleet_warm

            with keep_fleet_warm():
                _require_embedding_model()
                await ingest_stream(
                    _chain_plan_batches(first, plan_batches),
                    added,
                    updated,
                    failed,
                    skipped,
                    plan=state,
                    quiet=quiet,
                    on_progress=on_progress,
                    cancel=cancel,
                    flush_failed=flush_failed,
                    reasons=reasons,
                )
            if cancel is not None and cancel.is_set():
                # The stream stops feeding on cancel, so ingest can drain its
                # admitted files and return without raising. A cancelled run must
                # not go on to write skip markers or reconcile an unplanned corpus.
                raise asyncio.CancelledError
    finally:
        # Idempotent, and the only close when the stream is never consumed.
        await plan_batches.aclose()
    relocated = state.relocated

    # A flush failure is a transient store-side problem, not a verdict on the
    # file: leaving it unmarked re-plans it next sync instead of skipping it.
    marker_failed = [name for name in (*failed, *skipped) if name not in flush_failed]
    _persist_skip_markers(
        skip_markers, pending_hashes, succeeded=[*added, *updated], failed=marker_failed
    )
    # Record why each file this run skip-marked was marked (informational; the
    # hash markers above drive the resume logic). Only marker_failed files, so a
    # transient flush failure doesn't leave a stale reason behind.
    _persist_skip_reasons(skip_markers, {n: reasons[n] for n in marker_failed if n in reasons})

    if shard is None:
        # A worker's shard is merged before the indexes are built, so the passes
        # run once corpus-wide in the parent instead of once per shard.
        await _run_post_ingest_passes(
            _store,
            indexed_anything=bool(state.planned or relocated),
            # The old names of relocated sources ride along so the wiki index
            # subtracts them in the same pass that merges their new ones.
            touched=set(added) | set(updated) | set(relocated) | set(state.relocated_from),
            cancel=cancel,
        )

    # Reconciliation guard against silent data loss: any on-disk document file that
    # ended up in neither the index nor an accounting set was dropped without a
    # signal. Surface it loudly instead of letting a whole dataset vanish quietly.
    if missing := _reconcile_missing(
        disk_files, _store.get_sources(), failed, skipped, state.held_out
    ):
        log.warning(
            "Sync reconciliation: %d document file(s) on disk are absent from the index "
            "with no failure reported (possible silent drop): %s",
            len(missing),
            ", ".join(missing[:20]),
        )

    result = SyncResult(
        added=list(added),
        updated=list(updated),
        removed=ignored + refused,
        unchanged=state.unchanged,
        relocated=relocated,
        failed=list(failed),
        skipped=list(skipped),
        held_out=describe_skips(config.data_root, state.held_out),
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
            relocated=len(result.relocated),
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


def _build_admission(
    baseline: int, pages_done: list[int]
) -> tuple[asyncio.Semaphore | ResizableGate, int, asyncio.Task[None] | None]:
    """The batch's admission control, plus its task-window size and controller task.

    Static mode (the default) returns a fixed semaphore and no controller. Adaptive
    mode, when a GPU fleet is present to feed, returns a resizable gate and a running
    :class:`AdaptiveController` that tunes it toward this box's throughput knee; with
    no fleet it falls back to the static path so a GPU-less host is never affected.
    """
    profile = profile_for(resolve_mode())
    devices = enumerate_fleet_devices() if profile is not None else []
    if profile is None or not devices:
        return asyncio.Semaphore(baseline), baseline * _TASK_WINDOW_MULTIPLIER, None
    permit_max = max_workers()
    gate = ResizableGate(min(baseline, permit_max))
    controller = AdaptiveController(
        gate,
        profile,
        make_signal_sampler(devices),
        lambda: pages_done[0],
        permit_min=1,
        permit_max=permit_max,
    )
    task = asyncio.ensure_future(controller.run())
    # warning, not info: the default LILBEE_LOG_LEVEL is WARNING, so the
    # auto-chosen concurrency would otherwise never surface on a headless sync.
    log.warning(
        "Adaptive ingest concurrency (%s): start %d, max %d", profile.name, gate.limit, permit_max
    )
    return gate, permit_max * _TASK_WINDOW_MULTIPLIER, task


def _failed_result(
    exc: Exception,
    entry: FileToProcess,
    *,
    pages_done: list[int],
    on_progress: DetailedProgressCallback,
    cancel: CancelSignal | None,
) -> _IngestResult:
    """A file's failure as a result, or cancellation when the run is stopping.

    During shutdown, worker pools raise RuntimeError from submit(). Those are
    cancellation, not ingest failures: the cancel flag is the source of truth,
    and the executor's shutdown message covers the race where cancel was set
    after the submit.
    """
    if (cancel and cancel.is_set()) or is_executor_shutdown(exc):
        raise asyncio.CancelledError from exc
    # Suppress TaskCancelledError on the FILE_DONE notice: the user already
    # cancelled, and re-raising here would strand sibling tasks awaiting in
    # _collect_results.
    with contextlib.suppress(TaskCancelledError):
        on_progress(EventType.FILE_DONE, FileDoneEvent(file=entry.name, status="error", chunks=0))
    pages_done[0] += 1  # cleared the gate (as a failure); still a throughput tick
    return _IngestResult(entry.name, entry.path, 0, error=exc)


def _over_limit_result(
    exc: ChunkLimitError,
    entry: FileToProcess,
    *,
    pages_done: list[int],
    on_progress: DetailedProgressCallback,
) -> _IngestResult:
    """A file over the per-file chunk limit, recorded as skipped and skip-marked at its hash."""
    log.warning("Skipped %s: %s", entry.name, exc)
    with contextlib.suppress(TaskCancelledError):
        on_progress(EventType.FILE_DONE, FileDoneEvent(file=entry.name, status="skipped", chunks=0))
    pages_done[0] += 1
    return _IngestResult(
        entry.name,
        entry.path,
        0,
        error=None,
        file_hash=entry.file_hash,
        skip_reason=str(exc),
        needs_cleanup=entry.needs_cleanup,
    )


async def _stream_tasks(
    plan_batches: AsyncGenerator[list[FileToProcess]],
    make_task: Callable[[FileToProcess, int], Coroutine[Any, Any, _IngestResult]],
) -> AsyncGenerator[list[Coroutine[Any, Any, _IngestResult]]]:
    """Each plan batch's per-file coroutines, in plan order."""
    index = count(1)
    async for plan_batch in plan_batches:
        yield [make_task(entry, next(index)) for entry in plan_batch]


async def _chain_plan_batches(
    first: list[FileToProcess], rest: AsyncGenerator[list[FileToProcess]]
) -> AsyncGenerator[list[FileToProcess]]:
    """Yield an already-pulled plan batch, then the remainder of its stream."""
    yield first
    async for plan_batch in rest:
        yield plan_batch


async def ingest_stream(
    plan_batches: AsyncGenerator[list[FileToProcess]],
    added: dict[str, None],
    updated: dict[str, None],
    failed: dict[str, None],
    skipped: dict[str, None],
    *,
    plan: _StreamedPlan | None = None,
    quiet: bool = False,
    on_progress: DetailedProgressCallback = noop_callback,
    cancel: CancelSignal | None = None,
    flush_failed: set[str] | None = None,
    reasons: dict[str, str] | None = None,
) -> None:
    """Ingest a stream of planned file batches, optionally showing a Rich progress bar.

    Files are admitted as their batch is planned, so ingest starts on the first
    batch instead of waiting for the whole corpus to be diffed. Old chunks are
    deleted in the same transaction as the new write, so the two are atomic per
    file. When *cancel* is set, pending files raise CancelledError before starting.

    *plan* is the bookkeeping the batches were planned into; it carries the corpus
    the run is measured against. Without it progress is reported with no total,
    since a bare stream of batches does not say what corpus it came from.
    """
    # Honor LILBEE_INGEST_TRACE once per batch: it raises the trace loggers above
    # the default WARNING so per-file extraction lines actually surface.
    configure_trace_from_env()
    warn_if_table_model_ignored()
    # Throughput is measured in OCR pages, not documents: a document's cost scales
    # with its page count (a 500-page scan is 500x a memo), so pages are the unbiased
    # unit of GPU-feeding work for the adaptive controller to hill-climb on.
    pages_done = [0]
    # Sized off the files with no source row yet, not the planned count: the plan
    # streams in batches and its total is unknown until the stream drains, but the
    # pool has to be decided before the first batch is dispatched. Unindexed files
    # are the one part of the plan that is known without diffing, exact for a first
    # ingest or a rebuild and near zero for an incremental sync, so a small sync
    # over a large corpus stays in-process. Undercounting only keeps a run
    # in-process, which is the safe direction.
    admission, window, controller_task = _build_admission(_max_concurrent(), pages_done)

    async def _process_one(entry: FileToProcess, file_index: int) -> _IngestResult:
        name = entry.name
        async with admission:
            if cancel and cancel.is_set():
                raise asyncio.CancelledError

            try:
                on_progress(
                    EventType.FILE_START,
                    FileStartEvent(file=name, total_files=feed.planned, current_file=file_index),
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
                records, meta = await produce_records(
                    entry.path,
                    name,
                    entry.content_type,
                    quiet=quiet,
                    on_progress=on_progress,
                    page_texts_out=page_texts,
                )
                concept_records = await build_concept_records(records, name)
                entity_rows = await build_entity_records(records, name)
                on_progress(
                    EventType.FILE_DONE,
                    FileDoneEvent(file=name, status="ok", chunks=len(records)),
                )
                pages_done[0] += max(1, len(page_texts))  # OCR pages cleared: the throughput signal
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
                    entity_rows=entity_rows,
                    meta=meta,
                )
            except ChunkLimitError as exc:
                return _over_limit_result(
                    exc, entry, pages_done=pages_done, on_progress=on_progress
                )
            except (asyncio.CancelledError, TaskCancelledError) as exc:
                # TaskCancelledError is the TUI's cooperative cancel signal raised
                # by reporter.check_cancelled() inside on_progress; treat it as
                # asyncio cancellation so _collect_results can drain siblings
                # cleanly instead of orphaning their pending exceptions.
                raise asyncio.CancelledError from exc
            except Exception as exc:
                return _failed_result(
                    exc, entry, pages_done=pages_done, on_progress=on_progress, cancel=cancel
                )

    feed = _ResultFeed(_stream_tasks(plan_batches, _process_one), plan)
    collect = _collect_results if quiet else _collect_under_bar
    try:
        # extract_batching coalesces extractions into xberg batch calls when the
        # toggle is on (off by default); the per-file collect contract is unchanged.
        async with extract_batching():
            await collect(
                feed,
                added,
                updated,
                failed,
                skipped,
                window=window,
                on_progress=on_progress,
                flush_failed=flush_failed,
                reasons=reasons,
            )
    finally:
        # Stop the adaptive controller (if any) before returning: its background
        # loop must not outlive the batch it was tuning.
        if controller_task is not None:
            controller_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await controller_task


# Accumulate roughly this many chunks across documents before one batched
# LanceDB write. Bounds buffered-vector memory while amortizing the write lock
# and per-transaction overhead over many documents instead of one write per file.
_WRITE_FLUSH_CHUNKS = 2000


class _ResultFeed:
    """Pull-based source of per-file ingest coroutines over a streamed plan.

    ``take(wait=False)`` hands back an already-planned file without blocking, so
    the collector waits on the planner only when it has nothing left to run.
    Build-vs-buy: an ``asyncio.Queue`` is the stock bounded channel, but it would
    need a separate producer task to pump the plan-batch generator into it and a
    sentinel to close it; pulling ``anext`` on demand keeps the plan stream the
    single driver and needs neither. ``planned`` is the file count seen so far:
    the run's total once the stream is drained.
    """

    def __init__(
        self,
        plan_batches: AsyncGenerator[list[Coroutine[Any, Any, _IngestResult]]],
        plan: _StreamedPlan | None = None,
    ) -> None:
        self._plan_batches = plan_batches
        self._buffer: deque[Coroutine[Any, Any, _IngestResult]] = deque()
        self._pull: asyncio.Task[list[Coroutine[Any, Any, _IngestResult]] | None] | None = None
        self._drained = False
        self._plan = plan if plan is not None else _StreamedPlan()
        self.planned = 0

    @property
    def corpus_total(self) -> int:
        """Files the run's slice holds, or 0 when the caller declared no corpus."""
        return self._plan.corpus_total

    @property
    def resolved(self) -> int:
        """Files already disposed of by the plan, which never reach this feed."""
        return self._plan.resolved

    def pull(self) -> asyncio.Task[list[Coroutine[Any, Any, _IngestResult]] | None] | None:
        """The in-flight plan-batch prefetch, so a waiting collector wakes when it lands."""
        return self._pull

    async def take(self, *, wait: bool) -> Coroutine[Any, Any, _IngestResult] | None:
        """The next planned file, or None when the stream is drained (or, with
        *wait* False, when the next batch is not planned yet)."""
        while not self._buffer:
            if self._drained:
                return None
            if self._pull is None:
                self._pull = asyncio.ensure_future(anext(self._plan_batches, None))
            if not wait and not self._pull.done():
                return None
            plan_batch = await self._pull
            self._pull = None
            if plan_batch is None:
                self._drained = True
                return None
            self._buffer.extend(plan_batch)
            self.planned += len(plan_batch)
        return self._buffer.popleft()

    async def aclose(self) -> None:
        """Close the plan stream and discard files that were never started."""
        if self._pull is not None:
            self._pull.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                # A batch that landed before the cancel took effect still owns
                # coroutines; recover it so they are closed rather than leaked.
                plan_batch = await self._pull
                if plan_batch:
                    self._buffer.extend(plan_batch)
            self._pull = None
        for coro in self._buffer:
            coro.close()
        self._buffer.clear()
        await self._plan_batches.aclose()


async def _refill_window(
    in_flight: set[asyncio.Task[_IngestResult]],
    feed: _ResultFeed,
    window: int,
) -> None:
    """Top up the in-flight task set from *feed*, capped at *window* tasks.

    Waits on the planner only when nothing is running, so a slow batch never
    stalls files that are already planned.
    """
    while len(in_flight) < window:
        coro = await feed.take(wait=not in_flight)
        if coro is None:
            return
        in_flight.add(asyncio.ensure_future(coro))


async def _next_completions(
    in_flight: set[asyncio.Task[_IngestResult]], prefetch: asyncio.Future[Any] | None
) -> tuple[Iterable[asyncio.Future[Any]], set[asyncio.Task[_IngestResult]]]:
    """Wait for the next file to finish, returning (completed, still running).

    The feed's plan-batch prefetch waits alongside the running files, so work is
    admitted as soon as it is planned rather than on the next file completion,
    and it is filtered out of the still-running set here since it is not a file.
    """
    waiting: set[asyncio.Future[Any]] = set(in_flight)
    if prefetch is not None:
        waiting.add(prefetch)
    done, still_running = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
    # Explicit loop, like _cancel_in_flight: Nuitka miscompiled the comprehension
    # form of this task-set filtering.
    remaining: set[asyncio.Task[_IngestResult]] = set()
    for task in still_running:
        if task is not prefetch:
            remaining.add(cast("asyncio.Task[_IngestResult]", task))
    return done, remaining


async def _collect_results(
    feed: _ResultFeed,
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
    """Run *feed* through a bounded task window, batching writes and progress.

    At most *window* tasks exist at once: results are consumed as they complete
    and the window is refilled from the feed, so memory stays flat however many
    files a sync covers. Successful files are buffered and flushed to LanceDB in
    batches (one locked transaction per batch) rather than one write per file.
    The buffer is flushed on the way out too -- even on cancel -- so
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
        await _refill_window(in_flight, feed, window)
        while in_flight:
            prefetch = feed.pull()
            done, in_flight = await _next_completions(in_flight, prefetch)
            saw_cancel = False
            for fut in done:
                if fut is prefetch:
                    continue  # a planned batch landing, not a file result
                try:
                    result = fut.result()
                except asyncio.CancelledError:
                    # A user cancel completes several futures together. Flag it but
                    # keep draining `done` so a sibling that genuinely finished in
                    # the same batch is still buffered and flushed (the
                    # cancel-persists contract), then propagate after the loop. A
                    # non-cancel exception still propagates immediately, as before,
                    # so a genuine ingest bug surfaces and cancels the siblings.
                    saw_cancel = True
                    continue
                completed_count += 1
                status = _classify_result(result, added, updated, failed, skipped, reasons)
                if status is BatchStatus.INGESTED:
                    buffered_chunks = await _buffer_and_maybe_flush(
                        result,
                        buffer,
                        buffered_chunks,
                        added,
                        updated,
                        failed,
                        skipped,
                        flush_failed,
                    )
                elif status is BatchStatus.SKIPPED and result.needs_cleanup:
                    # Zero-text result is never buffered; collect it for the
                    # purge pass (see _purge_emptied_sources).
                    to_purge.append(result.name)
                _report_file_progress(
                    result,
                    status,
                    feed.resolved + completed_count,
                    feed.corpus_total,
                    on_progress,
                    progress,
                    ptask,
                )
            if saw_cancel:
                # Completed siblings in this batch are now buffered; propagate the
                # cancel so the finally flushes them and cancels still-running work.
                raise asyncio.CancelledError
            await _refill_window(in_flight, feed, window)
    finally:
        # The inner finally guarantees the sibling cancel even if the flush
        # itself raises (e.g. a cancellation landing on the to_thread await).
        try:
            await to_ingest_thread(
                _flush_writes, buffer, added, updated, failed, skipped, flush_failed
            )
            await to_ingest_thread(_purge_emptied_sources, to_purge)
        finally:
            try:
                await _cancel_in_flight(in_flight)
            finally:
                # Closing the feed stops the planner behind it, so a cancelled
                # sync does not keep hashing the rest of the corpus.
                await feed.aclose()


async def _collect_under_bar(
    feed: _ResultFeed,
    added: dict[str, None],
    updated: dict[str, None],
    failed: dict[str, None],
    skipped: dict[str, None],
    *,
    window: int,
    on_progress: DetailedProgressCallback = noop_callback,
    flush_failed: set[str] | None = None,
    reasons: dict[str, str] | None = None,
) -> None:
    """Run :func:`_collect_results` under a transient Rich progress bar."""
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        # The corpus the discovery walk found, known before the first batch is
        # planned. None only when the caller supplied no plan to measure against.
        ptask = progress.add_task("Ingesting documents...", total=feed.corpus_total or None)
        # The bar advances once per file (in _collect_results), so a single
        # multi-page scanned PDF would freeze at "0/1" through its whole
        # OCR + embed phase. Drive the spinner's description off the same
        # EXTRACT (OCR page i/N) and EMBED (chunk i/N) events the TUI uses
        # so the row visibly moves while one file is being worked.
        await _collect_results(
            feed,
            added,
            updated,
            failed,
            skipped,
            window=window,
            on_progress=_phase_progress_callback(progress, ptask, on_progress),
            progress=progress,
            ptask=ptask,
            flush_failed=flush_failed,
            reasons=reasons,
        )


async def _cancel_in_flight(in_flight: set[asyncio.Task[_IngestResult]]) -> None:
    """Cancel still-running tasks and await them so their CancelledErrors are retrieved."""
    # Explicit loop: Nuitka miscompiled the comprehension form of this cleanup.
    still_pending = []
    for t in in_flight:
        if not t.done():
            still_pending.append(t)
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
    skipped: dict[str, None],
    flush_failed: set[str] | None,
) -> int:
    """Buffer one ingested file, flushing at the chunk threshold; returns the new count."""
    buffer.append(result)
    # Zero-chunk files count one unit so the buffer stays bounded.
    buffered_chunks += max(result.chunk_count, 1)
    if buffered_chunks >= _WRITE_FLUSH_CHUNKS:
        await to_ingest_thread(_flush_writes, buffer, added, updated, failed, skipped, flush_failed)
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
    """Advance the Rich bar (when present) and emit one BATCH_PROGRESS event.

    *completed_count* counts every file the pass has disposed of and *total* is
    the corpus the discovery walk found, so the pair answers how much of the
    corpus is done rather than how much of the plan so far is.
    """
    if progress is not None and ptask is not None:
        desc = f"Ingested {result.name}" if result.error is None else f"Failed {result.name}"
        # Set, not advanced: files the plan resolved without ingest produce no
        # result of their own and would otherwise never reach the bar.
        progress.update(ptask, description=desc, completed=completed_count)
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

    Failures, refusals and zero-chunk files are tracked here; a successful file is
    reported as ``INGESTED`` and its chunks are persisted by the batched flush, so
    it stays in ``added`` / ``updated`` until then. When *reasons* is given, the
    human-readable cause is recorded there (filename → reason) for reporting.
    """
    if result.skip_reason is not None:
        added.pop(result.name, None)
        updated.pop(result.name, None)
        skipped[result.name] = None
        if reasons is not None:
            reasons[result.name] = result.skip_reason
        return BatchStatus.SKIPPED
    if result.error is not None:
        # A traceback here would bleed into the TUI chat pane; the full trace stays at DEBUG.
        log.warning("Failed to ingest %s: %s", result.name, result.error)
        log.debug("Traceback for failed ingest of %s", result.name, exc_info=result.error)
        added.pop(result.name, None)
        updated.pop(result.name, None)
        failed[result.name] = None
        if reasons is not None:
            reasons[result.name] = error_reason(result.error)
        return BatchStatus.FAILED
    if result.chunk_count == 0:
        # No searchable chunks: never report it as added/updated. With no page
        # texts either, nothing is persisted and the file retries next sync. With
        # page texts, it stays INGESTED so its pages persist (export/recon) and it
        # stops replanning, but it is reported as skipped since search can't see it.
        added.pop(result.name, None)
        updated.pop(result.name, None)
        skipped[result.name] = None
        if reasons is not None:
            reasons[result.name] = (
                "no text extracted (0 chunks)"
                if not result.page_texts
                else "stored page text only (0 searchable chunks)"
            )
        return BatchStatus.SKIPPED if not result.page_texts else BatchStatus.INGESTED
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
            meta=r.meta,
        )
        for r in buffer
    ]
    _retry_after_lock_timeout(lambda: store.write_chunks_batch(items))
    _flush_concept_records(buffer)
    _flush_entity_rows(buffer)


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


def _flush_entity_rows(buffer: list[_IngestResult]) -> None:
    """Write the flush unit's buffered entity rows in one batched pass.

    Runs after the chunk write, which also performed the per-source deletes,
    so replacement never leaves a source's stale entity rows behind. A write
    failure is logged and never fails the files, matching concept semantics.
    """
    rows = [row for r in buffer if r.entity_rows for row in r.entity_rows]
    if not rows:
        return
    try:
        get_services().store.add_entities(rows)
    except Exception:
        log.warning("Entity indexing failed for a %d-row batch", len(rows), exc_info=True)


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
    skipped: dict[str, None],
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
            # A page-text-only file was pre-marked skipped at classification; on a
            # flush failure it belongs in failed only, never both.
            skipped.pop(r.name, None)
            failed[r.name] = None
            if flush_failed is not None:
                flush_failed.add(r.name)
    finally:
        buffer.clear()
