"""Worker processes for bulk ingest: one GIL per worker.

Bulk ingest throughput is concurrent embeds divided by their latency, and a
single process is limited in how many it can keep in flight -- a ``--gil``
profile of the 8.8M-passage MS MARCO ingest charges that dispatch ~37% of
GIL-held time, file I/O ~13% and asyncio ~9%, diffuse with no single hotspot to
remove. More processes raise that ceiling only by raising the aggregate
concurrency: an 8-worker A/B that kept the total in flight unchanged measured
1.00x with the GPUs still at 63%, so the process count alone buys nothing.

Only per-file production (extract, chunk, embed) runs here. The parent keeps
the plan, the single LanceDB writer, skip markers, move detection,
reconciliation, progress and cancellation, so the one-index invariant holds by
construction rather than by a merge step.

Workers never build an engine; they attach to the parent's (see
``SwapManager.bind``), because a second fleet would double-book the GPUs. That
makes starting it the parent's job: the engine otherwise comes up lazily on the
first embed, and in this path the parent dispatches every file and never embeds,
so nothing would start it and every worker would fail to attach. The parent
warms it before the first worker spawns and holds it for the whole run.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lilbee.core.config import active_config
from lilbee.runtime.cpu import cpu_quota

if TYPE_CHECKING:
    from lilbee.core.config.model import Config
    from lilbee.data.ingest.types import ChunkRecord, ConceptRecords, PageTextRecord

log = logging.getLogger(__name__)

# Files per worker task. A worker runs its batch on one asyncio loop, so the
# batch must be big enough to overlap embed waits and small enough that the
# parent keeps flushing steadily rather than in rare large bursts. It also caps
# a worker's real concurrency: a batch of this many files cannot have more than
# this many embeds in flight, whatever the admission target says.
BATCH_FILES = 32

# Under this many files the pool costs more than it saves: every worker pays a
# fresh lilbee import (lancedb, kreuzberg) before its first file.
_MIN_FILES_FOR_POOL = 2000

# Under this many usable cores there is nothing to parallelise onto, and the
# workers would contend with the parent's flush thread.
_MIN_CPUS_FOR_POOL = 4

# Auto never asks for more than this, however many cores the box has. A sweep on
# 4xA40 with the 0.6B embedder measured 174 docs/sec at 1 process, 220 at 2, 218
# at 4 and 210 at 8: the win is real but it plateaus immediately and then decays,
# because what limits it is the fleet's dispatch evenness rather than CPU. Each
# worker routes with its own least-loaded view and they cannot see each other, so
# more of them pile onto the same replicas -- per-card utilisation was 90/90/50/40
# at 8. Scaling this with the core count would pick the wrong end of that curve.
_MAX_AUTO_PROCESSES = 4

# One short string, embedded in the parent to force the engine up before the
# workers (which may only attach) are spawned.
_ENGINE_PROBE = "lilbee ingest engine warm-up"


class WorkerIngestError(Exception):
    """A failure raised inside a worker, carrying its original type name.

    Exceptions do not all survive pickling, so the worker formats the origin
    into the message and the parent reports that verbatim (see
    :func:`error_reason`).
    """


def error_reason(error: BaseException) -> str:
    """The ``Type: message`` reason recorded for a failed file."""
    if isinstance(error, WorkerIngestError):
        return str(error)
    return f"{type(error).__name__}: {error}"


def resolve_process_count(file_count: int) -> int:
    """Worker processes for a plan of *file_count* files; 1 keeps ingest in-process.

    Auto (the default) opts a run in only when the plan is big enough to amortise
    worker startup and the box has cores to spare, so an interactive sync of a
    vault never pays for a pool, and it stops at ``_MAX_AUTO_PROCESSES``, which is
    the top of the plateau measured on one fleet (4xA40, 0.6B embedder). Whether
    the plateau sits there on other card counts or model sizes is unmeasured, so
    the cap is a conservative default rather than a known optimum: an explicit
    ``ingest_processes`` always wins, including past it.
    """
    configured = active_config().ingest_processes
    if configured:
        return max(1, configured)
    if file_count < _MIN_FILES_FOR_POOL:
        return 1
    usable = cpu_quota()
    if usable < _MIN_CPUS_FOR_POOL:
        return 1
    return min(usable, _MAX_AUTO_PROCESSES)


@dataclass(frozen=True)
class WorkerFile:
    """One file for a worker to produce records for."""

    path: Path
    name: str
    content_type: str


@dataclass
class WorkerOutcome:
    """What a worker produced for one file; ``error`` set means it failed."""

    name: str
    records: list[ChunkRecord] | None = None
    page_texts: list[PageTextRecord] | None = None
    concept_records: ConceptRecords | None = None
    entity_rows: list[dict] | None = None
    error: WorkerIngestError | None = None


class _WorkerBindings:
    """The contexts a worker holds open for its whole process lifetime."""

    def __init__(self) -> None:
        self._stack: ExitStack | None = None
        self.inflight = 1

    def enter(self, config: Config, cpu_share: int, inflight: int) -> None:
        """Bind *config* and this worker's budgets; replaces any previous binding."""
        from lilbee.core.config.context import config_scope
        from lilbee.providers.fleet.guest import bind_only_engine
        from lilbee.providers.fleet.ingest_warmth import keep_fleet_warm

        self.close()
        os.environ["LILBEE_CPU_QUOTA"] = str(cpu_share)
        stack = ExitStack()
        stack.enter_context(config_scope(config))
        # Attach to the engine the parent started; never build one. A worker that
        # built its own would put a second fleet on the same GPUs.
        stack.enter_context(bind_only_engine())
        # Hold the fleet resident for this worker's lifetime, matching the parent:
        # an unevenly loaded replica must not idle-unload mid-run. The worker binds
        # to the parent's engine, so its teardown drops the binding and stops nothing.
        stack.enter_context(keep_fleet_warm())
        self.inflight = max(1, inflight)
        self._stack = stack

    def close(self) -> None:
        """Release the bindings. Normally only tests call this; workers exit instead."""
        if self._stack is not None:
            self._stack.close()
            self._stack = None

    @property
    def bound(self) -> bool:
        return self._stack is not None


_bindings = _WorkerBindings()


def init_worker(config: Config, cpu_share: int, inflight: int) -> None:
    """Bind the parent's config and this worker's budgets, once per process.

    Both are the box's budget divided by the worker count: *cpu_share* because
    cores are shared, *inflight* because the embed fleet is shared and has a knee
    past which more concurrent requests lower its throughput. What N processes buy
    is the CPU headroom to sustain that aggregate, not a larger aggregate.
    """
    _bindings.enter(config, cpu_share, inflight)


def run_batch(files: list[WorkerFile]) -> list[WorkerOutcome]:
    """Produce records for *files* on this worker's own event loop.

    One loop per batch rather than per file: the loop setup and its self-pipe
    wakeups are themselves GIL-held work the profile charges ~9% to.
    """
    return asyncio.run(_produce_batch(files))


async def _produce_batch(files: list[WorkerFile]) -> list[WorkerOutcome]:
    """Run the batch concurrently so embed waits overlap within the worker."""
    limit = asyncio.Semaphore(_bindings.inflight)

    async def one(entry: WorkerFile) -> WorkerOutcome:
        async with limit:
            return await _produce_one(entry)

    return await asyncio.gather(*(one(entry) for entry in files))


async def _produce_one(entry: WorkerFile) -> WorkerOutcome:
    """Extract, chunk, embed and build side-table rows for a single file."""
    from lilbee.data.ingest.pipeline import (
        build_concept_records,
        build_entity_records,
        produce_records,
    )

    page_texts: list[PageTextRecord] = []
    try:
        records = await produce_records(
            entry.path, entry.name, entry.content_type, page_texts_out=page_texts
        )
        return WorkerOutcome(
            name=entry.name,
            records=records,
            page_texts=page_texts,
            concept_records=await build_concept_records(records, entry.name),
            entity_rows=await build_entity_records(records, entry.name),
        )
    except Exception as exc:
        return WorkerOutcome(name=entry.name, error=WorkerIngestError(error_reason(exc)))


def warm_parent_engine() -> None:
    """Bring the embed engine up in this process and hold it, before any worker spawns.

    The engine starts lazily on the first embed. Workers are attach-only, and the
    parent never embeds once it is dispatching, so without this nothing starts the
    engine and every worker fails to attach -- which is exactly how the first
    multiprocess A/B embedded 0 of 50k files.

    Embedding one probe string is what forces the acquisition ladder to run here:
    it starts (or binds) the engine and takes this process's membership, so
    last-user-out cannot stop it under the workers. It also fails fast, with the
    embedder's own error, before N processes are spawned.
    """
    from lilbee.app.services import get_services

    get_services().embedder.embed(_ENGINE_PROBE)


def build_pool(processes: int, config: Config, inflight: int) -> ProcessPoolExecutor:
    """A pool of *processes* workers bound to *config*, the CPU share, and *inflight*.

    Both budgets are divided, for different reasons. Cores are shared. In-flight
    files are shared too, because the fleet is what they queue at: an admission
    sweep on 2xA40 measured 32 in flight at 155.6 docs/sec and 76% GPU, then
    147.6 at 128, 137.9 at 256 and 134.2 at 512 -- past the knee, more concurrent
    requests make a small fleet slower, not faster. So the aggregate across
    workers targets the fleet's admission ceiling and the worker count divides it,
    rather than each worker taking the whole target and multiplying the load by
    ``processes``.

    Forced to ``spawn``. By the time ingest starts, this process holds the ingest
    thread pool, httpx connection pools and LanceDB; ``fork`` (the default on
    Linux, which is where bulk ingest runs) copies their locks without the
    threads that would release them, so a child can deadlock on its first embed.
    A fresh interpreter per worker is the cost, which is part of why the pool is
    gated to plans big enough to amortise it.
    """
    cpu_share = max(1, cpu_quota() // processes)
    return ProcessPoolExecutor(
        max_workers=processes,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=init_worker,
        initargs=(config, cpu_share, max(1, inflight // processes)),
    )


def batched(files: list[Any], size: int = BATCH_FILES) -> list[list[Any]]:
    """Split *files* into contiguous batches of at most *size*.

    Hand-rolled because ``itertools.batched`` is 3.12+ and lilbee supports 3.11.
    """
    return [files[i : i + size] for i in range(0, len(files), size)]


class BatchDispatcher:
    """Maps a file's position in the plan to the worker batch that produces it.

    Batches are submitted on first demand, so the parent's bounded task window
    is also what bounds how many batches are outstanding: nothing reads the
    whole plan into the pool up front.
    """

    def __init__(self, pool: ProcessPoolExecutor, files: list[WorkerFile]) -> None:
        self._pool = pool
        self._batches = batched(files)
        self._pending: dict[int, asyncio.Future[list[WorkerOutcome]]] = {}

    async def outcome_for(self, index: int) -> WorkerOutcome:
        """The outcome for the file at *index* (0-based) in the plan."""
        batch_index, offset = divmod(index, BATCH_FILES)
        batch = self._batches[batch_index]
        future = self._pending.get(batch_index)
        if future is None:
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(self._pool, run_batch, batch)
            self._pending[batch_index] = future
        outcomes = await future
        if offset == len(batch) - 1:
            # Last consumer of this batch: drop it so results are not retained
            # for the whole run.
            self._pending.pop(batch_index, None)
        return outcomes[offset]
