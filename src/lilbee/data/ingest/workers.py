"""Worker processes for bulk ingest: one GIL per worker.

Profiling the 8.8M-passage MS MARCO ingest found the ~155 docs/sec ceiling is
GIL-saturated and diffuse -- embed dispatch ~37% of GIL-held time, file I/O
~13%, asyncio ~9%, with no single hotspot to remove. One process is one GIL, so
throughput scales with processes, not with threads or GPUs.

Only per-file production (extract, chunk, embed) runs here. The parent keeps
the plan, the single LanceDB writer, skip markers, move detection,
reconciliation, progress and cancellation, so the one-index invariant holds by
construction rather than by a merge step.

Workers never build an engine: ``sync()`` forces the fleet up before ingest
starts, and the provider binds to that running engine (see
``SwapManager.bind``). A worker that could not bind would spawn a second fleet
and double-book the GPUs.
"""

from __future__ import annotations

import asyncio
import logging
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
# parent keeps flushing steadily rather than in rare large bursts.
BATCH_FILES = 32

# Under this many files the pool costs more than it saves: every worker pays a
# fresh lilbee import (lancedb, kreuzberg) before its first file.
_MIN_FILES_FOR_POOL = 2000

# Under this many usable cores there is nothing to parallelise onto, and the
# workers would contend with the parent's flush thread.
_MIN_CPUS_FOR_POOL = 4


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
    vault never pays for a pool. An explicit ``ingest_processes`` always wins.
    """
    configured = active_config().ingest_processes
    if configured:
        return max(1, configured)
    if file_count < _MIN_FILES_FOR_POOL:
        return 1
    usable = cpu_quota()
    return usable if usable >= _MIN_CPUS_FOR_POOL else 1


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

    def enter(self, config: Config, cpu_share: int) -> None:
        """Bind *config* and this worker's CPU share; replaces any previous binding."""
        from lilbee.core.config.context import config_scope
        from lilbee.providers.fleet.ingest_warmth import keep_fleet_warm

        self.close()
        os.environ["LILBEE_CPU_QUOTA"] = str(cpu_share)
        stack = ExitStack()
        stack.enter_context(config_scope(config))
        # Hold the fleet resident for this worker's lifetime, matching the parent:
        # an unevenly loaded replica must not idle-unload mid-run. The worker binds
        # to the parent's engine, so its teardown drops the binding and stops nothing.
        stack.enter_context(keep_fleet_warm())
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


def init_worker(config: Config, cpu_share: int) -> None:
    """Bind the parent's config and this worker's CPU share, once per process.

    The budgets are divided by the worker count before they are read: each
    worker otherwise sizes its own thread pool and admission to the whole box
    and the pool oversubscribes it N times over.
    """
    _bindings.enter(config, cpu_share)


def run_batch(files: list[WorkerFile]) -> list[WorkerOutcome]:
    """Produce records for *files* on this worker's own event loop.

    One loop per batch rather than per file: the loop setup and its self-pipe
    wakeups are themselves GIL-held work the profile charges ~9% to.
    """
    return asyncio.run(_produce_batch(files))


async def _produce_batch(files: list[WorkerFile]) -> list[WorkerOutcome]:
    """Run the batch concurrently so embed waits overlap within the worker."""
    limit = asyncio.Semaphore(max(1, cpu_quota()))

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


def build_pool(processes: int, config: Config) -> ProcessPoolExecutor:
    """A pool of *processes* workers, each bound to *config* and its CPU share."""
    share = max(1, cpu_quota() // processes)
    return ProcessPoolExecutor(
        max_workers=processes, initializer=init_worker, initargs=(config, share)
    )


def batched(files: list[Any], size: int = BATCH_FILES) -> list[list[Any]]:
    """Split *files* into contiguous batches of at most *size*."""
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
