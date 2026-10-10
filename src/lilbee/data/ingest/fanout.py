"""One ingest worker process per GPU, each writing its slice of the corpus to the one index."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import multiprocessing
import os
import queue
import shutil
import stat
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
)

from lilbee.core.config import active_config
from lilbee.data.ingest.errors import error_reason
from lilbee.data.types import ShardId, SyncResult
from lilbee.runtime.console import PlainConsole
from lilbee.runtime.cpu import available_cpu_count, cpu_quota
from lilbee.runtime.engine_lock import ENGINE_DIR_ENV
from lilbee.runtime.lock import (
    LockingUnsupportedError,
    ResetRefusedError,
    SyncRunningError,
    syncs_held_off,
)
from lilbee.runtime.progress import (
    BatchProgressEvent,
    BatchStatus,
    DetailedProgressCallback,
    EventType,
    ProgressEvent,
)
from lilbee.runtime.progress.columns import literal_text_column

if TYPE_CHECKING:
    from collections.abc import Sequence
    from multiprocessing.process import BaseProcess
    from multiprocessing.queues import Queue
    from multiprocessing.synchronize import Event

    from lilbee.core.config.model import Config
    from lilbee.runtime.cancellation import CancelSignal

log = logging.getLogger(__name__)

# Per-worker state (engine slots, log) under the parent data root. The index
# and the skip records are the corpus's, at the parent data root itself.
SHARDS_DIRNAME = "shards"
_DATA_ROOT_ENV = "LILBEE_DATA"
_CPU_QUOTA_ENV = "LILBEE_CPU_QUOTA"

# Below this many files on disk a fan-out costs more than it saves: every worker
# pays a fresh interpreter and its own engine.
_MIN_FILES_FOR_FANOUT = 2000

# Under two workers there is nothing to fan out to.
_MIN_FANOUT_WORKERS = 2

# How often a worker reports its counters to the parent.
_REPORT_INTERVAL_S = 0.25

# How long the parent sleeps between drains of the worker message queue and reads of the cancel.
_DRAIN_INTERVAL_S = 0.1

# Grace for the queue's feeder thread to flush a dead worker's last messages.
_FINAL_DRAIN_S = 1.0

# How long the workers get, together, to exit on a terminate before they are killed.
_WORKER_EXIT_GRACE_S = 30.0

# How often the parent looks for the workers' exit during that grace.
_EXIT_POLL_S = 0.01

# Where a worker's console output lands, under its own data root.
WORKER_LOG_NAME = "sync.log"

# Where each worker of an earlier lilbee kept a store of its own, under the shards directory.
_PRIVATE_STORE_GLOB = "w*/data"
_BYTES_PER_MB = 1024 * 1024
# How long a sync waits for the sync lock before it reads a holder as a running sync.
# A reset, or another sync that deletes the same stores, lets go well inside it.
_STORE_LOCK_WAIT_S = 5.0
_EARLIER_SYNC_RUNNING = (
    "A sync, an import, an add or a wiki build, possibly of an earlier lilbee, is running on "
    "this library. Run the sync again when it has finished."
)


@dataclass(frozen=True)
class ShardSpec:
    """One worker's slice, its card, and the state it owns."""

    shard: ShardId
    device: int
    config: Config
    engine_dir: Path
    cpu_share: int
    visible_devices: dict[str, str]


@dataclass(frozen=True)
class ShardOptions:
    """What every worker of one fan-out is told about the run it belongs to."""

    parent_pid: int


@dataclass(frozen=True)
class ShardProgress:
    """A worker's counters as it works."""

    kind: Literal["progress"]
    index: int
    done: int
    planned: int
    file: str
    status: BatchStatus


@dataclass(frozen=True)
class ShardDone:
    """A worker's verdict; *error* set means it did not finish its slice."""

    kind: Literal["done"]
    index: int
    result: SyncResult | None
    error: str | None


ShardMessage = ShardProgress | ShardDone


def resolve_process_count(devices: int) -> int:
    """Ingest worker processes for this run; 1 keeps ingest in this process.

    Auto (``ingest_processes = 0``) is one worker per visible card. An explicit
    count is honored past the card count, since two workers on one card is a
    legitimate configuration; they share that card's engine slot rather than
    putting a second fleet on it.
    """
    configured = active_config().ingest_processes
    if configured:
        return max(1, configured)
    return devices


def plan_fanout() -> list[ShardSpec]:
    """The workers for this sync, empty when it runs in this process."""
    from lilbee.data.ingest.discovery import corpus_has_at_least
    from lilbee.providers.fleet.gpu_env import apply_fleet_gpu_env
    from lilbee.providers.fleet.replicas import gpu_device_count

    # Applied before the cards are counted, so a gpu_devices pin is the space the
    # workers are dealt in: without it they would be dealt cards the pin excludes.
    apply_fleet_gpu_env()
    devices = gpu_device_count()
    processes = resolve_process_count(devices)
    if processes < _MIN_FANOUT_WORKERS or not corpus_has_at_least(_MIN_FILES_FOR_FANOUT):
        return []
    return shard_specs(active_config(), processes, devices)


def shard_specs(config: Config, processes: int, devices: int) -> list[ShardSpec]:
    """One spec per worker, dividing the corpus, the cards and the CPU pools."""
    from lilbee.providers.fleet.gpu_env import shard_visible_devices

    cpu_share = max(1, cpu_quota() // processes)
    plan_share = max(1, available_cpu_count() // processes)
    root = config.data_root / SHARDS_DIRNAME
    return [
        ShardSpec(
            shard=ShardId(index=index, count=processes, records_root=config.data_root),
            device=index % devices,
            config=_shard_config(config, root / f"w{index}", plan_share, processes),
            # Keyed by card, not by worker: workers sharing a card share one
            # fleet, workers on different cards never see each other's.
            engine_dir=root / f"gpu{index % devices}" / "engine",
            cpu_share=cpu_share,
            visible_devices=shard_visible_devices(index % devices),
        )
        for index in range(processes)
    ]


def _shard_config(config: Config, root: Path, plan_share: int, processes: int) -> Config:
    """*config* with a private data root and this worker's share of the CPU pools.

    ``documents_dir``, ``linked_roots`` and ``lancedb_dir`` are inherited: every
    worker reads the one corpus and writes the one index.
    """
    threads = config.extraction_threads
    return config.model_copy(
        update={
            "data_root": root,
            "ingest_workers": plan_share,
            "extraction_threads": max(1, threads // processes) if threads else 0,
        }
    )


def remove_private_stores(data_root: Path) -> None:
    """Delete the store each worker of an earlier lilbee kept under *data_root*.

    Raises ``SyncRunningError`` when the stores exist and the data root's sync
    mark is held: an earlier lilbee's fan-out sync writes and merges them under
    that mark, and a sync, an import, an add or a wiki build of this lilbee
    holds the same mark. The caller holds no sync mark of its own. A data root
    that cannot be locked keeps its stores for a later sync.
    """
    if not _private_stores(data_root):
        return
    try:
        with syncs_held_off(data_root, _EARLIER_SYNC_RUNNING, _STORE_LOCK_WAIT_S):
            _delete_stores(_private_stores(data_root), data_root)
    except SyncRunningError:
        # Another sync of this lilbee deleted them and now holds its own mark.
        if _private_stores(data_root):
            raise
    except (ResetRefusedError, LockingUnsupportedError) as exc:
        log.debug("Left the worker stores of an earlier lilbee under %s: %s", data_root, exc)


def _private_stores(data_root: Path) -> list[Path]:
    """Each store a worker of an earlier lilbee kept under *data_root*, in name order."""
    return sorted((data_root / SHARDS_DIRNAME).glob(_PRIVATE_STORE_GLOB))


def _delete_stores(stores: list[Path], data_root: Path) -> None:
    """Delete *stores*, which nothing reads, and log the stores that went and the space freed."""
    before = sum(_reclaimable_bytes(store) for store in stores)
    for store in stores:
        try:
            shutil.rmtree(store)
        except OSError as exc:
            log.warning("Could not delete the unused worker store %s: %s", store, exc)
    deleted = [store for store in stores if not store.exists()]
    if deleted:
        log.warning(
            "Deleted %d unused worker store(s) of an earlier lilbee under %s, freeing %.1f MB",
            len(deleted),
            data_root / SHARDS_DIRNAME,
            (before - sum(_reclaimable_bytes(store) for store in stores)) / _BYTES_PER_MB,
        )


def _reclaimable_bytes(store: Path) -> int:
    """The bytes deleting *store* frees: a file with a second hard link frees none.

    A file that cannot be read ends the count, which then states less than was freed.
    """
    total = 0
    try:
        for entry in store.rglob("*"):
            status = entry.lstat()
            if stat.S_ISREG(status.st_mode) and status.st_nlink == 1:
                total += status.st_size
    except OSError as exc:
        log.debug("Stopped measuring the worker store %s: %s", store, exc)
    return total


def _apply_shard_env(spec: ShardSpec) -> None:
    """Pin this process to the worker's card, engine slot, CPU share and log."""
    os.environ.update(spec.visible_devices)
    os.environ[ENGINE_DIR_ENV] = str(spec.engine_dir)
    os.environ[_DATA_ROOT_ENV] = str(spec.config.data_root)
    os.environ[_CPU_QUOTA_ENV] = str(spec.cpu_share)
    _redirect_output(spec.config.data_root / WORKER_LOG_NAME)


def _redirect_output(path: Path) -> None:
    """Send this process's console output to *path*.

    At the file descriptor, so the engine this worker spawns follows it: N
    workers logging onto the parent's terminal is the pile of log files the one
    aggregated bar exists to replace.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab", buffering=0) as handle:
        os.dup2(handle.fileno(), sys.stdout.fileno())
        os.dup2(handle.fileno(), sys.stderr.fileno())


class _ShardReporter:
    """Throttled relay of a worker's counters onto the parent's queue.

    The counters are the pipeline's own: how much of this worker's slice is done
    and how big that slice is. Counting files here instead would only re-derive
    the first, and the per-file events carry no slice size -- FILE_START's total
    is the plan so far, which grows all run.
    """

    def __init__(self, index: int, messages: Queue[ShardMessage]) -> None:
        self._index = index
        self._messages = messages
        self._done = 0
        self._planned = 0
        self._last_sent = 0.0

    def __call__(self, event_type: EventType, data: ProgressEvent) -> None:
        if event_type is not EventType.BATCH_PROGRESS or not isinstance(data, BatchProgressEvent):
            return
        self._done = data.current
        self._planned = data.total
        now = time.monotonic()
        if now - self._last_sent < _REPORT_INTERVAL_S:
            return
        self._last_sent = now
        self._send(data.file, data.status)

    def flush(self) -> None:
        """Send the final counters past the throttle, so the bar lands on its total."""
        self._send("", BatchStatus.INGESTED)

    def _send(self, file: str, status: BatchStatus) -> None:
        self._messages.put(
            ShardProgress(
                kind="progress",
                index=self._index,
                done=self._done,
                planned=self._planned,
                file=file,
                status=status,
            )
        )


class _Aggregate:
    """Every worker's latest counters, as one set of totals."""

    def __init__(self, on_progress: DetailedProgressCallback) -> None:
        self._latest: dict[int, ShardProgress] = {}
        self._on_progress = on_progress

    def update(self, message: ShardProgress) -> tuple[int, int]:
        """Record *message* and return the corpus-wide (done, planned)."""
        self._latest[message.index] = message
        done = sum(p.done for p in self._latest.values())
        planned = sum(p.planned for p in self._latest.values())
        self._on_progress(
            EventType.BATCH_PROGRESS,
            BatchProgressEvent(
                file=message.file, status=message.status, current=done, total=planned
            ),
        )
        return done, planned


def _drain(messages: Queue[ShardMessage]) -> list[ShardMessage]:
    """Every message queued right now, without blocking."""
    drained: list[ShardMessage] = []
    with contextlib.suppress(queue.Empty):
        while True:
            drained.append(messages.get_nowait())
    return drained


def _shard_progress_bar(quiet: bool) -> Progress:
    """The one bar a fan-out reports on, disabled when the caller wants no output."""
    return Progress(
        SpinnerColumn(),
        literal_text_column("{task.description}", style="progress.description"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=PlainConsole(),
        disable=quiet,
    )


async def _supervise(
    workers: Sequence[BaseProcess],
    messages: Queue[ShardMessage],
    *,
    quiet: bool,
    on_progress: DetailedProgressCallback,
    cancel: CancelSignal | None,
) -> dict[int, ShardDone]:
    """Drain worker messages until every worker has reported, keeping one bar current.

    Raises ``asyncio.CancelledError`` within one drain interval of a set *cancel*,
    whether or not a worker reports.
    """
    verdicts: dict[int, ShardDone] = {}
    aggregate = _Aggregate(on_progress)
    with _shard_progress_bar(quiet) as progress:
        task = progress.add_task(f"Ingesting on {len(workers)} workers", total=None)
        while len(verdicts) < len(workers):
            for message in _drain(messages):
                if message.kind == "done":
                    verdicts[message.index] = message
                else:
                    done, planned = aggregate.update(message)
                    progress.update(task, completed=done, total=planned or None)
            if cancel is not None and cancel.is_set():
                raise asyncio.CancelledError
            if not any(worker.is_alive() for worker in workers):
                verdicts.update(_final_verdicts(workers, messages, verdicts))
                break
            await asyncio.sleep(_DRAIN_INTERVAL_S)
    return verdicts


def _final_verdicts(
    workers: Sequence[BaseProcess],
    messages: Queue[ShardMessage],
    verdicts: dict[int, ShardDone],
) -> dict[int, ShardDone]:
    """Verdicts still in flight once every worker has exited, plus one per silent death.

    A worker the kernel killed (out of memory is the usual reason) reports
    nothing, so it is recorded as failed and the sync does not end as a success.
    """
    time.sleep(_FINAL_DRAIN_S)
    late = {m.index: m for m in _drain(messages) if m.kind == "done"}
    for index, worker in enumerate(workers):
        if index in verdicts or index in late:
            continue
        late[index] = ShardDone(
            kind="done",
            index=index,
            result=None,
            error=f"worker exited with code {worker.exitcode} before reporting",
        )
    return late


async def _stop_workers(workers: Sequence[BaseProcess], stop: Event) -> None:
    """Terminate every live worker, give them one grace period together, then kill the rest.

    A worker owns a GPU fleet, and its teardown can outlast a TERM; a plain join
    would hang the sync behind it instead of returning a result it already has.
    The wait yields to the event loop, and a cancel during it kills at once.
    """
    stop.set()
    for worker in workers:
        if worker.is_alive():
            worker.terminate()
    try:
        await _exited_or_grace_over(workers)
    finally:
        for worker in workers:
            if worker.is_alive():
                log.warning("Ingest worker %s did not exit; killing it", worker.name)
                worker.kill()
            worker.join()


async def _exited_or_grace_over(workers: Sequence[BaseProcess]) -> None:
    """Return once no worker is alive, or once the exit grace has passed."""
    deadline = time.monotonic() + _WORKER_EXIT_GRACE_S
    while time.monotonic() < deadline and any(worker.is_alive() for worker in workers):
        await asyncio.sleep(_EXIT_POLL_S)


async def run_workers(
    specs: list[ShardSpec],
    *,
    options: ShardOptions,
    quiet: bool,
    on_progress: DetailedProgressCallback,
    cancel: CancelSignal | None,
) -> list[ShardDone]:
    """Run every worker and return their verdicts, in shard order; a cancel terminates them."""
    context = multiprocessing.get_context("spawn")
    messages: Queue[ShardMessage] = context.Queue()
    stop = context.Event()
    workers = [
        context.Process(
            target=run_shard,
            args=(spec, options, messages, stop),
            name=f"lilbee-shard-{spec.shard.index}",
        )
        for spec in specs
    ]
    log.warning("Ingesting across %d worker processes, one per GPU", len(workers))
    for worker in workers:
        worker.start()
    try:
        verdicts = await _supervise(
            workers, messages, quiet=quiet, on_progress=on_progress, cancel=cancel
        )
    finally:
        await _stop_workers(workers, stop)
    return [verdicts[index] for index in sorted(verdicts)]


def aggregate_results(verdicts: list[ShardDone]) -> SyncResult:
    """The one result a fan-out reports, unioned from every worker's."""
    results = [verdict.result for verdict in verdicts if verdict.result is not None]
    return SyncResult(
        added=[name for r in results for name in r.added],
        updated=[name for r in results for name in r.updated],
        relocated=[name for r in results for name in r.relocated],
        failed=[name for r in results for name in r.failed],
        skipped=[name for r in results for name in r.skipped],
        skipped_ocr={name: ocr for r in results for name, ocr in r.skipped_ocr.items()},
        removed=[name for r in results for name in r.removed],
        held_out=[held for r in results for held in r.held_out],
        unchanged=sum(r.unchanged for r in results),
        truncated=sum(r.truncated for r in results),
        skip_records_error=next(
            (r.skip_records_error for r in results if r.skip_records_error is not None), None
        ),
    )


def run_shard(
    spec: ShardSpec, options: ShardOptions, messages: Queue[ShardMessage], stop: Event
) -> None:
    """Ingest this worker's slice in a fresh process, reporting onto *messages*."""
    from lilbee.app.services import build_services, services_scope
    from lilbee.core.config.context import config_scope
    from lilbee.data.ingest.pipeline import sync
    from lilbee.providers.fleet.child_guard import bind_lifetime_to_parent

    bind_lifetime_to_parent(options.parent_pid)
    _apply_shard_env(spec)
    index = spec.shard.index
    reporter = _ShardReporter(index, messages)
    try:
        with config_scope(spec.config), services_scope(build_services(spec.config)):
            result = asyncio.run(
                sync(
                    quiet=True,
                    on_progress=reporter,
                    cancel=stop,
                    shard=spec.shard,
                )
            )
        reporter.flush()
        messages.put(ShardDone(kind="done", index=index, result=result, error=None))
    except (Exception, asyncio.CancelledError) as exc:
        messages.put(ShardDone(kind="done", index=index, result=None, error=error_reason(exc)))
