"""One ingest worker process per GPU, each over its own slice of the corpus."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import multiprocessing
import os
import queue
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
    TextColumn,
    TimeElapsedColumn,
)

from lilbee.core.config import active_config
from lilbee.data.ingest.errors import error_reason
from lilbee.data.types import ShardId, SyncResult
from lilbee.runtime.cpu import available_cpu_count, cpu_quota
from lilbee.runtime.engine_lock import ENGINE_DIR_ENV
from lilbee.runtime.progress import (
    BatchProgressEvent,
    BatchStatus,
    DetailedProgressCallback,
    EventType,
    ProgressEvent,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from multiprocessing.process import BaseProcess
    from multiprocessing.queues import Queue
    from multiprocessing.synchronize import Event

    from lilbee.core.config.model import Config
    from lilbee.runtime.cancellation import CancelSignal

log = logging.getLogger(__name__)

# Per-worker state (store, skip markers, engine slots) under the parent data root.
SHARDS_DIRNAME = "shards"
_DATA_ROOT_ENV = "LILBEE_DATA"
_CPU_QUOTA_ENV = "LILBEE_CPU_QUOTA"

# Below this many files on disk a fan-out costs more than it saves: every worker
# pays a fresh interpreter, its own engine and a store of its own.
_MIN_FILES_FOR_FANOUT = 2000

# Under two workers there is nothing to fan out to.
_MIN_FANOUT_WORKERS = 2

# How often a worker reports its counters to the parent.
_REPORT_INTERVAL_S = 0.25

# How long the parent sleeps between drains of the worker message queue.
_DRAIN_INTERVAL_S = 0.1

# Grace for the queue's feeder thread to flush a dead worker's last messages.
_FINAL_DRAIN_S = 1.0

# How long a worker gets to exit on its own before it is killed.
_WORKER_EXIT_GRACE_S = 30.0

# Where a worker's console output lands, under its own data root.
WORKER_LOG_NAME = "sync.log"


@dataclass(frozen=True)
class ShardSpec:
    """One worker's slice, its card, and the private state it owns."""

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
    force_rebuild: bool = False
    retry_skipped: bool = False


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
    """A worker's verdict; *error* set means it produced no usable shard."""

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
            shard=ShardId(index=index, count=processes),
            device=index % devices,
            config=_shard_config(config, root / f"w{index}", plan_share),
            # Keyed by card, not by worker: workers sharing a card share one
            # fleet, workers on different cards never see each other's.
            engine_dir=root / f"gpu{index % devices}" / "engine",
            cpu_share=cpu_share,
            visible_devices=shard_visible_devices(index % devices),
        )
        for index in range(processes)
    ]


def _shard_config(config: Config, root: Path, plan_share: int) -> Config:
    """*config* with a private data root and this worker's share of the CPU pools.

    ``documents_dir`` and ``linked_roots`` are inherited: every worker reads the
    one shared corpus and only its own state is private.
    """
    return config.model_copy(
        update={
            "data_root": root,
            "lancedb_dir": root / "data" / "lancedb",
            "ingest_workers": plan_share,
        }
    )


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
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        disable=quiet,
    )


async def _supervise(
    workers: Sequence[BaseProcess],
    messages: Queue[ShardMessage],
    stop: Event,
    *,
    quiet: bool,
    on_progress: DetailedProgressCallback,
    cancel: CancelSignal | None,
) -> dict[int, ShardDone]:
    """Drain worker messages until every worker has reported, keeping one bar current."""
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
                stop.set()
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
    nothing, so its shard is recorded as failed rather than silently missing from
    the merge.
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


def _stop_workers(workers: Sequence[BaseProcess], stop: Event) -> None:
    """Ask every live worker to stop, then wait for it, then insist.

    A worker owns a GPU fleet, and its teardown can outlast a TERM; a plain join
    would hang the sync behind it instead of returning a result it already has.
    """
    stop.set()
    for worker in workers:
        if worker.is_alive():
            worker.terminate()
        worker.join(_WORKER_EXIT_GRACE_S)
        if worker.is_alive():
            log.warning("Ingest worker %s did not exit; killing it", worker.name)
            worker.kill()
            worker.join()


async def run_workers(
    specs: list[ShardSpec],
    *,
    options: ShardOptions,
    quiet: bool,
    on_progress: DetailedProgressCallback,
    cancel: CancelSignal | None,
) -> list[ShardDone]:
    """Run every worker to completion and return their verdicts, in shard order."""
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
            workers, messages, stop, quiet=quiet, on_progress=on_progress, cancel=cancel
        )
    finally:
        _stop_workers(workers, stop)
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
        unchanged=sum(r.unchanged for r in results),
        truncated=sum(r.truncated for r in results),
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
                    force_rebuild=options.force_rebuild,
                    quiet=True,
                    on_progress=reporter,
                    cancel=stop,
                    retry_skipped=options.retry_skipped,
                    shard=spec.shard,
                )
            )
        reporter.flush()
        messages.put(ShardDone(kind="done", index=index, result=result, error=None))
    except (Exception, asyncio.CancelledError) as exc:
        messages.put(ShardDone(kind="done", index=index, result=None, error=error_reason(exc)))
