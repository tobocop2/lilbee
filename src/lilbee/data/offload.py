"""Dedicated thread pool for ingest-side blocking work.

``asyncio.to_thread`` runs on the loop's default executor -- the same pool the
serving path (chat dispatch, handlers) offloads to. A fanned ingest can fill
that pool with extraction/OCR/embedding calls and starve request handling, so
ingest work runs on its own bounded executor instead.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import logging
import os
from collections.abc import Callable, Coroutine
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Any, ParamSpec, TypeVar

log = logging.getLogger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")

_MAX_WORKERS_ENV = "LILBEE_INGEST_MAX_WORKERS"
_POOL_THREAD_PREFIX = "lilbee-ingest"

# Files per embed replica to keep in flight during ingest. A replica interleaves
# its file's extraction with embedding, so a handful of files per card must be
# admitted at once or the GPU sits idle between requests. Scaling admission by
# the detected replica count auto-sizes a multi-GPU fleet with no manual cap,
# instead of the CPU-bound default that pins an 8-GPU box at ~4 files/card.
# Tuned against the 8x A100 MS MARCO fleet; override per run with
# ``ingest_max_inflight`` / ``LILBEE_INGEST_MAX_INFLIGHT``.
_EMBED_INFLIGHT_PER_REPLICA = 8


def fleet_inflight_target() -> int:
    """Admission that keeps every embed replica fed, from the detected fleet size.

    ``embed replicas x _EMBED_INFLIGHT_PER_REPLICA`` when a multi-replica fleet is
    resolvable, else 0 (single card or no fleet: the CPU-bound sizing already
    fits). A ``ProviderError`` propagates: no embed engine, or a device probe that
    timed out or found an unusable GPU. Any other error propagates too.
    """
    from lilbee.app.services import get_services
    from lilbee.providers.fleet.replicas import gpu_device_count, resolve_replica_count
    from lilbee.providers.roles import WorkerRole

    slots = resolve_replica_count(
        WorkerRole.EMBED,
        gpu_device_count(get_services().engines.for_role(WorkerRole.EMBED).adapter),
    )
    return slots * _EMBED_INFLIGHT_PER_REPLICA if slots > 1 else 0


def embed_inflight_target() -> int:
    """The fleet's admission target, or 0 when the fleet cannot be sized.

    A ``ProviderError`` returns 0: the embed call and the engine start raise it
    again with the user text. Any other error propagates.
    """
    from lilbee.providers.base import ProviderError

    try:
        return fleet_inflight_target()
    except ProviderError as exc:
        log.debug("embed fleet not sizable (%s); sizing admission from the CPU quota", exc)
        return 0


@functools.cache
def _fleet_sized_workers() -> int:
    """The pool size from a successful fleet sizing; a failed probe raises and is not kept.

    ``functools.cache`` keeps return values only, so the CPU default a failed probe
    gives is never remembered. The next call asks the device probe cache again, which
    holds the failure for its own window and then probes afresh.
    """
    from lilbee.core.config import active_config

    inflight = active_config().ingest_max_inflight or fleet_inflight_target()
    return max(_default_workers(), inflight)


def reset_max_workers() -> None:
    """Forget the fleet-sized pool count so the next ``max_workers`` call sizes it again."""
    _fleet_sized_workers.cache_clear()


def max_workers() -> int:
    """The ingest pool's worker count -- its hard concurrency ceiling.

    The adaptive-concurrency controller uses this as the upper bound on in-flight
    documents, since each one needs a pool thread to run its blocking extraction.
    Inside an ingest run it is the size of that run's pool, so the reported ceiling
    cannot drift away from the pool that actually exists. Outside a run, a fleet
    sizing that succeeded is kept; a failed device probe gives the CPU default only
    until the probe cache's failure window ends.

    Extraction rasterizes PDFs and drives OCR on this pool. The default caps at
    ``min(32, cpu_count + 4)``: a worker-count sweep on forced-OCR multi-page PDFs
    (4x H100) held throughput flat from 16 to 32 workers and *declining* past it,
    with the GPUs already ~85-90% busy the whole time. OCR ingest is GPU-bound, not
    extraction-bound, so extra threads only rasterize ahead into a buffer the GPUs
    cannot drain any faster while oversubscribing the box. The ``+4`` keeps headroom
    for threads parked on OCR/embed I/O; small hosts scale below the cap. The
    override lifts the ceiling for genuinely CPU-bound work (e.g. bulk text
    extraction) that can use more threads; non-positive or unparseable values warn
    and fall back to the default.

    It probes the embed fleet's devices, so call it off the event loop.
    """
    from lilbee.providers.base import ProviderError

    run_workers = _run_workers.get()
    if run_workers is not None:
        return run_workers
    override = _worker_override()
    if override is not None:
        return override
    # The pool (and thus the adaptive controller's permit_max, which is this
    # value) must be able to feed the admission ceiling, or in adaptive mode the
    # gate clamps back to 32 and a multi-GPU fleet stays starved. Size it to the
    # explicit ingest_max_inflight override, else auto-scale with the detected
    # embed fleet so no manual cap is needed on a multi-GPU box.
    try:
        return _fleet_sized_workers()
    except ProviderError as exc:
        log.debug("embed fleet not sizable (%s); sizing the pool from the CPU quota", exc)
        return _default_workers()


def _default_workers() -> int:
    """The CPU-bound worker count: ``min(32, cpu_count + 4)``."""
    return min(32, (os.cpu_count() or 4) + 4)


def _worker_override() -> int | None:
    """The ``LILBEE_INGEST_MAX_WORKERS`` count, or ``None`` when unset or not a positive int."""
    override = os.environ.get(_MAX_WORKERS_ENV)
    if override is None:
        return None
    try:
        value = int(override)
        if value > 0:
            return value
    except ValueError:
        pass  # bad override falls through to the warning + default below
    log.warning(
        "Ignoring %s=%r: must be a positive integer; using default.",
        _MAX_WORKERS_ENV,
        override,
    )
    return None


@functools.cache
def _ingest_executor() -> ThreadPoolExecutor:
    """The shared ingest pool for work outside an ingest run, created on first use.

    It is built on the event loop, so its size never probes the devices: no ingest run
    admits work onto it, so it needs no room for the embed fleet's admission ceiling.
    """
    workers = _worker_override() or _default_workers()
    return ThreadPoolExecutor(max_workers=workers, thread_name_prefix=_POOL_THREAD_PREFIX)


_run_pool: contextvars.ContextVar[ThreadPoolExecutor | None] = contextvars.ContextVar(
    "lilbee_ingest_run_pool", default=None
)
_run_workers: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "lilbee_ingest_run_workers", default=None
)


def owns_ingest_pool(
    run: Callable[_P, Coroutine[Any, Any, _R]],
) -> Callable[_P, Coroutine[Any, Any, _R]]:
    """Give each call of the async *run* its own ingest pool, shut down when the call ends."""

    @functools.wraps(run)
    async def _with_pool(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        workers = await asyncio.to_thread(max_workers)
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=_POOL_THREAD_PREFIX)
        token = _run_pool.set(pool)
        workers_token = _run_workers.set(workers)
        try:
            return await run(*args, **kwargs)
        finally:
            _run_workers.reset(workers_token)
            _run_pool.reset(token)
            pool.shutdown(wait=False, cancel_futures=True)

    return _with_pool


async def to_executor(
    executor: Executor, fn: Callable[_P, _R], /, *args: _P.args, **kwargs: _P.kwargs
) -> _R:
    """``asyncio.to_thread`` on *executor*, contextvars preserved.

    Extraction relies on contextvar propagation into its workers (cancel, config
    and progress context), which ``run_in_executor`` alone would drop; copy the
    context exactly like ``asyncio.to_thread`` does.
    """
    return await _submit(executor, fn, *args, **kwargs)


def _submit(
    executor: Executor, fn: Callable[_P, _R], /, *args: _P.args, **kwargs: _P.kwargs
) -> asyncio.Future[_R]:
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    call = functools.partial(ctx.run, fn, *args, **kwargs)
    return loop.run_in_executor(executor, call)


async def to_ingest_thread(fn: Callable[_P, _R], /, *args: _P.args, **kwargs: _P.kwargs) -> _R:
    """Run *fn* on the current ingest run's pool, or on the shared pool outside a run."""
    return await ingest_thread_future(fn, *args, **kwargs)


def ingest_thread_future(
    fn: Callable[_P, _R], /, *args: _P.args, **kwargs: _P.kwargs
) -> asyncio.Future[_R]:
    """Start *fn* on the ingest pool; the future is not a task, so no sweep of tasks cancels it."""
    return _submit(_run_pool.get() or _ingest_executor(), fn, *args, **kwargs)
