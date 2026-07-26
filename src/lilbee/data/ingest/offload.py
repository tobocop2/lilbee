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
from collections.abc import Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import ParamSpec, TypeVar

log = logging.getLogger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")

_MAX_WORKERS_ENV = "LILBEE_INGEST_MAX_WORKERS"

# Files per embed replica to keep in flight during ingest. A replica interleaves
# its file's extraction with embedding, so a handful of files per card must be
# admitted at once or the GPU sits idle between requests. Scaling admission by
# the detected replica count auto-sizes a multi-GPU fleet with no manual cap,
# instead of the CPU-bound default that pins an 8-GPU box at ~4 files/card.
# Tuned against the 8x A100 MS MARCO fleet; override per run with
# ``ingest_max_inflight`` / ``LILBEE_INGEST_MAX_INFLIGHT``.
_EMBED_INFLIGHT_PER_REPLICA = 8


def embed_inflight_target() -> int:
    """Admission that keeps every embed replica fed, from the detected fleet size.

    ``embed replicas x _EMBED_INFLIGHT_PER_REPLICA`` when a multi-replica fleet is
    resolvable, else 0 (single card or no fleet: the CPU-bound sizing already
    fits). Never raises -- a fleet that cannot be probed yet returns 0.
    """
    try:
        from lilbee.providers.fleet.replicas import gpu_device_count, resolve_replica_count
        from lilbee.providers.roles import WorkerRole

        slots = resolve_replica_count(WorkerRole.EMBED, gpu_device_count())
    except Exception:
        return 0
    return slots * _EMBED_INFLIGHT_PER_REPLICA if slots > 1 else 0


def _max_workers() -> int:
    """Concurrent slots for ingest offload work; honors ``LILBEE_INGEST_MAX_WORKERS``.

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
    """
    override = os.environ.get(_MAX_WORKERS_ENV)
    if override is not None:
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
    default = min(32, (os.cpu_count() or 4) + 4)
    # The pool (and thus the adaptive controller's permit_max, which is this
    # value) must be able to feed the admission ceiling, or in adaptive mode the
    # gate clamps back to 32 and a multi-GPU fleet stays starved. Size it to the
    # explicit ingest_max_inflight override, else auto-scale with the detected
    # embed fleet so no manual cap is needed on a multi-GPU box.
    from lilbee.core.config import active_config

    inflight = active_config().ingest_max_inflight or embed_inflight_target()
    return max(default, inflight)


def max_workers() -> int:
    """The ingest pool's worker count -- its hard concurrency ceiling.

    The adaptive-concurrency controller uses this as the upper bound on in-flight
    documents, since each one needs a pool thread to run its blocking extraction.
    """
    return _max_workers()


@functools.cache
def _ingest_executor() -> ThreadPoolExecutor:
    """The shared ingest pool, created on first use (cache makes it a singleton)."""
    return ThreadPoolExecutor(max_workers=_max_workers(), thread_name_prefix="lilbee-ingest")


async def to_executor(
    executor: Executor, fn: Callable[_P, _R], /, *args: _P.args, **kwargs: _P.kwargs
) -> _R:
    """``asyncio.to_thread`` on *executor*, contextvars preserved.

    Extraction relies on contextvar propagation into its workers (cancel, config
    and progress context), which ``run_in_executor`` alone would drop; copy the
    context exactly like ``asyncio.to_thread`` does.
    """
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    call = functools.partial(ctx.run, fn, *args, **kwargs)
    return await loop.run_in_executor(executor, call)


async def to_ingest_thread(fn: Callable[_P, _R], /, *args: _P.args, **kwargs: _P.kwargs) -> _R:
    """Run *fn* on the shared ingest executor."""
    return await to_executor(_ingest_executor(), fn, *args, **kwargs)
