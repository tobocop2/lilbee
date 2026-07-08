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
from concurrent.futures import ThreadPoolExecutor
from typing import ParamSpec, TypeVar

log = logging.getLogger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")

_MAX_WORKERS_ENV = "LILBEE_INGEST_MAX_WORKERS"


def _max_workers() -> int:
    """Concurrent slots for ingest offload work; honors ``LILBEE_INGEST_MAX_WORKERS``.

    Extraction rasterizes PDFs and drives OCR on this pool, so its width caps how
    many documents feed the GPU OCR/embed slots at once. The old fixed ``32``
    ceiling left most cores idle on a many-vCPU box and pinned full-corpus
    throughput below the vision fleet's capacity, so the default now scales with
    the vCPU count (``os.cpu_count() + 4``, the ``+4`` headroom for threads
    parked on OCR/embed I/O). The override accepts a positive integer; non-positive
    or unparseable values fall back to the default and a warning is logged.
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
    return (os.cpu_count() or 4) + 4


@functools.cache
def _ingest_executor() -> ThreadPoolExecutor:
    """The shared ingest pool, created on first use (cache makes it a singleton)."""
    return ThreadPoolExecutor(max_workers=_max_workers(), thread_name_prefix="lilbee-ingest")


async def to_ingest_thread(fn: Callable[_P, _R], /, *args: _P.args, **kwargs: _P.kwargs) -> _R:
    """``asyncio.to_thread`` on the ingest executor, contextvars preserved.

    Extraction relies on contextvar propagation into its workers (cancel and
    progress context), which ``run_in_executor`` alone would drop; copy the
    context exactly like ``asyncio.to_thread`` does.
    """
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    call = functools.partial(ctx.run, fn, *args, **kwargs)
    return await loop.run_in_executor(_ingest_executor(), call)
