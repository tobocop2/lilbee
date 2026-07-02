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
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

_T = TypeVar("_T")

_lock = threading.Lock()
_executor: ThreadPoolExecutor | None = None


def _max_workers() -> int:
    """Mirror the default-executor sizing; the point is isolation, not tuning."""
    return min(32, (os.cpu_count() or 4) + 4)


def _ingest_executor() -> ThreadPoolExecutor:
    global _executor
    with _lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=_max_workers(), thread_name_prefix="lilbee-ingest"
            )
        return _executor


async def to_ingest_thread(fn: Any, /, *args: Any, **kwargs: Any) -> Any:
    """``asyncio.to_thread`` on the ingest executor, contextvars preserved.

    Extraction relies on contextvar propagation into its workers (cancel and
    progress context), which ``run_in_executor`` alone would drop; copy the
    context exactly like ``asyncio.to_thread`` does.
    """
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    call = functools.partial(ctx.run, fn, *args, **kwargs)
    return await loop.run_in_executor(_ingest_executor(), call)
