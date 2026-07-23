"""Keep the fleet resident for the duration of a bulk ingest.

A bulk ingest fans embed requests unevenly across replicas: a replica that goes
briefly idle hits ``engine_idle_ttl_minutes``, unloads its weights, and reloads
cold on the next request. That cold reload is a connection-refused to the
dispatcher, which retries onto the surviving replicas, piling more load on them
so they, too, fall behind and unload -- a positive-feedback collapse (8 GPUs
drop to 3). Holding the whole fleet resident (llama-swap ttl 0) for the sync
avoids it. The signal is a ContextVar so it scopes to exactly one ingest and
propagates into the ingest thread pool (``to_ingest_thread`` copies the context).
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager

_INGEST_KEEP_WARM: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "ingest_keep_warm", default=False
)


def ingest_keep_warm() -> bool:
    """Whether an ingest is currently holding the fleet resident."""
    return _INGEST_KEEP_WARM.get()


@contextmanager
def keep_fleet_warm() -> Iterator[None]:
    """Hold the fleet resident (ttl 0) for the duration of the block."""
    token = _INGEST_KEEP_WARM.set(True)
    try:
        yield
    finally:
        _INGEST_KEEP_WARM.reset(token)
