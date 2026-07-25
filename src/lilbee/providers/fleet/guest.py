"""Attach-only engine access for processes that must never build a fleet.

Ingest worker processes share the engine the main process started. A worker that
built its own would load a second copy of the weights onto the same GPUs and
double-book their VRAM, which surfaces as an OOM or a throughput collapse rather
than an error -- and reads as the worker pool being at fault.

The signal is a ContextVar so it scopes to the worker and reaches the ingest
offload pool, which copies the context per call (``to_ingest_thread``), matching
:mod:`ingest_warmth`.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager

_BIND_ONLY: contextvars.ContextVar[bool] = contextvars.ContextVar("engine_bind_only", default=False)

# Names the problem and the model, not the dispatch path that raised it.
NO_ENGINE_TO_ATTACH = (
    "No running local model engine to attach to. Ingest worker processes share "
    "the engine the main lilbee process starts; they never start their own."
)


def bind_only_active() -> bool:
    """Whether this process may only attach to a running engine, never build one."""
    return _BIND_ONLY.get()


@contextmanager
def bind_only_engine() -> Iterator[None]:
    """Refuse to build an engine for the duration of the block; attach or fail."""
    token = _BIND_ONLY.set(True)
    try:
        yield
    finally:
        _BIND_ONLY.reset(token)
