"""Attach-only engine access for processes that must never build a fleet.

Ingest worker processes share the engine the main process started. A worker that
built its own would load a second copy of the weights onto the same GPUs and
double-book their VRAM, which surfaces as an OOM or a throughput collapse rather
than an error -- and reads as the worker pool being at fault.

The flag is process-wide, not a ContextVar. Being attach-only is a property of
the whole process, and the engine can be built from a thread the provider spawns
itself: ``warm_up_pool`` runs the acquisition ladder on a background thread, and
a new thread starts with a fresh context, so a ContextVar would read its default
there and the build would go ahead.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

# Names the problem and the model, not the dispatch path that raised it.
NO_ENGINE_TO_ATTACH = (
    "No running local model engine to attach to. Ingest worker processes share "
    "the engine the main lilbee process starts; they never start their own."
)


class _AttachOnly:
    """Whether this process may only attach to a running engine, never build one.

    Nesting-counted so an inner scope cannot clear an outer one, and locked
    because the reader is usually a different thread from the writer.
    """

    def __init__(self) -> None:
        self._depth = 0
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        with self._lock:
            return self._depth > 0

    @contextmanager
    def engaged(self) -> Iterator[None]:
        with self._lock:
            self._depth += 1
        try:
            yield
        finally:
            with self._lock:
                self._depth -= 1


_attach_only = _AttachOnly()


def bind_only_active() -> bool:
    """Whether this process may only attach to a running engine, never build one."""
    return _attach_only.active


@contextmanager
def bind_only_engine() -> Iterator[None]:
    """Refuse to build an engine for the duration of the block; attach or fail.

    Applies to every thread in the process, including ones the provider spawns.
    """
    with _attach_only.engaged():
        yield
