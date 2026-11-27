"""The names of backends whose provider is gone, waiting to leave xberg's registry."""

from __future__ import annotations

import collections
import itertools
import weakref
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

_serial = itertools.count(1)


def next_serial() -> int:
    """The next number for a backend name, counted across every table.

    xberg's registries are one per process, so every table that registers a backend
    takes its name number from this one counter.
    """
    return next(_serial)


class ReleasedNames:
    """A queue of backend names to unregister at the owner's next registration.

    A provider's finalizer only appends its name, because a finalizer can run in a
    garbage collection while xberg holds its registry lock and calls into Python.
    """

    def __init__(self) -> None:
        self._queue: collections.deque[str] = collections.deque()

    def release(self, name: str) -> None:
        """Queue *name* now."""
        self._queue.append(name)

    def release_when_collected(self, provider: object, name: str) -> None:
        """Queue *name* when *provider* is collected."""
        weakref.finalize(provider, self._queue.append, name)

    def take(self) -> Iterator[str]:
        """Each queued name once; other threads can take names at the same time."""
        while True:
            try:
                yield self._queue.popleft()
            except IndexError:
                return
