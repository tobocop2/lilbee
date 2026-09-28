"""A call run on a daemon thread that the caller can wait on, and that exit never joins.

concurrent.futures joins every executor worker at interpreter exit whatever its
daemon flag, so a worker stuck in a network read holds the process open.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Generic, TypeVar, cast

T = TypeVar("T")


class DaemonCall(Generic[T]):
    """Run *fn* on a started daemon thread; ``result()`` returns its value or re-raises."""

    def __init__(self, fn: Callable[[], T], *, name: str) -> None:
        self._fn = fn
        self._value: T | None = None
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self._value = self._fn()
        except BaseException as exc:  # re-raised to the waiting caller by result()
            self._error = exc

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the call finishes or *timeout* passes; return whether it finished."""
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def result(self) -> T:
        """Wait for the call, then return its value or re-raise its exception."""
        self.wait()
        if self._error is not None:
            raise self._error
        return cast(T, self._value)
