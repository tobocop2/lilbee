"""The adds that run under an id the client chose, so a later request can cancel one.

The HTTP add routes take an optional ``add_id``. While the add runs, the id maps
to the function that stops it, and ``POST /api/add/cancel`` calls that function.
A stop through the registry is the user's cancel: the add removes what it did
not finish. A dropped stream does not go through the registry, and the add
keeps its sources. One instance is held by ``Services``, beside the ingest
locks. Every call runs on the server's event loop.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager


class AddIdRunningError(ValueError):
    """Another add with the same id runs."""

    def __init__(self, add_id: str) -> None:
        super().__init__(f"An add with the add_id '{add_id}' is already running")


class AddRegistry:
    """Maps the id of each running add to the function that stops it as the user."""

    def __init__(self) -> None:
        self._stops: dict[str, Callable[[], None]] = {}

    def is_running(self, add_id: str) -> bool:
        """Whether an add with *add_id* runs now."""
        return add_id in self._stops

    @contextmanager
    def running(self, add_id: str | None, stop: Callable[[], None]) -> Iterator[None]:
        """Hold *add_id* with *stop* until the block ends; an add with no id holds nothing.

        Raises :class:`AddIdRunningError` when another add holds *add_id*.
        """
        if add_id is None:
            yield
            return
        if add_id in self._stops:
            raise AddIdRunningError(add_id)
        self._stops[add_id] = stop
        try:
            yield
        finally:
            del self._stops[add_id]

    def cancel(self, add_id: str) -> bool:
        """Stop the add with *add_id* as the user's cancel; False when none runs."""
        stop = self._stops.get(add_id)
        if stop is None:
            return False
        stop()
        return True
