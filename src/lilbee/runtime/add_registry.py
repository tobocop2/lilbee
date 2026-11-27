"""The adds that run under an id the client chose, so a later request can cancel one.

The HTTP add routes take an optional ``add_id``. The route reserves the id
before it answers, and the id maps to the function that stops the add until its
stream releases it. ``POST /api/add/cancel`` calls that function. A stop through
the registry is the user's cancel: the add removes what it did not finish. A
dropped stream does not go through the registry, and the add keeps its sources.
One instance is held by ``Services``, beside the ingest locks. Every call runs
on the server's event loop.
"""

from __future__ import annotations

from collections.abc import Callable


class AddIdRunningError(ValueError):
    """Another add with the same id runs."""

    def __init__(self, add_id: str) -> None:
        super().__init__(f"An add with the add_id '{add_id}' is already running")


def _hold_nothing() -> None:
    """The release of an add with no id."""


class AddRegistry:
    """Maps the id of each running add to the function that stops it as the user."""

    def __init__(self) -> None:
        self._stops: dict[str, Callable[[], None]] = {}

    def reserve(self, add_id: str | None, stop: Callable[[], None]) -> Callable[[], None]:
        """Hold *add_id* with *stop*; return the function that lets go of it.

        An add with no id holds nothing. The release frees the id only while
        this reservation holds it, so a second release does nothing. Raises
        :class:`AddIdRunningError` when another add holds *add_id*.
        """
        if add_id is None:
            return _hold_nothing
        if add_id in self._stops:
            raise AddIdRunningError(add_id)
        self._stops[add_id] = stop

        def release() -> None:
            if self._stops.get(add_id) is stop:
                del self._stops[add_id]

        return release

    def cancel(self, add_id: str) -> bool:
        """Stop the add with *add_id* as the user's cancel; False when none runs."""
        stop = self._stops.get(add_id)
        if stop is None:
            return False
        stop()
        return True
