"""Skip a node's Mount when the app exits during its compose.

Once the app is exiting, Textual mounts none of the children a compose
yields but still sends Mount. A mount handler that queries those children
then raises ``NoMatches``, and the quit ends with a traceback and exit code 1.
"""

from __future__ import annotations

from typing import TypeVar

from textual import events
from textual.message_pump import MessagePump
from textual.widgets import Select

_T = TypeVar("_T")


class MountWhenAttached(MessagePump):
    """Deliver Mount only while the node is attached to a running app.

    ``is_attached`` is False once the app exits, so the mount handlers of a
    subclass, and the work they schedule, run only when its children exist.
    """

    async def on_event(self, event: events.Event) -> None:
        if isinstance(event, events.Mount) and not self.is_attached:
            return
        await super().on_event(event)


class GuardedSelect(MountWhenAttached, Select[_T]):
    """A Select whose mount handler looks up its overlay only while the app runs."""
