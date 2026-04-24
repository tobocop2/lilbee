"""TextArea subclass that posts a ``Blurred`` message on focus loss.

Textual's built-in :class:`TextArea` only emits ``Changed`` and
``SelectionChanged`` messages. The settings screen needs to save list-typed
values on focus loss (same save-on-blur UX as :class:`Input`), so this
subclass posts a dedicated ``Blurred`` message after the default blur
handling runs.
"""

from __future__ import annotations

from textual.events import Blur
from textual.message import Message
from textual.widgets import TextArea


class ListTextArea(TextArea):
    """TextArea that posts a ``Blurred`` message when focus leaves it."""

    class Blurred(Message):
        """Posted after focus leaves the :class:`ListTextArea`."""

        def __init__(self, control: ListTextArea) -> None:
            super().__init__()
            self._control = control

        @property
        def control(self) -> ListTextArea:
            """The widget that lost focus. Enables ``@on`` selector matching."""
            return self._control

    def _on_blur(self, event: Blur) -> None:
        super()._on_blur(event)
        self.post_message(self.Blurred(self))
