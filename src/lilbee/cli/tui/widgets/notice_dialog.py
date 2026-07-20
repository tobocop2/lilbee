"""A minimal single-dismiss modal for an informational notice.

Unlike :class:`ConfirmDialog` there is nothing to decide: the modal states one
thing and closes. Used when a feature is turned off and its view is opened, so
the user learns why nothing happened rather than facing a dead screen.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Center, Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, Static

_CSS_FILE = Path(__file__).parent / "notice_dialog.tcss"


class _DismissPill(Static, can_focus=True):
    """Pill-styled clickable label that closes the notice."""

    def __init__(self, label: str) -> None:
        super().__init__(label, id="notice-dismiss", classes="notice-pill")

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.screen.dismiss(None)


class NoticeDialog(ModalScreen[None]):
    """Modal that shows a title and a message with a single dismiss action."""

    DEFAULT_CSS: ClassVar[str] = _CSS_FILE.read_text(encoding="utf-8")

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "dismiss_notice", "OK", show=True),
        Binding("escape", "dismiss_notice", "Close", show=False),
    ]

    def __init__(self, title: str, message: str, *, dismiss_label: str = "OK (enter)") -> None:
        super().__init__()
        self._title = title
        self._message = message
        self._dismiss_label = dismiss_label

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(self._title, id="notice-title")
            yield Label(self._message, id="notice-message")
            with Center():
                yield _DismissPill(self._dismiss_label)

    def action_dismiss_notice(self) -> None:
        self.dismiss(None)
