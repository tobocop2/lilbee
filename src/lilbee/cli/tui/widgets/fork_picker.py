"""Modal that picks where to fork a conversation: the whole of it, or before a question."""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.widgets.clamped_option_list import ClampedOptionList
from lilbee.sessions import MessageRole, SessionMessage, derive_title


class ForkPicker(ModalScreen[int | None]):
    """Dismisses with the number of leading messages to copy, or ``None`` on escape."""

    CSS_PATH = "fork_picker.tcss"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Close", show=True),
    ]

    def __init__(self, messages: tuple[SessionMessage, ...]) -> None:
        super().__init__()
        self._messages = messages
        # Row i forks with _counts[i] messages: all of them, then each question's log index.
        self._counts = [len(messages)] + [
            index for index, message in enumerate(messages) if message.role == MessageRole.USER
        ]

    def compose(self) -> ComposeResult:
        with Vertical(id="fork-root"):
            yield Static(msg.FORK_PICKER_TITLE, id="fork-title")
            yield ClampedOptionList(*self._options(), id="fork-list")
            yield Static(msg.FORK_PICKER_HINT, id="fork-hint")

    def _options(self) -> list[Option]:
        """One row per entry in ``_counts``: the whole conversation, then each question."""
        before = [
            msg.FORK_PICKER_BEFORE.format(line=derive_title(self._messages[index].content))
            for index in self._counts[1:]
        ]
        return [Option(label) for label in [msg.FORK_PICKER_WHOLE, *before]]

    def on_mount(self) -> None:
        self.query_one("#fork-list", OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.dismiss(self._counts[event.option_index])

    def action_cancel(self) -> None:
        self.dismiss(None)
