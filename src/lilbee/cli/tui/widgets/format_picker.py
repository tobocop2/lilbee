"""Modal that picks which weight format of a catalog row to install."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from lilbee.catalog.types import ModelFormat
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.format_rows import FormatRow, format_prompt
from lilbee.cli.tui.widgets.clamped_option_list import ClampedOptionList
from lilbee.cli.tui.widgets.dialog import CancelButton, Dialog, DialogFooter


class FormatPicker(Dialog[ModelFormat | None]):
    """Dismisses with the chosen format, or ``None`` on escape."""

    CSS_PATH = "format_picker.tcss"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Close", show=True),
    ]

    def __init__(self, name: str, rows: Sequence[FormatRow]) -> None:
        super().__init__()
        self._model_name = name
        self._rows = list(rows)

    def compose(self) -> ComposeResult:
        with Vertical(id="format-root"):
            yield Static(msg.FORMAT_PICKER_TITLE.format(name=self._model_name), id="format-title")
            yield ClampedOptionList(
                *(Option(format_prompt(row)) for row in self._rows), id="format-list"
            )
            with DialogFooter(id="format-footer"):
                yield Static(msg.FORMAT_PICKER_HINT, id="format-hint")
                yield CancelButton(msg.DIALOG_CLOSE_LABEL)

    def on_mount(self) -> None:
        self.query_one("#format-list", OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.dismiss(self._rows[event.option_index].fmt)

    def action_cancel(self) -> None:
        self.dismiss(None)
