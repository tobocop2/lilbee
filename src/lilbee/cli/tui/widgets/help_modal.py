"""Help modal — keybinding reference overlay."""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.command_registry import help_text as registry_help_text

_COMMANDS_BLOCK = registry_help_text()

_HELP_TEXT = msg.HELP_TEXT_TEMPLATE.format(commands_block=_COMMANDS_BLOCK)


class HelpModal(ModalScreen[None]):
    """Keybinding reference overlay."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "close", show=False),
        Binding("q", "close", show=False),
    ]

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(_HELP_TEXT)

    def action_close(self) -> None:
        self.dismiss(None)
