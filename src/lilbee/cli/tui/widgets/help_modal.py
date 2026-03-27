"""Help modal — keybinding reference overlay."""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

from lilbee.cli.tui.command_registry import help_text as registry_help_text

_COMMANDS_BLOCK = registry_help_text()

_HELP_TEXT = f"""\
[bold]Keys[/bold]

  F1             Help (this screen)
  F2             Model catalog
  F3             Knowledge base status
  F4             Settings
  Ctrl+T         Cycle theme
  Ctrl+C         Quit

  [bold]Chat[/bold]
  Enter          Send message
  Escape         Cancel stream
  j / k          Scroll (vim-style)
  Ctrl+D / Space Page down
  Ctrl+U         Page up
  Tab            Accept suggestion

  [bold]Commands[/bold]  (type / for suggestions)
{_COMMANDS_BLOCK}

  Press Escape or q to close.
"""


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
