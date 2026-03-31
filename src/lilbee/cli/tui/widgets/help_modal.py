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
[bold]Global[/bold]

  ? / F1 / ^h    Help (this screen)
  1              Chat view
  2              Model catalog
  3              Knowledge base status
  4              Settings
  F2 / ^n        Model catalog
  F3 / ^s        Knowledge base status
  F4 / ^e        Settings
  ^t             Cycle theme
  ^c             Quit

  [bold]Chat[/bold]
  Enter          Send message
  Escape         Cancel stream
  j / k          Scroll line (vim)
  g / G          Scroll to top / bottom
  ^d             Half-page down
  ^u             Half-page up
  PgUp / PgDn    Full page scroll
  Tab            Accept suggestion

  [bold]Catalog[/bold]
  j / k          Navigate list
  g / G          Jump to top / bottom
  1-4 / Ctrl+1-4  Switch tab (All/Chat/Embed/Vision)
  /              Focus search
  s              Cycle sort order
  Space          Page down
  ^d / ^u        Half-page down / up
  Enter          Install / select model
  q / Escape     Back

  [bold]Settings / Status[/bold]
  j / k          Navigate rows
  g / G          Jump to top / bottom
  q / Escape     Back

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
