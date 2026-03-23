"""Help modal — keybinding reference overlay."""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

_HELP_TEXT = """\
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
  /model [name]  Switch model (no arg = catalog)
  /vision [name] Set vision model (/vision off)
  /add path      Add file to knowledge base
  /delete name   Remove document from index
  /set key val   Change a setting
  /reset confirm Factory reset
  /theme name    Switch theme
  /version       Show version
  /quit          Exit

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
