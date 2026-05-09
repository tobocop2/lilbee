"""Always-visible chip that points new users at the slash-command catalog and the keybinding panel.

Sits above the Footer in BottomBars. Clicking it opens the
:class:`SlashCommandCatalog` modal; the chip is otherwise passive (it reflects
state, not stores it).
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from textual import events
from textual.content import Content
from textual.widgets import Static

_CSS_FILE = Path(__file__).parent / "help_hint.tcss"

HELP_HINT_COMMANDS = "/ commands"
HELP_HINT_KEYS = "? keys"
HELP_HINT_SEPARATOR = "  ·  "


class HelpHint(Static):
    """One-row chip rendering ``/ commands · ? keys`` above the chat Footer."""

    DEFAULT_CSS: ClassVar[str] = _CSS_FILE.read_text(encoding="utf-8")

    def __init__(self, *, id: str | None = None) -> None:
        body = Content.assemble(
            Content.styled(HELP_HINT_COMMANDS, "$success bold"),
            Content.styled(HELP_HINT_SEPARATOR, "$text-muted"),
            Content.styled(HELP_HINT_KEYS, "$primary bold"),
        )
        super().__init__(body, id=id)

    def on_click(self, event: events.Click) -> None:
        event.stop()
        from lilbee.cli.tui.widgets.slash_command_catalog import SlashCommandCatalog

        from_screen = self.screen
        catalog_screen = SlashCommandCatalog()

        def _on_pick(name: str | None) -> None:
            if name is None:
                return
            handler = getattr(from_screen, "insert_slash_command", None)
            if callable(handler):
                handler(name)

        self.app.push_screen(catalog_screen, _on_pick)
