"""Footer chip: passive ``type / for commands · F1 for keys`` hint; click opens the list."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from textual import events
from textual.content import Content
from textual.widgets import Static

from lilbee.cli.tui import messages as msg

_CSS_FILE = Path(__file__).parent / "help_hint.tcss"


class HelpHint(Static):
    """One-row chip rendering ``type / for commands · F1 for keys`` above the chat Footer."""

    DEFAULT_CSS: ClassVar[str] = _CSS_FILE.read_text(encoding="utf-8")

    def __init__(self, *, id: str | None = None) -> None:
        body = Content.assemble(
            Content.styled(msg.HELP_HINT_COMMANDS, "$success bold"),
            Content.styled(msg.HELP_HINT_SEPARATOR, "$text-muted"),
            Content.styled(msg.HELP_HINT_KEYS, "$primary bold"),
        )
        super().__init__(body, id=id)

    def on_click(self, event: events.Click) -> None:
        # Defer to the hosting screen so the chip stays UI-only and the
        # screen owns the modal-push + insert flow that already exists for
        # the /help slash command.
        from lilbee.cli.tui.screens.chat import ChatScreen

        screen = self.screen
        if not isinstance(screen, ChatScreen):
            return
        event.stop()
        screen.action_show_command_catalog()
