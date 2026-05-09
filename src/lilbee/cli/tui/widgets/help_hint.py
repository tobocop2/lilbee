"""Always-visible chip that points new users at the slash-command catalog and the keybinding panel.

Sits above the Footer in BottomBars. Clicking it asks the chat screen to open
the catalog modal; the chip is otherwise passive (it reflects state, not
stores it).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from textual import events
from textual.content import Content
from textual.widgets import Static

if TYPE_CHECKING:
    from lilbee.cli.tui.screens.chat import ChatScreen

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
        # Defer to the hosting screen so the chip stays UI-only and the
        # screen owns the modal-push + insert flow that already exists for
        # the /help slash command.
        from lilbee.cli.tui.screens.chat import ChatScreen

        screen: ChatScreen = self.screen  # type: ignore[assignment]
        if not isinstance(screen, ChatScreen):
            return
        event.stop()
        screen.action_show_command_catalog()
