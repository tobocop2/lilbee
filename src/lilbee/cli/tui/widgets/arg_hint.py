"""One-line status row beneath the chat input that describes the active slash command.

When the chat input contains a recognized slash command (e.g. ``/model``,
``/model gpt-4``), the row renders the command's argument syntax and a short
description. It hides itself for empty input, plain prose, an unknown command,
or a bare command name with no trailing space (so the user isn't crowded while
still typing the command itself).
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from textual.content import Content
from textual.widgets import Static

from lilbee.cli.tui.command_registry import COMMANDS

_CSS_FILE = Path(__file__).parent / "arg_hint.tcss"

_REGISTRY = {cmd.name: cmd for cmd in COMMANDS}
for _cmd in COMMANDS:
    for _alias in _cmd.aliases:
        _REGISTRY[_alias] = _cmd


class ArgHintLine(Static):
    """Reactive hint row that mirrors the active slash command's signature."""

    DEFAULT_CSS: ClassVar[str] = _CSS_FILE.read_text(encoding="utf-8")

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__("", id=id)
        self.display = False

    def update_for_input(self, text: str) -> None:
        """Render the appropriate hint for the current chat input contents."""
        rendered = _hint_for(text)
        if rendered is None:
            self._clear()
            return
        self.update(rendered)
        self.display = True

    def _clear(self) -> None:
        self.update("")
        self.display = False


def _hint_for(text: str) -> Content | None:
    """Build the hint content for *text*, or ``None`` when nothing should show."""
    if not text.startswith("/"):
        return None
    head = text.split(" ", 1)[0].lower()
    cmd = _REGISTRY.get(head)
    if cmd is None:
        return None
    # No trailing space yet means the user is still composing the command
    # name itself; do not crowd them with the hint until they advance.
    if " " not in text:
        return None

    name_part = Content.styled(f"  {cmd.name}", "$success")
    args_part = Content.styled(f" {cmd.args_hint}", "$text-muted") if cmd.args_hint else Content("")
    sep = Content.styled("  ·  ", "$text-muted")
    help_part = Content.styled(cmd.help_text, "$text-muted")
    return Content.assemble(name_part, args_part, sep, help_part)
