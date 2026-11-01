"""Status rows that mirror the active slash command's signature, and the analyze tip for /add."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from textual.content import Content
from textual.widgets import Static

from lilbee.app.analyze import tip_shows
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.command_registry import COMMANDS
from lilbee.cli.tui.thread_safe import call_from_thread
from lilbee.core.config import cfg

_CSS_FILE = Path(__file__).parent / "arg_hint.tcss"

_REGISTRY = {cmd.name: cmd for cmd in COMMANDS}
for _cmd in COMMANDS:
    for _alias in _cmd.aliases:
        _REGISTRY[_alias] = _cmd

TIP_COMMAND = "/add"
_TIP_GROUP = "analyze-tip"


class ArgHintLine(Static):
    """Reactive hint that mirrors the active slash command's signature.

    Under ``/add`` it adds the analyze tip while the project should see it, before ingest starts.
    """

    DEFAULT_CSS: ClassVar[str] = _CSS_FILE.read_text(encoding="utf-8")

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__("", id=id)
        self.display = False
        self._text = ""
        self._tip = False

    def update_for_input(self, text: str) -> None:
        """Render the appropriate hint for the current chat input contents."""
        if _command(text) == TIP_COMMAND and _command(self._text) != TIP_COMMAND:
            self.refresh_tip()
        self._text = text
        self._render_hint()

    def refresh_tip(self) -> None:
        """Re-read off the loop whether the project should see the analyze tip."""
        self.run_worker(self._read_tip, thread=True, group=_TIP_GROUP, exclusive=True)

    def _read_tip(self) -> None:
        call_from_thread(self, self._set_tip, tip_shows(cfg.data_root))

    def _set_tip(self, shows: bool) -> None:
        self._tip = shows
        self._render_hint()

    def _render_hint(self) -> None:
        rendered = _hint_for(self._text)
        if rendered is None:
            self.update("")
            self.display = False
            return
        if self._tip and _command(self._text) == TIP_COMMAND:
            rendered = Content("\n").join([rendered, _tip_line()])
        self.update(rendered)
        self.display = True


def _command(text: str) -> str | None:
    """The slash command *text* starts with, once a space ends the command name."""
    command, space, _argument = text.partition(" ")
    if not command.startswith("/") or not space:
        return None
    return command.lower()


def _tip_line() -> Content:
    return Content.assemble(
        Content.styled(f"  {msg.ANALYZE_TIP_LABEL}", "$accent"),
        Content.styled(f"  {msg.ANALYZE_TIP}", "$text-muted"),
    )


def _hint_for(text: str) -> Content | None:
    """Build the hint content for *text*, or ``None`` when nothing should show."""
    command = _command(text)
    cmd = None if command is None else _REGISTRY.get(command)
    if cmd is None:
        return None

    name_part = Content.styled(f"  {cmd.name}", "$success")
    args_part = Content.styled(f" {cmd.args_hint}", "$text-muted") if cmd.args_hint else Content("")
    sep = Content.styled("  ·  ", "$text-muted")
    help_part = Content.styled(cmd.help_text, "$text-muted")
    return Content.assemble(name_part, args_part, sep, help_part)
