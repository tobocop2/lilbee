"""Visible Stop button shown only while a chat stream is in flight."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from textual import events
from textual.binding import Binding, BindingType
from textual.message import Message
from textual.widgets import Static

from lilbee.cli.tui import messages as msg

_CSS_FILE = Path(__file__).parent / "chat_stop_button.tcss"


class ChatStopButton(Static, can_focus=True):
    """Pill button that cancels the active chat stream when activated."""

    DEFAULT_CSS: ClassVar[str] = _CSS_FILE.read_text(encoding="utf-8")

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "press", "Stop", show=False),
        Binding("space", "press", "Stop", show=False),
    ]

    @dataclass
    class Pressed(Message):
        """Posted when the stop button is activated by click or key."""

    def __init__(self, *, button_id: str = "chat-stop") -> None:
        super().__init__(msg.CHAT_STOP_BUTTON_LABEL, id=button_id)

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.action_press()

    def action_press(self) -> None:
        self.post_message(self.Pressed())
