"""Animated 'thinking…' indicator shown while a chat response is pending.

Lifecycle: mounted into the chat log immediately after the user's message.
Removed when the first response token arrives (the assistant message
takes over) or when the user cancels the stream.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from textual.timer import Timer
from textual.widgets import Static

from lilbee.cli.tui import messages as msg

_CSS_FILE = Path(__file__).parent / "thinking_indicator.tcss"

_BRAILLE_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
_FRAME_INTERVAL = 0.1


class ThinkingIndicator(Static):
    """One-line braille spinner that ticks while the LLM is generating."""

    DEFAULT_CSS: ClassVar[str] = _CSS_FILE.read_text(encoding="utf-8")

    def __init__(self) -> None:
        super().__init__(self._frame_text(0), classes="thinking-indicator")
        self._frame: int = 0
        self._timer: Timer | None = None

    @staticmethod
    def _frame_text(frame: int) -> str:
        return f"  {_BRAILLE_FRAMES[frame % len(_BRAILLE_FRAMES)]} {msg.CHAT_THINKING_LABEL}"

    def on_mount(self) -> None:
        self._timer = self.set_interval(_FRAME_INTERVAL, self._tick)

    def _tick(self) -> None:
        self._frame += 1
        self.update(self._frame_text(self._frame))

    def on_unmount(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
