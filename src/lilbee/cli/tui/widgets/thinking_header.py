"""Animated ``THINKING...`` header for the assistant bubble.

The widget owns a 100 ms timer that advances a single frame counter.
Each tick rebuilds the styled content: a moving block snake and a
shimmering letter on the ``THINKING...`` word, both in lockstep.

The header can either paint itself (default ``Static.update``) or
forward each frame's content to a callable -- so when reasoning
streams in and we mount a ``Collapsible`` below, the animator keeps
running by writing the latest frame to ``collapsible.title``.
``Collapsible.title`` is a plain string (not a widget), so this
indirection is the only way to keep one running animation across the
"no reasoning yet" and "reasoning streaming" states.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

from textual.content import Content
from textual.timer import Timer
from textual.widgets import Static

_CSS_FILE = Path(__file__).parent / "thinking_header.tcss"

# Block snake (5 cells, one filled) glides left-to-right; loops every 5 frames.
_BLOCK_FILLED = "▰"  # ▰
_BLOCK_EMPTY = "▱"  # ▱
_SNAKE_CELLS = 5

# Shimmer word travels one letter at a time, looping over the word length.
_THINKING_WORD = "THINKING…"  # THINKING…

# 100 ms per frame: fast enough to read as motion, slow enough not to thrash
# the renderer on large chat logs.
_FRAME_INTERVAL = 0.1

# Style strings, kept as constants so themes that swap palette keys see a
# single source of truth.
_DIM_STYLE = "$text-muted"
_BRIGHT_STYLE = "bold $primary"
_FILL_STYLE = "bold $success"
_SEPARATOR = "  "


def _frame_content(frame: int) -> Content:
    """Render the snake + shimmer for *frame* as styled content."""
    snake_pos = frame % _SNAKE_CELLS
    snake_parts: list[Content] = []
    for i in range(_SNAKE_CELLS):
        if i == snake_pos:
            snake_parts.append(Content.styled(_BLOCK_FILLED, _FILL_STYLE))
        else:
            snake_parts.append(Content.styled(_BLOCK_EMPTY, _DIM_STYLE))
    shimmer_pos = frame % len(_THINKING_WORD)
    word_parts: list[Content] = []
    for i, ch in enumerate(_THINKING_WORD):
        style = _BRIGHT_STYLE if i == shimmer_pos else _DIM_STYLE
        word_parts.append(Content.styled(ch, style))
    return Content.assemble(*snake_parts, _SEPARATOR, *word_parts)


class ThinkingHeader(Static):
    """Single Static that animates a moving snake and shimmering word."""

    DEFAULT_CSS: ClassVar[str] = _CSS_FILE.read_text(encoding="utf-8")

    def __init__(self) -> None:
        super().__init__(_frame_content(0), classes="thinking-header")
        self._frame: int = 0
        self._timer: Timer | None = None
        self._target: Callable[[Content], None] | None = None

    def on_mount(self) -> None:
        self._timer = self.set_interval(_FRAME_INTERVAL, self._tick)

    def on_unmount(self) -> None:
        self.stop()

    def stop(self) -> None:
        """Stop the animator timer; safe to call repeatedly."""
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def redirect_to(self, target: Callable[[Content], None] | None) -> None:
        """Send each frame's content to *target* instead of painting self."""
        self._target = target

    def _tick(self) -> None:
        self._frame += 1
        content = _frame_content(self._frame)
        if self._target is not None:
            self._target(content)
        else:
            self.update(content)
