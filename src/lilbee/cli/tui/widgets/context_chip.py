"""Live context-window usage, and what compaction is doing about it.

Two jobs in one chip, because they are the same question from the user's side:
how much of this conversation can the model still see, and why did it just pause.
Without the first, the window filling is invisible until turns start vanishing.
Without the second, compaction is an unexplained freeze -- seconds on a GPU, tens
of seconds on a CPU-only host, which reads as a hang.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from textual.content import Content
from textual.reactive import reactive
from textual.widget import Widget

from lilbee.cli.tui import messages as msg
from lilbee.core.config import cfg
from lilbee.retrieval.query.compaction import COMPACT_TRIGGER_FRACTION

_CSS_FILE = Path(__file__).with_suffix(".tcss")

# Below this the chip stays quiet: a half-empty window is not news, and a chip
# that is always shouting is one the eye stops reading.
_QUIET_BELOW = 0.5
# The chip turns amber exactly where compaction decides the window is full, so
# the warning and the action are the same fact. Duplicating the number here
# instead would let them drift: lower the trigger and the chip would go amber
# only after compaction had already fired, which is the gauge lying about the one
# thing it reports. With compaction off nothing fires, but the same point is
# still where the oldest turns start being at risk.
_PRESSURE_AT = COMPACT_TRIGGER_FRACTION


class ContextChip(Widget):
    """How full the chat's history budget is, and whether it is condensing now."""

    DEFAULT_CSS: ClassVar[str] = _CSS_FILE.read_text(encoding="utf-8")

    # layout=True, not just the default repaint: this chip's width is `auto`, so
    # its size comes from its text. A repaint redraws inside the box the widget
    # already has; only a layout recomputes that box. Without this the chip
    # renders empty at mount (usage 0 -> no text), is measured at zero columns,
    # and stays invisible forever however much the text changes afterwards.
    # No watch_ methods either: reactive already does both on change.
    usage: reactive[float] = reactive(0.0, layout=True)
    """Fraction of the history budget the conversation currently occupies."""

    compacting: reactive[bool] = reactive(False, layout=True)
    """True while a summarizing model call is in flight and blocking the turn."""

    def on_mount(self) -> None:
        self.tooltip = msg.CONTEXT_CHIP_TOOLTIP

    def render(self) -> Content:
        if self.compacting:
            return Content.styled(msg.CONTEXT_CHIP_COMPACTING, "$warning")
        if self.usage < _QUIET_BELOW:
            return Content("")
        percent = min(int(self.usage * 100), 100)
        if self.usage < _PRESSURE_AT:
            return Content.styled(msg.CONTEXT_CHIP_USAGE.format(percent=percent), "$text-muted")
        # Nearly full. With compaction on this resolves itself, so the number is
        # enough; with it off, turns are about to leave the model's view and the
        # user still has time to do something about it.
        template = (
            msg.CONTEXT_CHIP_USAGE if cfg.chat_compaction else msg.CONTEXT_CHIP_USAGE_DROPPING
        )
        return Content.styled(template.format(percent=percent), "$warning")
