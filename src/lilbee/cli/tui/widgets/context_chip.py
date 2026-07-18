"""Live context-window usage, and whether a summarize call is in flight.

One chip for both: they are the same question from the user's side -- how much
can the model still see, and why did it just pause.
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

# Below this the chip stays quiet: a half-empty window is not news.
_QUIET_BELOW = 0.5
# Linked, not duplicated: amber must mean exactly "where compaction fires",
# or the gauge lies when the trigger moves.
_PRESSURE_AT = COMPACT_TRIGGER_FRACTION


class ContextChip(Widget):
    """How full the chat's history budget is, and whether it is condensing now."""

    DEFAULT_CSS: ClassVar[str] = _CSS_FILE.read_text(encoding="utf-8")

    # layout=True is load-bearing: width is `auto`, and a repaint redraws
    # inside the already-measured box. Mounted empty (usage 0 -> no text) the
    # chip measures zero columns and stays invisible without a re-layout.
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
