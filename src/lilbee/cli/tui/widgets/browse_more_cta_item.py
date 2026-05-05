"""Focusable CTA row that triggers the bulk HF browse fetch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from textual import containers, widgets
from textual.app import ComposeResult
from textual.binding import Binding
from textual.events import Click
from textual.message import Message

from lilbee.cli.tui import messages as msg

_CSS_FILE = Path(__file__).parent / "browse_more_cta_item.tcss"


class BrowseMoreCtaItem(containers.VerticalGroup, can_focus=True):
    """A focusable row that fires the bulk HF browse worker when activated."""

    DEFAULT_CSS: ClassVar[str] = _CSS_FILE.read_text(encoding="utf-8")

    BINDINGS: ClassVar = [Binding("enter", "select", "Select", show=False)]

    @dataclass
    class Selected(Message):
        item: BrowseMoreCtaItem

        @property
        def control(self) -> BrowseMoreCtaItem:
            return self.item

    def action_select(self) -> None:
        self.post_message(self.Selected(self))

    def on_click(self, event: Click) -> None:
        event.stop()
        self.focus()
        self.post_message(self.Selected(self))

    def compose(self) -> ComposeResult:
        yield widgets.Static(msg.CATALOG_BROWSE_MORE, id="browse-more-label")
