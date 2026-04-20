"""Focusable CTA row that asks the user to search HuggingFace for the term."""

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

_CSS_FILE = Path(__file__).parent / "search_hf_cta_item.tcss"


class SearchHFCtaItem(containers.VerticalGroup, can_focus=True):
    """A focusable list row that fires the HF search worker when activated."""

    DEFAULT_CSS: ClassVar[str] = _CSS_FILE.read_text(encoding="utf-8")

    BINDINGS: ClassVar = [Binding("enter", "select", "Select", show=False)]

    @dataclass
    class Selected(Message):
        item: SearchHFCtaItem

        @property
        def control(self) -> SearchHFCtaItem:
            return self.item

        @property
        def term(self) -> str:
            return self.item.term

    def __init__(self, term: str) -> None:
        self._term = term
        super().__init__()

    @property
    def term(self) -> str:
        return self._term

    def action_select(self) -> None:
        self.post_message(self.Selected(self))

    def on_click(self, event: Click) -> None:
        event.stop()
        self.focus()
        self.post_message(self.Selected(self))

    def compose(self) -> ComposeResult:
        yield widgets.Static(msg.CATALOG_SEARCH_HF_CTA.format(query=self._term), id="cta-label")
