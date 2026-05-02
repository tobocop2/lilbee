"""Scope chip: search-only filter for raw / wiki / both."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from textual import events, on
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Select, Static

from lilbee.cli.tui.pill import pill
from lilbee.core.config import cfg
from lilbee.core.config.enums import ChatMode
from lilbee.data.store import SearchScope

_CSS_FILE = Path(__file__).parent / "scope_chip.tcss"

_SCOPE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("Both", SearchScope.BOTH.value),
    ("Wiki", SearchScope.WIKI.value),
    ("Raw", SearchScope.RAW.value),
)

_HIDDEN_CLASS = "-hidden"


class ScopeChip(Widget):
    """Search-only filter chip; visible when cfg.chat_mode=='search' and cfg.wiki."""

    DEFAULT_CSS: ClassVar[str] = _CSS_FILE.read_text(encoding="utf-8")

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Static(pill("Scope", "$accent", "$text"), classes="scope-chip-pill")
            yield Select[str](
                options=list(_SCOPE_OPTIONS),
                value=SearchScope.BOTH.value,
                id="scope-select",
                allow_blank=False,
            )

    @property
    def scope(self) -> SearchScope:
        """Current scope selection; consumed by ChatScreen for chunk_type."""
        select = self.query_one("#scope-select", Select)
        if select.value is Select.BLANK or select.value is None:
            return SearchScope.BOTH
        return SearchScope(str(select.value))

    def on_mount(self) -> None:
        self._refresh_visibility()
        from lilbee.cli.tui.app import LilbeeApp

        if isinstance(self.app, LilbeeApp):
            self.app.settings_changed_signal.subscribe(self, self._on_settings_changed)

    def on_unmount(self) -> None:
        for sel in self.query(Select):
            if sel.expanded:
                sel.expanded = False

    def _refresh_visibility(self) -> None:
        active = cfg.chat_mode == ChatMode.SEARCH.value and cfg.wiki
        self.set_class(not active, _HIDDEN_CLASS)

    def _on_settings_changed(self, payload: tuple[str, object]) -> None:
        key, _value = payload
        if key in {"chat_mode", "wiki"}:
            self._refresh_visibility()

    @on(Select.Changed, "#scope-select")
    def _swallow_blank(self, event: Select.Changed) -> None:
        """Drop spurious BLANK events Textual emits during option swaps."""
        if event.value is Select.BLANK or event.value is None:
            event.stop()

    def on_click(self, event: events.Click) -> None:
        """Forward outer-frame clicks to the select so the whole chip is hot."""
        if event.widget is self:
            self.query_one("#scope-select", Select).focus()
