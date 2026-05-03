"""Scope chip: search-only filter pill for raw / wiki / both."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.widget import Widget
from textual.widgets import Static

from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.app import apply_setting
from lilbee.core.config import cfg
from lilbee.core.config.enums import ChatMode
from lilbee.data.store import SearchScope

_CSS_FILE = Path(__file__).parent / "scope_chip.tcss"

_HIDDEN_CLASS = "-hidden"

# Display labels keyed by SearchScope; one source of truth for the pill text.
_SCOPE_LABELS: dict[SearchScope, str] = {
    SearchScope.BOTH: "Both",
    SearchScope.WIKI: "Wiki",
    SearchScope.RAW: "Raw",
}

# Cycle order: clicking the pill walks Both -> Wiki -> Raw -> Both.
_SCOPE_CYCLE: tuple[SearchScope, ...] = (
    SearchScope.BOTH,
    SearchScope.WIKI,
    SearchScope.RAW,
)


def _coerce_scope(value: object) -> SearchScope:
    """Map a stored ``cfg.scope`` value back to a SearchScope enum."""
    try:
        return SearchScope(str(value))
    except ValueError:
        return SearchScope.BOTH


class ScopeChip(Widget, can_focus=True):
    """Search-only filter chip; visible when cfg.chat_mode=='search' and cfg.wiki."""

    DEFAULT_CSS: ClassVar[str] = _CSS_FILE.read_text(encoding="utf-8")

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("s", "cycle_scope", "Cycle scope", show=False),
        Binding("enter", "cycle_scope", "Cycle scope", show=False),
        Binding("space", "cycle_scope", "Cycle scope", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield Static(self._pill_text(_coerce_scope(cfg.scope)), id="scope-chip-pill")

    @property
    def scope(self) -> SearchScope:
        """Current scope selection; consumed by ChatScreen for chunk_type."""
        return _coerce_scope(cfg.scope)

    def on_mount(self) -> None:
        self._refresh_visibility()
        self._repaint()
        from lilbee.cli.tui.app import LilbeeApp

        if isinstance(self.app, LilbeeApp):
            self.app.settings_changed_signal.subscribe(self, self._on_settings_changed)

    def _refresh_visibility(self) -> None:
        active = cfg.chat_mode == ChatMode.SEARCH.value and cfg.wiki
        self.set_class(not active, _HIDDEN_CLASS)

    def _on_settings_changed(self, payload: tuple[str, object]) -> None:
        key, _value = payload
        if key in {"chat_mode", "wiki"}:
            self._refresh_visibility()
        if key == "scope":
            self._repaint()

    def cycle(self) -> SearchScope:
        """Advance to the next scope in the cycle and persist; return new scope."""
        current = _coerce_scope(cfg.scope)
        idx = _SCOPE_CYCLE.index(current) if current in _SCOPE_CYCLE else 0
        new = _SCOPE_CYCLE[(idx + 1) % len(_SCOPE_CYCLE)]
        apply_setting(self.app, "scope", new.value)
        self._repaint()
        return new

    def on_click(self, event: events.Click) -> None:
        """Cycle the scope on click anywhere in the pill."""
        event.stop()
        self.cycle()

    def action_cycle_scope(self) -> None:
        """Keybinding handler that drives the same cycle as a click."""
        self.cycle()

    @staticmethod
    def _pill_text(scope: SearchScope) -> str:
        """Render the pill label for *scope*."""
        return msg.CHAT_SCOPE_PILL.format(scope=_SCOPE_LABELS[scope])

    def _repaint(self) -> None:
        try:
            label = self.query_one("#scope-chip-pill", Static)
        except Exception:
            return
        label.update(self._pill_text(_coerce_scope(cfg.scope)))
