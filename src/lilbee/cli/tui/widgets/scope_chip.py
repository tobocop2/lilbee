"""Scope chip: search-only filter pill for raw / wiki / both."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from textual import events
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static

from lilbee.cli.tui import messages as msg
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


class ScopeChip(Widget):
    """Search-only filter chip; visible when cfg.chat_mode=='search' and cfg.wiki.

    Scope is **session-only** state held on ``self._scope``; it is not
    persisted to ``cfg``. ``ChatScreen`` reads ``chip.scope`` at submit
    time to derive ``chunk_type``.
    """

    DEFAULT_CSS: ClassVar[str] = _CSS_FILE.read_text(encoding="utf-8")

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._scope: SearchScope = SearchScope.BOTH

    def compose(self) -> ComposeResult:
        yield Static(self._pill_text(self._scope), id="scope-chip-pill")

    @property
    def scope(self) -> SearchScope:
        """Current scope selection; consumed by ChatScreen for chunk_type."""
        return self._scope

    def on_mount(self) -> None:
        self._refresh_visibility()
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

    def cycle_scope(self) -> SearchScope:
        """Advance to the next scope in the cycle; return the new scope."""
        idx = _SCOPE_CYCLE.index(self._scope)
        self._scope = _SCOPE_CYCLE[(idx + 1) % len(_SCOPE_CYCLE)]
        self._repaint()
        return self._scope

    def on_click(self, event: events.Click) -> None:
        """Cycle the scope on click anywhere in the pill."""
        event.stop()
        self.cycle_scope()

    @staticmethod
    def _pill_text(scope: SearchScope) -> str:
        """Render the pill label for *scope*."""
        return msg.CHAT_SCOPE_PILL.format(scope=_SCOPE_LABELS[scope], glyph=msg.SCOPE_CYCLE_GLYPH)

    def _repaint(self) -> None:
        """Repaint the pill label after a scope change. Caller-safe post-mount."""
        self.query_one("#scope-chip-pill", Static).update(self._pill_text(self._scope))
