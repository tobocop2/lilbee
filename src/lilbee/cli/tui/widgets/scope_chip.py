"""Scope chip: search-only filter with three side-by-side pills."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from lilbee.cli.tui.app import LilbeeApp

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Static

from lilbee.cli.tui import messages as msg
from lilbee.core.config import cfg
from lilbee.core.config.enums import ChatMode
from lilbee.data.store import SearchScope

_CSS_FILE = Path(__file__).parent / "scope_chip.tcss"

_HIDDEN_CLASS = "-hidden"
_ACTIVE_CLASS = "-active"
_SCOPE_PILL_CLASS = "scope-pill"

_SCOPE_BOTH_PILL_ID = "scope-pill-both"
_SCOPE_WIKI_PILL_ID = "scope-pill-wiki"
_SCOPE_RAW_PILL_ID = "scope-pill-raw"

# Pill id -> scope value, used for click routing.
_PILL_TO_SCOPE: dict[str, SearchScope] = {
    _SCOPE_BOTH_PILL_ID: SearchScope.BOTH,
    _SCOPE_WIKI_PILL_ID: SearchScope.WIKI,
    _SCOPE_RAW_PILL_ID: SearchScope.RAW,
}

# Cycle order: Both -> Wiki -> Raw -> Both.
_SCOPE_CYCLE: tuple[SearchScope, ...] = (
    SearchScope.BOTH,
    SearchScope.WIKI,
    SearchScope.RAW,
)


class ScopePill(Static, can_focus=True):
    """Single focusable scope pill; Enter / Space activates the parent scope."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "select", "Pick", show=False),
        Binding("space", "select", "Pick", show=False),
    ]

    def action_select(self) -> None:
        """Tell the enclosing ScopeChip to switch to this pill's scope."""
        chip = next(
            (n for n in self.ancestors_with_self if isinstance(n, ScopeChip)),
            None,
        )
        if chip is None or self.id is None:
            return
        target = _PILL_TO_SCOPE.get(self.id)
        if target is not None:
            chip._set_scope(target)


class ScopeChip(Widget):
    """Three-pill search filter; visible when cfg.chat_mode=='search' and cfg.wiki.

    Scope is **session-only** state held on ``self._scope``; it is not
    persisted to ``cfg``. ``ChatScreen`` reads ``chip.scope`` at submit
    time to derive ``chunk_type``.
    """

    app: LilbeeApp  # type: ignore[assignment]

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
        with Horizontal():
            yield ScopePill(msg.SCOPE_PILL_BOTH, id=_SCOPE_BOTH_PILL_ID, classes=_SCOPE_PILL_CLASS)
            yield ScopePill(msg.SCOPE_PILL_WIKI, id=_SCOPE_WIKI_PILL_ID, classes=_SCOPE_PILL_CLASS)
            yield ScopePill(msg.SCOPE_PILL_RAW, id=_SCOPE_RAW_PILL_ID, classes=_SCOPE_PILL_CLASS)

    @property
    def scope(self) -> SearchScope:
        """Current scope selection; consumed by ChatScreen for chunk_type."""
        return self._scope

    def on_mount(self) -> None:
        self._refresh_visibility()
        self._refresh()
        self.app.settings_changed_signal.subscribe(self, self._on_settings_changed)

    def _refresh_visibility(self) -> None:
        active = cfg.chat_mode == ChatMode.SEARCH.value and cfg.wiki
        self.set_class(not active, _HIDDEN_CLASS)

    def _on_settings_changed(self, payload: tuple[str, object]) -> None:
        key, _value = payload
        if key in {"chat_mode", "wiki"}:
            self._refresh_visibility()

    def _refresh(self) -> None:
        """Toggle the ``-active`` class on each pill based on ``self._scope``."""
        for pill_id, scope in _PILL_TO_SCOPE.items():
            pill = self.query_one(f"#{pill_id}", ScopePill)
            pill.set_class(scope is self._scope, _ACTIVE_CLASS)

    def _set_scope(self, target: SearchScope) -> None:
        """Apply *target* and repaint if it differs from the current scope."""
        if self._scope is target:
            return
        self._scope = target
        self._refresh()

    def cycle_scope(self) -> SearchScope:
        """Advance to the next scope in the cycle; return the new scope."""
        idx = _SCOPE_CYCLE.index(self._scope)
        self._set_scope(_SCOPE_CYCLE[(idx + 1) % len(_SCOPE_CYCLE)])
        return self._scope

    def on_click(self, event: events.Click) -> None:
        """Route a child-pill click into the scope it represents."""
        widget = event.widget
        if widget is None:
            return
        target = _PILL_TO_SCOPE.get(widget.id or "")
        if target is None:
            return
        event.stop()
        self._set_scope(target)
