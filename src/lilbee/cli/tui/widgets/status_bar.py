"""ViewTabs: view tab strip with mode and active-model indicator."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.content import Content
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.pill import DOT_SEP, pill
from lilbee.config import cfg

_MODE_COLORS: dict[str, str] = {
    msg.MODE_NORMAL: "$primary",
    msg.MODE_INSERT: "$success",
}

_DEFAULT_MODE_COLOR = "$error"

# Settings keys that trigger a model-pill refresh.
_MODEL_PILL_KEYS = frozenset({"chat_model"})


class ViewTabs(Widget):
    """View tab strip with mode and active-model indicator."""

    # NOTE: no ``dock: bottom`` here. ViewTabs is always mounted inside a
    # ``BottomBars`` container that owns the dock; multiple dock-bottom
    # siblings overlap at the same row in Textual (see BottomBars docstring).
    DEFAULT_CSS = """
    ViewTabs {
        height: 1;
        width: 100%;
        background: $surface;
    }
    ViewTabs > Static {
        width: auto;
    }
    """
    active_view: reactive[str] = reactive(msg.DEFAULT_VIEW)
    mode_text: reactive[str] = reactive("")

    def compose(self) -> ComposeResult:
        yield Static(id="view-tabs-content")

    def on_mount(self) -> None:
        self.active_view = getattr(self.app, "active_view", msg.DEFAULT_VIEW)
        signal = getattr(self.app, "settings_changed_signal", None)
        if signal is not None:
            signal.subscribe(self, self._on_settings_changed)
        # Defer the initial paint: update() during on_mount can no-op while
        # the inner Static is still completing its mount cycle.
        self.call_after_refresh(self._refresh)

    def watch_active_view(self, value: str) -> None:
        self._refresh()

    def watch_mode_text(self, value: str) -> None:
        self._refresh()

    def _on_settings_changed(self, payload: tuple[str, object]) -> None:
        """Refresh the model pill when the active chat model changes."""
        key, _value = payload
        if key in _MODEL_PILL_KEYS:
            self._refresh()

    def _refresh(self) -> None:
        if not self.is_mounted:
            return
        parts: list[Content | str | tuple[str, str]] = []

        tab_parts: list[Content | str | tuple[str, str]] = []
        for name in msg.get_nav_views():
            if name == self.active_view:
                tab_parts.append(pill(f" {name} ", "$primary", "$text"))
            else:
                tab_parts.append((f" {name} ", "dim"))
        joined: list[Content | str | tuple[str, str]] = []
        for i, part in enumerate(tab_parts):
            if i > 0:
                joined.append((DOT_SEP, "$text-muted"))
            joined.append(part)
        parts.extend(joined)

        # ModelBar already shows the active chat model on the chat screen,
        # so the pill would just duplicate it there. Show it everywhere else.
        if cfg.chat_model and self.active_view != msg.DEFAULT_VIEW:
            parts.append("  ")
            parts.append(pill(f" {cfg.chat_model} ", "$accent", "$text"))

        if self.mode_text:
            color = _MODE_COLORS.get(self.mode_text, _DEFAULT_MODE_COLOR)
            parts.append("  ")
            parts.append(pill(f" {self.mode_text} ", color, "$text"))

        self.query_one("#view-tabs-content", Static).update(Content.assemble(*parts))
