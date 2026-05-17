"""ViewTabs: view tab strip with mode and active-model indicator."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from lilbee.cli.tui.app import LilbeeApp

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal
from textual.content import Content
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Label, Static

from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.pill import DOT_SEP, pill
from lilbee.core.config import cfg

_CSS_FILE = Path(__file__).parent / "status_bar.tcss"

_MODE_COLORS: dict[str, str] = {
    msg.MODE_NORMAL: "$primary",
    msg.MODE_INSERT: "$success",
}

_DEFAULT_MODE_COLOR = "$error"

# Settings keys that trigger a model-pill refresh.
_TRAILING_PILL_KEYS = frozenset({"chat_model", "lilbee_name", "show_lilbee_path"})


class ViewTab(Label, can_focus=True):
    """A focusable, clickable tab label inside ViewTabs.

    Owns its `view_name`. Click and Enter / Space when focused both
    fire the app's view switcher. Active and focus styling are
    handled in status_bar.tcss via the ``-active`` and ``:focus``
    pseudo-classes.
    """

    app: LilbeeApp  # type: ignore[assignment]

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "activate", "Switch view", show=False),
        Binding("space", "activate", "Switch view", show=False),
    ]

    def __init__(self, view_name: str) -> None:
        super().__init__(id=f"view-tab-{view_name.lower()}", classes="view-tab")
        self.view_name = view_name

    def set_active(self, active: bool) -> None:
        self.set_class(active, "-active")
        if active:
            # Bold $primary on a $surface background pill, mirroring the
            # Settings sub-tab aesthetic (#settings-tabs Tab.-active in
            # screens/settings.tcss). Background comes from the .-active
            # CSS class so the pill fills the padded label region.
            self.update(Content.styled(f"  {self.view_name}  ", "bold $primary"))
        else:
            self.update(Content.assemble((f"  {self.view_name}  ", "dim")))

    def on_click(self) -> None:
        self._switch()

    def action_activate(self) -> None:
        self._switch()

    def _switch(self) -> None:
        self.app.switch_view(self.view_name)


class ViewTabs(Widget):
    """View tab strip with mode and active-model indicator."""

    app: LilbeeApp  # type: ignore[assignment]

    # NOTE: no ``dock: bottom`` here. ViewTabs is always mounted inside a
    # ``BottomBars`` container that owns the dock; multiple dock-bottom
    # siblings overlap at the same row in Textual (see BottomBars docstring).
    DEFAULT_CSS: ClassVar[str] = _CSS_FILE.read_text(encoding="utf-8")
    active_view: reactive[str] = reactive(msg.DEFAULT_VIEW)
    mode_text: reactive[str] = reactive("")

    def compose(self) -> ComposeResult:
        # Compose every nav view including Wiki; visibility is toggled at
        # runtime via _apply_wiki_visibility so the user can flip the wiki
        # setting without restarting.
        all_views = [*msg._BASE_NAV_VIEWS, "Wiki"]
        with Horizontal(id="view-tabs-row"):
            for i, name in enumerate(all_views):
                if i > 0:
                    yield Static(
                        DOT_SEP,
                        classes="view-tab-sep",
                        id=f"view-tab-sep-{name.lower()}",
                    )
                yield ViewTab(name)
            yield Static(id="view-tabs-trailing")

    def on_mount(self) -> None:
        self.active_view = self.app.active_view
        self.app.settings_changed_signal.subscribe(self, self._on_settings_changed)
        # Wiki visibility AND the initial paint both deferred: query() during
        # on_mount can no-op while ViewTab children are still completing their
        # mount cycle, leaving the Wiki tab visible even when cfg.wiki=False.
        self.call_after_refresh(self._apply_wiki_visibility)
        self.call_after_refresh(self._refresh)

    def watch_active_view(self, value: str) -> None:
        self._refresh()

    def watch_mode_text(self, value: str) -> None:
        self._refresh()

    def _on_settings_changed(self, payload: tuple[str, object]) -> None:
        """Refresh the model pill, and toggle Wiki tab visibility on wiki."""
        key, _value = payload
        if key == "wiki":
            self._apply_wiki_visibility()
            return
        if key in _TRAILING_PILL_KEYS:
            self._refresh()

    def _apply_wiki_visibility(self) -> None:
        """Show or hide the Wiki tab and its preceding separator based on cfg.wiki."""
        if not self.is_mounted:
            return
        visible = bool(cfg.wiki)
        for selector in ("#view-tab-wiki", "#view-tab-sep-wiki"):
            for widget in self.query(selector):
                widget.display = visible

    def _refresh(self) -> None:
        if not self.is_mounted:
            return
        for tab in self.query(ViewTab):
            tab.set_active(tab.view_name == self.active_view)
        self._update_trailing()

    def _update_trailing(self) -> None:
        from lilbee.app.status import lilbee_label
        from lilbee.catalog import display_label_for_ref

        parts: list[Content | str | tuple[str, str]] = []
        # ModelBar already shows the active chat model on the chat screen,
        # so the pill would just duplicate it there. Show it everywhere else.
        if cfg.chat_model and self.active_view != msg.DEFAULT_VIEW:
            label = display_label_for_ref(cfg.chat_model) or cfg.chat_model
            parts.append("  ")
            parts.append(pill(label, "$accent", "$text"))
        if self.mode_text:
            color = _MODE_COLORS.get(self.mode_text, _DEFAULT_MODE_COLOR)
            parts.append("  ")
            parts.append(pill(self.mode_text, color, "$text"))
        parts.append("  ")
        parts.append(pill(lilbee_label(), "$secondary-muted", "$text"))
        self.query_one("#view-tabs-trailing", Static).update(Content.assemble(*parts))
