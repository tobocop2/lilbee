"""Settings screen — grouped, type-aware configuration editor."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import VerticalGroup, VerticalScroll
from textual.content import Content
from textual.screen import Screen
from textual.widgets import Checkbox, Input, Select, Static

from lilbee import settings
from lilbee.cli.settings_map import SETTINGS_MAP, SettingDef
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.pill import pill
from lilbee.config import cfg

log = logging.getLogger(__name__)

_TYPE_COLORS: dict[str, tuple[str, str]] = {
    "str": ("$secondary", "$text"),
    "int": ("$primary", "$text"),
    "float": ("$primary", "$text"),
    "bool": ("$success", "$text"),
    "select": ("$warning", "$text"),
}


_DEFAULTS_REMAP: dict[str, str] = {"top_k_sampling": "top_k"}


def _effective_value(key: str) -> str:
    """Return the effective value for a setting, including model defaults."""
    user_value = getattr(cfg, key, None)
    if user_value is not None:
        return str(user_value)
    defaults = cfg.model_defaults
    if defaults is None:
        return "None"
    defaults_key = _DEFAULTS_REMAP.get(key, key)
    default_val = getattr(defaults, defaults_key, None)
    if default_val is not None:
        return f"{default_val} (model default)"
    return "None"


def _is_writable(key: str) -> bool:
    """Check if a setting key is writable (derived from SETTINGS_MAP)."""
    defn = SETTINGS_MAP.get(key)
    return defn is not None and defn.writable


def _type_pill(defn: SettingDef) -> Content:
    """Create a colored pill badge for a setting's type."""
    type_name = defn.type.__name__
    if defn.choices:
        type_name = "select"
    bg, fg = _TYPE_COLORS.get(type_name, ("$surface", "$text"))
    return pill(type_name, bg, fg)


def _help_content(key: str, defn: SettingDef) -> Content:
    """Build help text with default value hint."""
    value = _effective_value(key)
    parts: list[Content | tuple[str, str]] = []
    if defn.help_text:
        parts.append(Content(defn.help_text))
    default_hint = f"  current: {value}"
    parts.append((default_hint, "$text-muted"))
    return Content.assemble(*parts)


def _group_settings() -> dict[str, list[tuple[str, SettingDef]]]:
    """Group settings by their group field, preserving insertion order."""
    groups: dict[str, list[tuple[str, SettingDef]]] = defaultdict(list)
    for key, defn in SETTINGS_MAP.items():
        groups[defn.group].append((key, defn))
    return dict(groups)


def _make_editor(key: str, defn: SettingDef) -> Input | Checkbox | Select[str]:
    """Create the appropriate editor widget for a setting."""
    value = _effective_value(key)
    if defn.choices:
        return _make_select(key, defn, value)
    if defn.type is bool:
        return _make_checkbox(key, value)
    return _make_input(key, value)


def _make_select(key: str, defn: SettingDef, value: str) -> Select[str]:
    """Create a Select widget for choice-based settings."""
    choices = [(c, c) for c in (defn.choices or ())]
    if value in {c[1] for c in choices}:
        return Select(choices, value=value, name=key, classes="setting-editor", id=f"ed-{key}")
    return Select(choices, name=key, classes="setting-editor", id=f"ed-{key}")


def _make_checkbox(key: str, value: str) -> Checkbox:
    """Create a Checkbox widget for boolean settings."""
    checked = value.lower() in ("true", "1", "yes", "on")
    return Checkbox(value=checked, name=key, classes="setting-editor", id=f"ed-{key}")


def _make_input(key: str, value: str) -> Input:
    """Create an Input widget for string/number settings."""
    display = "" if value == "None" else value.replace(" (model default)", "")
    return Input(value=display, name=key, classes="setting-editor", id=f"ed-{key}")


class SettingsScreen(Screen[None]):
    """Interactive settings viewer with grouped, type-aware editors."""

    CSS_PATH = "settings.tcss"
    AUTO_FOCUS = "#settings-scroll"
    HELP = "Browse and edit configuration.\n\nUse / to search, Enter to confirm, Escape to return to the list."

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "go_back", "Back", show=True),
        Binding("escape", "go_back", "Back", show=False),
        Binding("slash", "focus_search", "/ Search", show=True),
        Binding("j", "scroll_down", "Down", show=False),
        Binding("k", "scroll_up", "Up", show=False),
        Binding("g", "scroll_home", "Top", show=False),
        Binding("G", "scroll_end", "End", show=False),
    ]

    def compose(self) -> ComposeResult:
        from textual.widgets import Footer

        from lilbee.cli.tui.widgets.status_bar import ViewTabs

        yield Input(
            placeholder="Filter settings...",
            id="settings-search",
        )
        with VerticalScroll(id="settings-scroll"):
            yield from self._compose_groups()
        yield ViewTabs()
        yield Footer()

    def _compose_groups(self) -> ComposeResult:
        """Yield grouped setting sections."""
        for group_name, items in _group_settings().items():
            with VerticalGroup(classes="setting-group", id=f"group-{group_name.lower()}"):
                yield Static(group_name, classes="group-title")
                for key, defn in items:
                    yield from self._compose_setting(key, defn)

    def _compose_setting(self, key: str, defn: SettingDef) -> ComposeResult:
        """Yield widgets for a single setting row."""
        with VerticalGroup(
            classes="setting-row",
            name=f"{defn.group.lower()} {key}",
            id=f"row-{key}",
        ):
            yield Static(
                Content.assemble(Content(key + "  "), _type_pill(defn)),
                classes="setting-title",
            )
            yield Static(_help_content(key, defn), classes="setting-help")
            if defn.writable:
                yield _make_editor(key, defn)

    @on(Input.Submitted, "#settings-search")
    def _on_search_submitted(self) -> None:
        """Blur the search input when Enter is pressed."""
        self.query_one("#settings-scroll", VerticalScroll).focus()

    @on(Input.Changed, "#settings-search")
    def _filter_settings(self, event: Input.Changed) -> None:
        """Filter visible settings based on search input."""
        term = event.value.strip().lower()
        for group in self.query(".setting-group"):
            visible_count = 0
            for row in group.query(".setting-row"):
                matches = not term or term in (row.name or "")
                row.display = matches
                if matches:
                    visible_count += 1
            group.display = visible_count > 0

    @on(Input.Submitted, ".setting-editor")
    @on(Input.Blurred, ".setting-editor")
    def _on_input_save(self, event: Input.Submitted | Input.Blurred) -> None:
        """Save string/number input on submit or blur."""
        name = event.input.name
        if name is None:
            return
        defn = SETTINGS_MAP.get(name)
        if defn is None:
            return
        raw = event.value.strip()
        current = str(getattr(cfg, name, ""))
        if raw == current:
            return
        self._persist_value(name, defn, raw)

    @on(Checkbox.Changed, ".setting-editor")
    def _on_checkbox_save(self, event: Checkbox.Changed) -> None:
        """Save boolean on toggle."""
        name = event.checkbox.name
        if name is None:
            return
        defn = SETTINGS_MAP.get(name)
        if defn is None:
            return
        self._persist_value(name, defn, str(event.checkbox.value))

    @on(Select.Changed, ".setting-editor")
    def _on_select_save(self, event: Select.Changed) -> None:
        """Save select choice on change."""
        name = event.select.name
        if name is None:
            return
        defn = SETTINGS_MAP.get(name)
        if defn is None:
            return
        value = str(event.value) if event.value != Select.BLANK else ""
        self._persist_value(name, defn, value)

    def _persist_value(self, key: str, defn: SettingDef, raw: str) -> None:
        """Parse, apply, and persist a setting value."""
        try:
            parsed = self._parse_value(defn, raw)
            setattr(cfg, key, parsed)
            persisted = str(parsed) if parsed is not None else ""
            settings.set_value(cfg.data_root, key, persisted)
            self.notify(msg.CMD_SET_SUCCESS.format(key=key, value=parsed))
            self._refresh_help(key, defn)
            from lilbee.cli.tui.app import LilbeeApp

            if isinstance(self.app, LilbeeApp):  # test apps aren't LilbeeApp
                self.app.settings_changed_signal.publish((key, parsed))
        except (ValueError, TypeError) as exc:
            self.notify(msg.SETTINGS_INVALID_VALUE.format(error=exc), severity="error")

    def _parse_value(self, defn: SettingDef, raw: str) -> object:
        """Convert a raw string to the setting's target type."""
        if defn.nullable and raw.lower() in ("none", "null", ""):
            return None
        if defn.type is bool:
            return raw.lower() in ("true", "1", "yes", "on")
        return defn.type(raw)

    def _refresh_help(self, key: str, defn: SettingDef) -> None:
        """Update the help text after a value change."""
        try:
            row = self.query_one(f"#row-{key}", VerticalGroup)
            help_widget = row.query_one(".setting-help", Static)
            help_widget.update(_help_content(key, defn))
        except Exception:
            log.debug("Failed to refresh help for %s", key, exc_info=True)

    def action_focus_search(self) -> None:
        """Focus the search input -- bound to / key."""
        self.query_one("#settings-search", Input).focus()

    def action_go_back(self) -> None:
        search = self.query_one("#settings-search", Input)
        if self.focused is search:  # Escape from filter → blur, don't leave
            self.query_one("#settings-scroll", VerticalScroll).focus()
            return
        from lilbee.cli.tui.app import LilbeeApp

        if isinstance(self.app, LilbeeApp):  # test apps aren't LilbeeApp
            self.app.switch_view("Chat")
        else:
            self.app.pop_screen()

    def action_scroll_down(self) -> None:
        self.query_one("#settings-scroll", VerticalScroll).scroll_down()

    def action_scroll_up(self) -> None:
        self.query_one("#settings-scroll", VerticalScroll).scroll_up()

    def action_scroll_home(self) -> None:
        self.query_one("#settings-scroll", VerticalScroll).scroll_home()

    def action_scroll_end(self) -> None:
        self.query_one("#settings-scroll", VerticalScroll).scroll_end()
