"""Settings screen — grouped, type-aware configuration editor."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, VerticalGroup, VerticalScroll
from textual.content import Content
from textual.screen import Screen
from textual.widgets import Button, Checkbox, Input, Select, Static, TextArea

from lilbee import settings
from lilbee.cli.settings_map import SETTINGS_MAP, SettingDef, get_default
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.pill import pill
from lilbee.cli.tui.widgets.nav_aware_input import NavAwareInput
from lilbee.config import cfg

_ROW_ID_PREFIX = "row-"
_EDITOR_ID_PREFIX = "ed-"
_RESET_BUTTON_ID_PREFIX = "reset-"
_RESET_BUTTON_LABEL = "↺"

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


def _env_var_name(key: str) -> str:
    """Return the LILBEE_* env var name for a config key."""
    return f"LILBEE_{key.upper()}"


def _env_pill(key: str) -> Content | None:
    """Return a warning pill showing the literal env var when it's set.

    The pill appears only when the user has exported the corresponding
    env var, signalling that TUI edits won't persist because the env
    wins on next launch.
    """
    env_name = _env_var_name(key)
    if os.environ.get(env_name) is None:
        return None
    return pill(env_name, "$warning", "$text")


def _help_content(key: str, defn: SettingDef) -> Content:
    """Build help text; the editor widget already shows the current value."""
    if defn.help_text:
        return Content(defn.help_text)
    return Content("")


def _title_content(key: str, defn: SettingDef) -> Content:
    """Assemble the setting-row title: key name, type pill, and env pill when set."""
    parts: list[Content] = [Content(key + "  "), _type_pill(defn)]
    env_badge = _env_pill(key)
    if env_badge is not None:
        parts.append(Content("  "))
        parts.append(env_badge)
    return Content.assemble(*parts)


def _stringify_default(default: object) -> str:
    """Serialize a reset-to-default value for the TOML settings store.

    Lists become newline-joined so pydantic's splitlines validator round-trips
    cleanly on reload. None collapses to the empty string. Other scalars use
    str() directly.
    """
    if default is None:
        return ""
    if isinstance(default, list):
        return "\n".join(default)
    return str(default)


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
        return Select(
            choices,
            value=value,
            name=key,
            classes="setting-editor",
            id=f"{_EDITOR_ID_PREFIX}{key}",
        )
    return Select(choices, name=key, classes="setting-editor", id=f"{_EDITOR_ID_PREFIX}{key}")


def _make_checkbox(key: str, value: str) -> Checkbox:
    """Create a Checkbox widget for boolean settings."""
    checked = value.lower() in ("true", "1", "yes", "on")
    return Checkbox(
        value=checked, name=key, classes="setting-editor", id=f"{_EDITOR_ID_PREFIX}{key}"
    )


def _make_input(key: str, value: str) -> NavAwareInput:
    """Create an Input widget for string/number settings."""
    display = "" if value == "None" else value.replace(" (model default)", "")
    return NavAwareInput(
        value=display, name=key, classes="setting-editor", id=f"{_EDITOR_ID_PREFIX}{key}"
    )


class SettingsScreen(Screen[None]):
    """Interactive settings viewer with grouped, type-aware editors."""

    CSS_PATH = "settings.tcss"
    AUTO_FOCUS = "#settings-scroll"
    HELP = (
        "Browse and edit configuration.\n\n"
        "Use / to search, Enter to confirm, Escape to return to the list."
    )

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "go_back", "Back", show=True),
        Binding("escape", "go_back", "Back", show=False),
        Binding("slash", "focus_search", "Search", show=True),
        Binding("tab", "app.focus_next", "Next field", show=True),
        Binding("shift+tab", "app.focus_previous", "Prev field", show=True),
        Binding("ctrl+r", "reset_focused", "Reset", show=False),
        Binding("j", "scroll_down", "Down", show=False),
        Binding("k", "scroll_up", "Up", show=False),
        Binding("g", "scroll_home", "Top", show=False),
        Binding("G", "scroll_end", "End", show=False),
    ]

    def compose(self) -> ComposeResult:
        from textual.widgets import Footer

        from lilbee.cli.tui.widgets.bottom_bars import BottomBars
        from lilbee.cli.tui.widgets.status_bar import ViewTabs
        from lilbee.cli.tui.widgets.task_bar import TaskBar

        with Horizontal(id="settings-top-row"):
            yield NavAwareInput(
                placeholder="Filter settings...",
                id="settings-search",
            )
            yield Button(
                msg.SETTINGS_RESET_ALL_LABEL,
                id="reset-all-defaults",
                classes="reset-all-button",
            )
        with VerticalScroll(id="settings-scroll"):
            yield from self._compose_groups()
        with BottomBars():
            yield TaskBar()
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
            id=f"{_ROW_ID_PREFIX}{key}",
        ):
            yield Static(_title_content(key, defn), classes="setting-title")
            yield Static(_help_content(key, defn), classes="setting-help")
            if defn.writable:
                with Horizontal(classes="setting-editor-row"):
                    yield _make_editor(key, defn)
                    yield Button(
                        _RESET_BUTTON_LABEL,
                        id=f"{_RESET_BUTTON_ID_PREFIX}{key}",
                        classes="setting-reset-button",
                        tooltip=msg.SETTINGS_RESET_TO_DEFAULT_TOOLTIP,
                    )

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
        current = str(getattr(cfg, name, ""))
        if value == current:
            return
        self._persist_value(name, defn, value)

    def _persist_value(self, key: str, defn: SettingDef, raw: str, *, quiet: bool = False) -> None:
        """Parse, apply, and persist a setting value.

        When *quiet* is True, the per-field CMD_SET_SUCCESS toast is suppressed.
        Used by batch paths (e.g. reset-all) that fire one summary toast instead.
        """
        try:
            parsed = self._parse_value(defn, raw)
            setattr(cfg, key, parsed)
            persisted = str(parsed) if parsed is not None else ""
            settings.set_value(cfg.data_root, key, persisted)
            if not quiet:
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
            row = self.query_one(f"#{_ROW_ID_PREFIX}{key}", VerticalGroup)
            help_widget = row.query_one(".setting-help", Static)
            help_widget.update(_help_content(key, defn))
        except Exception:
            log.debug("Failed to refresh help for %s", key, exc_info=True)

    @on(Button.Pressed, ".setting-reset-button")
    def _on_reset_pressed(self, event: Button.Pressed) -> None:
        """Handle the small reset button embedded in each writable row."""
        button_id = event.button.id
        if button_id is None or not button_id.startswith(_RESET_BUTTON_ID_PREFIX):
            return
        key = button_id[len(_RESET_BUTTON_ID_PREFIX) :]
        self._reset_to_default(key)

    @on(Button.Pressed, "#reset-all-defaults")
    def _on_reset_all_pressed(self) -> None:
        """Open a destructive-confirm dialog before resetting every writable field."""
        from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog

        self.app.push_screen(
            ConfirmDialog(
                title=msg.SETTINGS_RESET_ALL_CONFIRM_TITLE,
                message=msg.SETTINGS_RESET_ALL_CONFIRM_MESSAGE,
            ),
            self._on_reset_all_confirmed,
        )

    def _on_reset_all_confirmed(self, confirmed: bool | None) -> None:
        """Apply reset-to-default to every writable SETTINGS_MAP entry on confirm.

        Writes are batched into one atomic config.toml rewrite via
        settings.update_values so a crash mid-loop cannot leave a partially
        reset config. Per-field toasts are suppressed; one summary fires at
        the end.

        The pydantic default is applied directly via setattr (skipping the
        _parse_value round-trip) so reset works even for fields where
        SETTINGS_MAP.nullable does not match the cfg type exactly. An invalid
        default is logged and skipped; no partial batch is committed for it.
        """
        if not confirmed:
            return
        updates: dict[str, str] = {}
        signal_payload: list[tuple[str, object]] = []
        for key, defn in SETTINGS_MAP.items():
            if not defn.writable:
                continue
            default = get_default(key)
            try:
                setattr(cfg, key, default)
            except (ValueError, TypeError):
                log.warning("Default for %s rejected by cfg; skipping in batch reset", key)
                continue
            updates[key] = _stringify_default(default)
            signal_payload.append((key, default))
            self._refresh_editor(key, defn, default)
            self._refresh_help(key, defn)
        if updates:
            settings.update_values(cfg.data_root, updates)
        from lilbee.cli.tui.app import LilbeeApp

        if isinstance(self.app, LilbeeApp):  # test apps aren't LilbeeApp
            for pub_key, pub_parsed in signal_payload:
                self.app.settings_changed_signal.publish((pub_key, pub_parsed))
        self.notify(msg.SETTINGS_RESET_ALL_SUCCESS)

    def action_reset_focused(self) -> None:
        """Reset the currently-focused setting row to its cfg default."""
        focused = self.focused
        if focused is None:
            return
        for ancestor in focused.ancestors_with_self:
            ancestor_id = getattr(ancestor, "id", None)
            if ancestor_id and ancestor_id.startswith(_ROW_ID_PREFIX):
                key = ancestor_id[len(_ROW_ID_PREFIX) :]
                self._reset_to_default(key)
                return

    def _reset_to_default(self, key: str) -> None:
        """Restore a single setting to its cfg default via the normal persist path.

        _persist_value already surfaces a `{key} = {value}` toast, so a second
        reset-specific toast would just double up for every per-row click.
        """
        defn = SETTINGS_MAP.get(key)
        if defn is None or not defn.writable:
            return
        default = get_default(key)
        stringified = _stringify_default(default)
        self._persist_value(key, defn, stringified)
        self._refresh_editor(key, defn, default)

    def _refresh_editor(self, key: str, defn: SettingDef, value: object) -> None:
        """Update the editor widget to reflect a new value (e.g. after reset)."""
        try:
            widget = self.query_one(f"#{_EDITOR_ID_PREFIX}{key}")
        except Exception:
            log.debug("Failed to refresh editor for %s", key, exc_info=True)
            return
        if isinstance(widget, Input):
            widget.value = "" if value is None else str(value)
        elif isinstance(widget, Checkbox):
            widget.value = bool(value)
        elif isinstance(widget, Select):
            if value is None:
                widget.clear()
            else:
                widget.value = str(value)
        elif isinstance(widget, TextArea):  # future-proofing: list/multiline defaults
            if isinstance(value, list):
                widget.load_text("\n".join(value))
            else:
                widget.load_text("" if value is None else str(value))

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
