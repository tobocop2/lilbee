"""Settings screen. Grouped, type-aware configuration editor."""

from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, VerticalGroup, VerticalScroll
from textual.content import Content
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    Input,
    Select,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)

from lilbee.cli.settings_map import SETTINGS_MAP, RenderStyle, SettingDef, get_default
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.pill import pill
from lilbee.cli.tui.widgets.list_text_area import ListTextArea
from lilbee.core import settings
from lilbee.core.config import DEFAULT_CRAWL_EXCLUDE_PATTERNS, cfg

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

_LIST_RESTORE_PREFIX = "list-restore-"
_LIST_ERROR_ID_PREFIX = "err-"
_LIST_ERROR_VISIBLE_CLASS = "-visible"

_API_KEYS_GROUP = "API-Keys"
_API_KEYS_WARNING_CLASS = "api-keys-warning"
_CONFIG_TOML_FILENAME = "config.toml"

# Model-config fields surface their value through a picker modal that
# lists every installed/available chat/embed/vision/rerank model the
# host can route, instead of a free-text Input. The mapping below is the
# single source of truth for the picker scope each model field uses.
_MODEL_FIELD_TO_PICKER_SCOPE: dict[str, str] = {
    "chat_model": "chat",
    "embedding_model": "embed",
    "vision_model": "vision",
    "reranker_model": "rerank",
}
_MODEL_PICKER_BUTTON_PREFIX = "model-pick-"
_MODEL_VALUE_NONE = "(none)"


def _model_picker_label(key: str) -> str:
    """Render the picker button label as the human-friendly model name."""
    from lilbee.catalog.formatting import display_label_for_ref

    ref = getattr(cfg, key, None) or ""
    label = display_label_for_ref(str(ref))
    return label or _MODEL_VALUE_NONE


@dataclass(frozen=True)
class _PaneGroup:
    """One settings tab: pane id, group label, ordered settings."""

    pane_id: str
    group_name: str
    items: list[tuple[str, SettingDef]]


class _LazyGroupBody(VerticalGroup):
    """Pane-body container that mounts its rows on first activation."""

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._populated = False

    @property
    def populated(self) -> bool:
        return self._populated

    def populate(self, build: Callable[[], list[Widget]]) -> None:
        """Build and mount this pane's row widgets exactly once."""
        if self._populated:
            return
        self._populated = True
        widgets = build()
        if widgets:
            self.mount_all(widgets)


def _config_toml_path() -> str:
    """Effective path to the config.toml lilbee reads and writes."""
    return str(cfg.data_dir / _CONFIG_TOML_FILENAME)


def _effective_value(key: str) -> str:
    """Return the effective value for a setting, including model defaults."""
    user_value = getattr(cfg, key, None)
    if user_value is not None:
        if isinstance(user_value, list):
            return f"{len(user_value)} lines"
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
    """Pill warning that an env var is overriding TUI edits, or None."""
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
    """Serialize a default for the TOML settings store."""
    if default is None:
        return ""
    if isinstance(default, list):
        return "\n".join(default)
    return str(default)


_FEATURE_GATED_GROUPS: dict[str, Callable[[], bool]] = {
    "API-Keys": lambda: _litellm_installed(),
    "Crawling": lambda: _crawler_installed(),
    "Wiki": lambda: bool(cfg.wiki),
}


def _litellm_installed() -> bool:
    from lilbee.providers.litellm_sdk import litellm_available

    return litellm_available()


def _crawler_installed() -> bool:
    from lilbee.crawler import crawler_available

    return crawler_available()


def _group_settings() -> dict[str, list[tuple[str, SettingDef]]]:
    """Group settings by group field, hiding entries gated by unavailable features."""
    groups: dict[str, list[tuple[str, SettingDef]]] = defaultdict(list)
    for key, defn in SETTINGS_MAP.items():
        gate = _FEATURE_GATED_GROUPS.get(defn.group)
        if gate is not None and not gate():
            continue
        groups[defn.group].append((key, defn))
    return dict(groups)


def _make_editor(key: str, defn: SettingDef) -> Widget:
    """Create the appropriate editor widget for a setting."""
    if defn.render is RenderStyle.LIST_COLLAPSED:
        return _make_list_editor(key)
    value = _effective_value(key)
    if defn.choices:
        return _make_select(key, defn, value)
    if defn.type is bool:
        return _make_checkbox(key, value)
    if defn.render is RenderStyle.MULTILINE:
        return _make_multiline_editor(key, value)
    return _make_input(key, value)


def _make_multiline_editor(key: str, value: str) -> ListTextArea:
    """Create a multi-line editor for string settings (system prompts, etc.).

    Reuses ListTextArea so we get the same blur-saves-on-focus-out behavior
    as list-of-strings settings without re-implementing the message bridge.
    """
    display = "" if value == "None" else value
    return ListTextArea(
        text=display,
        show_line_numbers=False,
        name=key,
        id=f"{_EDITOR_ID_PREFIX}{key}",
        classes="setting-editor setting-multiline-editor",
        soft_wrap=True,
    )


def _make_list_editor(key: str) -> Collapsible:
    """Create a Collapsible with a line-numbered TextArea for list[str] settings."""
    current = getattr(cfg, key, None) or []
    title = msg.SETTINGS_LIST_EDITOR_TITLE.format(key=key, count=len(current))
    editor = ListTextArea(
        text="\n".join(current),
        show_line_numbers=True,
        name=key,
        id=f"ed-{key}",
        classes="setting-list-editor",
    )
    error = Static("", id=f"{_LIST_ERROR_ID_PREFIX}{key}", classes="setting-list-error")
    reset = Button(
        msg.SETTINGS_LIST_EDITOR_RESTORE_DEFAULTS,
        id=f"{_LIST_RESTORE_PREFIX}{key}",
        classes="setting-list-restore",
    )
    return Collapsible(
        editor,
        error,
        reset,
        title=title,
        collapsed=True,
        id=f"collapsible-{key}",
    )


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


def _make_input(key: str, value: str) -> Input:
    """Create an Input widget for string/number settings."""
    display = "" if value == "None" else value.replace(" (model default)", "")
    return Input(value=display, name=key, classes="setting-editor", id=f"{_EDITOR_ID_PREFIX}{key}")


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
        # Tab walks every focusable on the screen (reset-all button,
        # the group Tabs strip, then editors inside the active
        # TabPane). Use ←/→ to switch tabs while focus is on the strip.
        Binding("tab", "app.focus_next", "Next field", show=True),
        Binding("shift+tab", "app.focus_previous", "Prev field", show=True),
        Binding("ctrl+r", "reset_focused", "Reset", show=False),
        Binding("j", "scroll_down", "Down", show=False),
        Binding("k", "scroll_up", "Up", show=False),
        Binding("g", "scroll_home", "Top", show=False),
        Binding("G", "scroll_end", "End", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        # Group definitions for lazy-mount on tab activation. Indexed
        # by pane id so the activated-pane handler can look up its
        # bundle in O(1). ``_eagerly_populate`` is the pane id whose
        # body gets populated in on_mount (the active-by-default first
        # pane); the rest fill in on first activation.
        self._pane_groups: dict[str, _PaneGroup] = {}
        self._eagerly_populate: str | None = None

    def compose(self) -> ComposeResult:
        from textual.widgets import Footer

        from lilbee.cli.tui.widgets.bottom_bars import BottomBars
        from lilbee.cli.tui.widgets.status_bar import ViewTabs
        from lilbee.cli.tui.widgets.task_bar import TaskBar
        from lilbee.cli.tui.widgets.top_bars import TopBars

        with TopBars():
            yield ViewTabs()
            with Horizontal(id="settings-top-row"):
                yield Button(
                    msg.SETTINGS_RESET_ALL_LABEL,
                    id="reset-all-defaults",
                    classes="reset-all-button",
                )
        with VerticalScroll(id="settings-scroll"), TabbedContent(id="settings-tabs"):
            yield from self._compose_group_tabs()
        with BottomBars():
            yield TaskBar()
            yield Footer()

    def _compose_group_tabs(self) -> ComposeResult:
        """Yield one TabPane per setting group; bodies populate on activation."""
        first = True
        for group_name, items in _group_settings().items():
            pane_id = f"settings-tab-{group_name.lower().replace('-', '_')}"
            self._pane_groups[pane_id] = _PaneGroup(
                pane_id=pane_id, group_name=group_name, items=items
            )
            yield TabPane(
                group_name,
                _LazyGroupBody(id=f"{pane_id}-body"),
                id=pane_id,
            )
            # The first pane is the one TabbedContent activates by
            # default; populate it eagerly so a user landing on
            # Settings sees content on first paint instead of an empty
            # active pane that fills in one frame later.
            if first:
                first = False
                self._eagerly_populate = pane_id

    def on_mount(self) -> None:
        """Populate the active pane's body so first paint shows content."""
        if self._eagerly_populate is not None:
            self._populate_pane(self._eagerly_populate)

    @on(TabbedContent.TabActivated)
    def _on_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        """Populate the activated pane's body on first activation."""
        pane = event.pane
        if pane is None or pane.id is None:
            return
        self._populate_pane(pane.id)

    def populate_all_panes(self) -> None:
        """Force every tab body to populate now (test/agent helper)."""
        for pane_id in self._pane_groups:
            self._populate_pane(pane_id)

    def _populate_pane(self, pane_id: str) -> None:
        """Populate a pane's body if known and the body widget is mounted."""
        group = self._pane_groups.get(pane_id)
        if group is None:
            return
        try:
            body = self.query_one(f"#{pane_id}-body", _LazyGroupBody)
        except Exception:
            log.debug("pane body %s not yet mounted", pane_id, exc_info=True)
            return
        body.populate(lambda: self._build_pane_widgets(group))

    def _build_pane_widgets(self, group: _PaneGroup) -> list[Widget]:
        """Return the body widgets for one settings tab."""
        widgets: list[Widget] = []
        if group.group_name == _API_KEYS_GROUP:
            widgets.append(
                Static(
                    msg.SETTINGS_API_KEYS_WARNING.format(path=_config_toml_path()),
                    classes=_API_KEYS_WARNING_CLASS,
                )
            )
        for key, defn in group.items:
            widgets.append(self._build_setting_row(key, defn))
        return widgets

    def _build_setting_row(self, key: str, defn: SettingDef) -> VerticalGroup:
        """Construct one setting row with its title, help, editor, and reset."""
        title = Static(_title_content(key, defn), classes="setting-title")
        help_widget = Static(_help_content(key, defn), classes="setting-help")
        children: list[Widget] = [title, help_widget]
        if key in _MODEL_FIELD_TO_PICKER_SCOPE:
            children.append(self._build_model_picker_row(key))
        elif defn.writable:
            editor_row = Horizontal(
                _make_editor(key, defn),
                Button(
                    _RESET_BUTTON_LABEL,
                    id=f"{_RESET_BUTTON_ID_PREFIX}{key}",
                    classes="setting-reset-button",
                    tooltip=msg.SETTINGS_RESET_TO_DEFAULT_TOOLTIP,
                ),
                classes="setting-editor-row",
            )
            children.append(editor_row)
        return VerticalGroup(
            *children,
            classes="setting-row",
            id=f"{_ROW_ID_PREFIX}{key}",
        )

    def _build_model_picker_row(self, key: str) -> Horizontal:
        """A button-style row that opens the same ModelPickerModal as the chat bar."""
        return Horizontal(
            Button(
                _model_picker_label(key),
                id=f"{_MODEL_PICKER_BUTTON_PREFIX}{key}",
                classes="setting-model-picker-button",
            ),
            classes="setting-editor-row",
        )

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

    @on(ListTextArea.Blurred, ".setting-multiline-editor")
    def _on_multiline_save(self, event: ListTextArea.Blurred) -> None:
        """Save multi-line string settings (system prompts) on blur."""
        ta = event.control
        name = ta.name
        if name is None:
            return
        defn = SETTINGS_MAP.get(name)
        if defn is None:
            return
        raw = ta.text
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
        """Parse, apply, and persist a setting value."""
        try:
            parsed = self._parse_value(defn, raw)
            from lilbee.cli.tui.app import LilbeeApp

            if isinstance(self.app, LilbeeApp):
                # set_setting handles theme live-apply, signal publish, etc.
                self.app.set_setting(key, parsed)
            else:
                # Test harnesses where self.app isn't a LilbeeApp.
                setattr(cfg, key, parsed)
                persisted = self._stringify_for_toml(parsed)
                settings.set_value(cfg.data_root, key, persisted)
            if not quiet:
                self.notify(msg.CMD_SET_SUCCESS.format(key=key, value=parsed))
            self._refresh_help(key, defn)
        except (ValueError, TypeError) as exc:
            self.notify(msg.SETTINGS_INVALID_VALUE.format(error=exc), severity="error")

    def _parse_value(self, defn: SettingDef, raw: str) -> object:
        """Convert a raw string to the setting's target type."""
        if defn.nullable and raw.lower() in ("none", "null", ""):
            return None
        if defn.type is bool:
            return raw.lower() in ("true", "1", "yes", "on")
        if defn.type is list:
            return [line.strip() for line in raw.split("\n") if line.strip()]
        return defn.type(raw)

    @staticmethod
    def _stringify_for_toml(parsed: object) -> str:
        """Serialize a parsed value for the TOML settings store."""
        if parsed is None:
            return ""
        if isinstance(parsed, list):
            return "\n".join(parsed)
        return str(parsed)

    @staticmethod
    def _validate_regex_list(lines: list[str]) -> tuple[int, str] | None:
        """Return the 1-indexed line number and error for the first bad regex, or None."""
        for i, line in enumerate(lines, 1):
            try:
                re.compile(line)
            except re.error as exc:
                return (i, str(exc))
        return None

    @on(ListTextArea.Blurred, ".setting-list-editor")
    def _on_list_blur_save(self, event: ListTextArea.Blurred) -> None:
        """Validate and save list values when a ListTextArea loses focus."""
        ta = event.control
        key = ta.name
        if key is None:
            return
        defn = SETTINGS_MAP.get(key)
        if defn is None:
            return
        raw = ta.text
        parsed = self._parse_value(defn, raw)
        assert isinstance(parsed, list)  # noqa: S101 -- mypy narrowing, defn.type is list above
        err = self._validate_regex_list(parsed)
        error_widget = self.query_one(f"#{_LIST_ERROR_ID_PREFIX}{key}", Static)
        if err is not None:
            line_no, err_text = err
            error_widget.update(
                msg.SETTINGS_LIST_EDITOR_INVALID_REGEX.format(n=line_no, error=err_text)
            )
            error_widget.add_class(_LIST_ERROR_VISIBLE_CLASS)
            return
        error_widget.remove_class(_LIST_ERROR_VISIBLE_CLASS)
        self._persist_value(key, defn, raw)
        self._refresh_list_title(key, len(parsed))

    @on(Button.Pressed, ".setting-list-restore")
    def _on_list_restore(self, event: Button.Pressed) -> None:
        """Restore defaults for a LIST_COLLAPSED setting."""
        btn_id = event.button.id
        if btn_id is None or not btn_id.startswith(_LIST_RESTORE_PREFIX):
            return
        key = btn_id.removeprefix(_LIST_RESTORE_PREFIX)
        defn = SETTINGS_MAP.get(key)
        if defn is None:
            return
        defaults = list(DEFAULT_CRAWL_EXCLUDE_PATTERNS)
        text = "\n".join(defaults)
        ta = self.query_one(f"#ed-{key}", ListTextArea)
        ta.load_text(text)
        self._persist_value(key, defn, text)
        error_widget = self.query_one(f"#{_LIST_ERROR_ID_PREFIX}{key}", Static)
        error_widget.remove_class(_LIST_ERROR_VISIBLE_CLASS)
        self._refresh_list_title(key, len(defaults))

    def _refresh_list_title(self, key: str, count: int) -> None:
        """Update the Collapsible title to reflect the current line count."""
        try:
            collapsible = self.query_one(f"#collapsible-{key}", Collapsible)
            collapsible.title = msg.SETTINGS_LIST_EDITOR_TITLE.format(key=key, count=count)
        except Exception:
            log.debug("Failed to refresh collapsible title for %s", key, exc_info=True)

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

    @on(Button.Pressed, ".setting-model-picker-button")
    def _on_model_picker_pressed(self, event: Button.Pressed) -> None:
        """Open ModelPickerModal for the model field this button represents."""
        button_id = event.button.id
        if button_id is None or not button_id.startswith(_MODEL_PICKER_BUTTON_PREFIX):
            return
        key = button_id[len(_MODEL_PICKER_BUTTON_PREFIX) :]
        scope = _MODEL_FIELD_TO_PICKER_SCOPE.get(key)
        if scope is None:
            return
        self._open_model_picker_for_field(key, scope)

    def _open_model_picker_for_field(self, key: str, scope: str) -> None:
        """Discover models for *scope*, push the picker, persist on dismiss."""
        from lilbee.cli.tui.screens.model_picker import ModelPickerModal, PickerScope
        from lilbee.cli.tui.widgets.model_bar import (
            ModelOption,
            classify_installed_models_full,
        )
        from lilbee.modelhub.models import ModelTask

        scope_to_task: dict[str, ModelTask] = {
            "chat": ModelTask.CHAT,
            "embed": ModelTask.EMBEDDING,
            "vision": ModelTask.VISION,
            "rerank": ModelTask.RERANK,
        }
        task = scope_to_task[scope]
        buckets = classify_installed_models_full()
        options = list(buckets.get(task, []))
        if not options:
            options = [ModelOption(label=_MODEL_VALUE_NONE, ref="")]
        modal_scope: PickerScope = scope  # type: ignore[assignment]
        self.app.push_screen(
            ModelPickerModal(scope=modal_scope, options=options),
            lambda ref: self._on_model_picker_dismissed(key, ref),
        )

    def _on_model_picker_dismissed(self, key: str, ref: str | None) -> None:
        """Persist the picker selection and refresh the button label."""
        if ref is None or not ref:
            return
        from lilbee.cli.tui.app import apply_active_model

        apply_active_model(self.app, key, ref)
        try:
            button = self.query_one(f"#{_MODEL_PICKER_BUTTON_PREFIX}{key}", Button)
            button.label = _model_picker_label(key)
        except Exception:
            log.debug("Failed to refresh model picker label for %s", key, exc_info=True)

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
        """Reset every writable setting to its cfg default atomically."""
        if not confirmed:
            return
        writable = [(key, defn) for key, defn in SETTINGS_MAP.items() if defn.writable]
        snapshot = {key: getattr(cfg, key) for key, _ in writable}
        updates, signal_payload, skipped = self._apply_batch_defaults(writable)
        if updates and not self._persist_batch(writable, snapshot, updates):
            return
        self._refresh_batch(writable, skipped)
        self._publish_batch_signals(signal_payload)
        self._notify_batch_result(skipped)

    def _apply_batch_defaults(
        self, writable: list[tuple[str, SettingDef]]
    ) -> tuple[dict[str, str], list[tuple[str, object]], list[str]]:
        """Mutate cfg in-memory for every writable key; track updates + skips."""
        updates: dict[str, str] = {}
        signal_payload: list[tuple[str, object]] = []
        skipped: list[str] = []
        for key, _defn in writable:
            default = get_default(key)
            try:
                setattr(cfg, key, default)
            except (ValueError, TypeError) as exc:
                log.warning("Default for %s rejected by cfg (%s); skipping", key, exc)
                skipped.append(key)
                continue
            updates[key] = _stringify_default(default)
            signal_payload.append((key, default))
        return updates, signal_payload, skipped

    def _persist_batch(
        self,
        writable: list[tuple[str, SettingDef]],
        snapshot: dict[str, object],
        updates: dict[str, str],
    ) -> bool:
        """Persist the batch; roll back cfg + UI on disk error. Returns True on success."""
        try:
            settings.update_values(cfg.data_root, updates)
        except OSError as exc:
            self._rollback_batch(writable, snapshot)
            self.notify(msg.SETTINGS_INVALID_VALUE.format(error=exc), severity="error")
            return False
        return True

    def _rollback_batch(
        self, writable: list[tuple[str, SettingDef]], snapshot: dict[str, object]
    ) -> None:
        """Restore cfg and editor widgets from snapshot after a failed persist."""
        for key, prev in snapshot.items():
            try:
                setattr(cfg, key, prev)
            except (ValueError, TypeError):
                log.exception("Failed to roll back cfg.%s", key)
        for key, defn in writable:
            self._refresh_editor(key, defn, snapshot[key])
            self._refresh_help(key, defn)

    def _refresh_batch(self, writable: list[tuple[str, SettingDef]], skipped: list[str]) -> None:
        """Refresh editor + help for each successfully-reset writable key."""
        for key, defn in writable:
            if key in skipped:
                continue
            default = get_default(key)
            self._refresh_editor(key, defn, default)
            self._refresh_help(key, defn)

    def _publish_batch_signals(self, signal_payload: list[tuple[str, object]]) -> None:
        """Fan out settings_changed signals for every successfully-reset key."""
        from lilbee.cli.tui.app import LilbeeApp

        if not isinstance(self.app, LilbeeApp):  # test apps aren't LilbeeApp
            return
        for pub_key, pub_parsed in signal_payload:
            self.app.settings_changed_signal.publish((pub_key, pub_parsed))

    def _notify_batch_result(self, skipped: list[str]) -> None:
        """Surface a single summary toast; warning severity when any key skipped."""
        if skipped:
            self.notify(
                msg.SETTINGS_RESET_ALL_PARTIAL.format(skipped=", ".join(skipped)),
                severity="warning",
            )
        else:
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
        """Restore a single setting to its cfg default."""
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

    def action_go_back(self) -> None:
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
