"""Editor-row builders and label helpers for the Settings screen."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING

from textual.content import Content
from textual.widget import Widget
from textual.widgets import Button, Checkbox, Collapsible, Input, Select, Static, TextArea

from lilbee.app.settings import OCR_SETTING_KEYS, scanned_pages_line
from lilbee.app.settings_map import SETTINGS_MAP, RenderStyle, SettingDef, SettingGroup
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.pill import pill
from lilbee.cli.tui.widgets.list_text_area import ListTextArea
from lilbee.core.config import cfg
from lilbee.core.config.defaults import env_var_name
from lilbee.core.config.enums import SettingSource
from lilbee.core.config.resolve import read_profile_table
from lilbee.crawler import crawler_available

if TYPE_CHECKING:
    from lilbee.catalog.types import ModelTask
    from lilbee.cli.tui.screens.model_picker import PickerScope

ROW_ID_PREFIX = "row-"
EDITOR_ID_PREFIX = "ed-"
RESET_BUTTON_ID_PREFIX = "reset-"
RESET_BUTTON_LABEL = "↺"

_TYPE_COLORS: dict[str, tuple[str, str]] = {
    "str": ("$secondary", "$text"),
    "int": ("$primary", "$text"),
    "float": ("$primary", "$text"),
    "bool": ("$success", "$text"),
    "select": ("$warning", "$text"),
}

_DEFAULTS_REMAP: dict[str, str] = {"top_k_sampling": "top_k"}
MODEL_DEFAULT_MARKER = " (model default)"

LIST_RESTORE_PREFIX = "list-restore-"
LIST_ERROR_ID_PREFIX = "err-"
LIST_ERROR_VISIBLE_CLASS = "-visible"
ADVANCED_COLLAPSIBLE_ID_PREFIX = "settings-advanced-"
ADVANCED_COLLAPSIBLE_CLASS = "settings-advanced-collapsible"

API_KEYS_GROUP = SettingGroup.API_KEYS
API_KEYS_WARNING_CLASS = "api-keys-warning"
CONFIG_TOML_FILENAME = "config.toml"


def model_field_to_picker_scope() -> dict[str, PickerScope]:
    """Single source of truth for the picker scope each model field uses."""
    mapping: dict[str, PickerScope] = {
        "chat_model": "chat",
        "embedding_model": "embed",
        "vision_model": "vision",
        "reranker_model": "rerank",
    }
    return mapping


def picker_scope_to_task(scope: PickerScope) -> ModelTask:
    """Map a picker scope to the ``ModelTask`` bucket it discovers from."""
    from lilbee.catalog.types import ModelTask as _ModelTask

    return {
        "chat": _ModelTask.CHAT,
        "embed": _ModelTask.EMBEDDING,
        "vision": _ModelTask.VISION,
        "rerank": _ModelTask.RERANK,
    }[scope]


MODEL_PICKER_BUTTON_PREFIX = "model-pick-"


def set_widget_value(widget: Widget, value: object) -> None:
    """Push *value* into a settings-row editor widget."""
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
            widget.load_text("\n".join(str(item) for item in value))
        else:
            widget.load_text("" if value is None else str(value))


SettingEditor = Input | TextArea | Checkbox | Select[str]
"""Every widget kind that edits one setting's value."""
EDITOR_KINDS = (Input, TextArea, Checkbox, Select)


def displayed_text(widget: SettingEditor) -> str:
    """The raw text an editor shows: a Select's blank is "", a Checkbox is "True" or "False"."""
    if isinstance(widget, Input):
        return widget.value
    if isinstance(widget, TextArea):
        return widget.text
    if isinstance(widget, Checkbox):
        return str(widget.value)
    return "" if widget.value == Select.BLANK else str(widget.value)


def list_editor_text(key: str, value: list[object] | None = None) -> str:
    """The newline-joined text a list setting's editor shows for *value*.

    *value* defaults to the current cfg value; a caller that already read it
    (to also compute a count, say) passes it through instead of re-reading cfg.
    """
    items = (getattr(cfg, key, None) or []) if value is None else value
    return "\n".join(str(item) for item in items)


def select_shown_value(defn: SettingDef, value: str) -> str:
    """The value a Select shows for *value*: the matching choice, else blank.

    A Select's ``value`` is not set from its constructor until it mounts, so this does not read it.
    """
    if value in (defn.choices or ()):
        return value
    return ""


def strip_model_default_marker(value: str) -> str:
    """Drop the model-default marker and the sentinel "None" display text."""
    return "" if value == "None" else value.replace(MODEL_DEFAULT_MARKER, "")


def model_picker_label(key: str) -> str:
    """Render the picker button label as the human-friendly model name."""
    from lilbee.catalog.formatting import display_label_for_ref

    ref = getattr(cfg, key, None) or ""
    label = display_label_for_ref(str(ref))
    return label or msg.MODEL_VALUE_NONE


def config_toml_path() -> str:
    """Effective path to the config.toml lilbee reads and writes."""
    return str(cfg.data_dir / CONFIG_TOML_FILENAME)


def effective_value(key: str) -> str:
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
        return f"{default_val}{MODEL_DEFAULT_MARKER}"
    return "None"


def is_writable(key: str) -> bool:
    """Check if a setting key is writable (derived from SETTINGS_MAP)."""
    defn = SETTINGS_MAP.get(key)
    return defn is not None and defn.writable


def type_pill(defn: SettingDef) -> Content:
    """Create a colored pill badge for a setting's type."""
    type_name = defn.type.__name__
    if defn.choices:
        type_name = "select"
    bg, fg = _TYPE_COLORS.get(type_name, ("$surface", "$text"))
    return pill(type_name, bg, fg)


def user_pill() -> Content:
    """The accent pill that marks a value set by you."""
    return pill(msg.SETTINGS_SOURCE_USER_PILL, "$accent", "$text")


def _user_pill(_key: str, _profile_name: str | None) -> Content:
    return user_pill()


def _env_pill(key: str, _profile_name: str | None) -> Content:
    return pill(env_var_name(key), "$warning", "$text")


def _profile_pill(_key: str, profile_name: str | None) -> Content:
    label = (
        msg.SETTINGS_SOURCE_PROFILE_NAMED_PILL.format(name=profile_name)
        if profile_name
        else msg.SETTINGS_SOURCE_PROFILE_PILL
    )
    return pill(label, "$secondary", "$text")


_SOURCE_PILLS: dict[SettingSource, Callable[[str, str | None], Content]] = {
    SettingSource.USER: _user_pill,
    SettingSource.ENV: _env_pill,
    SettingSource.PROFILE: _profile_pill,
}


def active_profile_name() -> str | None:
    """The active profile's name, one config.toml read; call once per render, not per row."""
    return read_profile_table(cfg.data_root).name


def source_pill(key: str, source: SettingSource, profile_name: str | None = None) -> Content | None:
    """Pill for a value the user, an env var, or a profile overrides; None for any other source.

    *profile_name* is the caller's own ``active_profile_name()`` read, passed in rather than
    read here, so a pane with several profile-set rows shares one config.toml read.
    """
    build = _SOURCE_PILLS.get(source)
    return None if build is None else build(key, profile_name)


def help_content(key: str, defn: SettingDef) -> Content:
    """Build help text, plus the scanned-pages line on an OCR row; the editor shows the value."""
    help_text = Content(defn.help_text)
    if key not in OCR_SETTING_KEYS:
        return help_text
    return Content.assemble(help_text, "\n", scanned_pages_line())


def title_content(
    key: str, defn: SettingDef, source: SettingSource, profile_name: str | None = None
) -> Content:
    """Assemble the setting-row title: key name, type pill, and the source pill for an override."""
    parts: list[Content] = [Content(key + "  "), type_pill(defn)]
    source_badge = source_pill(key, source, profile_name)
    if source_badge is not None:
        parts.append(Content("  "))
        parts.append(source_badge)
    return Content.assemble(*parts)


def _providers_installed() -> bool:
    from lilbee.providers import availability

    return availability.provider_sdk_available()


def _crawler_installed() -> bool:
    return crawler_available()


def _wiki_enabled() -> bool:
    return bool(cfg.wiki)


_FEATURE_GATED_GROUPS: dict[SettingGroup, Callable[[], bool]] = {
    SettingGroup.API_KEYS: _providers_installed,
    SettingGroup.LOCAL_SERVERS: _providers_installed,
    SettingGroup.CRAWLING: _crawler_installed,
    SettingGroup.WIKI: _wiki_enabled,
}


def _tesseract_reads() -> bool:
    return not cfg.vision_model


# Rows shown only while their gate holds; the screen re-checks them on an OCR key change.
GATED_ROWS: dict[str, Callable[[], bool]] = {"ocr_language": _tesseract_reads}


def row_visible(key: str) -> bool:
    """Whether a setting row shows now: ocr_language only while Tesseract reads."""
    gate = GATED_ROWS.get(key)
    return gate is None or gate()


def group_settings() -> dict[SettingGroup, list[tuple[str, SettingDef]]]:
    """Group settings by group field, skipping hidden entries and gated features."""
    groups: dict[SettingGroup, list[tuple[str, SettingDef]]] = defaultdict(list)
    for key, defn in SETTINGS_MAP.items():
        if defn.hidden:
            continue
        gate = _FEATURE_GATED_GROUPS.get(defn.group)
        if gate is not None and not gate():
            continue
        groups[defn.group].append((key, defn))
    return dict(groups)


def make_editor(key: str, defn: SettingDef) -> Collapsible | SettingEditor:
    """Create the appropriate editor widget for a setting."""
    # A list never reaches an Input: str(list) would be saved back as the value.
    if defn.type is list:
        return make_list_editor(key)
    value = effective_value(key)
    if defn.choices:
        return make_select(key, defn, value)
    if defn.type is bool:
        return make_checkbox(key, value)
    if defn.render is RenderStyle.MULTILINE:
        return make_multiline_editor(key, value)
    return make_input(key, value, secret=defn.secret)


def make_multiline_editor(key: str, value: str) -> ListTextArea:
    """Create a multi-line editor for string settings (system prompts, etc.)."""
    display = strip_model_default_marker(value)
    return ListTextArea(
        text=display,
        show_line_numbers=False,
        name=key,
        id=f"{EDITOR_ID_PREFIX}{key}",
        classes="setting-editor setting-multiline-editor",
        soft_wrap=True,
    )


def make_list_editor(key: str) -> Collapsible:
    """Create a Collapsible with a line-numbered TextArea for a list setting."""
    current = getattr(cfg, key, None) or []
    title = msg.SETTINGS_LIST_EDITOR_TITLE.format(key=key, count=len(current))
    editor = ListTextArea(
        text=list_editor_text(key, current),
        show_line_numbers=True,
        name=key,
        id=f"{EDITOR_ID_PREFIX}{key}",
        classes="setting-list-editor",
    )
    error = Static(
        "", id=f"{LIST_ERROR_ID_PREFIX}{key}", classes="setting-list-error", markup=False
    )
    reset = Button(
        msg.SETTINGS_LIST_EDITOR_RESTORE_DEFAULTS,
        id=f"{LIST_RESTORE_PREFIX}{key}",
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


def make_select(key: str, defn: SettingDef, value: str) -> Select[str]:
    """Create a Select widget for choice-based settings."""
    labels = msg.SETTING_CHOICE_LABELS.get(key, {})
    choices = [(labels.get(c, c), c) for c in (defn.choices or ())]
    if value in {c[1] for c in choices}:
        return Select(
            choices,
            value=select_shown_value(defn, value),
            allow_blank=defn.nullable,
            name=key,
            classes="setting-editor",
            id=f"{EDITOR_ID_PREFIX}{key}",
        )
    return Select(choices, name=key, classes="setting-editor", id=f"{EDITOR_ID_PREFIX}{key}")


def make_checkbox(key: str, value: str) -> Checkbox:
    """Create a Checkbox widget for boolean settings."""
    checked = value.lower() in ("true", "1", "yes", "on")
    return Checkbox(
        value=checked, name=key, classes="setting-editor", id=f"{EDITOR_ID_PREFIX}{key}"
    )


def make_input(key: str, value: str, *, secret: bool = False) -> Input:
    """Create an Input widget for string/number settings.

    ``secret`` renders the field masked. Textual masks only the rendering, so
    the real value is still submitted and saved, and text pasted into a masked
    field never appears on screen.
    """
    display = strip_model_default_marker(value)
    return Input(
        value=display,
        name=key,
        password=secret,
        classes="setting-editor",
        id=f"{EDITOR_ID_PREFIX}{key}",
    )
