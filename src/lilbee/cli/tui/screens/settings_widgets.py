"""Editor-row builders and label helpers for the Settings screen."""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING

from textual.content import Content
from textual.widget import Widget
from textual.widgets import Button, Checkbox, Collapsible, Input, Select, Static, TextArea

from lilbee.app.settings_map import SETTINGS_MAP, RenderStyle, SettingDef, SettingGroup
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.pill import pill
from lilbee.cli.tui.widgets.list_text_area import ListTextArea
from lilbee.core.config import cfg

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

LIST_RESTORE_PREFIX = "list-restore-"
LIST_ERROR_ID_PREFIX = "err-"
LIST_ERROR_VISIBLE_CLASS = "-visible"

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
            widget.load_text("\n".join(value))
        else:
            widget.load_text("" if value is None else str(value))


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


def env_var_name(key: str) -> str:
    """Return the LILBEE_* env var name for a config key."""
    return f"LILBEE_{key.upper()}"


def env_pill(key: str) -> Content | None:
    """Pill warning that an env var is overriding TUI edits, or None."""
    env_name = env_var_name(key)
    if os.environ.get(env_name) is None:
        return None
    return pill(env_name, "$warning", "$text")


def help_content(_key: str, defn: SettingDef) -> Content:
    """Build help text; the editor widget already shows the current value."""
    if defn.help_text:
        return Content(defn.help_text)
    return Content("")


def title_content(key: str, defn: SettingDef) -> Content:
    """Assemble the setting-row title: key name, type pill, and env pill when set."""
    parts: list[Content] = [Content(key + "  "), type_pill(defn)]
    env_badge = env_pill(key)
    if env_badge is not None:
        parts.append(Content("  "))
        parts.append(env_badge)
    return Content.assemble(*parts)


def stringify_default(default: object) -> str:
    """Serialize a default for the TOML settings store."""
    if default is None:
        return ""
    if isinstance(default, list):
        return "\n".join(default)
    return str(default)


def _litellm_installed() -> bool:
    from lilbee.providers.litellm_sdk import litellm_available

    return litellm_available()


def _crawler_installed() -> bool:
    from lilbee.crawler import crawler_available

    return crawler_available()


def _wiki_enabled() -> bool:
    return bool(cfg.wiki)


_FEATURE_GATED_GROUPS: dict[SettingGroup, Callable[[], bool]] = {
    SettingGroup.API_KEYS: _litellm_installed,
    SettingGroup.CRAWLING: _crawler_installed,
    SettingGroup.WIKI: _wiki_enabled,
}


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


def make_editor(key: str, defn: SettingDef) -> Widget:
    """Create the appropriate editor widget for a setting."""
    if defn.render is RenderStyle.LIST_COLLAPSED:
        return make_list_editor(key)
    value = effective_value(key)
    if defn.choices:
        return make_select(key, defn, value)
    if defn.type is bool:
        return make_checkbox(key, value)
    if defn.render is RenderStyle.MULTILINE:
        return make_multiline_editor(key, value)
    return make_input(key, value)


def make_multiline_editor(key: str, value: str) -> ListTextArea:
    """Create a multi-line editor for string settings (system prompts, etc.)."""
    display = "" if value == "None" else value
    return ListTextArea(
        text=display,
        show_line_numbers=False,
        name=key,
        id=f"{EDITOR_ID_PREFIX}{key}",
        classes="setting-editor setting-multiline-editor",
        soft_wrap=True,
    )


def make_list_editor(key: str) -> Collapsible:
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
    error = Static("", id=f"{LIST_ERROR_ID_PREFIX}{key}", classes="setting-list-error")
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
    choices = [(c, c) for c in (defn.choices or ())]
    if value in {c[1] for c in choices}:
        return Select(
            choices,
            value=value,
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


def make_input(key: str, value: str) -> Input:
    """Create an Input widget for string/number settings."""
    display = "" if value == "None" else value.replace(" (model default)", "")
    return Input(value=display, name=key, classes="setting-editor", id=f"{EDITOR_ID_PREFIX}{key}")
