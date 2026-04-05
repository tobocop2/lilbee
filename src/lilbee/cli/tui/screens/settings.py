"""Settings screen — view and edit configuration."""

from __future__ import annotations

import os
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Input, Static

from lilbee import settings
from lilbee.cli.settings_map import SETTINGS_MAP
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.widgets.nav_bar import NavBar
from lilbee.config import cfg

_MAX_VALUE_LEN = 60
_TOKEN_VISIBLE_CHARS = 4  # number of leading characters shown in masked tokens
_HF_TOKEN_KEY = "hf_token"
_LOCKED_TAG = "[locked]"
_READONLY_FIELDS = frozenset(
    {
        "data_dir",
        "lancedb_dir",
        "data_root",
        "documents_dir",
        "models_dir",
    }
)

_MODEL_INFO_KEYS = (
    "chat_model_arch",
    "embed_model_arch",
    "vision_projector",
    "active_chat_handler",
)


def mask_token(token: str) -> str:
    """Mask a secret token, showing only the first few characters."""
    if not token:
        return "not set"
    visible = min(_TOKEN_VISIBLE_CHARS, len(token))
    return token[:visible] + "****"


def _get_hf_token_display() -> str:
    """Get a masked display of the HuggingFace token, or 'not set'."""
    token = os.environ.get("LILBEE_HF_TOKEN") or os.environ.get("HF_TOKEN") or ""
    if not token:
        import contextlib

        with contextlib.suppress(Exception):
            from huggingface_hub import get_token

            token = get_token() or ""
    return mask_token(token)


def _truncate(value: str) -> str:
    """Truncate a value for display in the table."""
    if len(value) > _MAX_VALUE_LEN:
        return value[:_MAX_VALUE_LEN] + "..."
    return value


def _effective_value(cfg_attr: str) -> str:
    """Return the effective value for a setting, including model defaults.

    If the user hasn't set a value (None) but a model default exists,
    returns the model default with a '(model default)' suffix.
    """
    user_value = getattr(cfg, cfg_attr, None)
    if user_value is not None:
        return str(user_value)
    defaults = cfg._model_defaults
    if defaults is None:
        return "None"
    default_map = _defaults_field_map()
    if cfg_attr in default_map:
        default_val = getattr(defaults, default_map[cfg_attr], None)
        if default_val is not None:
            return f"{default_val} (model default)"
    return "None"


def _defaults_field_map() -> dict[str, str]:
    """Map config attr names to ModelDefaults field names.

    top_k_sampling maps to top_k in ModelDefaults.
    """
    return {
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k_sampling": "top_k",
        "repeat_penalty": "repeat_penalty",
        "num_ctx": "num_ctx",
        "max_tokens": "max_tokens",
        "seed": "seed",
    }


def _get_model_info() -> dict[str, str]:
    """Collect model architecture info by reading GGUF metadata from registry."""
    info: dict[str, str] = {
        "chat_model_arch": "unknown",
        "embed_model_arch": "unknown",
        "vision_projector": "unknown",
        "active_chat_handler": "not loaded",
    }
    try:
        from lilbee.providers.llama_cpp_provider import _read_gguf_metadata, _resolve_model_path

        # Chat model
        try:
            path = _resolve_model_path(cfg.chat_model)
            meta = _read_gguf_metadata(path)
            if meta:
                info["chat_model_arch"] = meta.get("architecture", "unknown")
                info["active_chat_handler"] = "llama-cpp"
        except Exception:
            pass

        # Embedding model
        try:
            path = _resolve_model_path(cfg.embedding_model)
            meta = _read_gguf_metadata(path)
            if meta:
                info["embed_model_arch"] = meta.get("architecture", "unknown")
        except Exception:
            pass

        # Vision projector
        if cfg.vision_model:
            try:
                from lilbee.providers.llama_cpp_provider import (
                    _find_mmproj_for_model,
                    _read_mmproj_projector_type,
                )

                path = _resolve_model_path(cfg.vision_model)
                mmproj = _find_mmproj_for_model(path)
                proj_type = _read_mmproj_projector_type(mmproj)
                info["vision_projector"] = proj_type or "unknown"
            except Exception:
                pass
    except ImportError:
        pass  # llama-cpp not available (litellm-only mode)
    return info


def _is_writable(key: str) -> bool:
    """Check if a setting key is writable."""
    if key in _READONLY_FIELDS or key == _HF_TOKEN_KEY:
        return False
    if key in _MODEL_INFO_KEYS:
        return False
    defn = SETTINGS_MAP.get(key)
    if defn is None:
        return False
    return defn.writable


class SettingsScreen(Screen[None]):
    """Interactive settings viewer with inline editing."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "pop_screen", "Back", show=True),
        Binding("escape", "dismiss_or_back", "Back", show=False),
        Binding("enter", "edit_row", "Edit", show=True),
        Binding("j", "cursor_down", "Nav", show=False),
        Binding("k", "cursor_up", "Nav", show=False),
        Binding("g", "jump_top", "Top", show=False),
        Binding("G", "jump_bottom", "End", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._editing_key: str | None = None

    def compose(self) -> ComposeResult:
        yield NavBar(id="global-nav-bar")
        yield Header()
        yield DataTable(id="settings-table")
        yield Horizontal(id="edit-bar")
        yield Static("", id="setting-detail")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#settings-table", DataTable)
        table.add_columns(
            ("Setting", "setting"),
            ("Value", "value"),
            ("Type", "type"),
        )
        table.cursor_type = "row"
        self._populate_settings(table)
        self._populate_model_info(table)
        type_label = "str " + _LOCKED_TAG
        table.add_row(_HF_TOKEN_KEY, _get_hf_token_display(), type_label, key=_HF_TOKEN_KEY)
        self.query_one("#edit-bar", Horizontal).display = False

    def _populate_settings(self, table: DataTable) -> None:
        """Add config settings rows to the table."""
        for key, defn in SETTINGS_MAP.items():
            value = _effective_value(defn.cfg_attr)
            type_label = defn.type.__name__
            if not defn.writable:
                type_label += " " + _LOCKED_TAG
            table.add_row(key, _truncate(value), type_label, key=key)

    def _populate_model_info(self, table: DataTable) -> None:
        """Add model architecture info as read-only rows."""
        info = _get_model_info()
        for key in _MODEL_INFO_KEYS:
            value = info.get(key, "unknown")
            table.add_row(key, value, "str " + _LOCKED_TAG, key=key)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Open editor when a row is clicked."""
        self.action_edit_row()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if self._editing_key:
            return
        detail = self.query_one("#setting-detail", Static)
        if event.row_key and event.row_key.value:
            key = str(event.row_key.value)
            if key == _HF_TOKEN_KEY:
                detail.update(f"{_HF_TOKEN_KEY}\n{_get_hf_token_display()}\nUse /login to set")
                return
            if key in _MODEL_INFO_KEYS:
                info = _get_model_info()
                detail.update(f"{key} (read-only)\n{info.get(key, 'unknown')}")
                return
            defn = SETTINGS_MAP.get(key)
            if defn:
                value = _effective_value(defn.cfg_attr)
                ro = " (read-only)" if not defn.writable else ""
                detail.update(f"{key} ({defn.type.__name__}){ro}\n{value}")
                return
        detail.update("")

    def action_dismiss_or_back(self) -> None:
        """Escape: cancel edit if editing, otherwise pop screen."""
        if self._editing_key is not None:
            self._close_editor()
        else:
            self.app.pop_screen()

    def action_edit_row(self) -> None:
        """Enter: open inline editor for the highlighted row."""
        if self._editing_key is not None:
            return
        table = self.query_one("#settings-table", DataTable)
        row_idx = table.cursor_row
        if row_idx is None:
            return
        key_cell = table.get_row_at(row_idx)
        key = str(key_cell[0])

        if not _is_writable(key):
            self.notify(msg.SETTINGS_READ_ONLY, severity="warning")
            return

        defn = SETTINGS_MAP[key]
        current_value = str(getattr(cfg, defn.cfg_attr, ""))
        if current_value == "None":
            current_value = ""
        self._editing_key = key

        bar = self.query_one("#edit-bar", Horizontal)
        bar.display = True
        editor = Input(
            value=current_value,
            placeholder=f"Enter value for {key}",
            id="settings-editor",
        )
        bar.mount(Static(f"  {key}: ", id="edit-label"))
        bar.mount(editor)
        editor.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Save edited value on Enter."""
        if event.input.id != "settings-editor" or self._editing_key is None:
            return
        key = self._editing_key
        new_value = event.value.strip()
        defn = SETTINGS_MAP[key]

        try:
            if defn.type is bool:
                parsed = new_value.lower() in ("true", "1", "yes", "on")
            elif defn.nullable and new_value.lower() in ("none", "null", ""):
                parsed = None
            else:
                parsed = defn.type(new_value)
            setattr(cfg, defn.cfg_attr, parsed)
            persisted = str(parsed) if parsed is not None else ""
            settings.set_value(cfg.data_root, defn.cfg_attr, persisted)

            table = self.query_one("#settings-table", DataTable)
            display = _truncate(persisted)
            table.update_cell(key, "value", display)
            self.notify(msg.CMD_SET_SUCCESS.format(key=key, value=parsed))
        except (ValueError, TypeError) as exc:
            self.notify(msg.SETTINGS_INVALID_VALUE.format(error=exc), severity="error")

        self._close_editor()

    def _close_editor(self) -> None:
        """Remove the inline editor and restore table focus."""
        self._editing_key = None
        bar = self.query_one("#edit-bar", Horizontal)
        bar.remove_children()
        bar.display = False
        self.query_one("#settings-table", DataTable).focus()

    def action_pop_screen(self) -> None:
        if self._editing_key:
            self._close_editor()
        else:
            self.app.pop_screen()

    def action_cursor_down(self) -> None:
        self.query_one("#settings-table", DataTable).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one("#settings-table", DataTable).action_cursor_up()

    def action_jump_top(self) -> None:
        self.query_one("#settings-table", DataTable).move_cursor(row=0)

    def action_jump_bottom(self) -> None:
        table = self.query_one("#settings-table", DataTable)
        table.move_cursor(row=table.row_count - 1)
