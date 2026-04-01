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
from lilbee.config import cfg

_MAX_VALUE_LEN = 60
_HF_TOKEN_KEY = "hf_token"
_READONLY_FIELDS = frozenset(
    {
        "data_dir",
        "lancedb_dir",
        "data_root",
        "documents_dir",
        "models_dir",
    }
)


def _get_hf_token_display() -> str:
    """Get a masked display of the HuggingFace token, or 'not set'."""
    token = os.environ.get("LILBEE_HF_TOKEN") or os.environ.get("HF_TOKEN") or ""
    if not token:
        try:
            from huggingface_hub import get_token

            token = get_token() or ""
        except Exception:
            token = ""
    if not token:
        return "not set"
    return token[:4] + "..." + token[-4:]


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
        for key, defn in SETTINGS_MAP.items():
            value = str(getattr(cfg, defn.cfg_attr, "?"))
            truncated = value[:_MAX_VALUE_LEN] + "..." if len(value) > _MAX_VALUE_LEN else value
            table.add_row(key, truncated, defn.type.__name__, key=key)
        table.add_row(_HF_TOKEN_KEY, _get_hf_token_display(), "str", key=_HF_TOKEN_KEY)
        self.query_one("#edit-bar", Horizontal).display = False

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if self._editing_key:
            return
        detail = self.query_one("#setting-detail", Static)
        if event.row_key and event.row_key.value:
            key = str(event.row_key.value)
            if key == _HF_TOKEN_KEY:
                detail.update(f"{_HF_TOKEN_KEY}\n{_get_hf_token_display()}\nUse /login to set")
                return
            defn = SETTINGS_MAP.get(key)
            if defn:
                value = str(getattr(cfg, defn.cfg_attr, "?"))
                ro = " (read-only)" if key in _READONLY_FIELDS else ""
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

        if key in _READONLY_FIELDS or key == _HF_TOKEN_KEY:
            self.notify("read-only", severity="warning")
            return

        if key not in SETTINGS_MAP:
            self.notify("read-only", severity="warning")
            return

        defn = SETTINGS_MAP[key]
        current_value = str(getattr(cfg, defn.cfg_attr, ""))
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
            if len(persisted) > _MAX_VALUE_LEN:
                display = persisted[:_MAX_VALUE_LEN] + "..."
            else:
                display = persisted
            table.update_cell(key, "value", display)
            self.notify(f"{key} = {parsed}")
        except (ValueError, TypeError) as exc:
            self.notify(f"Invalid value: {exc}", severity="error")

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
