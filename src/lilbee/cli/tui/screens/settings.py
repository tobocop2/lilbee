"""Settings screen — view and edit configuration."""

from __future__ import annotations

import os
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Input, Static

from lilbee.cli.settings_map import SETTINGS_MAP
from lilbee.config import cfg

_MAX_VALUE_LEN = 60
_HF_TOKEN_KEY = "hf_token"
_READ_ONLY = {"data_dir", "lancedb_dir", "data_root", "documents_dir", "models_dir"}


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
    """Interactive settings editor. Press Enter to edit a value."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "pop_screen", "Back", show=True),
        Binding("escape", "cancel_or_back", "Back", show=False),
        Binding("enter", "edit_setting", "Edit", show=True),
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
        yield Input(placeholder="New value...", id="edit-input")
        yield Static("", id="setting-detail")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#edit-input", Input).display = False
        table = self.query_one("#settings-table", DataTable)
        table.add_columns("Setting", "Value", "Type")
        table.cursor_type = "row"
        for key, defn in SETTINGS_MAP.items():
            value = str(getattr(cfg, defn.cfg_attr, "?"))
            display = value[:_MAX_VALUE_LEN] + "..." if len(value) > _MAX_VALUE_LEN else value
            table.add_row(key, display, defn.type.__name__, key=key)
        table.add_row(_HF_TOKEN_KEY, _get_hf_token_display(), "str", key=_HF_TOKEN_KEY)

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
                ro = " (read-only)" if key in _READ_ONLY else ""
                detail.update(f"{key} ({defn.type.__name__}){ro}\n{value}")
                return
        detail.update("")

    def action_edit_setting(self) -> None:
        """Open inline editor for the highlighted setting."""
        table = self.query_one("#settings-table", DataTable)
        row_key = table.cursor_row
        if row_key is None:
            return
        # Get the key from the row
        try:
            cells = table.get_row_at(row_key)
            key = str(cells[0])
        except Exception:
            return

        if key == _HF_TOKEN_KEY:
            self.notify("Use /login to set HuggingFace token")
            return
        if key in _READ_ONLY:
            self.notify(f"{key} is read-only", severity="warning")
            return

        defn = SETTINGS_MAP.get(key)
        if not defn:
            return

        self._editing_key = key
        current = str(getattr(cfg, defn.cfg_attr, ""))
        edit_input = self.query_one("#edit-input", Input)
        edit_input.value = current
        edit_input.display = True
        edit_input.focus()
        self.query_one("#setting-detail", Static).update(f"Editing: {key} ({defn.type.__name__})")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Save the edited value."""
        if event.input.id != "edit-input" or not self._editing_key:
            return
        key = self._editing_key
        value = event.value.strip()

        from lilbee import settings

        try:
            settings.set_value(cfg.data_root, key, value)
            defn = SETTINGS_MAP[key]
            setattr(cfg, defn.cfg_attr, defn.type(value))
            self.notify(f"{key} = {value}")
        except Exception as exc:
            self.notify(f"Error: {exc}", severity="error")

        self._finish_edit()
        self._refresh_table()

    def _finish_edit(self) -> None:
        """Close the edit input and return focus to table."""
        self._editing_key = None
        edit_input = self.query_one("#edit-input", Input)
        edit_input.display = False
        edit_input.value = ""
        self.query_one("#settings-table", DataTable).focus()

    def _refresh_table(self) -> None:
        """Rebuild table with current values."""
        table = self.query_one("#settings-table", DataTable)
        table.clear()
        for key, defn in SETTINGS_MAP.items():
            value = str(getattr(cfg, defn.cfg_attr, "?"))
            display = value[:_MAX_VALUE_LEN] + "..." if len(value) > _MAX_VALUE_LEN else value
            table.add_row(key, display, defn.type.__name__, key=key)
        table.add_row(_HF_TOKEN_KEY, _get_hf_token_display(), "str", key=_HF_TOKEN_KEY)

    def action_cancel_or_back(self) -> None:
        """Escape: cancel edit if editing, otherwise go back."""
        if self._editing_key:
            self._finish_edit()
        else:
            self.app.pop_screen()

    def action_pop_screen(self) -> None:
        if self._editing_key:
            self._finish_edit()
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
