"""Settings screen — view and edit configuration."""

from __future__ import annotations

import os
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from lilbee.cli.settings_map import SETTINGS_MAP
from lilbee.config import cfg

_MAX_VALUE_LEN = 60
_HF_TOKEN_KEY = "hf_token"


def _get_hf_token_display() -> str:
    """Get a masked display of the HuggingFace token, or 'not set'."""
    token = os.environ.get("LILBEE_HF_TOKEN") or os.environ.get("HF_TOKEN") or ""
    if not token:
        try:
            from huggingface_hub import HfFolder

            token = HfFolder.get_token() or ""
        except Exception:
            token = ""
    if not token:
        return "not set"
    return token[:4] + "..." + token[-4:]


class SettingsScreen(Screen[None]):
    """Interactive settings viewer with detail panel for long values."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "pop_screen", "Back", show=True),
        Binding("escape", "pop_screen", "Back", show=False),
        Binding("j", "cursor_down", "Nav", show=False),
        Binding("k", "cursor_up", "Nav", show=False),
        Binding("g", "jump_top", "Top", show=False),
        Binding("G", "jump_bottom", "End", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="settings-table")
        yield Static("", id="setting-detail")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#settings-table", DataTable)
        table.add_columns("Setting", "Value", "Type")
        table.cursor_type = "row"
        for key, defn in SETTINGS_MAP.items():
            value = str(getattr(cfg, defn.cfg_attr, "?"))
            display = value[:_MAX_VALUE_LEN] + "..." if len(value) > _MAX_VALUE_LEN else value
            table.add_row(key, display, defn.type.__name__, key=key)
        table.add_row(_HF_TOKEN_KEY, _get_hf_token_display(), "str", key=_HF_TOKEN_KEY)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        detail = self.query_one("#setting-detail", Static)
        if event.row_key and event.row_key.value:
            key = str(event.row_key.value)
            if key == _HF_TOKEN_KEY:
                detail.update(
                    f"{_HF_TOKEN_KEY} (str)\n{_get_hf_token_display()}\n"
                    "Use /login <token> to set your HuggingFace token"
                )
                return
            defn = SETTINGS_MAP.get(key)
            if defn:
                value = str(getattr(cfg, defn.cfg_attr, "?"))
                detail.update(f"{key} ({defn.type.__name__})\n{value}")
                return
        detail.update("")

    def action_pop_screen(self) -> None:
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
