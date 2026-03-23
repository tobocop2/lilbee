"""Settings screen — view and edit configuration."""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from lilbee.cli.settings_map import SETTINGS_MAP
from lilbee.config import cfg

_MAX_VALUE_LEN = 60


class SettingsScreen(Screen[None]):
    """Interactive settings viewer with detail panel for long values."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "pop_screen", "Back", show=True),
        Binding("escape", "pop_screen", "Back", show=False),
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

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        detail = self.query_one("#setting-detail", Static)
        if event.row_key and event.row_key.value:
            key = str(event.row_key.value)
            defn = SETTINGS_MAP.get(key)
            if defn:
                value = str(getattr(cfg, defn.cfg_attr, "?"))
                detail.update(f"{key} ({defn.type.__name__})\n{value}")
                return
        detail.update("")

    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    def key_j(self) -> None:
        self.query_one("#settings-table", DataTable).action_cursor_down()

    def key_k(self) -> None:
        self.query_one("#settings-table", DataTable).action_cursor_up()
