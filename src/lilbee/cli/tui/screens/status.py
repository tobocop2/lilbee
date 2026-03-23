"""Status screen — knowledge base info and document listing."""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from lilbee.config import cfg


class StatusScreen(Screen[None]):
    """Knowledge base status view."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "pop_screen", "Back", show=True),
        Binding("escape", "pop_screen", "Back", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="status-info")
        yield DataTable(id="docs-table")
        yield Footer()

    def on_mount(self) -> None:
        self._load_status()

    def _load_status(self) -> None:
        info_parts = [
            f"Data: {cfg.data_dir}",
            f"Model: {cfg.chat_model}",
            f"Embedding: {cfg.embedding_model}",
            f"Chunk size: {cfg.chunk_size}",
        ]
        self.query_one("#status-info", Static).update("\n".join(info_parts))

        table = self.query_one("#docs-table", DataTable)
        table.add_columns("Document", "Chunks", "Type")
        table.cursor_type = "row"

        try:
            from lilbee import store

            sources = store.get_sources()
            for src in sources:
                table.add_row(
                    src.get("source", "?"),
                    str(src.get("chunk_count", 0)),
                    src.get("content_type", "?"),
                )
        except Exception:
            table.add_row("(unable to read store)", "", "")

    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    def key_j(self) -> None:
        self.query_one("#docs-table", DataTable).action_cursor_down()

    def key_k(self) -> None:
        self.query_one("#docs-table", DataTable).action_cursor_up()
