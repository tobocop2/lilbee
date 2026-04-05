"""Status screen — knowledge base info with collapsible sections."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import VerticalScroll
from textual.content import Content
from textual.screen import Screen
from textual.widgets import Collapsible, DataTable, Footer, Header, Static

from lilbee.cli.tui.pill import pill
from lilbee.cli.tui.widgets.nav_bar import NavBar
from lilbee.config import cfg
from lilbee.model_info import ModelArchInfo, get_model_architecture
from lilbee.store import SourceRecord

log = logging.getLogger(__name__)


def _model_pill(name: str) -> Content:
    """Return a green 'loaded' pill if name is set, red 'not set' otherwise."""
    if name:
        return pill("loaded", "$success", "$text")
    return pill("not set", "$error", "$text")


def _config_line(label: str, value: str, status: Content) -> Content:
    """Assemble a single config info line with label, value, and status pill."""
    return Content.assemble(
        (f"{label}: ", "bold"),
        value,
        " ",
        status,
    )


def _data_dir_pill() -> Content:
    """Return a pill based on whether the data directory exists."""
    if Path(cfg.data_dir).exists():
        return pill("exists", "$success", "$text")
    return pill("missing", "$error", "$text")


def _build_config_content() -> Content:
    """Build the configuration section content."""
    lines = [
        _config_line("Data dir", str(cfg.data_dir), _data_dir_pill()),
        _config_line("Chat model", cfg.chat_model, _model_pill(cfg.chat_model)),
        _config_line("Embed model", cfg.embedding_model, _model_pill(cfg.embedding_model)),
        _config_line("Vision model", cfg.vision_model or "(none)", _model_pill(cfg.vision_model)),
    ]
    return Content("\n").join(lines)


def _build_storage_content(doc_count: int) -> Content:
    """Build the storage section content."""
    lines = [
        Content.assemble(("Documents: ", "bold"), str(doc_count)),
        Content.assemble(("Data dir: ", "bold"), str(cfg.data_dir)),
        Content.assemble(("Models dir: ", "bold"), str(cfg.models_dir)),
    ]
    return Content("\n").join(lines)


def _build_arch_content(info: ModelArchInfo) -> Content:
    """Build the model architecture section from GGUF metadata."""
    lines = [
        Content.assemble(("Chat arch: ", "bold"), info.chat_arch),
        Content.assemble(("Embed arch: ", "bold"), info.embed_arch),
        Content.assemble(("Handler: ", "bold"), info.active_handler),
    ]
    if cfg.vision_model:
        lines.append(Content.assemble(("Vision proj: ", "bold"), info.vision_projector))
    return Content("\n").join(lines)


class StatusScreen(Screen[None]):
    """Knowledge base status view with collapsible sections."""

    CSS_PATH = "status.tcss"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "pop_screen", "Back", show=True),
        Binding("escape", "pop_screen", "Back", show=False),
        Binding("j", "cursor_down", "Nav", show=False),
        Binding("k", "cursor_up", "Nav", show=False),
        Binding("g", "jump_top", "Top", show=False),
        Binding("G", "jump_bottom", "End", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield NavBar(id="global-nav-bar")
        yield Header()
        yield VerticalScroll(
            Collapsible(Static(id="config-info"), title="Configuration", id="config-section"),
            Collapsible(DataTable(id="docs-table"), title="Documents", id="docs-section"),
            Collapsible(Static(id="arch-info"), title="Model Architecture", id="arch-section"),
            Collapsible(Static(id="storage-info"), title="Storage", id="storage-section"),
            id="status-scroll",
        )
        yield Footer()

    def on_mount(self) -> None:
        self._load_config()
        sources = self._fetch_sources()
        self._load_documents(sources)
        self._load_arch()
        self._load_storage(len(sources))

    def _fetch_sources(self) -> list[SourceRecord]:
        """Fetch sources once from the store."""
        try:
            from lilbee.services import get_services

            return get_services().store.get_sources()
        except Exception:
            log.debug("Failed to read store for status screen", exc_info=True)
            return []

    def _load_config(self) -> None:
        """Populate the configuration section."""
        self.query_one("#config-info", Static).update(_build_config_content())

    def _load_documents(self, sources: list[SourceRecord]) -> None:
        """Populate the documents table."""
        table = self.query_one("#docs-table", DataTable)
        table.add_columns("Document", "Chunks")
        table.cursor_type = "row"
        self._fill_doc_rows(table, sources)

    def _fill_doc_rows(self, table: DataTable, sources: list[SourceRecord]) -> None:
        """Fill the documents table with source data."""
        if not sources:
            table.add_row("(unable to read store)", "")
            return
        for src in sources:
            table.add_row(src.get("filename", "?"), str(src.get("chunk_count", 0)))

    def _load_arch(self) -> None:
        """Populate the model architecture section."""
        info = get_model_architecture()
        self.query_one("#arch-info", Static).update(_build_arch_content(info))

    def _load_storage(self, doc_count: int) -> None:
        """Populate the storage section."""
        self.query_one("#storage-info", Static).update(_build_storage_content(doc_count))

    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    def action_cursor_down(self) -> None:
        self.query_one("#docs-table", DataTable).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one("#docs-table", DataTable).action_cursor_up()

    def action_jump_top(self) -> None:
        self.query_one("#docs-table", DataTable).move_cursor(row=0)

    def action_jump_bottom(self) -> None:
        table = self.query_one("#docs-table", DataTable)
        table.move_cursor(row=table.row_count - 1)
