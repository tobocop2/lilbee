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
from textual.widgets import Collapsible, DataTable, Static

from lilbee.cli.tui.pill import pill
from lilbee.config import cfg
from lilbee.model_info import ModelArchInfo, get_model_architecture
from lilbee.services import get_services
from lilbee.store import SourceRecord

log = logging.getLogger(__name__)


def _model_pill(name: str) -> Content:
    """Return a green 'loaded' pill if name is set, red 'not set' otherwise."""
    if name:
        return pill("loaded", "$success", "$text")
    return pill("not set", "$error", "$text")


# Label-column width used across the status sections so keys line up
# when scanned vertically. Values past this column render bold.
_KV_LABEL_WIDTH = 14


def _kv_line(label: str, value: str | Content, status: Content | None = None) -> Content:
    """Assemble one key/value row: dim padded label, bold value, optional pill."""
    padded = label.ljust(_KV_LABEL_WIDTH)
    parts: list[Content] = [Content.styled(padded, "$text-muted")]
    if isinstance(value, Content):
        parts.append(value)
    else:
        parts.append(Content.styled(value, "bold"))
    if status is not None:
        parts.append(Content("  "))
        parts.append(status)
    return Content.assemble(*parts)


def _collapse_home(path: Path | str) -> str:
    """Replace the user's home prefix with '~' so long paths stay scannable."""
    text = str(path)
    home = str(Path.home())
    return text.replace(home, "~", 1) if text.startswith(home) else text


def _ocr_label() -> str:
    """Return a human-readable OCR status string."""
    if cfg.enable_ocr is True:
        return "enabled"
    if cfg.enable_ocr is False:
        return "disabled"
    return "auto"


def _ocr_pill() -> Content:
    """Return a pill reflecting OCR status."""
    if cfg.enable_ocr is True:
        return pill("on", "$success", "$text")
    if cfg.enable_ocr is False:
        return pill("off", "$warning", "$text")
    return pill("auto", "$accent", "$text")


def _data_dir_pill() -> Content:
    """Return a pill based on whether the data directory exists."""
    if Path(cfg.data_dir).exists():
        return pill("exists", "$success", "$text")
    return pill("missing", "$error", "$text")


def _build_config_content() -> Content:
    """Build the configuration section content."""
    lines = [
        _kv_line("Data dir", _collapse_home(cfg.data_dir), _data_dir_pill()),
        _kv_line("Chat model", cfg.chat_model or "—", _model_pill(cfg.chat_model)),
        _kv_line("Embed model", cfg.embedding_model or "—", _model_pill(cfg.embedding_model)),
        _kv_line("OCR", _ocr_label(), _ocr_pill()),
    ]
    return Content("\n").join(lines)


def _build_storage_content(doc_count: int) -> Content:
    """Build the storage section content."""
    lines = [
        _kv_line("Documents", str(doc_count)),
        _kv_line("Data dir", _collapse_home(cfg.data_dir)),
        _kv_line("Models dir", _collapse_home(cfg.models_dir)),
    ]
    return Content("\n").join(lines)


def _build_arch_content(info: ModelArchInfo) -> Content:
    """Build the model architecture section from GGUF metadata."""
    lines = [
        _kv_line("Chat arch", info.chat_arch),
        _kv_line("Embed arch", info.embed_arch),
        _kv_line("Handler", pill(info.active_handler, "$accent", "$text")),
    ]
    if info.vision_projector:
        lines.append(_kv_line("Vision proj", info.vision_projector))
    return Content("\n").join(lines)


class StatusScreen(Screen[None]):
    """Knowledge base status view with collapsible sections."""

    CSS_PATH = "status.tcss"
    AUTO_FOCUS = "CollapsibleTitle"
    HELP = (
        "Knowledge base status.\n\n"
        "View configuration, documents, model architecture, and storage info."
    )

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "go_back", "Back", show=True),
        Binding("escape", "go_back", "Back", show=False),
        Binding("tab", "app.focus_next", "Next section", show=True),
        Binding("shift+tab", "app.focus_previous", "Prev section", show=True),
        Binding("j", "cursor_down", "Nav", show=False),
        Binding("k", "cursor_up", "Nav", show=False),
        Binding("g", "jump_top", "Top", show=False),
        Binding("G", "jump_bottom", "End", show=False),
    ]

    def compose(self) -> ComposeResult:
        from textual.widgets import Footer

        from lilbee.cli.tui.widgets.status_bar import ViewTabs
        from lilbee.cli.tui.widgets.task_bar import TaskBar

        yield VerticalScroll(
            Collapsible(Static(id="config-info"), title="Configuration", id="config-section"),
            Collapsible(DataTable(id="docs-table"), title="Documents", id="docs-section"),
            Collapsible(Static(id="arch-info"), title="Model Architecture", id="arch-section"),
            Collapsible(Static(id="storage-info"), title="Storage", id="storage-section"),
            id="status-scroll",
        )
        yield TaskBar()
        yield ViewTabs()
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

    def action_go_back(self) -> None:
        from lilbee.cli.tui.app import LilbeeApp

        if isinstance(self.app, LilbeeApp):  # test apps aren't LilbeeApp
            self.app.switch_view("Chat")
        else:
            self.app.pop_screen()

    def action_cursor_down(self) -> None:
        self.query_one("#status-scroll", VerticalScroll).scroll_down()

    def action_cursor_up(self) -> None:
        self.query_one("#status-scroll", VerticalScroll).scroll_up()

    def action_jump_top(self) -> None:
        self.query_one("#status-scroll", VerticalScroll).scroll_home()

    def action_jump_bottom(self) -> None:
        self.query_one("#status-scroll", VerticalScroll).scroll_end()
