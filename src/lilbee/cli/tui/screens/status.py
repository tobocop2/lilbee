"""Status screen: knowledge base info with collapsible sections."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from lilbee.cli.tui.app import LilbeeApp

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import VerticalScroll
from textual.content import Content
from textual.screen import Screen
from textual.widgets import Collapsible, DataTable, Static
from textual.worker import Worker, WorkerState

from lilbee.app.services import get_services
from lilbee.app.settings import SCANNED_PAGES_LABEL, scanned_pages_state
from lilbee.app.status import held_out_sources
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.browse_bindings import BROWSE_LIST_BINDINGS, browse_back_bindings
from lilbee.cli.tui.pill import pill
from lilbee.core.config import cfg
from lilbee.data.store import DOCUMENTS_PAGE_DEFAULT, SourceRecord
from lilbee.data.types import SkippedSource
from lilbee.modelhub.model_info import ModelArchInfo, get_model_architecture

log = logging.getLogger(__name__)

# Rows per page of the Documents table. Must exceed the table's visible height
# (``max-height`` in status.tcss) or the table never scrolls and never asks for more.
_DOCS_PAGE_SIZE = DOCUMENTS_PAGE_DEFAULT
# Load the next page when the table is scrolled this far down, or when the cursor
# reaches this far into the loaded rows. A cursor move scrolls the table only after
# its layout settles, so the cursor carries its own trigger.
_DOCS_SCROLL_PREFETCH_RATIO = 0.85
_WORKER_FETCH_SOURCES = "status_fetch_sources"
_WORKER_FETCH_DOCS_PAGE = "status_fetch_docs_page"


@dataclass
class _DocsResult:
    """Outcome of the background sources read.

    ``load_failed`` distinguishes "store opened but empty" (``sources == []``)
    from "the read raised" so the UI shows the right placeholder.
    """

    sources: list[SourceRecord]
    load_failed: bool
    document_count: int = 0
    held_out: list[SkippedSource] = field(default_factory=list)
    held_out_total: int = 0


@dataclass
class _DocsPage:
    """One further page of sources, or the fact that reading it failed."""

    sources: list[SourceRecord]
    load_failed: bool = False


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


def _data_dir_pill() -> Content:
    """Return a pill based on whether the data directory exists."""
    if Path(cfg.data_dir).exists():
        return pill("exists", "$success", "$text")
    return pill("missing", "$error", "$text")


def _build_config_content() -> Content:
    """Build the configuration section content."""
    lines = [
        _kv_line("Data dir", _collapse_home(cfg.data_dir), _data_dir_pill()),
        _kv_line("Chat model", cfg.chat_model or "(disabled)", _model_pill(cfg.chat_model)),
        _kv_line(
            "Embed model", cfg.embedding_model or "(disabled)", _model_pill(cfg.embedding_model)
        ),
        _kv_line("Vision model", cfg.vision_model or "(disabled)", _model_pill(cfg.vision_model)),
        _kv_line("Reranker", cfg.reranker_model or "(disabled)", _model_pill(cfg.reranker_model)),
        _kv_line(SCANNED_PAGES_LABEL, scanned_pages_state()),
    ]
    return Content("\n").join(lines)


def _build_storage_content(doc_count: int) -> Content:
    """Build the storage section content."""
    lines = [
        _kv_line("Documents", str(doc_count)),
        _kv_line("Data dir", _collapse_home(cfg.data_dir)),
        _kv_line("Models dir", _collapse_home(cfg.models_dir)),
    ]
    if cfg.entity_extraction:
        from lilbee.app.status import entity_status

        section = entity_status()
        if section is not None:
            names = ", ".join(section.types) or "schema pending"
            lines.append(_kv_line("Entities", f"{section.rows} extracted ({names})"))
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

    app: LilbeeApp  # type: ignore[assignment]

    CSS_PATH = "status.tcss"
    AUTO_FOCUS = "CollapsibleTitle"
    HELP = (
        "Knowledge base status.\n\n"
        "View configuration, documents, model architecture, and storage info."
    )

    BINDINGS: ClassVar[list[BindingType]] = [
        *browse_back_bindings(),
        Binding("tab", "app.focus_next", "Next section", show=False),
        Binding(
            "shift+tab",
            "app.focus_previous",
            "Prev section",
            show=False,
        ),
        *BROWSE_LIST_BINDINGS,
    ]

    def __init__(self) -> None:
        super().__init__()
        self._sections_mounted: bool = False
        self._pending_docs: _DocsResult | None = None
        self._pending_arch: ModelArchInfo | None = None
        self._docs_loaded: int = 0
        self._docs_total: int = 0
        self._docs_has_more: bool = False
        self._docs_loading: bool = False
        self._docs_page_failed: bool = False
        self._docs_scroll_watched: bool = False

    def compose(self) -> ComposeResult:
        from textual.widgets import Footer

        from lilbee.cli.tui.widgets.bottom_bars import BottomBars
        from lilbee.cli.tui.widgets.status_bar import ViewTabs
        from lilbee.cli.tui.widgets.task_bar import TaskBar
        from lilbee.cli.tui.widgets.top_bars import TopBars

        with TopBars():
            yield ViewTabs()
        # Mount only the first (Configuration) collapsible up front so the
        # screen paints fast on push. Documents/arch/storage hydrate via
        # ``call_after_refresh`` once the screen is visible -- their
        # backing widgets are still cheap to mount, but the synchronous
        # cost of mounting all four under a single VerticalScroll spiked
        # screen-switch latency to ~1s on cold caches.
        yield VerticalScroll(
            Collapsible(
                Static(id="config-info"),
                title="Configuration",
                id="config-section",
                collapsed=False,
            ),
            id="status-scroll",
        )
        with BottomBars():
            yield TaskBar()
            yield Footer()

    def on_mount(self) -> None:
        # ``cfg`` reads are in-memory and cheap. Anything that touches
        # disk runs in a worker so the screen paints instantly.
        # ``get_model_architecture`` opens up to three GGUF files and
        # parses their headers (~hundreds of ms each cold); ``get_sources``
        # reads LanceDB (seconds on cold caches).
        self._load_config()
        self.call_after_refresh(self._mount_remaining_sections)
        self._fetch_sources_worker()
        self._fetch_arch_worker()

    async def _mount_remaining_sections(self) -> None:
        """Mount Documents/Architecture/Storage once the screen is visible."""
        if not self.is_mounted:
            return
        scroll = self.query_one("#status-scroll", VerticalScroll)
        await scroll.mount_all(
            [
                Collapsible(
                    DataTable(id="docs-table"),
                    Static(id="docs-more"),
                    title=msg.STATUS_DOCS_TITLE,
                    id="docs-section",
                    collapsed=False,
                ),
                Collapsible(
                    DataTable(id="held-out-table"),
                    title=msg.STATUS_HELD_OUT_TITLE,
                    id="held-out-section",
                    collapsed=True,
                ),
                Collapsible(
                    Static(id="arch-info"),
                    title="Model Architecture",
                    id="arch-section",
                    collapsed=False,
                ),
                Collapsible(
                    Static(id="storage-info"),
                    title="Storage",
                    id="storage-section",
                    collapsed=False,
                ),
            ]
        )
        self._sections_mounted = True
        # Yield once so Textual gets a chance to compose the freshly-
        # mounted Collapsibles' children. Without this, querying
        # #docs-table immediately after mount_all races on Windows.
        await asyncio.sleep(0)
        self._show_loading_placeholders()
        # Replay any worker callbacks that arrived before the deferred
        # mount completed.
        if self._pending_docs is not None:
            self._apply_docs(self._pending_docs)
            self._pending_docs = None
        if self._pending_arch is not None:
            self._load_arch(self._pending_arch)
            self._pending_arch = None

    def _show_loading_placeholders(self) -> None:
        """Surface a 'Loading…' marker for sections backed by workers.

        Wrapped in NoMatches suppression because Collapsible children
        compose on the next refresh tick, which on Windows can outlast
        the synchronous return from mount_all. Worker callbacks repaint
        the same widgets when they arrive, so a missed placeholder is
        only a brief cosmetic gap.
        """
        from textual.css.query import NoMatches

        with contextlib.suppress(NoMatches):
            table = self.query_one("#docs-table", DataTable)
            table.add_columns("Document", "Chunks")
            table.cursor_type = "row"
            table.add_row("Loading...", "")
        with contextlib.suppress(NoMatches):
            self.query_one("#storage-info", Static).update(
                Content.styled("Loading...", "$text-muted")
            )
        with contextlib.suppress(NoMatches):
            self.query_one("#arch-info", Static).update(Content.styled("Loading...", "$text-muted"))

    @work(thread=True, name=_WORKER_FETCH_SOURCES, exit_on_error=False)
    def _fetch_sources_worker(self) -> _DocsResult:
        """Read the source count and the first page off the UI thread.

        ``load_failed`` is True only when the store read actually raised;
        an empty store with zero documents is the routine first-run state.
        The read is one count and one page, so a library of millions of sources
        costs the same screen as a library of ten.
        """
        try:
            held_out, held_out_total = held_out_sources()
            store = get_services().store
            return _DocsResult(
                sources=store.get_sources(limit=_DOCS_PAGE_SIZE, offset=0),
                load_failed=False,
                document_count=store.count_sources(),
                held_out=held_out,
                held_out_total=held_out_total,
            )
        except Exception:
            log.debug("Failed to read store for status screen", exc_info=True)
            return _DocsResult(sources=[], load_failed=True)

    @work(thread=True, name=_WORKER_FETCH_DOCS_PAGE, exit_on_error=False)
    def _fetch_docs_page_worker(self, offset: int) -> _DocsPage:
        """Read one further page of sources off the UI thread."""
        try:
            page = get_services().store.get_sources(limit=_DOCS_PAGE_SIZE, offset=offset)
            return _DocsPage(sources=page)
        except Exception:
            log.debug("Failed to read a page of sources for status screen", exc_info=True)
            return _DocsPage(sources=[], load_failed=True)

    @work(thread=True, name="status_fetch_arch", exit_on_error=False)
    def _fetch_arch_worker(self) -> ModelArchInfo:
        try:
            return get_model_architecture()
        except Exception:
            log.debug("Failed to read model architecture for status", exc_info=True)
            return ModelArchInfo()

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.state != WorkerState.SUCCESS:
            return
        if event.worker.name == _WORKER_FETCH_DOCS_PAGE:
            page = event.worker.result
            self._append_docs_page(page if isinstance(page, _DocsPage) else _DocsPage([], True))
        elif event.worker.name == _WORKER_FETCH_SOURCES:
            result = event.worker.result
            docs = result if isinstance(result, _DocsResult) else _DocsResult([], True)
            if self._sections_mounted:
                self._apply_docs(docs)
            else:
                self._pending_docs = docs
        elif event.worker.name == "status_fetch_arch":
            arch = event.worker.result
            if isinstance(arch, ModelArchInfo):
                if self._sections_mounted:
                    self._load_arch(arch)
                else:
                    self._pending_arch = arch

    def _apply_docs(self, docs: _DocsResult) -> None:
        """Render *docs* into the Documents table + storage section."""
        self._load_documents(docs)
        self._load_held_out(docs)
        self._load_storage(docs.document_count)

    def _load_held_out(self, docs: _DocsResult) -> None:
        """Fill the held-out table; the section opens only when something is held out."""
        from textual.css.query import NoMatches

        with contextlib.suppress(NoMatches):
            table = self.query_one("#held-out-table", DataTable)
            table.clear(columns=True)
            table.add_columns("File", "Reason")
            if not docs.held_out:
                table.add_row(msg.STATUS_HELD_OUT_EMPTY, "")
                return
            for held in docs.held_out:
                table.add_row(Text(held.filename), Text(held.reason))
            hidden = docs.held_out_total - len(docs.held_out)
            if hidden > 0:
                table.add_row(msg.STATUS_HELD_OUT_MORE.format(count=hidden), "")
            self.query_one("#held-out-section", Collapsible).collapsed = False

    def _load_arch(self, info: ModelArchInfo) -> None:
        """Populate the model architecture section from worker result."""
        from textual.css.query import NoMatches

        with contextlib.suppress(NoMatches):
            self.query_one("#arch-info", Static).update(_build_arch_content(info))

    def _load_config(self) -> None:
        """Populate the configuration section."""
        self.query_one("#config-info", Static).update(_build_config_content())

    def _load_documents(self, docs: _DocsResult) -> None:
        """Fill the table with the first page of sources and arm the paging."""
        from textual.css.query import NoMatches

        with contextlib.suppress(NoMatches):
            table = self.query_one("#docs-table", DataTable)
            table.clear()
            self._docs_loaded = len(docs.sources)
            self._docs_total = docs.document_count
            self._docs_has_more = self._docs_loaded < self._docs_total
            self._docs_loading = False
            self._docs_page_failed = False
            if not self._docs_scroll_watched:
                self.watch(table, "scroll_y", self._on_docs_scrolled, init=False)
                self._docs_scroll_watched = True
            self._show_docs_hint()
            if not docs.sources:
                placeholder = (
                    msg.STATUS_DOCS_LOAD_FAILED if docs.load_failed else msg.STATUS_DOCS_EMPTY
                )
                table.add_row(placeholder, "")
                return
            for src in docs.sources:
                table.add_row(Text(src["filename"]), str(src["chunk_count"]))

    def _append_docs_page(self, page: _DocsPage) -> None:
        """Add a further page of sources under the rows already shown."""
        from textual.css.query import NoMatches

        self._docs_loading = False
        with contextlib.suppress(NoMatches):
            table = self.query_one("#docs-table", DataTable)
            for src in page.sources:
                table.add_row(Text(src["filename"]), str(src["chunk_count"]))
            self._docs_loaded += len(page.sources)
            self._docs_page_failed = page.load_failed
            # An empty page means the library shrank under the count; stop asking.
            self._docs_has_more = bool(page.sources) and self._docs_loaded < self._docs_total
            self._show_docs_hint()

    def _show_docs_hint(self) -> None:
        """Say how many sources are loaded while more remain, and that a read failed."""
        hint = self.query_one("#docs-more", Static)
        if self._docs_page_failed:
            hint.update(msg.STATUS_DOCS_LOAD_FAILED)
        elif self._docs_has_more:
            hint.update(msg.STATUS_DOCS_MORE.format(count=self._docs_loaded))
        hint.display = self._docs_page_failed or self._docs_has_more

    def _load_more_docs(self) -> None:
        """Start the next page unless one is already loading or none is left."""
        if self._docs_loading or not self._docs_has_more:
            return
        self._docs_loading = True
        self._fetch_docs_page_worker(self._docs_loaded)

    def _on_docs_scrolled(self, _scroll_y: float) -> None:
        """Load the next page when the user scrolls near the bottom of the table."""
        table = self.query_one("#docs-table", DataTable)
        max_y = table.max_scroll_y
        if max_y > 0 and table.scroll_y / max_y >= _DOCS_SCROLL_PREFETCH_RATIO:
            self._load_more_docs()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Load the next page when the cursor nears the last loaded source."""
        near_end = event.cursor_row >= self._docs_loaded * _DOCS_SCROLL_PREFETCH_RATIO
        if event.data_table.id == "docs-table" and near_end:
            self._load_more_docs()

    def _load_storage(self, doc_count: int) -> None:
        """Populate the storage section."""
        from textual.css.query import NoMatches

        with contextlib.suppress(NoMatches):
            self.query_one("#storage-info", Static).update(_build_storage_content(doc_count))

    def action_go_back(self) -> None:
        self.app.go_back()

    def action_cursor_down(self) -> None:
        self.query_one("#status-scroll", VerticalScroll).scroll_down()

    def action_cursor_up(self) -> None:
        self.query_one("#status-scroll", VerticalScroll).scroll_up()

    def action_jump_top(self) -> None:
        self.query_one("#status-scroll", VerticalScroll).scroll_home(animate=False)

    def action_jump_bottom(self) -> None:
        self.query_one("#status-scroll", VerticalScroll).scroll_end(animate=False)
