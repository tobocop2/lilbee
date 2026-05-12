"""Status screen: knowledge base info with collapsible sections."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from lilbee.cli.tui.app import LilbeeApp

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import VerticalScroll
from textual.content import Content
from textual.screen import Screen
from textual.widgets import Collapsible, DataTable, Static
from textual.worker import Worker, WorkerState

from lilbee.app.services import get_services
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.pill import pill
from lilbee.core.config import cfg
from lilbee.data.store import SourceRecord
from lilbee.modelhub.model_info import ModelArchInfo, get_model_architecture

log = logging.getLogger(__name__)

# Rows appended to the Documents table per refresh tick. Each ``add_row`` runs
# on the UI thread, so dumping a whole ingested wiki in one loop froze the
# screen for seconds; rendering a bounded batch per ``call_after_refresh`` lets
# the user scroll and interact while the rest streams in.
_DOC_RENDER_BATCH = 100


@dataclass
class _DocsResult:
    """Outcome of the background sources read.

    ``load_failed`` distinguishes "store opened but empty" (``sources == []``)
    from "the read raised" so the UI shows the right placeholder.
    """

    sources: list[SourceRecord]
    load_failed: bool


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
        _kv_line("Chat model", cfg.chat_model or "(disabled)", _model_pill(cfg.chat_model)),
        _kv_line(
            "Embed model", cfg.embedding_model or "(disabled)", _model_pill(cfg.embedding_model)
        ),
        _kv_line("Vision model", cfg.vision_model or "(disabled)", _model_pill(cfg.vision_model)),
        _kv_line("Reranker", cfg.reranker_model or "(disabled)", _model_pill(cfg.reranker_model)),
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

    app: LilbeeApp  # type: ignore[assignment]

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

    def __init__(self) -> None:
        super().__init__()
        self._sections_mounted: bool = False
        self._pending_docs: _DocsResult | None = None
        self._pending_arch: ModelArchInfo | None = None
        # Sources still waiting to be appended to the table by the batched
        # renderer. Bumped each time a new docs result arrives so a stale
        # render chain (from a previous load) stops itself.
        self._docs_render_queue: list[SourceRecord] = []
        self._docs_render_gen: int = 0

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
            Collapsible(Static(id="config-info"), title="Configuration", id="config-section"),
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
                    DataTable(id="docs-table"), title=msg.STATUS_DOCS_TITLE, id="docs-section"
                ),
                Collapsible(Static(id="arch-info"), title="Model Architecture", id="arch-section"),
                Collapsible(Static(id="storage-info"), title="Storage", id="storage-section"),
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

    @work(thread=True, name="status_fetch_sources", exit_on_error=False)
    def _fetch_sources_worker(self) -> _DocsResult:
        """Read the full source list off the UI thread.

        ``load_failed`` is True only when the store read actually raised;
        an empty store with zero documents is the routine first-run state.
        Rendering the (potentially large) list happens back on the UI thread
        in bounded batches via :meth:`_render_doc_batch`.
        """
        try:
            return _DocsResult(sources=get_services().store.get_sources(), load_failed=False)
        except Exception:
            log.debug("Failed to read store for status screen", exc_info=True)
            return _DocsResult(sources=[], load_failed=True)

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
        if event.worker.name == "status_fetch_sources":
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
        """Render *docs* into the Documents table (batched) + storage section."""
        self._load_documents(docs)
        self._load_storage(len(docs.sources))

    def _load_arch(self, info: ModelArchInfo) -> None:
        """Populate the model architecture section from worker result."""
        from textual.css.query import NoMatches

        with contextlib.suppress(NoMatches):
            self.query_one("#arch-info", Static).update(_build_arch_content(info))

    def _load_config(self) -> None:
        """Populate the configuration section."""
        self.query_one("#config-info", Static).update(_build_config_content())

    def _load_documents(self, docs: _DocsResult) -> None:
        """Clear the table, then stream rows in batches over successive refreshes.

        Streaming via ``call_after_refresh`` keeps the screen responsive even
        with a whole ingested wiki in the store: each batch is small, and the
        user can scroll/interact between batches. Suppresses NoMatches because
        the deferred Collapsible composes its inner DataTable a refresh tick
        after ``mount_all`` returns.
        """
        from textual.css.query import NoMatches

        with contextlib.suppress(NoMatches):
            table = self.query_one("#docs-table", DataTable)
            table.clear()
            if not docs.sources:
                placeholder = (
                    msg.STATUS_DOCS_LOAD_FAILED if docs.load_failed else msg.STATUS_DOCS_EMPTY
                )
                table.add_row(placeholder, "")
                self._docs_render_queue = []
                return
        # Bump the generation so any in-flight render chain from a previous
        # load stops itself, then kick off a fresh one.
        self._docs_render_gen += 1
        self._docs_render_queue = list(docs.sources)
        self._render_doc_batch(self._docs_render_gen)

    def _render_doc_batch(self, generation: int) -> None:
        """Append up to ``_DOC_RENDER_BATCH`` rows; reschedule itself if more remain."""
        if generation != self._docs_render_gen or not self._docs_render_queue:
            return
        from textual.css.query import NoMatches

        with contextlib.suppress(NoMatches):
            table = self.query_one("#docs-table", DataTable)
            batch = self._docs_render_queue[:_DOC_RENDER_BATCH]
            del self._docs_render_queue[:_DOC_RENDER_BATCH]
            for src in batch:
                table.add_row(src.get("filename", "?"), str(src.get("chunk_count", 0)))
        if self._docs_render_queue:
            self.call_after_refresh(self._render_doc_batch, generation)

    def _load_storage(self, doc_count: int) -> None:
        """Populate the storage section."""
        from textual.css.query import NoMatches

        with contextlib.suppress(NoMatches):
            self.query_one("#storage-info", Static).update(_build_storage_content(doc_count))

    def action_go_back(self) -> None:
        self.app.switch_view("Chat")

    def action_cursor_down(self) -> None:
        self.query_one("#status-scroll", VerticalScroll).scroll_down()

    def action_cursor_up(self) -> None:
        self.query_one("#status-scroll", VerticalScroll).scroll_up()

    def action_jump_top(self) -> None:
        self.query_one("#status-scroll", VerticalScroll).scroll_home()

    def action_jump_bottom(self) -> None:
        self.query_one("#status-scroll", VerticalScroll).scroll_end()
