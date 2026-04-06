"""Catalog screen -- browse and install models via grid or list view."""

from __future__ import annotations

import contextlib
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import VerticalScroll
from textual.events import Click
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Input, Static
from textual.worker import Worker, WorkerState

from lilbee.catalog import (
    CatalogModel,
    ModelFamily,
    ModelVariant,
    clean_display_name,
    get_catalog,
    get_families,
)
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.widgets.grid_select import GridSelect
from lilbee.cli.tui.widgets.model_card import ModelCard
from lilbee.config import cfg
from lilbee.model_manager import RemoteModel, get_model_manager
from lilbee.models import ModelTask

log = logging.getLogger(__name__)

_HF_PAGE_SIZE = 25
_ALL_TASKS = tuple(ModelTask)

COLUMNS = ("Name", "Task", "Params", "Size", "Quant", "Downloads")


def _parse_param_label(name: str) -> str:
    """Extract parameter count label from model name (e.g. '8B', '0.6B')."""
    match = re.search(r"(\d+\.?\d*)B", name, re.IGNORECASE)
    return f"{match.group(1)}B" if match else "--"


def _format_downloads(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def _format_size_mb(size_mb: int) -> str:
    """Format size in MB to a human-readable string."""
    if size_mb == 0:
        return "--"
    if size_mb >= 1024:
        return f"{size_mb / 1024:.1f} GB"
    return f"{size_mb} MB"


def _format_size_gb(size_gb: float) -> str:
    """Format size in GB to a human-readable string."""
    if size_gb <= 0:
        return "--"
    return f"{size_gb:.1f} GB"


@dataclass
class TableRow:
    """A row in the catalog DataTable with source metadata."""

    name: str
    task: str
    params: str
    size: str
    quant: str
    downloads: str
    featured: bool
    installed: bool
    sort_downloads: int
    sort_size: float
    variant: ModelVariant | None = None
    family: ModelFamily | None = None
    catalog_model: CatalogModel | None = None
    remote_model: RemoteModel | None = None


def _variant_to_row(v: ModelVariant, f: ModelFamily, installed: bool) -> TableRow:
    """Convert a ModelVariant + family to a TableRow."""
    prefix = "* " if v.recommended else ""
    return TableRow(
        name=f"{prefix}{f.name} {v.param_count}",
        task=f.task,
        params=v.param_count,
        size=_format_size_mb(v.size_mb),
        quant=v.quant or "--",
        downloads="--",
        featured=True,
        installed=installed,
        sort_downloads=0,
        sort_size=v.size_mb / 1024,
        variant=v,
        family=f,
    )


def _catalog_to_row(m: CatalogModel, installed: bool) -> TableRow:
    """Convert a CatalogModel to a TableRow."""
    display = clean_display_name(m.hf_repo)
    quant = _extract_quant_from_filename(m.gguf_filename)
    return TableRow(
        name=display,
        task=m.task,
        params=_parse_param_label(m.name),
        size=_format_size_gb(m.size_gb),
        quant=quant or "--",
        downloads=_format_downloads(m.downloads) if m.downloads > 0 else "--",
        featured=m.featured,
        installed=installed,
        sort_downloads=m.downloads,
        sort_size=m.size_gb,
        catalog_model=m,
    )


def _remote_to_row(rm: RemoteModel) -> TableRow:
    """Convert a RemoteModel to a TableRow."""
    return TableRow(
        name=rm.name,
        task=rm.task,
        params=rm.parameter_size or "--",
        size="--",
        quant="--",
        downloads="--",
        featured=False,
        installed=True,
        sort_downloads=0,
        sort_size=0.0,
        remote_model=rm,
    )


def _extract_quant_from_filename(filename: str) -> str:
    """Extract quantization label from a GGUF filename pattern."""
    m = re.search(r"(Q\d[A-Z0-9_]*)", filename, re.IGNORECASE)
    return m.group(1).upper() if m else ""


def _row_display_name(row: TableRow) -> str:
    """Build the display name with featured/installed markers."""
    parts: list[str] = []
    if row.featured:
        parts.append("\u2605")
    parts.append(row.name)
    if row.installed:
        parts.append("[installed]")
    return " ".join(parts)


# Column sort key extractors
_SORT_KEYS = {
    "Name": lambda r: r.name.lower(),
    "Task": lambda r: r.task,
    "Params": lambda r: _param_sort_value(r.params),
    "Size": lambda r: r.sort_size,
    "Quant": lambda r: r.quant,
    "Downloads": lambda r: r.sort_downloads,
}


def _param_sort_value(params: str) -> float:
    """Convert param label to sortable float (e.g. '8B' -> 8.0)."""
    match = re.search(r"(\d+\.?\d*)", params)
    return float(match.group(1)) if match else 0.0


class CatalogScreen(Screen[None]):
    """Model catalog with grid (default) and list views."""

    CSS_PATH = "catalog.tcss"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "go_back", "Back", show=True),
        Binding("escape", "go_back", "Back", show=False),
        Binding("v", "toggle_view", "View", show=True),
        Binding("slash", "focus_search", "Search", show=True),
        Binding("d", "delete_model", "Delete", show=True),
        Binding("x", "delete_model", "Delete", show=False),
        Binding("j", "cursor_down", "Nav", show=False),
        Binding("k", "cursor_up", "Nav", show=False),
        Binding("g", "jump_top", "Top", show=False),
        Binding("G", "jump_bottom", "End", show=False),
        Binding("space", "page_down", "PgDn", show=False),
        Binding("ctrl+d", "page_down", "PgDn", show=False),
        Binding("ctrl+u", "page_up", "PgUp", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._families: list[ModelFamily] = get_families()
        self._hf_models: list[CatalogModel] = []
        self._remote_models: list[RemoteModel] = []
        self._hf_offset = 0
        self._hf_has_more = True
        self._rows: list[TableRow] = []
        self._sort_column: str = "Name"
        self._sort_ascending: bool = True
        self._pending_delete: str | None = None
        self._installed_names: set[str] = set()
        self._grid_view: bool = True
        self._hf_fetched: bool = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="sort-label", shrink=True)
        yield VerticalScroll(id="catalog-grid")
        yield DataTable(id="catalog-table", cursor_type="row")
        yield Input(placeholder=msg.CATALOG_FILTER_PLACEHOLDER, id="catalog-search")
        yield Static("", id="model-detail")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#catalog-search", Input).display = False
        table = self.query_one("#catalog-table", DataTable)
        for col in COLUMNS:
            table.add_column(col, key=col)
        self._fetch_installed_names()
        self.add_class("-grid-view")
        self._refresh_grid()
        self._fetch_remote_models()

    def _fetch_installed_names(self) -> None:
        """Populate installed source repos/filenames from registry manifests."""
        with contextlib.suppress(Exception):
            from lilbee.registry import ModelRegistry

            registry = ModelRegistry(cfg.models_dir)
            self._installed_names = set()
            for m in registry.list_installed():
                # Store both name:tag and source_repo/source_filename for matching
                self._installed_names.add(f"{m.name}:{m.tag}")
                if m.source_repo and m.source_filename:
                    self._installed_names.add(f"{m.source_repo}/{m.source_filename}")

    def action_toggle_view(self) -> None:
        """Toggle between grid and list view."""
        if self._grid_view:
            self._grid_view = False
            self.remove_class("-grid-view")
            self.add_class("-list-view")
            if not self._hf_fetched:
                self._hf_fetched = True
                self._fetch_all_hf_models()
            self._refresh_table()
            with contextlib.suppress(Exception):
                self.query_one("#catalog-table", DataTable).focus()
        else:
            self._grid_view = True
            self.remove_class("-list-view")
            self.add_class("-grid-view")
            self._refresh_grid()

    def action_focus_search(self) -> None:
        """Focus the filter input -- bound to / key."""
        filter_input = self.query_one("#catalog-search", Input)
        filter_input.display = True
        filter_input.focus()

    @on(Input.Changed, "#catalog-search")
    def _on_search_changed(self, event: Input.Changed) -> None:
        """Filter models when search input changes."""
        if self._grid_view:
            self._filter_grid()
        else:
            self._refresh_table()

    @on(Input.Submitted, "#catalog-search")
    def _on_search_submitted(self, event: Input.Submitted) -> None:
        """Close filter on Enter."""
        event.input.display = False
        with contextlib.suppress(Exception):
            self.query_one("#catalog-table", DataTable).focus()

    def _fetch_hf_page(self) -> list[CatalogModel]:
        """Fetch one page of HF models for all task types (runs in worker thread)."""
        all_models: list[CatalogModel] = []
        seen_repos: set[str] = set()
        for task in _ALL_TASKS:
            result = get_catalog(
                task=task,
                featured=False,
                limit=_HF_PAGE_SIZE,
                offset=self._hf_offset,
            )
            for m in result.models:
                if not m.featured and m.hf_repo not in seen_repos:
                    seen_repos.add(m.hf_repo)
                    all_models.append(m)
        self._hf_has_more = len(all_models) >= _HF_PAGE_SIZE
        return all_models

    @work(thread=True)
    def _fetch_all_hf_models(self) -> list[CatalogModel]:
        """Fetch HF models for all task types (replaces current list)."""
        return self._fetch_hf_page()

    @work(thread=True)
    def _fetch_remote_models(self) -> list[RemoteModel]:
        from lilbee.model_manager import classify_remote_models

        return classify_remote_models(cfg.litellm_base_url)

    @work(thread=True)
    def _fetch_more_hf(self) -> list[CatalogModel]:
        """Fetch next page of HF models for all task types (extends current list)."""
        return self._fetch_hf_page()

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.state != WorkerState.SUCCESS:
            return
        result = event.worker.result
        if event.worker.name == "_fetch_all_hf_models" and isinstance(result, list):
            self._hf_models = result
            self._refresh_view()
        elif event.worker.name == "_fetch_more_hf" and isinstance(result, list):
            self._hf_models.extend(result)
            self._refresh_view()
        elif event.worker.name == "_fetch_remote_models" and isinstance(result, list):
            self._remote_models = result
            self._refresh_view()

    def _get_search_text(self) -> str:
        return self.query_one("#catalog-search", Input).value.strip().lower()

    def _build_rows(self) -> list[TableRow]:
        """Build all table rows from current data sources."""
        search = self._get_search_text()
        rows: list[TableRow] = []
        rows.extend(self._build_family_rows(search))
        rows.extend(self._build_hf_rows(search))
        rows.extend(self._build_remote_rows(search))
        return rows

    def _build_family_rows(self, search: str) -> list[TableRow]:
        """Build rows from featured model families."""
        rows: list[TableRow] = []
        for fam in self._families:
            for v in fam.variants:
                installed = self._is_installed(
                    f"{fam.name}:{v.param_count}", repo=v.hf_repo, filename=v.filename
                )
                row = _variant_to_row(v, fam, installed)
                if _matches_search(row, search):
                    rows.append(row)
        return rows

    def _build_hf_rows(self, search: str) -> list[TableRow]:
        """Build rows from HuggingFace models."""
        rows: list[TableRow] = []
        for m in self._hf_models:
            installed = self._is_installed(m.name, repo=m.hf_repo, filename=m.gguf_filename)
            row = _catalog_to_row(m, installed)
            if _matches_search(row, search):
                rows.append(row)
        return rows

    def _build_remote_rows(self, search: str) -> list[TableRow]:
        """Build rows from remote (inference-only) models."""
        rows: list[TableRow] = []
        for rm in self._remote_models:
            row = _remote_to_row(rm)
            if _matches_search(row, search):
                rows.append(row)
        return rows

    def _is_installed(self, name: str, repo: str = "", filename: str = "") -> bool:
        """Check if a model is installed by name or source repo/filename."""
        if name in self._installed_names:
            return True
        if repo and filename:
            return f"{repo}/{filename}" in self._installed_names
        return False

    def _sort_rows(self, rows: list[TableRow]) -> list[TableRow]:
        """Sort rows: featured first, then by current sort column."""
        key_fn = _SORT_KEYS.get(self._sort_column, _SORT_KEYS["Name"])
        # Stable sort: featured always first, then by column
        return sorted(
            rows,
            key=lambda r: (not r.featured, key_fn(r)),
            reverse=not self._sort_ascending,
        )

    def _refresh_view(self) -> None:
        """Refresh the active view (grid or list)."""
        if self._grid_view:
            self._refresh_grid()
        else:
            self._refresh_table()

    def _refresh_grid(self) -> None:
        """Rebuild the grid view with all cards (called when data changes)."""
        container = self.query_one("#catalog-grid", VerticalScroll)
        container.remove_children()
        family_rows = self._build_family_rows("")
        remote_rows = self._build_remote_rows("")
        hf_rows = self._build_hf_rows("") if self._hf_fetched else []
        all_rows = family_rows + remote_rows + hf_rows
        widgets_to_mount: list[Static | GridSelect] = []
        for section in _group_rows_for_grid(all_rows):
            if not section.rows:
                continue
            widgets_to_mount.append(Static(section.heading, classes="section-heading"))
            cards = [ModelCard(row) for row in section.rows]
            grid = GridSelect(*cards, min_column_width=30, max_column_width=50)
            widgets_to_mount.append(grid)
        if not self._hf_fetched:
            widgets_to_mount.append(
                Static(
                    msg.CATALOG_BROWSE_MORE,
                    classes="grid-cta browse-more-hf",
                )
            )
        widgets_to_mount.append(
            Static(
                msg.CATALOG_VIEW_TOGGLE_GRID,
                classes="grid-cta view-toggle-cta",
            )
        )
        container.mount_all(widgets_to_mount)

    def _filter_grid(self) -> None:
        """Filter visible cards by search text without recreating widgets."""
        search = self._get_search_text()
        for card in self.query(ModelCard):
            card.display = _matches_search(card.row, search)
        container = self.query_one("#catalog-grid", VerticalScroll)
        children = list(container.children)
        for i, child in enumerate(children):
            if not child.has_class("section-heading"):
                continue
            grid = children[i + 1] if i + 1 < len(children) else None
            if isinstance(grid, GridSelect):
                has_visible = any(c.display for c in grid.children)
                child.display = has_visible
                grid.display = has_visible

    @on(Click, ".browse-more-hf")
    def _on_browse_more_clicked(self) -> None:
        """Fetch all models when the browse-more card is clicked."""
        if not self._hf_fetched:
            self._hf_fetched = True
            self._fetch_all_hf_models()

    @on(GridSelect.Selected)
    def _on_grid_selected(self, event: GridSelect.Selected) -> None:
        """Handle model selection from the grid view."""
        if isinstance(event.widget, ModelCard):
            self._select_row(event.widget.row)

    def _refresh_table(self) -> None:
        """Rebuild the DataTable from current data."""
        self._rows = self._sort_rows(self._build_rows())
        table = self.query_one("#catalog-table", DataTable)
        table.clear()
        for row in self._rows:
            table.add_row(
                _row_display_name(row),
                row.task,
                row.params,
                row.size,
                row.quant,
                row.downloads,
            )
        self._update_sort_label()

    def _update_sort_label(self) -> None:
        """Update the sort indicator label."""
        direction = "asc" if self._sort_ascending else "desc"
        n_total = len(self._rows)
        more = "+" if self._hf_has_more else ""
        self.query_one("#sort-label", Static).update(
            f"Sort: {self._sort_column} ({direction})  |  "
            f"{n_total}{more} models  |  {msg.CATALOG_VIEW_TOGGLE_TABLE}"
        )

    @on(DataTable.HeaderSelected, "#catalog-table")
    def _on_header_selected(self, event: DataTable.HeaderSelected) -> None:
        """Sort by the clicked column header, toggling asc/desc."""
        col_key = str(event.column_key)
        if col_key == self._sort_column:
            self._sort_ascending = not self._sort_ascending
        else:
            self._sort_column = col_key
            self._sort_ascending = True
        self._refresh_table()

    @on(DataTable.RowSelected, "#catalog-table")
    def _on_row_selected(self, event: DataTable.RowSelected) -> None:
        """Install/select the model on the highlighted row."""
        row_index = event.cursor_row
        if row_index < 0 or row_index >= len(self._rows):
            return
        row = self._rows[row_index]
        self._select_row(row)

    def _select_row(self, row: TableRow) -> None:
        """Handle row selection: install or use the model."""
        if row.variant and row.family:
            self._install_variant(row.variant, row.family)
        elif row.catalog_model:
            self._install_model(row.catalog_model)
        elif row.remote_model:
            cfg.chat_model = row.remote_model.name
            self.notify(msg.CATALOG_USING_REMOTE.format(name=row.remote_model.name))

    def _load_more(self) -> None:
        """Load next page of HF models."""
        self._hf_offset += _HF_PAGE_SIZE
        self._fetch_more_hf()

    def _install_variant(self, variant: ModelVariant, family: ModelFamily) -> None:
        """Convert a variant back to a CatalogModel and trigger install."""
        entry = CatalogModel(
            name=f"{family.name} {variant.param_count}",
            hf_repo=variant.hf_repo,
            gguf_filename=variant.filename,
            size_gb=variant.size_mb / 1024,
            min_ram_gb=max(2.0, (variant.size_mb / 1024) * 1.5),
            description=family.description,
            featured=True,
            downloads=0,
            task=family.task,
        )
        self._install_model(entry)

    def _install_model(self, model: CatalogModel) -> None:
        from lilbee.catalog import resolve_filename

        try:
            filename = resolve_filename(model)
            dest = cfg.models_dir / filename
            if dest.exists():
                self.notify(msg.CATALOG_ALREADY_INSTALLED.format(name=model.name))
                return
        except Exception:
            log.debug("Could not resolve filename", exc_info=True)

        self._enqueue_download(model)

    def _enqueue_download(self, model: CatalogModel) -> None:
        """Enqueue a model download in the app's TaskBar."""
        from lilbee.cli.tui.app import LilbeeApp

        if not isinstance(self.app, LilbeeApp):
            self.notify(msg.CATALOG_NO_TASK_BAR, severity="error")
            return
        task_bar = self.app.task_bar
        task_id = task_bar.add_task(f"Downloading {model.name}", "download")
        task_bar.queue.advance("download")
        self.notify(msg.CATALOG_QUEUED_DOWNLOAD.format(name=model.name))
        self._run_download(model, task_id, task_bar)

    def _make_progress_callback(self, task_id: str, bar: object) -> Callable[[int, int], None]:
        """Build a progress callback that reports download progress to the TaskBar."""
        from lilbee.cli.tui.widgets.task_bar import TaskBar

        tb: TaskBar = bar  # type: ignore[assignment]

        def on_progress(downloaded: int, total: int) -> None:
            mb_done = downloaded / (1024 * 1024)
            if total > 0:
                pct = min(int(downloaded * 100 / total), 100)
                mb_total = total / (1024 * 1024)
                self._safe_call(tb.update_task, task_id, pct, f"{mb_done:.0f}/{mb_total:.0f} MB")
            else:
                self._safe_call(tb.update_task, task_id, 0, f"{mb_done:.0f} MB")

        return on_progress

    @work(thread=True)
    def _run_download(self, model: CatalogModel, task_id: str, task_bar: object) -> None:
        """Download a model in a background thread, reporting to TaskBar."""
        from lilbee.catalog import download_model
        from lilbee.cli.tui.widgets.task_bar import TaskBar

        bar: TaskBar = task_bar  # type: ignore[assignment]

        try:
            download_model(model, on_progress=self._make_progress_callback(task_id, bar))
            self._safe_call(bar.complete_task, task_id)
            self._safe_call(self.notify, msg.CATALOG_INSTALLED_OK.format(name=model.name))
        except PermissionError:
            detail = msg.CATALOG_GATED_REPO.format(name=model.name)
            log.warning("Gated repo: %s", model.hf_repo)
            self._safe_call(bar.fail_task, task_id, detail)
            self._safe_call(self.notify, detail, severity="warning")
        except Exception:
            log.warning("Download failed for %s", model.name, exc_info=True)
            detail = msg.CATALOG_DOWNLOAD_FAILED.format(name=model.name)
            self._safe_call(bar.fail_task, task_id, detail)
            self._safe_call(self.notify, detail, severity="error")

    def _safe_call(self, fn: Any, *args: Any, **kwargs: Any) -> None:
        """Call fn via call_from_thread, suppressing errors if app context is gone."""
        try:
            self.app.call_from_thread(fn, *args, **kwargs)
        except Exception:
            log.debug(
                "_safe_call failed for %s",
                fn.__name__ if hasattr(fn, "__name__") else fn,
                exc_info=True,
            )

    def action_go_back(self) -> None:
        from lilbee.cli.tui.app import LilbeeApp

        if isinstance(self.app, LilbeeApp):
            self.app.switch_view("Chat")
        else:
            self.app.pop_screen()

    def action_delete_model(self) -> None:
        """Delete an installed model. First press asks confirmation, second confirms."""
        if isinstance(self.focused, Input):
            return
        model_name = self._get_highlighted_model_name()
        if model_name is None:
            self.notify(msg.CATALOG_SELECT_TO_DELETE, severity="warning")
            return

        mgr = get_model_manager()
        if not mgr.is_installed(model_name):
            self.notify(msg.CATALOG_NOT_INSTALLED.format(name=model_name), severity="warning")
            return

        if self._pending_delete == model_name:
            self._pending_delete = None
            self._run_delete(model_name)
        else:
            self._pending_delete = model_name
            self.notify(msg.CATALOG_CONFIRM_DELETE.format(name=model_name))

    def _get_highlighted_model_name(self) -> str | None:
        """Return the registry-compatible model name for the highlighted row.

        For registry models, returns 'Name:latest' to match manifest format.
        For remote models, returns the remote model name.
        """
        table = self.query_one("#catalog-table", DataTable)
        row_idx = table.cursor_row
        if row_idx < 0 or row_idx >= len(self._rows):
            return None
        row = self._rows[row_idx]
        if row.variant and row.family:
            # Registry name matches the catalog entry name used in _register_model
            return f"{row.family.name} {row.variant.param_count}:latest"
        if row.remote_model:
            return row.remote_model.name
        if row.catalog_model:
            return f"{row.catalog_model.name}:latest"
        return None

    @work(thread=True)
    def _run_delete(self, model_name: str) -> None:
        """Remove a model in a background thread."""
        try:
            removed = get_model_manager().remove(model_name)
            if removed:
                self._safe_call(self.notify, msg.CATALOG_DELETED.format(name=model_name))
                self._safe_call(self._refresh_after_delete)
            else:
                self._safe_call(
                    self.notify,
                    msg.CATALOG_DELETE_FAILED.format(error=model_name),
                    severity="error",
                )
        except Exception as exc:
            log.warning("Delete failed for %s", model_name, exc_info=True)
            self._safe_call(
                self.notify, msg.CATALOG_DELETE_FAILED.format(error=exc), severity="error"
            )

    def _refresh_after_delete(self) -> None:
        """Re-fetch remote models and refresh after deletion."""
        self._fetch_installed_names()
        self._refresh_view()
        self._fetch_remote_models()

    def action_page_down(self) -> None:
        if isinstance(self.focused, Input) or self._grid_view:
            return
        table = self.query_one("#catalog-table", DataTable)
        for _ in range(10):
            table.action_cursor_down()

    def action_page_up(self) -> None:
        if isinstance(self.focused, Input) or self._grid_view:
            return
        table = self.query_one("#catalog-table", DataTable)
        for _ in range(10):
            table.action_cursor_up()

    def action_cursor_down(self) -> None:
        if isinstance(self.focused, Input) or self._grid_view:
            return
        self.query_one("#catalog-table", DataTable).action_cursor_down()

    def action_cursor_up(self) -> None:
        if isinstance(self.focused, Input) or self._grid_view:
            return
        self.query_one("#catalog-table", DataTable).action_cursor_up()

    def action_jump_top(self) -> None:
        if isinstance(self.focused, Input) or self._grid_view:
            return
        table = self.query_one("#catalog-table", DataTable)
        table.move_cursor(row=0)

    def action_jump_bottom(self) -> None:
        if isinstance(self.focused, Input) or self._grid_view:
            return
        table = self.query_one("#catalog-table", DataTable)
        if self._rows:
            table.move_cursor(row=len(self._rows) - 1)

    def key_left(self) -> None:
        """Navigate to previous view instead of switching tabs."""
        from lilbee.cli.tui.app import LilbeeApp

        if isinstance(self.app, LilbeeApp):
            self.app.action_nav_prev()

    def key_right(self) -> None:
        """Navigate to next view instead of switching tabs."""
        from lilbee.cli.tui.app import LilbeeApp

        if isinstance(self.app, LilbeeApp):
            self.app.action_nav_next()


@dataclass
class GridSection:
    """A named group of rows for the grid view."""

    heading: str
    rows: list[TableRow]


def _group_rows_for_grid(rows: list[TableRow]) -> list[GridSection]:
    """Group rows into sections for the grid view."""
    recommended = [r for r in rows if r.featured]
    installed = [r for r in rows if r.installed and not r.featured]
    chat = [r for r in rows if r.task == ModelTask.CHAT and not r.featured and not r.installed]
    embedding = [
        r for r in rows if r.task == ModelTask.EMBEDDING and not r.featured and not r.installed
    ]
    vision = [r for r in rows if r.task == ModelTask.VISION and not r.featured and not r.installed]
    return [
        GridSection("Our picks", recommended),
        GridSection("Installed", installed),
        GridSection("Chat", chat),
        GridSection("Embedding", embedding),
        GridSection("Vision", vision),
    ]


def _matches_search(row: TableRow, search: str) -> bool:
    """Return True if the row matches the search text."""
    if not search:
        return True
    return (
        search in row.name.lower()
        or search in row.task.lower()
        or search in row.params.lower()
        or search in row.quant.lower()
    )
