"""Catalog screen -- browse and install models via grid or list view."""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import ClassVar

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import VerticalScroll
from textual.events import Click
from textual.screen import Screen
from textual.widgets import Input, Static
from textual.worker import Worker, WorkerState

from lilbee.catalog import (
    CatalogModel,
    ModelFamily,
    ModelVariant,
    get_catalog,
    get_families,
)
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.screens.catalog_utils import (
    SORT_KEYS,
    TableRow,
    catalog_to_row,
    matches_search,
    remote_to_row,
    variant_to_row,
)
from lilbee.cli.tui.widgets.grid_select import GridSelect
from lilbee.cli.tui.widgets.model_card import ModelCard
from lilbee.cli.tui.widgets.model_list_item import ModelListItem
from lilbee.cli.tui.widgets.nav_aware_input import NavAwareInput
from lilbee.cli.tui.widgets.search_hf_cta_item import SearchHFCtaItem
from lilbee.config import cfg
from lilbee.model_manager import RemoteModel, get_model_manager
from lilbee.models import ModelTask
from lilbee.providers.model_ref import OLLAMA_PREFIX
from lilbee.providers.sdk_backend import OLLAMA_BACKEND_NAME

log = logging.getLogger(__name__)

_HF_PAGE_SIZE = 25
# When the highlighted row is within this many rows of the end we
# auto-fetch the next page. Small enough that the request is already
# in flight by the time the user reaches the bottom.
_HF_LOAD_MORE_TRIGGER = 5
# Long enough to register; short enough to clear before a warm-cache fetch.
_NOTIFY_SEARCHING_TIMEOUT_SECONDS = 4
_ALL_TASKS = tuple(ModelTask)

_WORKER_FETCH_HF = "fetch_hf_models"
_WORKER_FETCH_MORE_HF = "fetch_more_hf"
_WORKER_FETCH_REMOTE = "fetch_remote_models"
_WORKER_FETCH_SEARCH = "fetch_hf_search"

_GRID_PAGE_ROWS = 3
_LIST_PAGE_ROWS = 10

# Sort columns cycled by the `s` keybinding in list view.
_SORT_CYCLE: tuple[str, ...] = ("Name", "Downloads", "Size", "Params")


class CatalogScreen(Screen[None]):
    """Model catalog with grid (default) and list views."""

    CSS_PATH = "catalog.tcss"
    AUTO_FOCUS = ""  # GridSelect is mounted dynamically; focused in on_mount

    HELP = (
        "# Catalog\n"
        "Browse and install models.\n\n"
        "Use arrows to navigate the grid, Enter to install."
    )

    _ACTION_GROUP = Binding.Group("Actions", compact=True)
    _SCROLL_GROUP = Binding.Group("Scroll", compact=True)

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "go_back", "Back", show=True, group=_ACTION_GROUP),
        Binding("escape", "go_back", "Back", show=True),
        Binding("v", "toggle_view", "View", show=True, group=_ACTION_GROUP),
        Binding("slash", "focus_search", "Search", show=True, group=_ACTION_GROUP),
        Binding("d", "delete_model", "Delete", show=True, group=_ACTION_GROUP),
        Binding("x", "delete_model", "Delete", show=False),
        Binding("j", "cursor_down", "Nav", show=False, group=_SCROLL_GROUP),
        Binding("k", "cursor_up", "Nav", show=False, group=_SCROLL_GROUP),
        Binding("g", "jump_top", "Top", show=False, group=_SCROLL_GROUP),
        Binding("G", "jump_bottom", "End", show=False, group=_SCROLL_GROUP),
        Binding("space", "page_down", "PgDn", show=False, group=_SCROLL_GROUP),
        Binding("ctrl+d", "page_down", "PgDn", show=False, group=_SCROLL_GROUP),
        Binding("ctrl+u", "page_up", "PgUp", show=False, group=_SCROLL_GROUP),
        # Hidden from the footer so catalog still has <=5 visible bindings;
        # the sort-label surfaces "press n for more" and "press s to sort"
        # to the user instead.
        Binding("n", "load_more", "More", show=False, group=_ACTION_GROUP),
        Binding("s", "cycle_sort", "Sort", show=False, group=_ACTION_GROUP),
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
        self._loading_more: bool = False
        self._grid_cache_key: tuple[tuple[tuple[str, bool], ...], str] | tuple = ()
        self._search_in_flight: bool = False

    def compose(self) -> ComposeResult:
        from textual.widgets import Footer

        from lilbee.cli.tui.widgets.bottom_bars import BottomBars
        from lilbee.cli.tui.widgets.status_bar import ViewTabs
        from lilbee.cli.tui.widgets.task_bar import TaskBar
        from lilbee.cli.tui.widgets.top_bars import TopBars

        with TopBars():
            yield ViewTabs()
            yield NavAwareInput(placeholder=msg.CATALOG_FILTER_PLACEHOLDER, id="catalog-search")
        yield Static("", id="sort-label", shrink=True)
        yield VerticalScroll(id="catalog-grid")
        yield VerticalScroll(id="catalog-list")
        yield Static("", id="model-detail")
        with BottomBars():
            yield TaskBar()
            yield Footer()

    def on_mount(self) -> None:
        self._fetch_installed_names()
        self.add_class("-grid-view")
        self._refresh_grid()
        self._focus_first_grid()
        self._fetch_remote_models()

    def _focus_first_grid(self) -> None:
        """Focus the first GridSelect widget if available."""
        with contextlib.suppress(Exception):
            self.query_one(GridSelect).focus()

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
            self._refresh_list()
            self._focus_list_item(0)
        else:
            self._grid_view = True
            self.remove_class("-list-view")
            self.add_class("-grid-view")
            self._refresh_grid()
            with contextlib.suppress(Exception):
                self.query_one("#catalog-grid GridSelect", GridSelect).focus()

    def action_focus_search(self) -> None:
        """Focus the filter input -- bound to / key."""
        self.query_one("#catalog-search", Input).focus()

    @on(Input.Changed, "#catalog-search")
    def _on_search_changed(self, event: Input.Changed) -> None:
        """Filter models when search input changes."""
        if self._grid_view:
            self._filter_grid()
            self._sync_grid_search_cta()
        else:
            self._filter_list()

    @on(Input.Submitted, "#catalog-search")
    def _on_search_submitted(self, event: Input.Submitted) -> None:
        """Enter installs the first visible match; falls through to the HF CTA
        when nothing matches locally so the obvious intent ('search for this')
        doesn't require the user to Tab over to the CTA row first."""
        if self._grid_view:
            if any(card.display for card in self.query(ModelCard)):
                self._select_first_visible_grid_card()
                return
        elif any(item.display for item in self.query(ModelListItem)):
            self._select_first_visible_list_item()
            return
        self._trigger_remote_search(self._get_search_text())

    def _trigger_remote_search(self, query: str) -> None:
        """Fire the HF search worker, unless one is already in flight."""
        if self._search_in_flight or not query:
            return
        self._search_in_flight = True
        self._update_sort_label()
        # Sort label is hidden in grid view, so the toast is the only feedback there.
        self.notify(msg.CATALOG_SEARCHING_HF, timeout=_NOTIFY_SEARCHING_TIMEOUT_SECONDS)
        self._fetch_hf_search(query)

    @on(SearchHFCtaItem.Selected)
    def _on_search_hf_cta_selected(self, event: SearchHFCtaItem.Selected) -> None:
        self._trigger_remote_search(event.term)

    @on(Click, ".search-hf-cta")
    def _on_search_hf_cta_clicked(self) -> None:
        self._trigger_remote_search(self._get_search_text())

    def _select_first_visible_grid_card(self) -> None:
        """Focus the first grid with a visible match and trigger its install.

        Without the "first visible" walk, focusing any grid with
        ``highlighted = 0`` could land on a card the filter just hid,
        and Enter would install the wrong model. Setting
        ``highlighted`` to the first visible index guarantees the
        install fires on what the user can actually see.
        """
        with contextlib.suppress(Exception):
            for grid in self.query(GridSelect):
                visible = [i for i, card in enumerate(grid.children) if card.display]
                if visible:
                    grid.focus()
                    grid.highlighted = visible[0]
                    grid.action_select()
                    return

    def _select_first_visible_list_item(self) -> None:
        """List-view counterpart: focus + install the first visible row."""
        with contextlib.suppress(Exception):
            for item in self.query(ModelListItem):
                if item.display:
                    item.focus()
                    item.action_select()
                    return

    def _fetch_hf_page(self) -> list[CatalogModel]:
        """Fetch one page of HF models for all task types (runs in worker thread)."""
        all_models: list[CatalogModel] = []
        seen_repos: set[str] = set()
        any_has_more = False
        for task in _ALL_TASKS:
            result = get_catalog(
                task=task,
                featured=False,
                limit=_HF_PAGE_SIZE,
                offset=self._hf_offset,
            )
            if result.has_more:
                any_has_more = True
            for m in result.models:
                if not m.featured and m.hf_repo not in seen_repos:
                    seen_repos.add(m.hf_repo)
                    all_models.append(m)
        self._hf_has_more = any_has_more
        return all_models

    @work(thread=True, name=_WORKER_FETCH_HF)
    def _fetch_all_hf_models(self) -> list[CatalogModel]:
        """Fetch HF models for all task types (replaces current list)."""
        return self._fetch_hf_page()

    @work(thread=True, name=_WORKER_FETCH_REMOTE)
    def _fetch_remote_models(self) -> list[RemoteModel]:
        from lilbee.model_manager import classify_remote_models

        return classify_remote_models(cfg.remote_base_url)

    @work(thread=True, name=_WORKER_FETCH_MORE_HF)
    def _fetch_more_hf(self) -> list[CatalogModel]:
        """Fetch next page of HF models for all task types (extends current list)."""
        return self._fetch_hf_page()

    @work(thread=True, name=_WORKER_FETCH_SEARCH, exit_on_error=False)
    def _fetch_hf_search(self, query: str) -> list[CatalogModel]:
        """Fetch HF models matching the user's search term (runs in worker thread)."""
        existing_repos = {m.hf_repo for m in self._hf_models}
        new_models: list[CatalogModel] = []
        for task in _ALL_TASKS:
            result = get_catalog(
                task=task,
                featured=False,
                search=query,
                limit=_HF_PAGE_SIZE,
                offset=0,
            )
            for m in result.models:
                if not m.featured and m.hf_repo not in existing_repos:
                    existing_repos.add(m.hf_repo)
                    new_models.append(m)
        return new_models

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        # PENDING/RUNNING fire here too; only ERROR/CANCELLED should release latches.
        if event.state in (WorkerState.ERROR, WorkerState.CANCELLED):
            if event.worker.name == _WORKER_FETCH_MORE_HF:
                self._loading_more = False
            if event.worker.name == _WORKER_FETCH_SEARCH:
                self._search_in_flight = False
                self._update_sort_label()
            return
        if event.state != WorkerState.SUCCESS:
            return
        result = event.worker.result
        if not isinstance(result, list):
            return
        name = event.worker.name
        if name == _WORKER_FETCH_HF:
            self._hf_models = result
        elif name == _WORKER_FETCH_MORE_HF:
            self._hf_models.extend(result)
            self._loading_more = False
        elif name == _WORKER_FETCH_SEARCH:
            self._hf_fetched = True
            self._hf_models.extend(result)
            self._search_in_flight = False
            self._update_sort_label()
        elif name == _WORKER_FETCH_REMOTE:
            self._remote_models = result
        else:
            return
        self._refresh_view()

    def _get_search_text(self) -> str:
        # Preserve the user's casing for display (e.g. the CTA label); matching
        # callers normalize via _normalize_for_search.
        return self.query_one("#catalog-search", Input).value.strip()

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
                    f"{fam.slug}:{v.tag}", repo=v.hf_repo, filename=v.filename
                )
                row = variant_to_row(v, fam, installed)
                if matches_search(row, search):
                    rows.append(row)
        return rows

    def _build_hf_rows(self, search: str) -> list[TableRow]:
        """Build rows from HuggingFace models."""
        rows: list[TableRow] = []
        for m in self._hf_models:
            installed = self._is_installed(m.ref, repo=m.hf_repo, filename=m.gguf_filename)
            row = catalog_to_row(m, installed)
            if matches_search(row, search):
                rows.append(row)
        return rows

    def _build_remote_rows(self, search: str) -> list[TableRow]:
        """Build rows from remote (inference-only) models."""
        rows: list[TableRow] = []
        for rm in self._remote_models:
            row = remote_to_row(rm)
            if matches_search(row, search):
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
        key_fn = SORT_KEYS.get(self._sort_column, SORT_KEYS["Name"])
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
            self._refresh_list()

    def _refresh_grid(self) -> None:
        """Rebuild the grid view with all cards (called when data changes)."""
        family_rows = self._build_family_rows("")
        remote_rows = self._build_remote_rows("")
        hf_rows = self._build_hf_rows("") if self._hf_fetched else []
        all_rows = family_rows + remote_rows + hf_rows
        # Include the full search text so toggle-back + value-change combinations
        # rebuild the grid (and therefore the CTA) with the current query.
        row_key = (
            tuple((r.name, r.installed) for r in all_rows),
            self._get_search_text(),
        )
        if self._grid_cache_key == row_key:
            return
        self._grid_cache_key = row_key
        container = self.query_one("#catalog-grid", VerticalScroll)
        container.remove_children()
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
        search = self._get_search_text()
        if search:
            widgets_to_mount.append(
                Static(
                    msg.CATALOG_SEARCH_HF_CTA.format(query=search),
                    classes="grid-cta search-hf-cta",
                )
            )
        widgets_to_mount.append(
            Static(
                msg.CATALOG_VIEW_TOGGLE_GRID,
                classes="grid-cta",
            )
        )
        container.mount_all(widgets_to_mount)

    def _sync_grid_search_cta(self) -> None:
        """Mount/remove/update the grid-view search-HF CTA in response to typing."""
        search = self._get_search_text()
        existing = self.query("#catalog-grid > .search-hf-cta")
        if not search:
            for w in existing:
                w.remove()
            return
        cta_text = msg.CATALOG_SEARCH_HF_CTA.format(query=search)
        if existing:
            for w in existing:
                if isinstance(w, Static):
                    w.update(cta_text)
            return
        container = self.query_one("#catalog-grid", VerticalScroll)
        container.mount(Static(cta_text, classes="grid-cta search-hf-cta"))

    def _filter_grid(self) -> None:
        """Filter visible cards by search text without recreating widgets."""
        search = self._get_search_text()
        for card in self.query(ModelCard):
            card.display = matches_search(card.row, search)
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

    @on(GridSelect.LeaveDown)
    def _on_grid_leave_down(self, event: GridSelect.LeaveDown) -> None:
        """Move focus to the next GridSelect or focusable widget."""
        self.focus_next()

    @on(GridSelect.LeaveUp)
    def _on_grid_leave_up(self, event: GridSelect.LeaveUp) -> None:
        """Move focus to the previous GridSelect or focusable widget."""
        self.focus_previous()

    @on(GridSelect.Selected)
    def _on_grid_selected(self, event: GridSelect.Selected) -> None:
        """Handle model selection from the grid view."""
        if isinstance(event.widget, ModelCard):
            self._select_row(event.widget.row)

    @on(ModelListItem.Selected)
    def _on_list_item_selected(self, event: ModelListItem.Selected) -> None:
        """Handle model selection from the list view."""
        self._select_row(event.item.row)

    def _refresh_list(self) -> None:
        """Rebuild the list view from current data; append HF search CTA when filtering."""
        self._rows = self._sort_rows(self._build_rows())
        container = self.query_one("#catalog-list", VerticalScroll)
        container.remove_children()
        widgets_to_mount: list[ModelListItem | SearchHFCtaItem] = [
            ModelListItem(row) for row in self._rows
        ]
        search = self._get_search_text()
        if search:
            widgets_to_mount.append(SearchHFCtaItem(search))
        if widgets_to_mount:
            container.mount_all(widgets_to_mount)
        self._update_sort_label()

    def _filter_list(self) -> None:
        """Filter visible list items by search without rebuilding the list.

        Per-keystroke path: toggles .display on existing ModelListItems
        and mounts/removes the HF CTA row as needed. Only _refresh_list
        (data change, sort change) remounts.
        """
        search = self._get_search_text()
        for item in self.query(ModelListItem):
            item.display = matches_search(item.row, search)
        self._sync_list_search_cta(search)
        self._update_sort_label()

    def _sync_list_search_cta(self, search: str) -> None:
        """Ensure the search-HF CTA row exists iff a search term is active."""
        container = self.query_one("#catalog-list", VerticalScroll)
        existing = list(container.query(SearchHFCtaItem))
        for widget in existing:
            widget.remove()
        if search:
            container.mount(SearchHFCtaItem(search))

    def _update_sort_label(self) -> None:
        """Update the sort indicator label."""
        direction = "asc" if self._sort_ascending else "desc"
        n_total = len(self._rows)
        if self._loading_more:
            count = f"{n_total} models · loading more…"
        elif self._hf_has_more:
            count = f"{n_total} models · press [b]n[/b] for more"
        else:
            count = f"{n_total} models"
        hint = msg.CATALOG_SEARCHING_HF if self._search_in_flight else msg.CATALOG_VIEW_TOGGLE_LIST
        self.query_one("#sort-label", Static).update(
            f"Sort: {self._sort_column} ({direction})  |  {count}  |  {hint}"
        )

    def action_cycle_sort(self) -> None:
        """Cycle the list-view sort column ascending: Name, Downloads, Size, Params."""
        if isinstance(self.focused, Input):
            return
        if self._grid_view:
            self.notify(msg.CATALOG_SORT_LIST_ONLY)
            return
        try:
            idx = _SORT_CYCLE.index(self._sort_column)
        except ValueError:
            idx = -1
        self._sort_column = _SORT_CYCLE[(idx + 1) % len(_SORT_CYCLE)]
        self._sort_ascending = True
        self._refresh_list()
        # _refresh_list replaces the list children asynchronously via
        # mount_all; focusing before the new widgets settle can leave focus
        # on the filter Input, which swallows the next `s` press as text
        # . Defer the focus move until after Textual's next refresh
        # so _list_items() actually returns the new rows.
        self.call_after_refresh(self._focus_list_item, 0)

    def _select_row(self, row: TableRow) -> None:
        """Handle row selection: install or use the model."""
        if row.variant and row.family:
            self._install_variant(row.variant, row.family)
        elif row.catalog_model:
            self._install_model(row.catalog_model)
        elif row.remote_model:
            ref = (
                f"{OLLAMA_PREFIX}{row.remote_model.name}"
                if row.remote_model.provider == OLLAMA_BACKEND_NAME
                else row.remote_model.name
            )
            from lilbee.cli.tui.app import apply_active_model

            apply_active_model(self.app, "chat_model", ref)
            self.notify(msg.CATALOG_USING_REMOTE.format(name=row.remote_model.name))

    def _load_more(self) -> None:
        """Load next page of HF models, if any remain and no fetch is in flight."""
        if self._loading_more or not self._hf_has_more:
            return
        self._loading_more = True
        self._hf_offset += _HF_PAGE_SIZE
        self._fetch_more_hf()

    def action_load_more(self) -> None:
        """Keyboard trigger (``n``) so users can page without scrolling."""
        self._load_more()

    def _install_variant(self, variant: ModelVariant, family: ModelFamily) -> None:
        """Convert a variant back to a CatalogModel and trigger install."""
        entry = CatalogModel(
            name=family.slug,
            tag=variant.tag,
            display_name=f"{family.name} {variant.param_count}",
            hf_repo=variant.hf_repo,
            gguf_filename=variant.filename,
            size_gb=variant.size_mb / 1024,
            min_ram_gb=max(2.0, (variant.size_mb / 1024) * 1.5),
            description=family.description,
            featured=True,
            downloads=0,
            task=family.task,
            recommended=variant.recommended,
        )
        self._install_model(entry)

    def _install_model(self, model: CatalogModel) -> None:
        from lilbee.catalog import resolve_filename

        try:
            filename = resolve_filename(model)
            dest = cfg.models_dir / filename
            if dest.exists():
                self.notify(msg.CATALOG_ALREADY_INSTALLED.format(name=model.display_name))
                return
        except Exception:
            log.debug("Could not resolve filename", exc_info=True)

        self._enqueue_download(model)

    def _enqueue_download(self, model: CatalogModel) -> None:
        """Submit the download to the app-level TaskBarController.

        The controller owns the worker thread; this screen just fires the
        request and returns. Progress is visible from every screen and
        survives navigation.
        """
        from lilbee.cli.tui.app import LilbeeApp

        if not isinstance(self.app, LilbeeApp):  # test apps aren't LilbeeApp
            self.notify(msg.CATALOG_NO_TASK_BAR, severity="error")
            return
        self.app.task_bar.start_download(model)
        self.notify(msg.CATALOG_QUEUED_DOWNLOAD.format(name=model.display_name))

    def action_go_back(self) -> None:
        # First Escape press unfocuses the filter input; without this
        # the screen-level `s` / `v` keys get typed into the input and the only
        # way to regain screen focus is to leave the screen entirely.
        if isinstance(self.focused, Input):
            self._focus_list_or_grid()
            return
        from lilbee.cli.tui.app import LilbeeApp

        if isinstance(self.app, LilbeeApp):  # test apps aren't LilbeeApp
            self.app.switch_view("Chat")
        else:
            self.app.pop_screen()

    def _focus_list_or_grid(self) -> None:
        """Move focus from the filter input to the active view's list/grid."""
        if self._grid_view:
            self._focus_first_grid()
        else:
            self._focus_list_item(0)

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
        """Return the registry-compatible model ref for the focused/highlighted row."""
        if isinstance(self.focused, ModelListItem):
            return self.focused.row.ref or None
        focused_grid = self._focused_grid()
        if focused_grid is None or focused_grid.highlighted is None:
            return None
        child = focused_grid.children[focused_grid.highlighted]
        if isinstance(child, ModelCard):
            return child.row.ref or None
        return None

    @work(thread=True)
    def _run_delete(self, model_name: str) -> None:
        """Remove a model in a background thread."""
        from lilbee.cli.tui.thread_safe import call_from_thread

        try:
            removed = get_model_manager().remove(model_name)
            if removed:
                call_from_thread(self, self.notify, msg.CATALOG_DELETED.format(name=model_name))
                call_from_thread(self, self._refresh_after_delete)
            else:
                call_from_thread(
                    self,
                    self.notify,
                    msg.CATALOG_DELETE_FAILED.format(error=model_name),
                    severity="error",
                )
        except Exception as exc:
            log.warning("Delete failed for %s", model_name, exc_info=True)
            call_from_thread(
                self,
                self.notify,
                msg.CATALOG_DELETE_FAILED.format(error=exc),
                severity="error",
            )

    def _refresh_after_delete(self) -> None:
        """Re-fetch remote models and refresh after deletion."""
        self._fetch_installed_names()
        self._refresh_view()
        self._fetch_remote_models()

    def _focused_grid(self) -> GridSelect | None:
        """Return the focused GridSelect (grid view), else None."""
        if self._grid_view and isinstance(self.focused, GridSelect):
            return self.focused
        return None

    def _list_items(self) -> list[ModelListItem]:
        """Return all visible list items in the list view."""
        return [item for item in self.query(ModelListItem) if item.display]

    def _focus_list_item(self, index: int) -> None:
        """Focus the list item at *index*, clamped to the visible range."""
        items = self._list_items()
        if not items:
            return
        clamped = max(0, min(index, len(items) - 1))
        items[clamped].focus()

    def _focused_list_index(self) -> int | None:
        """Index of the focused ModelListItem among visible list items."""
        if not isinstance(self.focused, ModelListItem):
            return None
        items = self._list_items()
        try:
            return items.index(self.focused)
        except ValueError:
            return None

    def _nudge_list(self, delta: int) -> None:
        idx = self._focused_list_index()
        if idx is None:
            self._focus_list_item(0)
            return
        self._focus_list_item(idx + delta)
        self._maybe_prefetch_on_nav()

    def _maybe_prefetch_on_nav(self) -> None:
        if self._grid_view or not self._hf_has_more or self._loading_more:
            return
        idx = self._focused_list_index()
        if idx is None:
            return
        if idx >= len(self._list_items()) - _HF_LOAD_MORE_TRIGGER:
            self._load_more()

    def _page_rows(self) -> int:
        """How many cursor steps make up one 'page' in the active view."""
        return _GRID_PAGE_ROWS if self._grid_view else _LIST_PAGE_ROWS

    def action_page_down(self) -> None:
        if isinstance(self.focused, Input):
            return
        if self._grid_view:
            if (grid := self._focused_grid()) is not None:
                for _ in range(self._page_rows()):
                    grid.action_cursor_down()
        else:
            self._nudge_list(self._page_rows())

    def action_page_up(self) -> None:
        if isinstance(self.focused, Input):
            return
        if self._grid_view:
            if (grid := self._focused_grid()) is not None:
                for _ in range(self._page_rows()):
                    grid.action_cursor_up()
        else:
            self._nudge_list(-self._page_rows())

    def action_cursor_down(self) -> None:
        if isinstance(self.focused, Input):
            return
        if self._grid_view:
            if (grid := self._focused_grid()) is not None:
                grid.action_cursor_down()
        else:
            self._nudge_list(1)

    def action_cursor_up(self) -> None:
        if isinstance(self.focused, Input):
            return
        if self._grid_view:
            if (grid := self._focused_grid()) is not None:
                grid.action_cursor_up()
        else:
            self._nudge_list(-1)

    def action_jump_top(self) -> None:
        if isinstance(self.focused, Input):
            return
        if self._grid_view:
            if (grid := self._focused_grid()) is not None:
                grid.highlight_first()
        else:
            self._focus_list_item(0)

    def action_jump_bottom(self) -> None:
        if isinstance(self.focused, Input):
            return
        if self._grid_view:
            if (grid := self._focused_grid()) is not None:
                grid.highlight_last()
        else:
            items = self._list_items()
            if items:
                self._focus_list_item(len(items) - 1)


@dataclass
class GridSection:
    """A named group of rows for the grid view."""

    heading: str
    rows: list[TableRow]


_TASK_BUCKET_ORDER = (ModelTask.CHAT, ModelTask.EMBEDDING, ModelTask.VISION, ModelTask.RERANK)


def _group_rows_for_grid(rows: list[TableRow]) -> list[GridSection]:
    """Group rows into sections for the grid view."""
    recommended: list[TableRow] = []
    installed: list[TableRow] = []
    by_task: dict[str, list[TableRow]] = {task: [] for task in _TASK_BUCKET_ORDER}
    # Display order is fixed by _TASK_BUCKET_ORDER, but any ModelTask value
    # not in that tuple still renders in its own section after the known
    # ones, so adding a new task variant never silently drops rows.
    extras: dict[str, list[TableRow]] = {}
    for row in rows:
        if row.featured:
            recommended.append(row)
            continue
        if row.installed:
            installed.append(row)
            continue
        bucket = by_task.get(row.task)
        if bucket is not None:
            bucket.append(row)
        else:
            extras.setdefault(row.task, []).append(row)
    return [
        GridSection(msg.HEADING_OUR_PICKS, recommended),
        GridSection(msg.HEADING_INSTALLED, installed),
        *[GridSection(task.capitalize(), by_task[task]) for task in _TASK_BUCKET_ORDER],
        *[GridSection(task.capitalize(), extras[task]) for task in extras],
    ]
