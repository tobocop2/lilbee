"""Catalog screen -- browse and install models via grid or list view."""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass
from typing import ClassVar

from textual import getters, on, work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import VerticalScroll
from textual.events import Click
from textual.screen import Screen
from textual.timer import Timer
from textual.widgets import Input, Static, TabbedContent, TabPane
from textual.worker import Worker, WorkerState

from lilbee.catalog import (
    CatalogModel,
    ModelFamily,
    ModelVariant,
    get_catalog,
    get_families,
    resolve_filename,
)
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.app import LilbeeApp, apply_active_model
from lilbee.cli.tui.screens.catalog_utils import (
    SORT_KEYS,
    CatalogRow,
    FrontierCatalogRow,
    KeyStatus,
    LocalCatalogRow,
    catalog_to_row,
    frontier_row_from_remote,
    matches_search,
    remote_to_row,
    variant_to_row,
)
from lilbee.cli.tui.thread_safe import call_from_thread
from lilbee.cli.tui.widgets.bottom_bars import BottomBars
from lilbee.cli.tui.widgets.grid_select import GridSelect
from lilbee.cli.tui.widgets.model_card import ModelCard
from lilbee.cli.tui.widgets.model_list import ModelList, ModelListSection
from lilbee.cli.tui.widgets.status_bar import ViewTabs
from lilbee.cli.tui.widgets.task_bar import TaskBar
from lilbee.cli.tui.widgets.top_bars import TopBars
from lilbee.core.config import cfg
from lilbee.core.services import get_services
from lilbee.modelhub.model_manager import RemoteModel, classify_remote_models
from lilbee.modelhub.models import ModelTask
from lilbee.providers.model_ref import OLLAMA_PREFIX
from lilbee.providers.sdk_backend import OLLAMA_BACKEND_NAME

log = logging.getLogger(__name__)

_HF_PAGE_SIZE = 25
_HF_LOAD_MORE_TRIGGER = 5
_NOTIFY_SEARCHING_TIMEOUT_SECONDS = 4
_ALL_TASKS = tuple(ModelTask)

_WORKER_FETCH_HF = "fetch_hf_models"
_WORKER_FETCH_MORE_HF = "fetch_more_hf"
_WORKER_FETCH_REMOTE = "fetch_remote_models"
_WORKER_FETCH_SEARCH = "fetch_hf_search"
_WORKER_FETCH_FRONTIER = "fetch_frontier_models"

_GRID_PAGE_ROWS = 3
_LIST_PAGE_ROWS = 10

_LOCAL_TAB_ID = "local"
_FRONTIER_TAB_ID = "frontier"
_FRONTIER_LIST_ID = "frontier-list"

_SORT_CYCLE: tuple[str, ...] = ("Name", "Downloads", "Size", "Params")

_RowCacheKey = tuple[int, int, int, bool, int, int]


@dataclass(frozen=True)
class _RowCacheEntry:
    """Memoized output of one ``_all_*_rows`` builder."""

    key: _RowCacheKey
    rows: list[LocalCatalogRow]


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

    _search_input = getters.query_one("#catalog-search", Input)
    _grid_container = getters.query_one("#catalog-grid", VerticalScroll)
    _list_widget = getters.query_one("#catalog-list", ModelList)

    def __init__(self) -> None:
        super().__init__()
        self._families: list[ModelFamily] = get_families()
        self._hf_models: list[CatalogModel] = []
        self._remote_models: list[RemoteModel] = []
        self._hf_offset = 0
        self._hf_has_more = True
        self._rows: list[LocalCatalogRow] = []
        self._sort_column: str = "Name"
        self._sort_ascending: bool = True
        self._pending_delete: str | None = None
        self._installed_names: set[str] = set()
        self._grid_view: bool = True
        self._hf_fetched: bool = False
        self._loading_more: bool = False
        self._grid_cache_key: tuple = ()
        self._list_cache_key: tuple = ()
        self._search_in_flight: bool = False
        self._frontier_rows: list[FrontierCatalogRow] = []
        # Bumped on every worker callback so the _all_*_rows caches
        # invalidate even when collection lengths happen to coincide.
        self._data_version: int = 0
        self._family_rows_cache: _RowCacheEntry | None = None
        self._hf_rows_cache: _RowCacheEntry | None = None
        self._remote_rows_cache: _RowCacheEntry | None = None
        self._view_switching: bool = False
        self._frontier_refresh_timer: Timer | None = None
        self._search_filter_timer: Timer | None = None
        self._scroll_prefetch_armed_at: float = 0.0

    def compose(self) -> ComposeResult:
        from textual.widgets import Footer

        with TopBars():
            yield ViewTabs()
            yield Input(placeholder=msg.CATALOG_FILTER_PLACEHOLDER, id="catalog-search")
        yield Static("", id="sort-label", shrink=True)
        with (
            TabbedContent(initial=_LOCAL_TAB_ID, id="catalog-tabs"),
            TabPane(msg.CATALOG_TAB_LOCAL, id=_LOCAL_TAB_ID),
        ):
            yield VerticalScroll(id="catalog-grid")
            yield ModelList(id="catalog-list")
        yield Static("", id="model-detail")
        with BottomBars():
            yield TaskBar()
            yield Footer()

    def on_mount(self) -> None:
        self._fetch_installed_names()
        self.add_class("-grid-view")
        # Defer the card mount so the catalog frame paints first, then the
        # cards stream in on the next refresh tick. With ~30 ModelCards the
        # synchronous mount adds ~600 ms of stylesheet work; deferring drops
        # perceived "frozen" time on Catalog open by half.
        self.call_after_refresh(self._refresh_grid)
        self.call_after_refresh(self._initial_focus_first_grid)
        self._fetch_remote_models()
        self._fetch_frontier_models()
        if isinstance(self.app, LilbeeApp):
            self.app.provider_availability_changed_signal.subscribe(
                self, self._on_provider_availability_changed
            )
        # Auto-load more HF rows when scrolled near the bottom of the list.
        self.watch(self._list_widget, "scroll_y", self._on_list_scrolled, init=False)

    def on_unmount(self) -> None:
        if isinstance(self.app, LilbeeApp):
            with contextlib.suppress(Exception):
                self.app.provider_availability_changed_signal.unsubscribe(self)

    _FRONTIER_REFRESH_DEBOUNCE = 1.0

    def _on_provider_availability_changed(self, _payload: tuple[str, object]) -> None:
        """Debounced refetch of frontier rows when an API key changes."""
        if self._frontier_refresh_timer is not None:
            self._frontier_refresh_timer.stop()
        self._frontier_refresh_timer = self.set_timer(
            self._FRONTIER_REFRESH_DEBOUNCE, self._fetch_frontier_models
        )

    def _focus_first_grid(self) -> None:
        """Focus the first GridSelect widget if available."""
        with contextlib.suppress(Exception):
            self.query_one(GridSelect).focus()

    def _initial_focus_first_grid(self) -> None:
        """on_mount initial focus: skip if a later refresh-tick has already
        landed focus elsewhere (e.g. a test focused #catalog-search before
        the streaming-section mount drained its scheduled callbacks)."""
        if self.focused is not None:
            return
        self._focus_first_grid()

    def _fetch_installed_names(self) -> None:
        """Populate installed identities from the shared ModelManager cache.

        The set contains both the canonical ref (``hf_repo/filename``) and
        the bare ``hf_repo`` so catalog rows whose ref is the repo alone
        still light up as installed when at least one quant of that repo
        has a manifest.
        """
        with contextlib.suppress(Exception):
            self._installed_names = set(get_services().model_manager.list_native_identities())
            self._data_version += 1

    def _active_tab_id(self) -> str:
        """Return the active TabbedContent pane id, or Local when not yet mounted."""
        try:
            return self.query_one("#catalog-tabs", TabbedContent).active or _LOCAL_TAB_ID
        except Exception:
            return _LOCAL_TAB_ID

    def action_toggle_view(self) -> None:
        """Toggle between grid and list view (Local tab only).

        Mid-toggle re-entry would tear the DOM (one toggle's mount_all
        running while the previous toggle's remove_children is still in
        flight). The _view_switching gate makes the toggle atomic.
        """
        if self._active_tab_id() != _LOCAL_TAB_ID:
            return
        if self._view_switching:
            return
        self._view_switching = True
        try:
            if self._grid_view:
                self._grid_view = False
                self.remove_class("-grid-view")
                self.add_class("-list-view")
                if not self._hf_fetched:
                    self._hf_fetched = True
                    self._fetch_all_hf_models()
                with self.app.batch_update():
                    self._refresh_list()
                self._focus_list_item(0)
            else:
                self._grid_view = True
                self.remove_class("-list-view")
                self.add_class("-grid-view")
                with self.app.batch_update():
                    self._refresh_grid()
                with contextlib.suppress(Exception):
                    self.query_one("#catalog-grid GridSelect", GridSelect).focus()
        finally:
            self._view_switching = False

    def action_focus_search(self) -> None:
        """Focus the filter input -- bound to / key."""
        self._search_input.focus()

    _SEARCH_FILTER_DEBOUNCE_SECONDS = 0.08

    @on(Input.Changed, "#catalog-search")
    def _on_search_changed(self, event: Input.Changed) -> None:
        """Schedule a filter pass after a short debounce.

        Per keystroke the filter walks every visible ``ModelCard`` /
        ``ModelListItem`` and toggles ``display``, which Textual treats
        as a layout invalidation. Without the debounce, a 5-char term
        produces 5 full layout passes; with it, typing collapses to a
        single pass once the user pauses.
        """
        if self._search_filter_timer is not None:
            self._search_filter_timer.stop()
        self._search_filter_timer = self.set_timer(
            self._SEARCH_FILTER_DEBOUNCE_SECONDS,
            self._apply_search_filter,
        )

    def _apply_search_filter(self) -> None:
        if self._active_tab_id() == _FRONTIER_TAB_ID:
            self._populate_frontier_list()
            return
        if self._grid_view:
            self._filter_grid()
            self._sync_grid_search_cta()
        else:
            self._filter_list()

    @on(Input.Submitted, "#catalog-search")
    def _on_search_submitted(self, event: Input.Submitted) -> None:
        """Enter installs the first visible match; falls through to a remote
        HF search when nothing matches locally."""
        if self._grid_view:
            if any(card.display for card in self.query(ModelCard)):
                self._select_first_visible_grid_card()
                return
        elif self._list_widget.option_count:
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
        """List-view counterpart: highlight + select the first row."""
        with contextlib.suppress(Exception):
            if self._list_widget.option_count:
                self._list_widget.highlighted = 0
                self._list_widget.focus()
                self._list_widget.action_select()

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
        return classify_remote_models(cfg.remote_base_url)

    @work(thread=True, name=_WORKER_FETCH_FRONTIER, exit_on_error=False)
    def _fetch_frontier_models(self) -> list[FrontierCatalogRow]:
        """Discover cloud chat models off the UI thread.

        ``discover_api_models`` imports litellm (heavy, >50ms) and probes
        every provider key, totaling several hundred ms even when no
        keys are set. Running it on the main thread froze the catalog
        on mount and on every signal-driven refresh; the worker keeps
        the screen responsive."""
        from lilbee.modelhub.model_manager import discover_api_models

        try:
            groups = discover_api_models()
        except Exception:
            log.debug("discover_api_models failed in worker", exc_info=True)
            return []

        rows: list[FrontierCatalogRow] = []
        for display_name, models in groups.items():
            provider_id = display_name.lower()
            key_field = f"{provider_id}_api_key"
            has_key = bool(getattr(cfg, key_field, ""))
            status = KeyStatus.READY if has_key else KeyStatus.MISSING_KEY
            for rm in models:
                rows.append(
                    frontier_row_from_remote(rm, provider_id=provider_id, key_status=status)
                )
        rows.sort(key=lambda r: (r.provider, r.name.lower()))
        return rows

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
            self._handle_worker_error_or_cancel(event.worker.name)
            return
        if event.state != WorkerState.SUCCESS:
            return
        result = event.worker.result
        if not isinstance(result, list):
            return
        worker_name = event.worker.name
        if not self._apply_worker_result(worker_name, result):
            return
        # FETCH_MORE_HF appends to the list (and is invisible to the capped grid).
        # Skip the full _refresh_view rebuild; just mount the new rows at the end.
        if worker_name == _WORKER_FETCH_MORE_HF and not self._grid_view:
            self._append_more_hf_to_list(result)
            return
        self._refresh_view()

    def _append_more_hf_to_list(self, new_models: list[CatalogModel]) -> None:
        """Append newly-arrived HF rows to the virtualized list view."""
        new_rows = [
            catalog_to_row(m, installed=self._is_installed(m.ref, m.hf_repo, m.gguf_filename))
            for m in new_models
        ]
        new_rows = self._sort_rows(new_rows)
        if not new_rows:
            self._update_sort_label()
            return
        self._rows.extend(new_rows)
        self._list_widget.append_rows(list(new_rows))
        self._list_cache_key = (
            tuple((r.name, r.installed) for r in self._rows),
            self._get_search_text(),
        )
        self._update_sort_label()

    def _handle_worker_error_or_cancel(self, name: str) -> None:
        if name == _WORKER_FETCH_MORE_HF:
            self._loading_more = False
        if name == _WORKER_FETCH_SEARCH:
            self._search_in_flight = False
            self._update_sort_label()

    def _apply_worker_result(self, name: str, result: list) -> bool:
        """Land worker results into the screen's caches.

        Returns True when the screen should refresh its view, False when
        the worker name is unrecognized (defensive: a future @work
        decorator name won't silently rebuild the grid)."""
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
        elif name == _WORKER_FETCH_FRONTIER:
            self._frontier_rows = result
            self._sync_frontier_tab()
        else:
            return False
        self._data_version += 1
        return True

    def _sync_frontier_tab(self) -> None:
        """Mount the Frontier tab when rows exist; unmount it when none."""
        try:
            tabs = self.query_one("#catalog-tabs", TabbedContent)
        except Exception:
            return
        existing = tabs.query(f"#{_FRONTIER_TAB_ID}")
        if not self._frontier_rows:
            for pane in existing:
                pane.remove()
            return
        if existing:
            self._populate_frontier_list()
            return
        pane = TabPane(
            msg.CATALOG_TAB_FRONTIER,
            ModelList(id=_FRONTIER_LIST_ID),
            id=_FRONTIER_TAB_ID,
        )
        tabs.add_pane(pane)
        self.call_after_refresh(self._populate_frontier_list)

    def _populate_frontier_list(self) -> None:
        try:
            ml = self.query_one(f"#{_FRONTIER_LIST_ID}", ModelList)
        except Exception:
            return
        ml.set_rows(_group_frontier_rows(self._build_frontier_rows(self._get_search_text())))

    def _get_search_text(self) -> str:
        return self._search_input.value.strip()

    def _local_rows_data_key(self) -> _RowCacheKey:
        """Cache key over the inputs that drive row construction.

        ``_data_version`` covers replacements and extensions both;
        search text deliberately omitted (we filter cached rows).
        """
        return (
            len(self._families),
            len(self._hf_models),
            len(self._remote_models),
            self._hf_fetched,
            len(self._installed_names),
            self._data_version,
        )

    def _all_family_rows(self) -> list[LocalCatalogRow]:
        key = self._local_rows_data_key()
        cached = self._family_rows_cache
        if cached is not None and cached.key == key:
            return cached.rows
        rows: list[LocalCatalogRow] = []
        for fam in self._families:
            for v in fam.variants:
                installed = self._is_installed(v.hf_repo, repo=v.hf_repo, filename=v.filename)
                rows.append(variant_to_row(v, fam, installed))
        self._family_rows_cache = _RowCacheEntry(key=key, rows=rows)
        return rows

    def _all_hf_rows(self) -> list[LocalCatalogRow]:
        key = self._local_rows_data_key()
        cached = self._hf_rows_cache
        if cached is not None and cached.key == key:
            return cached.rows
        rows: list[LocalCatalogRow] = []
        for m in self._hf_models:
            installed = self._is_installed(m.ref, repo=m.hf_repo, filename=m.gguf_filename)
            rows.append(catalog_to_row(m, installed))
        self._hf_rows_cache = _RowCacheEntry(key=key, rows=rows)
        return rows

    def _all_remote_rows(self) -> list[LocalCatalogRow]:
        key = self._local_rows_data_key()
        cached = self._remote_rows_cache
        if cached is not None and cached.key == key:
            return cached.rows
        rows = [remote_to_row(rm) for rm in self._remote_models]
        self._remote_rows_cache = _RowCacheEntry(key=key, rows=rows)
        return rows

    def _build_rows(self) -> list[LocalCatalogRow]:
        """Build filtered table rows from current data sources."""
        search = self._get_search_text()
        rows: list[LocalCatalogRow] = []
        rows.extend(self._build_family_rows(search))
        rows.extend(self._build_hf_rows(search))
        rows.extend(self._build_remote_rows(search))
        return rows

    def _build_family_rows(self, search: str) -> list[LocalCatalogRow]:
        """Filter the cached family rows against the active search."""
        if not search:
            return self._all_family_rows()
        return [r for r in self._all_family_rows() if matches_search(r, search)]

    def _build_hf_rows(self, search: str) -> list[LocalCatalogRow]:
        """Filter the cached HF rows against the active search."""
        if not search:
            return self._all_hf_rows()
        return [r for r in self._all_hf_rows() if matches_search(r, search)]

    def _build_remote_rows(self, search: str) -> list[LocalCatalogRow]:
        """Filter the cached remote rows against the active search."""
        if not search:
            return self._all_remote_rows()
        return [r for r in self._all_remote_rows() if matches_search(r, search)]

    def _build_frontier_rows(self, search: str) -> list[FrontierCatalogRow]:
        """Filter the cached frontier rows against the active search.

        The discovery itself runs in :meth:`_fetch_frontier_models` (a
        worker) because litellm import + key probing blocks the UI
        thread. Renderers call this synchronously to read the
        already-discovered rows, so no I/O happens here.
        """
        if not self._frontier_rows:
            return []
        return [row for row in self._frontier_rows if matches_search(row, search)]

    def _is_installed(self, name: str, repo: str = "", filename: str = "") -> bool:
        """Check if a model is installed by name or source repo/filename."""
        if name in self._installed_names:
            return True
        if repo and filename:
            return f"{repo}/{filename}" in self._installed_names
        return False

    def _sort_rows(self, rows: list[LocalCatalogRow]) -> list[LocalCatalogRow]:
        """Sort rows: featured first, then by current sort column."""
        key_fn = SORT_KEYS.get(self._sort_column, SORT_KEYS["Name"])
        # Stable sort: featured always first, then by column
        return sorted(
            rows,
            key=lambda r: (not r.featured, key_fn(r)),
            reverse=not self._sort_ascending,
        )

    def _refresh_view(self) -> None:
        """Refresh the active view (grid or list).

        Mount/remove of dozens of widgets is wrapped in batch_update so
        Textual coalesces layout passes; without it, the worker callback
        path can land inside an in-flight grid-list toggle and tear the
        DOM."""
        with self.app.batch_update():
            if self._grid_view:
                self._refresh_grid()
            else:
                self._refresh_list()

    # Cap protects against ItemGrid layout cost (~5 subwidgets per
    # card * N cards locks the UI). List view has no cap.
    _GRID_HF_BUDGET = 24

    def _refresh_grid(self) -> None:
        """Rebuild the grid view with all cards (called when data changes)."""
        family_rows = self._build_family_rows("")
        remote_rows = self._build_remote_rows("")
        hf_rows_full = self._build_hf_rows("") if self._hf_fetched else []
        hf_overflow = max(0, len(hf_rows_full) - self._GRID_HF_BUDGET)
        hf_rows = hf_rows_full[: self._GRID_HF_BUDGET]
        all_rows = family_rows + remote_rows + hf_rows
        row_key = (
            tuple((r.name, r.installed) for r in all_rows),
            self._get_search_text(),
        )
        if self._grid_cache_key == row_key:
            return
        self._grid_cache_key = row_key
        container = self._grid_container
        container.remove_children()
        sections = [s for s in _group_rows_for_grid(all_rows) if s.rows]
        if not sections:
            self._mount_grid_ctas(hf_overflow)
            return
        # Mount the first section immediately (cheap; user sees content fast),
        # then schedule the rest one section per refresh tick so each section
        # paints in its own frame instead of one ~600 ms freeze.
        self._mount_grid_section(sections[0])
        self._stream_grid_sections(sections[1:], hf_overflow)

    def _mount_grid_section(self, section: GridSection) -> None:
        cards = [ModelCard(row) for row in section.rows]
        grid = GridSelect(*cards, min_column_width=30, max_column_width=50)
        self._grid_container.mount_all(
            [
                Static(section.heading, classes="section-heading"),
                grid,
            ]
        )

    def _stream_grid_sections(self, remaining: list[GridSection], hf_overflow: int) -> None:
        if not remaining:
            self.call_after_refresh(self._mount_grid_ctas, hf_overflow)
            return
        next_section = remaining[0]
        rest = remaining[1:]
        self.call_after_refresh(self._stream_one_grid_section, next_section, rest, hf_overflow)

    def _stream_one_grid_section(
        self, section: GridSection, rest: list[GridSection], hf_overflow: int
    ) -> None:
        self._mount_grid_section(section)
        self._stream_grid_sections(rest, hf_overflow)

    def _mount_grid_ctas(self, hf_overflow: int) -> None:
        ctas: list[Static] = []
        if not self._hf_fetched:
            ctas.append(Static(msg.CATALOG_BROWSE_MORE, classes="grid-cta browse-more-hf"))
        elif hf_overflow:
            ctas.append(
                Static(
                    msg.CATALOG_GRID_OVERFLOW.format(count=hf_overflow),
                    classes="grid-cta",
                )
            )
        search = self._get_search_text()
        if search:
            ctas.append(
                Static(
                    msg.CATALOG_SEARCH_HF_CTA.format(query=search),
                    classes="grid-cta search-hf-cta",
                )
            )
        ctas.append(Static(msg.CATALOG_VIEW_TOGGLE_GRID, classes="grid-cta"))
        self._grid_container.mount_all(ctas)

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
        container = self._grid_container
        container.mount(Static(cta_text, classes="grid-cta search-hf-cta"))

    def _filter_grid(self) -> None:
        """Filter visible cards by search text without recreating widgets.

        Walks the grid container once per section: toggles each card's
        ``display`` and accumulates ``has_visible`` in the same pass, so
        we avoid a second ``self.query(ModelCard)`` DOM walk that would
        match the same set the section iteration already enumerates.
        """
        search = self._get_search_text()
        children = list(self._grid_container.children)
        for i, child in enumerate(children):
            if not child.has_class("section-heading"):
                continue
            grid = children[i + 1] if i + 1 < len(children) else None
            if not isinstance(
                grid, GridSelect
            ):  # pragma: no cover - heading always paired with grid
                continue
            has_visible = False
            for card in grid.children:
                if not isinstance(
                    card, ModelCard
                ):  # pragma: no cover - grid only mounts ModelCards
                    continue
                visible = matches_search(card.row, search)
                card.display = visible
                if visible:
                    has_visible = True
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

    @on(ModelList.Selected)
    def _on_model_list_selected(self, event: ModelList.Selected) -> None:
        """Handle model selection from any ModelList (Local list view or Frontier tab)."""
        self._select_row(event.row)

    def _refresh_list(self) -> None:
        """Rebuild the list view with local rows only (frontier lives in its own tab)."""
        self._rows = self._sort_rows(self._build_rows())
        search = self._get_search_text()
        list_key = (
            tuple((r.name, r.installed) for r in self._rows),
            search,
        )
        if self._list_cache_key == list_key:
            self._update_sort_label()
            return
        self._list_cache_key = list_key
        visible = [r for r in self._rows if not search or matches_search(r, search)]
        self._list_widget.set_rows([ModelListSection(heading=None, rows=list(visible))])
        self._update_sort_label()

    def _filter_list(self) -> None:
        """Filter the list view to rows matching the active search."""
        search = self._get_search_text()
        visible = [r for r in self._rows if not search or matches_search(r, search)]
        self._list_widget.set_rows([ModelListSection(heading=None, rows=list(visible))])
        # Cache key reflects the filtered shape so a no-op _refresh_list
        # immediately after a filter pass does not double-render.
        self._list_cache_key = (
            tuple((r.name, r.installed) for r in self._rows),
            search,
        )
        self._update_sort_label()

    def _update_sort_label(self) -> None:
        """Update the sort indicator label, switching copy by active tab."""
        label = self.query_one("#sort-label", Static)
        if self._active_tab_id() == _FRONTIER_TAB_ID:
            label.update(self._frontier_label_text())
            return
        direction = "asc" if self._sort_ascending else "desc"
        n_total = len(self._rows)
        if self._loading_more:
            count = f"{n_total} models · loading more…"
        elif self._hf_has_more:
            count = f"{n_total} models · press [b]n[/b] for more"
        else:
            count = f"{n_total} models"
        hint = msg.CATALOG_SEARCHING_HF if self._search_in_flight else msg.CATALOG_VIEW_TOGGLE_LIST
        label.update(f"Sort: {self._sort_column} ({direction})  |  {count}  |  {hint}")

    def _frontier_label_text(self) -> str:
        provider_count = len({r.provider for r in self._frontier_rows})
        return msg.CATALOG_FRONTIER_SUMMARY.format(
            count=len(self._frontier_rows), providers=provider_count
        )

    def action_cycle_sort(self) -> None:
        """Cycle the list-view sort column ascending: Name, Downloads, Size, Params."""
        if isinstance(self.focused, Input):
            return
        if self._active_tab_id() != _LOCAL_TAB_ID:
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
        # mount_all is async; focus the first row after Textual's next
        # refresh so the filter Input doesn't swallow the next `s` press.
        self.call_after_refresh(self._focus_list_item, 0)

    def _select_row(self, row: CatalogRow) -> None:
        """Handle row selection: install, switch model, or open settings."""
        if isinstance(row, FrontierCatalogRow):  # sealed-union dispatch
            self._select_frontier_row(row)
            return
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
            apply_active_model(self.app, "chat_model", ref)
            self.notify(msg.CATALOG_USING_REMOTE.format(name=row.remote_model.name))

    def _select_frontier_row(self, row: FrontierCatalogRow) -> None:
        """Activate a cloud model, or jump to settings when the key is missing."""
        if row.key_status == KeyStatus.READY:
            apply_active_model(self.app, "chat_model", row.ref)
            self.notify(msg.CATALOG_USING_FRONTIER.format(name=row.name, provider=row.provider))
            return
        key_field = f"{row.provider_id}_api_key"
        self.notify(
            msg.CATALOG_NEEDS_KEY.format(provider=row.provider, key_field=key_field),
            severity="warning",
            timeout=10,
        )
        if isinstance(self.app, LilbeeApp):
            self.app.switch_view("Settings")

    def _load_more(self) -> None:
        """Load next page of HF models, if any remain and no fetch is in flight."""
        if self._loading_more or not self._hf_has_more:
            return
        self._loading_more = True
        self._hf_offset += _HF_PAGE_SIZE
        self._fetch_more_hf()

    def action_load_more(self) -> None:
        """Keyboard trigger (``n``) so users can page without scrolling."""
        if self._active_tab_id() != _LOCAL_TAB_ID:
            return
        self._load_more()

    @on(TabbedContent.TabActivated, "#catalog-tabs")
    def _on_catalog_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        """Refresh sort label and re-populate the active pane."""
        self._update_sort_label()
        if event.pane.id == _FRONTIER_TAB_ID:
            self._populate_frontier_list()

    def _install_variant(self, variant: ModelVariant, family: ModelFamily) -> None:
        """Convert a variant back to a CatalogModel and trigger install."""
        entry = CatalogModel(
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
        if not isinstance(self.app, LilbeeApp):  # test apps aren't LilbeeApp
            self.notify(msg.CATALOG_NO_TASK_BAR, severity="error")
            return
        self.app.task_bar.start_download(model)
        self.notify(msg.CATALOG_QUEUED_DOWNLOAD.format(name=model.display_name))

    def action_go_back(self) -> None:
        # Escape unfocuses the filter input first so screen-level keys
        # (s / v) reach the screen, not the input.
        if isinstance(self.focused, Input):
            self._focus_list_or_grid()
            return
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

        mgr = get_services().model_manager
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
        if not self._grid_view and self._list_widget.has_focus:
            row = self._list_widget.highlighted_row()
            return row.ref or None if row else None
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
        try:
            removed = get_services().model_manager.remove(model_name)
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

    def _list_count(self) -> int:
        """Total options currently shown in the list view (excluding headings)."""
        return self._list_widget.row_count

    def _focus_list_item(self, index: int) -> None:
        """Highlight the row at *index*, clamped to the visible range."""
        count = self._list_widget.option_count
        if not count:
            return
        clamped = max(0, min(index, count - 1))
        self._list_widget.highlighted = clamped
        self._list_widget.focus()

    def _focused_list_index(self) -> int | None:
        """Index of the highlighted list row, or None when nothing is highlighted."""
        return self._list_widget.highlighted

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
        if idx >= self._list_widget.option_count - _HF_LOAD_MORE_TRIGGER:
            self._load_more()

    _SCROLL_PREFETCH_RATIO = 0.85
    _SCROLL_PREFETCH_COOLDOWN = 0.8

    def _on_list_scrolled(self, _scroll_y: float) -> None:
        """Trigger _load_more when the user scrolls near the bottom of the list."""
        if not self._scroll_prefetch_due():
            return
        self._scroll_prefetch_armed_at = time.monotonic()
        self._load_more()

    def _scroll_prefetch_due(self) -> bool:
        # Cooldown blocks a runaway cascade where appending rows shifts
        # max_scroll_y, the watcher refires, and load_more kicks off the
        # next fetch before the user notices.
        if self._grid_view or not self._hf_has_more or self._loading_more:
            return False
        if self._scroll_prefetch_armed_at:
            elapsed = time.monotonic() - self._scroll_prefetch_armed_at
            if elapsed < self._SCROLL_PREFETCH_COOLDOWN:
                return False
        max_y = self._list_widget.max_scroll_y
        if max_y <= 0:
            return False
        return self._list_widget.scroll_y / max_y >= self._SCROLL_PREFETCH_RATIO

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
            count = self._list_widget.option_count
            if count:
                self._focus_list_item(count - 1)
                self._maybe_prefetch_on_nav()


@dataclass
class GridSection:
    """A named group of rows for the grid view."""

    heading: str
    rows: list[CatalogRow]


_TASK_BUCKET_ORDER = (ModelTask.CHAT, ModelTask.EMBEDDING, ModelTask.VISION, ModelTask.RERANK)


def _group_frontier_rows(
    frontier_rows: list[FrontierCatalogRow],
) -> list[ModelListSection]:
    """Group frontier rows into provider-headed sections, alphabetical within."""
    if not frontier_rows:
        return []
    per_provider: dict[str, list[FrontierCatalogRow]] = {}
    for row in frontier_rows:
        per_provider.setdefault(row.provider, []).append(row)
    sections: list[ModelListSection] = []
    for provider in sorted(per_provider):
        rows = sorted(per_provider[provider], key=lambda r: r.name.lower())
        sections.append(ModelListSection(heading=provider, rows=list(rows)))
    return sections


def _group_rows_for_grid(local_rows: list[LocalCatalogRow]) -> list[GridSection]:
    """Group local rows into sections for the grid view."""
    recommended: list[CatalogRow] = []
    installed: list[CatalogRow] = []
    by_task: dict[str, list[CatalogRow]] = {task: [] for task in _TASK_BUCKET_ORDER}
    extras: dict[str, list[CatalogRow]] = {}
    for row in local_rows:
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
