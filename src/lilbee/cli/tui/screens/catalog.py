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
from textual.containers import Container, Horizontal, VerticalScroll
from textual.events import Click, Key, MouseScrollDown
from textual.message import Message
from textual.screen import Screen
from textual.timer import Timer
from textual.widgets import Footer, Input, Static, TabbedContent, TabPane
from textual.worker import Worker, WorkerState

from lilbee.app.services import get_services
from lilbee.catalog import (
    CatalogModel,
    ModelFamily,
    ModelVariant,
    get_catalog,
    get_families,
    resolve_filename,
)
from lilbee.catalog.types import ModelSource, ModelTask
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.app import LilbeeApp, apply_active_model
from lilbee.cli.tui.screens.catalog_grouping import (
    GridSection,
    for_you_sort_key,
    group_frontier_rows,
    group_rows_for_grid,
    group_task_rows_with_picks,
    row_cache_signature,
)
from lilbee.cli.tui.screens.catalog_utils import (
    SORT_KEYS,
    TAB_CHAT,
    TAB_DISCOVER,
    TAB_EMBED,
    TAB_ID_TO_TASK,
    TAB_LIBRARY,
    TAB_RERANK,
    TAB_VISION,
    TASK_TAB_IDS,
    CatalogRow,
    CatalogRowKind,
    FrontierCatalogRow,
    KeyStatus,
    LocalCatalogRow,
    SourceMode,
    catalog_to_row,
    family_to_size_variants,
    frontier_row_from_remote,
    matches_search,
    next_source_mode,
    remote_to_row,
    row_delete_id,
    variant_to_row,
)
from lilbee.cli.tui.thread_safe import call_from_thread
from lilbee.cli.tui.widgets.bottom_bars import BottomBars
from lilbee.cli.tui.widgets.catalog_detail import CatalogDetailDrawer
from lilbee.cli.tui.widgets.discover_rails import DiscoverRails
from lilbee.cli.tui.widgets.grid_select import GridSelect
from lilbee.cli.tui.widgets.model_card import ModelCard
from lilbee.cli.tui.widgets.model_grid import ModelGrid
from lilbee.cli.tui.widgets.model_list import ModelList, ModelListSection
from lilbee.cli.tui.widgets.status_bar import ViewTabs
from lilbee.cli.tui.widgets.task_bar import TaskBar
from lilbee.cli.tui.widgets.top_bars import TopBars
from lilbee.core.config import cfg
from lilbee.modelhub.model_manager import RemoteModel, classify_remote_models
from lilbee.providers.sdk_backend import get_provider_api_key
from lilbee.runtime.hardware import available_memory_for_fit, compute_fit

log = logging.getLogger(__name__)

# Models fetched per task per page. We make one /api/models call per
# task (chat / embedding / vision / rerank), so the user-visible page
# size is _HF_PAGE_SIZE * 4. Small pages keep each HF round-trip well
# under a second on a typical connection and keep the freshly-rendered
# row count low so layout reflow stays cheap.
_HF_PAGE_SIZE = 4
_HF_LOAD_MORE_TRIGGER = 4
_NOTIFY_SEARCHING_TIMEOUT_SECONDS = 4
_ALL_TASKS = tuple(ModelTask)

_WORKER_FETCH_HF = "fetch_hf_models"
_WORKER_FETCH_MORE_HF = "fetch_more_hf"
_WORKER_FETCH_REMOTE = "fetch_remote_models"
_WORKER_FETCH_SEARCH = "fetch_hf_search"
_WORKER_FETCH_FRONTIER = "fetch_frontier_models"
_WORKER_FETCH_INSTALLED = "fetch_installed_names"

_GRID_PAGE_ROWS = 3
_LIST_PAGE_ROWS = 10

# Per-tab DOM ids: f"grid-{tab_id}" / f"list-{tab_id}". Memoized on the
# screen so each access is one dict lookup, not a DOM walk.
_GRID_ID_PREFIX = "grid-"
_LIST_ID_PREFIX = "list-"

_SORT_CYCLE: tuple[str, ...] = ("Name", "Downloads", "Size", "Params")

# Braille spinner frames for the catalog pagination/search loading
# indicator. Cycled on a 100 ms timer while the catalog is fetching
# more HF rows or a remote search is in flight, so the user always
# has a moving signal during the wait instead of an empty pane.
_SPINNER_FRAMES: tuple[str, ...] = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
_SPINNER_INTERVAL_S = 0.1

_RowCacheKey = tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class _RowCacheEntry:
    """Memoized output of one ``_all_*_rows`` builder."""

    key: _RowCacheKey
    rows: list[LocalCatalogRow]


class CatalogScreen(Screen[None]):
    """Model catalog with grid (default) and list views."""

    app: LilbeeApp  # type: ignore[assignment]

    CSS_PATH = "catalog.tcss"
    AUTO_FOCUS = ""  # GridSelect is mounted dynamically; focused in on_mount

    HELP = (
        "# Catalog\n"
        "Six tabs: Discover (curated landing), Chat / Embed / Vision / Rerank,\n"
        "and Library (your installed local + activated cloud APIs).\n\n"
        "## Navigation\n"
        "- Arrows / j k h l: move the card cursor.\n"
        "- 1-6: jump to tab N.\n"
        "- Tab / Shift+Tab: cycle focus.\n\n"
        "## Actions\n"
        "- Enter: install the highlighted model (or activate, if cloud).\n"
        "- Space: toggle select.\n"
        "- d / Backspace / x: delete an installed model (two presses to confirm).\n"
        "- i: open the info modal for the highlighted card.\n"
        "- Right Arrow: expand a family card to show its size variants.\n\n"
        "## Filters and views\n"
        "- /: filter the active tab (Esc clears).\n"
        "- s: cycle sort (Name / Downloads / Size / Params).\n"
        "- v: toggle Grid vs List view on a task tab.\n"
        "- c: cycle source chip [local | cloud | both] on a task tab.\n"
        "- n: load more HF rows (or just keep scrolling).\n\n"
        "## Detail drawer\n"
        "- Ctrl+B: toggle the right-pane detail drawer.\n"
        "  Shows fit chip, size variants with per-variant fit, license, description.\n\n"
        "## Fit chip\n"
        "- Green 'fits +N GB': model fits with at least 1 GB headroom.\n"
        "- Amber 'tight +N GB': model fits but within the 0..1 GB band.\n"
        '- Red "won\'t N GB": model overflows available memory by N GB.\n\n'
        "## Other\n"
        "- q / Esc: back."
    )

    _ACTION_GROUP = Binding.Group("Actions", compact=True)
    _SCROLL_GROUP = Binding.Group("Scroll", compact=True)

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "go_back", "Back", show=True, group=_ACTION_GROUP),
        Binding("escape", "dismiss_filter", "", show=False),
        # Surfaced outside _ACTION_GROUP so the "Grid/List" affordance prints
        # in full in the footer instead of collapsing into the compact pill.
        # The "(faster)" tag tells users list view paginates without the
        # card layout overhead.
        Binding("v", "toggle_view", "Grid/List (faster)", show=True),
        Binding("slash", "focus_search", "Search", show=True, group=_ACTION_GROUP),
        # Delete sits outside _ACTION_GROUP so the footer renders it as
        # its own "D Delete" entry rather than collapsing it into the
        # compact "qv/di Actions" pill. Removing an installed model
        # needs to be obvious, not buried.
        Binding("d", "delete_model", "Delete", show=True),
        Binding("backspace", "delete_model", "Delete", show=False),
        Binding("x", "delete_model", "Delete", show=False),
        Binding("i", "show_info", "Info", show=True, group=_ACTION_GROUP),
        Binding("j", "cursor_down", "Nav", show=False, group=_SCROLL_GROUP),
        Binding("k", "cursor_up", "Nav", show=False, group=_SCROLL_GROUP),
        # Arrows move the card cursor too (auto-scrolls into view) so
        # the highlight follows the visible region. Decoupling them
        # into pure viewport scroll left a stale highlight on the
        # previously-focused card.
        Binding("down", "cursor_down", "Down", show=False, group=_SCROLL_GROUP),
        Binding("up", "cursor_up", "Up", show=False, group=_SCROLL_GROUP),
        # priority=True so vim jump-to-top/bottom always wins over the
        # focused ModelGrid's enter/select binding when keys collide.
        Binding("g", "jump_top", "Top", show=False, group=_SCROLL_GROUP, priority=True),
        Binding("G", "jump_bottom", "End", show=False, group=_SCROLL_GROUP, priority=True),
        Binding("space", "page_down", "PgDn", show=False, group=_SCROLL_GROUP),
        Binding("ctrl+d", "page_down", "PgDn", show=False, group=_SCROLL_GROUP),
        Binding("ctrl+u", "page_up", "PgUp", show=False, group=_SCROLL_GROUP),
        # Hidden from the footer so catalog still has <=5 visible bindings;
        # the sort-label surfaces "press n for more" and "press s to sort"
        # to the user instead.
        Binding("n", "load_more", "More", show=False, group=_ACTION_GROUP),
        Binding("s", "cycle_sort", "Sort", show=False, group=_ACTION_GROUP),
        Binding("ctrl+b", "toggle_drawer", "Detail", show=False, group=_ACTION_GROUP),
        Binding("c", "cycle_source", "Source", show=False, group=_ACTION_GROUP),
        # Numeric tab shortcuts; 1-6 jump to the corresponding tab in
        # ALL_TAB_IDS order (Discover, Chat, Embed, Vision, Rerank, Library).
        # priority=True so they win against any focused-widget binding that
        # might already grab digits (Textual's Tabs/ContentTabs has its own
        # numeric handling), and over-the-air shortcut feel matches the plan.
        Binding("1", "select_tab(0)", "Discover", show=False, priority=True),
        Binding("2", "select_tab(1)", "Chat", show=False, priority=True),
        Binding("3", "select_tab(2)", "Embed", show=False, priority=True),
        Binding("4", "select_tab(3)", "Vision", show=False, priority=True),
        Binding("5", "select_tab(4)", "Rerank", show=False, priority=True),
        Binding("6", "select_tab(5)", "Library", show=False, priority=True),
        # Discoverable tab cycling. The numeric jumps (1-6) above are quick
        # but hidden; > / < show in the footer so users learn the affordance.
        # ctrl+arrow conflicts with macOS desktop-space shortcuts, hence
        # vim-style angle brackets. priority=True so the active ModelGrid's
        # own focus cycling doesn't swallow them.
        Binding("greater_than_sign", "cycle_tab(1)", "Next tab", show=True, priority=True),
        Binding("less_than_sign", "cycle_tab(-1)", "Prev tab", show=True, priority=True),
    ]

    _search_input = getters.query_one("#catalog-search", Input)

    def __init__(self) -> None:
        super().__init__()
        self._families: list[ModelFamily] = get_families()
        self._hf_models: list[CatalogModel] = []
        self._remote_models: list[RemoteModel] = []
        # Per-task pagination state. Each task tab tracks its own HF offset
        # and has-more flag so paginating in one tab (e.g. Chat) only fetches
        # that task's next page; sibling tabs stay untouched.
        self._hf_offset_by_task: dict[ModelTask, int] = dict.fromkeys(_ALL_TASKS, 0)
        self._hf_has_more_by_task: dict[ModelTask, bool] = dict.fromkeys(_ALL_TASKS, True)
        self._hf_fetched_tasks: set[ModelTask] = set()
        self._rows: list[LocalCatalogRow] = []
        self._sort_column: str = "Name"
        self._sort_ascending: bool = True
        self._pending_delete: str | None = None
        self._installed_names: set[str] = set()
        self._grid_view: bool = True
        self._loading_more: bool = False
        # Per-tab grid/list cache keys. Each tab tracks its own last-rendered
        # shape; switching between already-populated tabs is a no-op refresh.
        self._grid_cache_keys: dict[str, tuple] = {}
        self._list_cache_keys: dict[str, tuple] = {}
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
        self._spinner_timer: Timer | None = None
        self._spinner_frame: int = 0
        # Active-tab cache + per-tab widget memoization. Avoids a second
        # query_one on every _grid_container / _list_widget access. Default
        # matches the TabbedContent's initial= value below.
        self._active_tab_id_cache: str = TAB_CHAT
        self._tab_grid_cache: dict[str, VerticalScroll] = {}
        self._tab_list_cache: dict[str, ModelList] = {}
        # During initial mount Textual fires TabActivated for whichever pane
        # ends up first in compose order (Discover) before our explicit
        # call_after_refresh setter activates Chat. Suppressing cache writes
        # while this flag is False keeps the cache pinned to its TAB_CHAT
        # __init__ default through the race; user-driven tab switches after
        # mount flip the flag and re-arm normal cache updates.
        self._activation_settled: bool = False
        # Per-tab source mode (local / cloud / both). Defaults to LOCAL on
        # every task tab so the catalog opens on the same row set the
        # mega-grid era surfaced; users opt into cloud-mixed views via `c`.
        self._source_modes: dict[str, SourceMode] = {
            tab_id: SourceMode.LOCAL for tab_id in TASK_TAB_IDS
        }
        # Hardware-fit baseline. Captured once at construction so the
        # cached row-build path can stamp each row's fit chip without
        # re-probing on every refresh.
        self._available_memory_bytes: int | None = available_memory_for_fit()

    def _grid_for_tab(self, tab_id: str) -> VerticalScroll:
        """Return (and memoize) the VerticalScroll for *tab_id*.

        Discover has no grid; falls through to TAB_CHAT so callers that
        access ``_grid_container`` while Discover is active never crash.
        Cached references are validated via ``is_running`` so a stale
        post-remount handle gets refreshed transparently.
        """
        target = TAB_CHAT if tab_id == TAB_DISCOVER else tab_id
        cached = self._tab_grid_cache.get(target)
        if cached is not None and cached.is_running:
            return cached
        container = self.query_one(f"#{_GRID_ID_PREFIX}{target}", VerticalScroll)
        self._tab_grid_cache[target] = container
        return container

    def _list_for_tab(self, tab_id: str) -> ModelList:
        """Return (and memoize) the ModelList for *tab_id*. Same fallthrough as _grid_for_tab."""
        target = TAB_CHAT if tab_id == TAB_DISCOVER else tab_id
        cached = self._tab_list_cache.get(target)
        if cached is not None and cached.is_running:
            return cached
        widget = self.query_one(f"#{_LIST_ID_PREFIX}{target}", ModelList)
        self._tab_list_cache[target] = widget
        return widget

    @property
    def _grid_container(self) -> VerticalScroll:
        return self._grid_for_tab(self._active_tab_id_cache)

    @property
    def _list_widget(self) -> ModelList:
        return self._list_for_tab(self._active_tab_id_cache)

    @property
    def _search_focused(self) -> bool:
        """True when the search Input widget owns focus.

        Used to short-circuit digit / single-character action handlers so the
        keystroke lands in the search field instead of activating a tab.
        """
        return isinstance(self.focused, Input)

    def compose(self) -> ComposeResult:
        from lilbee.cli.tui.widgets.grid_list_toggle import GridListToggle

        with TopBars():
            yield ViewTabs()
            yield Input(
                placeholder=msg.CATALOG_FILTER_PLACEHOLDER,
                id="catalog-search",
                classes="-hidden",
            )
        with Horizontal(id="catalog-toolbar"):
            yield GridListToggle()
            yield Static("", id="sort-label", shrink=True)
            yield Static("", id="catalog-loading-spinner")
        # Horizontal split: TabbedContent fills, CatalogDetailDrawer docks
        # right at fixed width and toggles via the -collapsed class. Each
        # per-task tab has its own VerticalScroll + ModelList so prefetch
        # only extends the active tab's grid; the single mega-grid was the
        # source of cross-section viewport jumps on pagination.
        with Horizontal(id="catalog-body"):
            with (
                Container(id="catalog-tabs-wrap"),
                TabbedContent(initial=TAB_CHAT, id="catalog-tabs"),
            ):
                with TabPane(msg.CATALOG_TAB_DISCOVER, id=TAB_DISCOVER):
                    yield DiscoverRails(id="discover-rails")
                with TabPane(msg.CATALOG_TAB_CHAT, id=TAB_CHAT):
                    yield VerticalScroll(
                        id=f"{_GRID_ID_PREFIX}{TAB_CHAT}", classes="catalog-grid-pane"
                    )
                    yield ModelList(id=f"{_LIST_ID_PREFIX}{TAB_CHAT}")
                with TabPane(msg.CATALOG_TAB_EMBED, id=TAB_EMBED):
                    yield VerticalScroll(
                        id=f"{_GRID_ID_PREFIX}{TAB_EMBED}", classes="catalog-grid-pane"
                    )
                    yield ModelList(id=f"{_LIST_ID_PREFIX}{TAB_EMBED}")
                with TabPane(msg.CATALOG_TAB_VISION, id=TAB_VISION):
                    yield VerticalScroll(
                        id=f"{_GRID_ID_PREFIX}{TAB_VISION}", classes="catalog-grid-pane"
                    )
                    yield ModelList(id=f"{_LIST_ID_PREFIX}{TAB_VISION}")
                with TabPane(msg.CATALOG_TAB_RERANK, id=TAB_RERANK):
                    yield VerticalScroll(
                        id=f"{_GRID_ID_PREFIX}{TAB_RERANK}", classes="catalog-grid-pane"
                    )
                    yield ModelList(id=f"{_LIST_ID_PREFIX}{TAB_RERANK}")
                with TabPane(msg.CATALOG_TAB_LIBRARY, id=TAB_LIBRARY):
                    yield VerticalScroll(
                        id=f"{_GRID_ID_PREFIX}{TAB_LIBRARY}", classes="catalog-grid-pane"
                    )
                    yield ModelList(id=f"{_LIST_ID_PREFIX}{TAB_LIBRARY}")
            yield CatalogDetailDrawer(id="catalog-detail-drawer", classes="-collapsed")
        with BottomBars():
            yield TaskBar()
            yield Footer()

    def on_mount(self) -> None:
        self._fetch_installed_names()
        # Force Chat as the initial active tab. `TabbedContent(initial=...)`
        # doesn't take effect when panes are added via `with TabPane(...)`
        # (Textual resolves initial at construction time but the panes mount
        # after), so we set active explicitly via call_after_refresh so the
        # TabActivated cascade has already settled before our setter runs.
        # Chat is the most common landing destination; users opt into
        # Discover via keyboard shortcut.
        self.call_after_refresh(self._activate_initial_tab)
        self.add_class("-grid-view")

    def _activate_initial_tab(self) -> None:
        try:
            tabs = self.query_one("#catalog-tabs", TabbedContent)
        except Exception:
            self._activation_settled = True
            return
        if self._active_tab_id_cache == TAB_CHAT and tabs.active != TAB_CHAT:
            tabs.active = TAB_CHAT
        if not self._activation_settled:
            self._activation_settled = True
        self.call_after_refresh(self._refresh_grid)
        self.call_after_refresh(self._initial_focus_first_grid)
        self._fetch_remote_models()
        self._fetch_frontier_models()
        # Eagerly load the HF catalog for the initial chat tab. Sibling
        # task tabs fetch lazily on first activation (see
        # `_on_catalog_tab_activated`) so opening the catalog only costs
        # one HF round-trip instead of four.
        self._ensure_task_initial_fetch(ModelTask.CHAT)
        self.app.provider_availability_changed_signal.subscribe(
            self, self._on_provider_availability_changed
        )
        # Auto-load more HF rows when scrolled near the bottom in either view.
        # Watch every per-task tab's container plus the Library container.
        # Inactive tabs never scroll, so the handler runs only for the active
        # tab; this is cheaper than tearing down and re-installing the watch
        # on every tab activation.
        for tab_id in (*TASK_TAB_IDS, TAB_LIBRARY):
            with contextlib.suppress(Exception):
                self.watch(
                    self._list_for_tab(tab_id), "scroll_y", self._on_list_scrolled, init=False
                )
                self.watch(
                    self._grid_for_tab(tab_id), "scroll_y", self._on_grid_scrolled, init=False
                )

    def on_unmount(self) -> None:
        with contextlib.suppress(Exception):
            self.app.provider_availability_changed_signal.unsubscribe(self)
        self._stop_spinner_timer()

    def on_screen_suspend(self) -> None:
        """Pause the spinner timer while the screen is offscreen.

        Without this the 100 ms braille tick keeps firing for the full
        TUI session even when the catalog is not visible, costing ~4%
        of main-thread CPU forever.
        """
        self._stop_spinner_timer()

    def on_screen_resume(self) -> None:
        """Re-arm the spinner only if a fetch is still in flight."""
        if self._loading_more or self._search_in_flight:
            self._sync_loading_spinner()

    def _stop_spinner_timer(self) -> None:
        if self._spinner_timer is not None:
            self._spinner_timer.stop()
            self._spinner_timer = None

    _FRONTIER_REFRESH_DEBOUNCE = 1.0

    def _on_provider_availability_changed(self, _payload: tuple[str, object]) -> None:
        """Debounced refetch of frontier rows when an API key changes."""
        if self._frontier_refresh_timer is not None:
            self._frontier_refresh_timer.stop()
        self._frontier_refresh_timer = self.set_timer(
            self._FRONTIER_REFRESH_DEBOUNCE, self._fetch_frontier_models
        )

    def _focus_first_grid(self) -> None:
        """Focus the first grid widget in the active tab's container."""
        for cls in (ModelGrid, GridSelect):
            with contextlib.suppress(Exception):
                self._grid_container.query(cls).first().focus()
                return

    def _initial_focus_first_grid(self) -> None:
        """on_mount initial focus: skip if a later refresh-tick has already
        landed focus elsewhere (e.g. a test focused #catalog-search before
        the streaming-section mount drained its scheduled callbacks)."""
        if self.focused is not None:
            return
        self._focus_first_grid()

    @work(thread=True, name=_WORKER_FETCH_INSTALLED, exit_on_error=False)
    def _fetch_installed_names(self) -> list[str]:
        """Populate installed identities from the shared ModelManager cache.

        Runs off the UI thread because ``list_native_identities`` walks the
        registry on disk; on Windows with Defender real-time scanning the
        walk can take seconds and ``on_mount`` would otherwise call it on
        the main loop. The result lands back via ``_apply_worker_result``
        so it picks up the same ``_data_version`` bump + view refresh as
        every other catalog worker. The set contains both the canonical
        ref (``hf_repo/filename``) and the bare ``hf_repo`` so catalog
        rows whose ref is the repo alone still light up as installed when
        at least one quant of that repo has a manifest.
        """
        try:
            return list(get_services().model_manager.list_native_identities())
        except Exception:
            log.debug("Failed to fetch installed model identities", exc_info=True)
            return []

    def _active_tab_id(self) -> str:
        """Return the cached active tab id; falls back to TAB_CHAT pre-mount.

        The cache is updated by ``_on_catalog_tab_activated`` so this is a
        bare attribute read, not a DOM walk. Prefer this over a fresh
        ``TabbedContent.active`` lookup on every check.
        """
        return self._active_tab_id_cache

    def _active_task(self) -> ModelTask | None:
        """Return the active tab's task, or None on Discover / Library."""
        return TAB_ID_TO_TASK.get(self._active_tab_id())

    def _active_task_has_more(self) -> bool:
        """True iff the active task tab has another HF page available.

        Discover and Library tabs return False; neither paginates.
        """
        task = self._active_task()
        if task is None:
            return False
        return self._hf_has_more_by_task.get(task, False)

    def _hf_fetched_any(self) -> bool:
        """True iff any task has had its first HF page fetched.

        Renders gate HF sections on this so the catalog doesn't paint
        empty HF rows before the first per-task fetch lands.
        """
        return bool(self._hf_fetched_tasks)

    def _ensure_task_initial_fetch(self, task: ModelTask) -> None:
        """Fire the per-task initial HF fetch once; idempotent on repeats."""
        if task in self._hf_fetched_tasks:
            return
        self._hf_fetched_tasks.add(task)
        self._fetch_initial_hf_models_for_task(task)

    def action_toggle_view(self) -> None:
        """Toggle between grid and list view on the active task tab.

        Mid-toggle re-entry would tear the DOM (one toggle's mount_all
        running while the previous toggle's remove_children is still in
        flight). The _view_switching gate makes the toggle atomic.
        Discover and Library tabs don't expose the toggle.
        """
        if self._active_tab_id() not in TASK_TAB_IDS:
            return
        if self._view_switching:
            return
        self._view_switching = True
        try:
            if self._grid_view:
                self._grid_view = False
                self.remove_class("-grid-view")
                self.add_class("-list-view")
                active_task = TAB_ID_TO_TASK.get(self._active_tab_id())
                if active_task is not None:
                    self._ensure_task_initial_fetch(active_task)
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
                    self._grid_container.query_one(ModelGrid).focus()
        finally:
            self._view_switching = False
        self._sync_grid_list_toggle()

    def _sync_grid_list_toggle(self) -> None:
        from lilbee.cli.tui.widgets.grid_list_toggle import GridListToggle

        with contextlib.suppress(Exception):
            self.query_one(GridListToggle).set_grid(self._grid_view)

    def action_focus_search(self) -> None:
        """Reveal and focus the filter input. Bound to / key."""
        self._search_input.remove_class("-hidden")
        self._search_input.focus()

    _SEARCH_FILTER_DEBOUNCE_SECONDS = 0.08

    @on(Input.Changed, "#catalog-search")
    def _on_search_changed(self, event: Input.Changed) -> None:
        """Schedule a filter pass after a short debounce.

        Each keystroke triggers a grid re-render or a list redraw, both of
        which Textual treats as layout invalidations. Without the debounce
        a 5-char term produces 5 full passes; with it, typing collapses
        to a single pass once the user pauses.
        """
        if self._search_filter_timer is not None:
            self._search_filter_timer.stop()
        self._search_filter_timer = self.set_timer(
            self._SEARCH_FILTER_DEBOUNCE_SECONDS,
            self._apply_search_filter,
        )

    def _apply_search_filter(self) -> None:
        if self._active_tab_id() == TAB_LIBRARY:
            self._populate_library_list()
            return
        if self._active_tab_id() == TAB_DISCOVER:
            return
        if self._grid_view:
            self._filter_grid()
        else:
            self._filter_list()

    @on(Input.Submitted, "#catalog-search")
    def _on_search_submitted(self, event: Input.Submitted) -> None:
        """Enter installs the first visible match; falls through to a remote
        HF search when nothing matches locally."""
        if self._grid_view:
            if any(grid.rows for grid in self._grid_container.query(ModelGrid)):
                self._select_first_visible_grid_card()
                return
        elif self._list_widget.option_count:
            self._select_first_visible_list_item()
            return
        self._trigger_remote_search(self._get_search_text())

    def _trigger_remote_search(self, query: str) -> None:
        """Fire the HF search worker for the active task, unless one is in flight.

        Search is task-scoped so typing on the Chat tab only surfaces chat
        models; embedding/vision/rerank rows can never leak into the active
        list. Non-task tabs (Discover/Library) can't reach this path because
        the search Input is hidden on them.
        """
        if self._search_in_flight or not query:
            return
        active_task = TAB_ID_TO_TASK.get(self._active_tab_id())
        if active_task is None:
            return
        self._search_in_flight = True
        self._update_sort_label()
        self._sync_loading_spinner()
        # Sort label is hidden in grid view, so the toast is the only feedback there.
        self.notify(msg.CATALOG_SEARCHING_HF, timeout=_NOTIFY_SEARCHING_TIMEOUT_SECONDS)
        self._fetch_hf_search(query, active_task)

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
            for grid in self._grid_container.query(ModelGrid):
                if grid.rows:
                    grid.focus()
                    grid.highlighted = 0
                    grid.action_select()
                    return

    def _select_first_visible_list_item(self) -> None:
        """List-view counterpart: highlight + select the first row."""
        with contextlib.suppress(Exception):
            if self._list_widget.option_count:
                self._list_widget.highlighted = 0
                self._list_widget.focus()
                self._list_widget.action_select()

    def _fetch_hf_page_for_task(self, task: ModelTask) -> list[CatalogModel]:
        """Fetch one HF page for *task* at the task's own offset.

        Dedupes against repos already in ``self._hf_models`` so re-fetches
        from a stale offset don't double-count rows. Writes the per-task
        ``has_more`` directly on the screen from the worker thread; the
        dict assignment is GIL-atomic and the main thread only reads.
        """
        offset = self._hf_offset_by_task[task]
        result = get_catalog(
            task=task,
            featured=False,
            limit=_HF_PAGE_SIZE,
            offset=offset,
        )
        self._hf_has_more_by_task[task] = result.has_more
        existing_repos = {m.hf_repo for m in self._hf_models}
        return [m for m in result.models if not m.featured and m.hf_repo not in existing_repos]

    @work(thread=True, name=_WORKER_FETCH_HF)
    def _fetch_initial_hf_models_for_task(self, task: ModelTask) -> list[CatalogModel]:
        """Fetch the first HF page for *task* (extends the merged store)."""
        return self._fetch_hf_page_for_task(task)

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
            has_key = get_provider_api_key(provider_id) is not None
            status = KeyStatus.READY if has_key else KeyStatus.MISSING_KEY
            for rm in models:
                rows.append(
                    frontier_row_from_remote(rm, provider_id=provider_id, key_status=status)
                )
        rows.sort(key=lambda r: (r.provider, r.name.lower()))
        return rows

    @work(thread=True, name=_WORKER_FETCH_MORE_HF)
    def _fetch_more_hf_for_task(self, task: ModelTask) -> list[CatalogModel]:
        """Fetch the next HF page for *task* (extends the merged store)."""
        return self._fetch_hf_page_for_task(task)

    @work(thread=True, name=_WORKER_FETCH_SEARCH, exit_on_error=False)
    def _fetch_hf_search(self, query: str, task: ModelTask) -> list[CatalogModel]:
        """Fetch HF models matching *query* for *task* only (worker thread)."""
        existing_repos = {m.hf_repo for m in self._hf_models}
        result = get_catalog(
            task=task,
            featured=False,
            search=query,
            limit=_HF_PAGE_SIZE,
            offset=0,
        )
        return [m for m in result.models if not m.featured and m.hf_repo not in existing_repos]

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
        # A fast worker can complete before TabbedContent finishes mounting
        # its panes; tolerate that and let the deferred _refresh_grid that
        # _activate_initial_tab schedules rebuild against the applied state.
        from textual.css.query import NoMatches

        with contextlib.suppress(NoMatches):
            # FETCH_MORE_HF appends to the active view's tail; skip the full
            # _refresh_view rebuild so scroll position and focus are preserved.
            if worker_name == _WORKER_FETCH_MORE_HF:
                if self._grid_view:
                    self._refresh_grid()
                else:
                    self._append_more_hf_to_list(result)
                return
            self._refresh_view()

    def _append_more_hf_to_list(self, new_models: list[CatalogModel]) -> None:
        """Append newly-arrived HF rows to the active task tab's list.

        Falls back to a full ``_refresh_view`` on the rare tab-switch
        race where the worker's payload no longer matches the active
        task; otherwise a blind extend would leak foreign rows into a
        sibling tab's list.
        """
        active_task = self._active_task()
        if active_task is None or any(m.task != active_task for m in new_models):
            self._refresh_view()
            return
        new_rows = self._sort_rows(
            [
                catalog_to_row(m, installed=self._is_installed(m.ref, m.hf_repo, m.gguf_filename))
                for m in new_models
            ]
        )
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
        self._sync_loading_spinner()

    def _apply_worker_result(self, name: str, result: list) -> bool:
        """Land worker results into the screen's caches.

        Returns True when the screen should refresh its view, False when
        the worker name is unrecognized (defensive: a future @work
        decorator name won't silently rebuild the grid)."""
        if name == _WORKER_FETCH_HF:
            # Per-task initial fetches all share this worker name; each
            # one carries dedup-filtered new rows (see
            # ``_fetch_hf_page_for_task``) so extend is correct here.
            self._hf_models.extend(result)
            self._loading_more = False
        elif name == _WORKER_FETCH_MORE_HF:
            self._hf_models.extend(result)
            self._loading_more = False
        elif name == _WORKER_FETCH_SEARCH:
            self._hf_models.extend(result)
            self._search_in_flight = False
            self._update_sort_label()
        elif name == _WORKER_FETCH_REMOTE:
            self._remote_models = result
        elif name == _WORKER_FETCH_FRONTIER:
            self._frontier_rows = result
            self._populate_library_list()
        elif name == _WORKER_FETCH_INSTALLED:
            self._installed_names = set(result)
        else:
            return False
        self._data_version += 1
        self._sync_loading_spinner()
        # If the user is parked on Discover, re-populate the rails so the
        # Fresh-on-the-Hub strip fills as HF rows arrive. Without this the
        # rail stays empty for the lifetime of the Discover view because
        # _populate_discover_rails fires only on tab activation.
        if self._active_tab_id_cache == TAB_DISCOVER:
            self._populate_discover_rails()
        return True

    def _populate_library_list(self) -> None:
        """Render the Library tab: installed local + activated cloud APIs in both views."""
        search = self._get_search_text()
        installed_rows: list[LocalCatalogRow] = []
        for source in (self._all_family_rows, self._all_hf_rows, self._all_remote_rows):
            with contextlib.suppress(AttributeError):
                installed_rows.extend(r for r in source() if r.installed)
        if search:
            installed_rows = [r for r in installed_rows if matches_search(r, search)]
        frontier: list[FrontierCatalogRow] = []
        with contextlib.suppress(AttributeError):
            frontier = self._build_frontier_rows(search)
        self._render_library_list(installed_rows, frontier)
        self._render_library_grid(installed_rows, frontier)

    def _render_library_list(
        self,
        installed_rows: list[LocalCatalogRow],
        frontier: list[FrontierCatalogRow],
    ) -> None:
        try:
            ml = self._list_for_tab(TAB_LIBRARY)
        except Exception:
            return
        sections: list[ModelListSection] = []
        if installed_rows:
            sections.append(
                ModelListSection(heading=msg.HEADING_INSTALLED, rows=list(installed_rows))
            )
        sections.extend(group_frontier_rows(frontier))
        ml.set_rows(sections)

    def _render_library_grid(
        self,
        installed_rows: list[LocalCatalogRow],
        frontier: list[FrontierCatalogRow],
    ) -> None:
        try:
            container = self._grid_for_tab(TAB_LIBRARY)
        except Exception:
            return
        sections: list[GridSection] = []
        if installed_rows:
            sections.append(GridSection(heading=msg.HEADING_INSTALLED, rows=list(installed_rows)))
        if frontier:
            sections.append(GridSection(heading="Cloud", rows=list(frontier)))
        existing_grids = list(container.query(ModelGrid))
        existing_headings = [
            w for w in container.query(".section-heading") if isinstance(w, Static)
        ]
        if existing_grids and len(existing_grids) == len(sections):
            for grid, heading, section in zip(
                existing_grids, existing_headings, sections, strict=False
            ):
                heading.update(section.heading)
                grid.set_rows(section.rows)
            return
        container.remove_children()
        for section in sections:
            container.mount_all(
                [
                    Static(section.heading, classes="section-heading"),
                    ModelGrid(section.rows, name=section.heading, classes="catalog-section"),
                ]
            )

    def _get_search_text(self) -> str:
        # Deferred refresh callbacks can land while the screen is between
        # mount cycles (e.g. switch_view chaining); the descriptor query
        # would otherwise raise NoMatches and crash the callback.
        try:
            return self._search_input.value.strip()
        except Exception:
            return ""

    def _local_rows_data_key(self) -> _RowCacheKey:
        """Cache key over the inputs that drive row construction.

        ``_data_version`` covers replacements and extensions both;
        search text deliberately omitted (we filter cached rows).
        """
        return (
            len(self._families),
            len(self._hf_models),
            len(self._remote_models),
            len(self._hf_fetched_tasks),
            len(self._installed_names),
            self._data_version,
        )

    def _all_family_rows(self) -> list[LocalCatalogRow]:
        """One row per featured family, aggregating its quants into size_variants.

        The mega-grid era emitted one row per ``ModelVariant``; the same
        family showed up three or four times stacked next to each other,
        once per quant. The redesign collapses each family into a single
        card whose ``size_variants`` strip carries every quant. Primary
        variant (recommended; otherwise the smallest) drives the card's
        primary metadata + fit chip; the strip lets users pick a
        non-primary size without leaving the grid.
        """
        key = self._local_rows_data_key()
        cached = self._family_rows_cache
        if cached is not None and cached.key == key:
            return cached.rows
        rows: list[LocalCatalogRow] = []
        for fam in self._families:
            if not fam.variants:
                continue
            primary = next(
                (v for v in fam.variants if v.recommended),
                min(fam.variants, key=lambda v: v.size_mb),
            )
            family_installed = any(
                self._is_installed(v.hf_repo, repo=v.hf_repo, filename=v.filename)
                for v in fam.variants
            )
            row = variant_to_row(primary, fam, family_installed)
            row.size_variants = family_to_size_variants(fam)
            rows.append(row)
        self._stamp_fit(rows)
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
        self._stamp_fit(rows)
        self._hf_rows_cache = _RowCacheEntry(key=key, rows=rows)
        return rows

    def _all_remote_rows(self) -> list[LocalCatalogRow]:
        key = self._local_rows_data_key()
        cached = self._remote_rows_cache
        if cached is not None and cached.key == key:
            return cached.rows
        rows = [remote_to_row(rm) for rm in self._remote_models]
        # Remote rows don't carry a known size; _stamp_fit no-ops on those.
        self._stamp_fit(rows)
        self._remote_rows_cache = _RowCacheEntry(key=key, rows=rows)
        return rows

    def _stamp_fit(self, rows: list[LocalCatalogRow]) -> None:
        """Stamp each row's hardware-fit chip in place.

        Runs only inside the cached row builders, so this is one pass per
        data refresh, not per render. Rows whose ``sort_size`` is zero
        (remote / unknown size) leave ``fit`` as ``None`` and the card
        renderer omits the chip. Available-memory probe is captured once
        at __init__; if the probe failed, every row falls through chip-less.
        """
        if self._available_memory_bytes is None:
            return
        bytes_per_gb = 1024**3
        for row in rows:
            if row.sort_size <= 0:
                continue
            row.fit = compute_fit(
                model_size_bytes=int(row.sort_size * bytes_per_gb),
                available_bytes=self._available_memory_bytes,
            )

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

    def _refresh_grid(self) -> None:
        """Rebuild grid view; extend in-place when sections already mounted.

        Initial paint mounts everything (first time a tab is opened).
        Subsequent dataset updates (HF pagination, sort change, filter)
        update each existing ModelGrid via set_rows rather than tearing
        the container down and re-mounting from scratch. Avoids a 100%
        CPU spike on every "Browse more" return.
        """
        prep = self._prepare_grid_refresh()
        if prep is None:
            self._update_sort_label()
            return
        sections, hf_count = prep
        if not sections:
            self._grid_container.remove_children()
            self._mount_grid_ctas(hf_count=hf_count)
            self._update_sort_label()
            return
        if self._extend_grid_sections_in_place(sections, hf_count):
            return
        self._remount_grid_sections(sections, hf_count)
        self._update_sort_label()

    def _prepare_grid_refresh(self) -> tuple[list[GridSection], int] | None:
        """Build sections + cache them. Returns None when the cache is hot.

        On the None branch the caller refreshes the sort label so the
        cached path still picks up sort-toggle clicks.
        """
        search = self._get_search_text()
        family_rows = self._build_family_rows(search)
        remote_rows = self._build_remote_rows(search)
        hf_rows = self._build_hf_rows(search) if self._hf_fetched_any() else []
        all_rows = family_rows + remote_rows + hf_rows
        active_tab = self._active_tab_id_cache
        tab_rows = self._rows_for_active_tab(all_rows, active_tab)
        # Keep self._rows in sync (locals-only) so the toolbar sort-label
        # can render "{n} loaded" whichever view (grid or list) is active.
        # Frontier rows render in their own Cloud section but don't count
        # toward the local-row tally.
        local_tab_rows: list[LocalCatalogRow] = [
            r for r in tab_rows if r.kind == CatalogRowKind.LOCAL
        ]
        self._rows = local_tab_rows
        row_key = (
            tuple(row_cache_signature(r) for r in tab_rows),
            search,
        )
        # Per-tab cache key: switching back to an already-rendered tab
        # is a no-op refresh; only sort-label refreshes. Keyed by
        # active_tab so other tabs' caches survive in-place.
        if self._grid_cache_keys.get(active_tab) == row_key:
            return None
        self._grid_cache_keys[active_tab] = row_key
        if active_tab in TASK_TAB_IDS:
            active_task = TAB_ID_TO_TASK[active_tab]
            task_label = active_task.value.capitalize()
            # Split locals and frontier so the picks/installed grouping
            # only sees LocalCatalogRow (it reads .featured / .installed
            # which FrontierCatalogRow doesn't carry). Frontier rows land
            # under their own "Cloud" section appended below.
            frontier_only = [r for r in tab_rows if r.kind == CatalogRowKind.FRONTIER]
            sections = [s for s in group_task_rows_with_picks(local_tab_rows, task_label) if s.rows]
            if frontier_only:
                sections.append(GridSection(heading="Cloud", rows=list(frontier_only)))
            hf_count = sum(1 for r in hf_rows if r.task == active_task.value)
        else:
            sections = [s for s in group_rows_for_grid(local_tab_rows) if s.rows]
            hf_count = len(hf_rows)
        return sections, hf_count

    def _extend_grid_sections_in_place(self, sections: list[GridSection], hf_count: int) -> bool:
        """Update existing ModelGrids in place when section count matches.

        Returns True iff the in-place path applied; the caller falls
        through to a teardown + remount on False.
        """
        existing_grids = list(self._grid_container.query(ModelGrid))
        existing_headings = [
            w for w in self._grid_container.query(".section-heading") if isinstance(w, Static)
        ]
        if not existing_grids or len(existing_grids) != len(sections):
            return False
        # Heading + grid mounts each compose on their own frame, so a
        # partially-mounted state can land here with the heading list
        # one short of the grid list. Drop strict=True so we cleanly
        # update whatever pairs we have without forcing a full remount.
        for grid, heading, section in zip(
            existing_grids, existing_headings, sections, strict=False
        ):
            heading.update(section.heading)
            grid.set_rows(section.rows)
        self._refresh_grid_ctas(hf_count=hf_count)
        self._update_sort_label()
        return True

    def _remount_grid_sections(self, sections: list[GridSection], hf_count: int) -> None:
        """Teardown + remount the grid for a section-count change.

        Captures the user's current cursor + scroll position before the
        teardown so both can be restored after remount; otherwise the
        ``_focus_first_grid`` fallback snaps the cursor back to the top
        of the catalog mid-keypress, and the layout shift from extra
        sections drifts the visible window away from where the user was
        looking.
        """
        focus_anchor = self._capture_focused_section()
        container = self._grid_container
        prior_scroll_y = container.scroll_y
        container.remove_children()
        self._mount_grid_section(sections[0])
        self.call_after_refresh(
            self._mount_remaining_grid_sections,
            sections[1:],
            hf_count=hf_count,
            focus_anchor=focus_anchor,
            prior_scroll_y=prior_scroll_y,
        )

    def _capture_focused_section(self) -> tuple[str, int | None] | None:
        """Return ``(heading, highlighted_index)`` for the focused grid.

        Heading is read from ``ModelGrid.name`` (set by
        ``_mount_grid_section``). Used to restore the cursor across a
        teardown+remount in ``_refresh_grid`` so paginated loads don't
        yank the user back to the top of the catalog.
        """
        focused = self._focused_grid()
        if not isinstance(focused, ModelGrid) or focused.name is None:
            return None
        return (focused.name, focused.highlighted)

    def _restore_focused_section(self, anchor: tuple[str, int | None] | None) -> bool:
        """Refocus the grid whose ``name`` matches the captured anchor.

        Returns True when the previous focus position was successfully
        restored; False when no anchor was given or the matching section
        no longer exists (caller falls back to ``_focus_first_grid``).
        """
        if anchor is None:
            return False
        target_heading, target_highlighted = anchor
        for grid in self._grid_container.query(ModelGrid):
            if grid.name != target_heading:
                continue
            grid.focus()
            if target_highlighted is not None and grid.rows:
                grid.highlighted = min(target_highlighted, len(grid.rows) - 1)
            return True
        return False

    def _mount_grid_section(self, section: GridSection) -> None:
        # ``name=section.heading`` doubles as the section identity used by
        # ``_capture_focused_section`` / ``_restore_focused_section`` to
        # preserve the cursor across teardown + remount.
        grid = ModelGrid(section.rows, name=section.heading, classes="catalog-section")
        self._grid_container.mount_all(
            [
                Static(section.heading, classes="section-heading"),
                grid,
            ]
        )

    def _mount_remaining_grid_sections(
        self,
        remaining: list[GridSection],
        hf_count: int,
        focus_anchor: tuple[str, int | None] | None = None,
        prior_scroll_y: float = 0.0,
    ) -> None:
        for section in remaining:
            self._mount_grid_section(section)
        self._mount_grid_ctas(hf_count=hf_count)
        # Restore the prior viewport position; mounting fresh sections shifts
        # the layout and ``focus()`` below would otherwise overshoot.
        if prior_scroll_y:
            self._grid_container.scroll_to(y=prior_scroll_y, animate=False)
        # Lock focus onto a grid once mount completes so j / k / PgDn /
        # PgUp dispatch correctly. Without this, on first paint the focus
        # race can leave nothing focused and the catalog feels frozen
        # until the user toggles to list view and back. When the previous
        # paint had a focused grid, restore the cursor to the same
        # section + highlighted index instead of jumping to the top.
        if not self._grid_view or self._focused_grid() is not None:
            return
        if self._restore_focused_section(focus_anchor):
            return
        self._focus_first_grid()

    def _grid_scroll_hint_text(self, hf_count: int) -> str:
        """Pick the bottom scroll-hint text based on fetch state."""
        if self._loading_more:
            return msg.CATALOG_GRID_LOADING_MORE.format(frame=_SPINNER_FRAMES[self._spinner_frame])
        if self._active_task_has_more():
            return msg.CATALOG_GRID_LOAD_MORE.format(count=hf_count)
        return msg.CATALOG_GRID_ALL_LOADED.format(count=hf_count)

    def _mount_grid_ctas(self, *, hf_count: int) -> None:
        try:
            container = self._grid_container
        except Exception:
            return
        ctas: list[Static] = [
            Static(
                self._grid_scroll_hint_text(hf_count),
                classes="grid-cta scroll-hint",
            )
        ]
        search = self._get_search_text()
        if search:
            ctas.append(
                Static(
                    msg.CATALOG_SEARCH_HF_CTA.format(query=search),
                    classes="grid-cta search-hf-cta",
                )
            )
        container.mount_all(ctas)

    def _refresh_grid_ctas(self, *, hf_count: int) -> None:
        """Update the bottom CTA strip in place; remount when class changes."""
        try:
            container = self._grid_container
        except Exception:
            return
        existing = list(container.query(".grid-cta"))
        for w in existing:
            with contextlib.suppress(Exception):
                w.remove()
        self._mount_grid_ctas(hf_count=hf_count)

    def _rows_for_active_tab(
        self, all_rows: list[LocalCatalogRow], active_tab: str
    ) -> list[CatalogRow]:
        """Slice the source row list for what the active task tab should render.

        Library/Discover bypass this (their refresh paths build their own
        slices). For task tabs, returns rows for the matching ModelTask
        further filtered by the per-tab SourceMode chip; CLOUD and BOTH
        also union the matching frontier rows.
        """
        if active_tab not in TASK_TAB_IDS:
            return list(all_rows)
        active_task = TAB_ID_TO_TASK[active_tab]
        mode = self._source_modes.get(active_tab, SourceMode.LOCAL)
        local_for_task: list[CatalogRow] = []
        if mode is not SourceMode.CLOUD:
            local_for_task = [r for r in all_rows if r.task == active_task.value]
        frontier_for_task: list[CatalogRow] = []
        if mode is not SourceMode.LOCAL:
            frontier_for_task = [r for r in self._frontier_rows if r.task == active_task.value]
        return local_for_task + frontier_for_task

    def _filter_grid(self) -> None:
        """Re-render the grid with the current filter applied via _refresh_grid."""
        self._refresh_grid()

    @on(ModelGrid.Highlighted)
    def _on_grid_highlighted(self, event: ModelGrid.Highlighted) -> None:
        """Run keyboard-driven prefetch on every grid cursor move and, when
        the cursor lands on the last row of the last grid, scroll the parent
        VerticalScroll to its end so the inline scroll-hint Static comes into
        view (matches the natural overshoot mouse-scroll past the cards
        already produces). Also re-renders the detail drawer for the newly
        highlighted row.
        """
        self._maybe_prefetch_on_grid_nav()
        self._reveal_scroll_hint_at_catalog_end()
        self._update_drawer_for_grid(event.grid, event.index)

    def _update_drawer_for_grid(self, grid: ModelGrid, index: int) -> None:
        """Push the focused row into the drawer; no-op if drawer is detached."""
        try:
            drawer = self.query_one("#catalog-detail-drawer", CatalogDetailDrawer)
        except Exception:
            return
        rows = grid.rows
        row = rows[index] if 0 <= index < len(rows) else None
        drawer.update_for_row(row)

    def on_key(self, event: Key) -> None:
        """Intercept 1-6 to jump tabs even when a focused widget owns digits.

        Bindings with priority=True should win against focused-widget
        bindings, but Textual's TabbedContent's inner ContentTabs swallows
        numeric keypresses before they reach screen-level bindings. An
        explicit on_key handler intercepts the digit at the bubbling stage,
        triggers ``action_select_tab``, and stops further dispatch so the
        digit doesn't bleed into the search Input or another widget.
        """
        if self._search_focused:
            return
        digit_to_index = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5}
        index = digit_to_index.get(event.key)
        if index is None:
            return
        event.stop()
        event.prevent_default()
        self.action_select_tab(index)

    def action_select_tab(self, index: int) -> None:
        """Activate the tab at *index* in ALL_TAB_IDS (0..5)."""
        from lilbee.cli.tui.screens.catalog_utils import ALL_TAB_IDS

        if self._search_focused:
            return
        if not 0 <= index < len(ALL_TAB_IDS):
            return
        target = ALL_TAB_IDS[index]
        try:
            tabs = self.query_one("#catalog-tabs", TabbedContent)
        except Exception:
            return
        self.set_focus(None)
        if tabs.active != target:
            tabs.active = target
        self._active_tab_id_cache = target

    def action_cycle_tab(self, delta: int) -> None:
        """Step the active tab by *delta*, wrapping around the strip.

        ctrl+right -> next, ctrl+left -> prev. Wraps so the user can spin
        either direction without hitting an end stop.
        """
        from lilbee.cli.tui.screens.catalog_utils import ALL_TAB_IDS

        if self._search_focused:
            return
        try:
            current = ALL_TAB_IDS.index(self._active_tab_id_cache)
        except ValueError:
            current = 0
        next_index = (current + delta) % len(ALL_TAB_IDS)
        self.action_select_tab(next_index)

    def action_cycle_source(self) -> None:
        """Cycle the active task tab's source mode: LOCAL -> CLOUD -> BOTH.

        No-op outside the four task tabs (Discover/Library aren't filtered
        by source). Per-tab mode means flipping Chat to BOTH doesn't drag
        Embed along; users can keep different views per task.
        """
        if self._search_focused:
            return
        active = self._active_tab_id_cache
        if active not in TASK_TAB_IDS:
            return
        self._source_modes[active] = next_source_mode(self._source_modes[active])
        # Force a rebuild on this tab; cache key for this tab is now stale
        # because the source filter changed but the upstream row data didn't.
        self._grid_cache_keys.pop(active, None)
        self._list_cache_keys.pop(active, None)
        self._refresh_view()

    def action_toggle_drawer(self) -> None:
        """Toggle the detail drawer's visibility via the -collapsed class.

        Default state is collapsed; users opt in. Class toggle is a single
        layout pass; we don't dynamically mount/unmount the drawer because
        rendering it offscreen costs zero (display: none).
        """
        try:
            drawer = self.query_one("#catalog-detail-drawer", CatalogDetailDrawer)
        except Exception:
            return
        drawer.toggle_class("-collapsed")

    def _reveal_scroll_hint_at_catalog_end(self) -> None:
        """Scroll the catalog container to the end when the keyboard cursor
        is on the last row of the bottom-most grid; otherwise no-op so the
        ``watch_highlighted`` cell-into-view scroll keeps tracking the cursor.

        ``immediate=True`` so the overshoot lands in the same compositor
        frame as the cell-into-view scroll above it; deferred would let a
        subsequent ``parent.scroll_to_region`` re-pin scroll_y to the cell.
        """
        focused = self._focused_grid()
        if not isinstance(focused, ModelGrid) or focused.highlighted is None:
            return
        grids = list(self._grid_container.query(ModelGrid))
        if not grids or focused is not grids[-1]:
            return
        cols = max(1, focused.columns_per_row)
        last_row = (len(focused.rows) - 1) // cols
        if focused.highlighted // cols < last_row:
            return
        self._grid_container.scroll_end(animate=False, immediate=True)

    @on(GridSelect.LeaveDown)
    @on(ModelGrid.LeaveDown)
    def _on_grid_leave_down(self, event: Message) -> None:
        """Move focus to the next grid widget, or fetch more if at the end.

        On the bottom-most grid we expose the inline scroll-hint Static
        (mounted below the last grid via ``_mount_grid_ctas``) by scrolling
        the parent VerticalScroll to its end. That mirrors the way mouse
        wheel naturally overshoots past the last card to reveal the hint.
        Cursor stays parked on the last cell.
        """
        if isinstance(event, ModelGrid.LeaveDown):
            grids = list(self._grid_container.query(ModelGrid))
            if grids and event.grid is grids[-1]:
                self._grid_container.scroll_end(animate=False, immediate=True)
                if self._active_task_has_more() and not self._loading_more:
                    self._load_more()
                return
        self.focus_next()

    @on(GridSelect.LeaveUp)
    @on(ModelGrid.LeaveUp)
    def _on_grid_leave_up(self, event: Message) -> None:
        """Move focus to the previous grid widget.

        On the topmost grid, return without moving focus so the cursor
        stays parked at the top row instead of leaking focus upward.
        """
        if isinstance(event, ModelGrid.LeaveUp):
            grids = list(self._grid_container.query(ModelGrid))
            if grids and event.grid is grids[0]:
                return
        self.focus_previous()

    @on(GridSelect.Selected)
    def _on_grid_select_selected(self, event: GridSelect.Selected) -> None:
        """Handle model selection from a GridSelect (setup wizard path)."""
        widget = event.widget
        if isinstance(widget, ModelCard):
            self._select_row(widget.row)

    @on(ModelGrid.Selected)
    def _on_grid_selected(self, event: ModelGrid.Selected) -> None:
        """Handle model selection from the catalog grid view."""
        self._select_row(event.row)

    @on(ModelList.Selected)
    def _on_model_list_selected(self, event: ModelList.Selected) -> None:
        """Handle model selection from any ModelList (Local list view or Frontier tab)."""
        self._select_row(event.row)

    def _refresh_list(self) -> None:
        """Rebuild the list view for the active tab; per-tab cache key skips no-op rebuilds."""
        active_tab = self._active_tab_id_cache
        all_rows = self._sort_rows(self._build_rows())
        if active_tab in TASK_TAB_IDS:
            active_task = TAB_ID_TO_TASK[active_tab]
            self._rows = [r for r in all_rows if r.task == active_task.value]
        else:
            self._rows = list(all_rows)
        search = self._get_search_text()
        list_key = (
            tuple((r.name, r.installed) for r in self._rows),
            search,
        )
        if self._list_cache_keys.get(active_tab) == list_key:
            self._update_sort_label()
            return
        self._list_cache_keys[active_tab] = list_key
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
        self._list_cache_keys[self._active_tab_id_cache] = (
            tuple((r.name, r.installed) for r in self._rows),
            search,
        )
        self._update_sort_label()

    def _sync_loading_spinner(self) -> None:
        """Show/hide the toolbar spinner based on active fetch state.

        Visible when a paginated HF fetch or a remote search is in
        flight (both grid and list views share the same toolbar
        widget). Cycles braille frames on a 100 ms timer so the
        wait reads as "moving" rather than "frozen".
        """
        try:
            spinner = self.query_one("#catalog-loading-spinner", Static)
        except Exception:
            return
        active = self._loading_more or self._search_in_flight
        if active:
            spinner.styles.display = "block"
            spinner.update(f"{_SPINNER_FRAMES[self._spinner_frame]} loading…")
            if self._spinner_timer is None:
                self._spinner_timer = self.set_interval(
                    _SPINNER_INTERVAL_S, self._tick_loading_spinner
                )
            # Mirror the spinner into the inline scroll-hint so users
            # waiting at the bottom of the grid see the activity in the
            # same place mouse scroll surfaces it.
            if self._loading_more:
                with contextlib.suppress(Exception):
                    hint = self._grid_container.query_one(".scroll-hint", Static)
                    hint.update(
                        msg.CATALOG_GRID_LOADING_MORE.format(
                            frame=_SPINNER_FRAMES[self._spinner_frame]
                        )
                    )
        else:
            spinner.update("")
            spinner.styles.display = "none"
            if self._spinner_timer is not None:
                self._spinner_timer.stop()
                self._spinner_timer = None
            self._spinner_frame = 0
            # Restore the post-load CTA text now that the fetch settled.
            # Count is per active task tab so the hint matches what's rendered.
            hf_rows = self._build_hf_rows(self._get_search_text()) if self._hf_fetched_any() else []
            active_task = self._active_task()
            hf_count = (
                sum(1 for r in hf_rows if r.task == active_task.value)
                if active_task is not None
                else len(hf_rows)
            )
            self._refresh_grid_ctas(hf_count=hf_count)

    def _tick_loading_spinner(self) -> None:
        """Advance the spinner one braille frame; called by the interval timer."""
        self._spinner_frame = (self._spinner_frame + 1) % len(_SPINNER_FRAMES)
        with contextlib.suppress(Exception):
            spinner = self.query_one("#catalog-loading-spinner", Static)
            spinner.update(f"{_SPINNER_FRAMES[self._spinner_frame]} loading…")
        if self._loading_more:
            with contextlib.suppress(Exception):
                hint = self._grid_container.query_one(".scroll-hint", Static)
                hint.update(
                    msg.CATALOG_GRID_LOADING_MORE.format(frame=_SPINNER_FRAMES[self._spinner_frame])
                )

    def _update_sort_label(self) -> None:
        """Update the sort indicator label, switching copy by active tab.

        Wrapped in NoMatches suppression because the worker callbacks that
        trigger an update (``_fetch_remote_models``, ``_fetch_frontier_models``)
        can fire on the next loop tick after a screen switch, before the
        new screen's ``compose`` has finished mounting ``#sort-label``.
        On Windows that race lands often enough to fail CI.
        """
        from textual.css.query import NoMatches

        try:
            label = self.query_one("#sort-label", Static)
        except NoMatches:
            return
        if self._active_tab_id() == TAB_LIBRARY:
            label.update(self._frontier_label_text())
            return
        direction = "asc" if self._sort_ascending else "desc"
        n_total = len(self._rows)
        if self._loading_more:
            count = f"{n_total} models · loading more…"
        elif self._active_task_has_more():
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
        if self._search_focused:
            return
        if self._active_tab_id() not in TASK_TAB_IDS:
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
        if row.kind == CatalogRowKind.FRONTIER:  # sealed-union dispatch
            self._select_frontier_row(row)
            return
        if row.variant and row.family:
            self._install_variant(row.variant, row.family)
        elif row.catalog_model:
            self._install_model(row.catalog_model)
        elif row.remote_model:
            apply_active_model(self.app, "chat_model", row.ref)
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
        self.app.switch_view("Settings")

    def _load_more(self) -> None:
        """Load the next HF page for the active task tab.

        Pagination is per-task: only the active tab's offset advances, only
        the active tab's task is fetched. Discover and Library short-circuit
        because they have no associated task and can't paginate.
        """
        if self._loading_more:
            return
        task = self._active_task()
        if task is None or not self._hf_has_more_by_task.get(task, False):
            return
        self._loading_more = True
        self._sync_loading_spinner()
        self._hf_offset_by_task[task] += _HF_PAGE_SIZE
        self._fetch_more_hf_for_task(task)

    def action_load_more(self) -> None:
        """Keyboard trigger (``n``) so users can page without scrolling."""
        if self._active_tab_id() not in TASK_TAB_IDS:
            return
        self._load_more()

    @on(TabbedContent.TabActivated, "#catalog-tabs")
    def _on_catalog_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        """Update active-tab cache, refresh sort label, populate the active pane.

        Cache update is the load-bearing line: every later check that asks
        ``_active_tab_id()`` reads this cache, not a fresh DOM query, so
        per-render overhead stays constant regardless of tab count.
        """
        new_tab = event.pane.id or TAB_CHAT
        if not self._activation_settled:
            return
        self._active_tab_id_cache = new_tab
        # Stale per-tab widget caches survive across tab activations,
        # but if the user switched after a remount, the cached handle
        # may be detached. _grid_for_tab/_list_for_tab validate via
        # is_running and refetch as needed.
        self._update_sort_label()
        if new_tab == TAB_LIBRARY:
            self._populate_library_list()
        elif new_tab == TAB_DISCOVER:
            self._populate_discover_rails()
        elif new_tab in TASK_TAB_IDS:
            # Lazy first-fetch: tabs other than Chat skip their HF round-trip
            # at mount and hit the API only when first activated. Cached
            # after, so re-activations stay free.
            self._ensure_task_initial_fetch(TAB_ID_TO_TASK[new_tab])
            # Refresh the newly active task tab. Per-tab cache key skips
            # the rebuild when the row shape hasn't changed since last paint.
            self._refresh_view()

    def _populate_discover_rails(self) -> None:
        """Push three curated row slices into the Discover landing.

        - For You: featured rows ranked by fit (FITS first, TIGHT, then
          WONT_RUN), capped at 6 to keep the rail compact.
        - Your Collection: every installed local row + every activated
          cloud API. Mirrors the Library tab's spirit but capped to a
          single rail-friendly slice.
        - Fresh on the Hub: most-downloaded non-featured HF rows as a
          recency-ish proxy (the API doesn't expose 'newly uploaded' as
          a sort key today; downloads-desc surfaces buzzy recent uploads).
        """
        try:
            rails = self.query_one("#discover-rails", DiscoverRails)
        except Exception:
            return
        family_rows = self._all_family_rows()
        hf_rows = self._all_hf_rows() if self._hf_fetched_any() else []
        remote_rows = self._all_remote_rows()
        for_you = sorted(
            (r for r in family_rows + hf_rows if r.featured),
            key=for_you_sort_key,
        )[:6]
        collection = [r for r in family_rows + remote_rows if r.installed][:6]
        fresh = sorted(
            (r for r in hf_rows if not r.featured),
            key=lambda r: -r.sort_downloads,
        )[:6]
        rails.set_rails(for_you=for_you, collection=collection, fresh=fresh)

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
        self.app.task_bar.start_download(model)
        self.notify(msg.CATALOG_QUEUED_DOWNLOAD.format(name=model.display_name))

    def action_go_back(self) -> None:
        # Escape from a focused filter input collapses the input back to
        # hidden and restores focus to the grid/list, so screen-level
        # keys (s / v) reach the screen instead of the (now-hidden) input.
        if self._search_focused:
            self._search_input.value = ""
            self._search_input.add_class("-hidden")
            self._focus_list_or_grid()
            return
        self.app.switch_view("Chat")

    def action_dismiss_filter(self) -> None:
        """Esc: hide the filter Input + restore grid/list focus; never dismiss.

        Heavy-interaction QA showed a stray Esc (from info-modal-then-Esc
        cycles, drawer thrash, filter typing chains) would dismiss the
        catalog mid-task and leak subsequent keystrokes into the chat
        Input on the next screen. Esc now only handles the filter; the
        dismiss path is `q` (action_go_back) which still does both.
        """
        if self._search_focused:
            self._search_input.value = ""
            self._search_input.add_class("-hidden")
            self._focus_list_or_grid()

    def _focus_list_or_grid(self) -> None:
        """Move focus from the filter input to the active view's list/grid."""
        if self._grid_view:
            self._focus_first_grid()
        else:
            self._focus_list_item(0)

    def action_show_info(self) -> None:
        """Pop up an info modal for the highlighted catalog row."""
        if self._search_focused:
            return
        row = self._highlighted_row()
        if row is None:
            self.notify(msg.CATALOG_SELECT_FOR_INFO, severity="warning")
            return
        if row.kind != CatalogRowKind.LOCAL:
            self.notify(msg.CATALOG_FRONTIER_NO_INFO, severity="warning")
            return
        from lilbee.cli.tui.screens.model_info import ModelInfoModal

        self.app.push_screen(ModelInfoModal(row))

    def _highlighted_row(self) -> CatalogRow | None:
        """Return the focused row in either grid or list view, or None."""
        if not self._grid_view and self._list_widget.has_focus:
            return self._list_widget.highlighted_row()
        focused_grid = self._focused_grid()
        if focused_grid is None or focused_grid.highlighted is None:
            return None
        if isinstance(focused_grid, ModelGrid):
            rows = focused_grid.rows
            index = focused_grid.highlighted
            return rows[index] if 0 <= index < len(rows) else None
        child = focused_grid.children[focused_grid.highlighted]
        if isinstance(child, ModelCard):
            return child.row
        return None

    def action_delete_model(self) -> None:
        """Delete an installed model. First press asks confirmation, second confirms."""
        if self._search_focused:
            return
        model_name = self._get_highlighted_model_name()
        if model_name is None:
            self.notify(msg.CATALOG_SELECT_TO_DELETE, severity="warning")
            return

        if not self._row_is_installed(model_name):
            self.notify(msg.CATALOG_NOT_INSTALLED.format(name=model_name), severity="warning")
            return

        if self._pending_delete == model_name:
            self._pending_delete = None
            self._run_delete(model_name)
        else:
            self._pending_delete = model_name
            self.notify(msg.CATALOG_CONFIRM_DELETE.format(name=model_name))

    def _row_is_installed(self, model_name: str) -> bool:
        """True if *model_name* names an installed native or remote model.

        ``_installed_names`` carries both the full ``<repo>/<file>.gguf``
        ref and the bare ``hf_repo`` for every installed native model,
        so it answers either ref shape; remote presence is asked of the
        manager directly.
        """
        if model_name in self._installed_names:
            return True
        return get_services().model_manager.is_installed(model_name, ModelSource.REMOTE)

    def _resolve_delete_ref(self, identity: str) -> str:
        """Pick the single registry ref that deleting *identity* maps to.

        Featured / HF browse rows surface a bare hf_repo while the
        registry deletes by ``<hf_repo>/<file>.gguf``. Bare repos
        resolve to the lexicographically-first matching installed
        manifest; full refs and remote names pass through.
        """
        if "/" in identity and identity.endswith(".gguf"):
            return identity
        prefix = identity + "/"
        matches = sorted(n for n in self._installed_names if n.startswith(prefix))
        if matches:
            return matches[0]
        return identity

    def _get_highlighted_model_name(self) -> str | None:
        """Return the registry-compatible model ref for the focused/highlighted row."""
        if not self._grid_view and self._list_widget.has_focus:
            row = self._list_widget.highlighted_row()
            return row_delete_id(row) if row else None
        focused_grid = self._focused_grid()
        if focused_grid is None or focused_grid.highlighted is None:
            return None
        if isinstance(focused_grid, ModelGrid):
            rows = focused_grid.rows
            index = focused_grid.highlighted
            if 0 <= index < len(rows):
                return row_delete_id(rows[index])
            return None
        # GridSelect path: cards are direct children indexed positionally.
        child = focused_grid.children[focused_grid.highlighted]
        if isinstance(child, ModelCard):
            return row_delete_id(child.row)
        return None

    @work(thread=True)
    def _run_delete(self, model_name: str) -> None:
        """Remove a model in a background thread."""
        delete_ref = self._resolve_delete_ref(model_name)
        try:
            removed = get_services().model_manager.remove(delete_ref)
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

    def _focused_grid(self) -> ModelGrid | GridSelect | None:
        """Return the focused grid widget (grid view), else None."""
        if self._grid_view and isinstance(self.focused, (ModelGrid, GridSelect)):
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
        if self._grid_view or not self._active_task_has_more() or self._loading_more:
            return
        idx = self._focused_list_index()
        if idx is None:
            return
        if idx >= self._list_widget.option_count - _HF_LOAD_MORE_TRIGGER:
            self._load_more()

    def _maybe_prefetch_on_grid_nav(self) -> None:
        """Fire ``_load_more`` when the keyboard cursor lands within the last
        rows of the catalog. Mouse wheel triggers via ``_on_grid_scrolled`` at
        the 85 % scroll threshold, but cell-by-cell keyboard nav advances
        scroll_y too gradually to ever cross that threshold; this check
        guarantees keyboard reaches the same prefetch trigger.
        """
        if not self._grid_view or not self._active_task_has_more() or self._loading_more:
            return
        grids = list(self._grid_container.query(ModelGrid))
        if not grids:
            return
        focused = self._focused_grid()
        if not isinstance(focused, ModelGrid) or focused.highlighted is None:
            return
        # Absolute cursor position = cards in earlier grids + cursor in this grid.
        try:
            grid_index = grids.index(focused)
        except ValueError:
            return
        cards_before = sum(len(g.rows) for g in grids[:grid_index])
        absolute = cards_before + focused.highlighted
        total = sum(len(g.rows) for g in grids)
        if total <= 0:
            return
        if absolute >= total - _HF_LOAD_MORE_TRIGGER:
            self._load_more()

    _SCROLL_PREFETCH_RATIO = 0.85
    _SCROLL_PREFETCH_COOLDOWN = 0.8

    def _on_list_scrolled(self, _scroll_y: float) -> None:
        """Trigger _load_more when the user scrolls near the bottom of the list."""
        if not self._scroll_prefetch_due(self._list_widget):
            return
        self._scroll_prefetch_armed_at = time.monotonic()
        self._load_more()

    def _on_grid_scrolled(self, _scroll_y: float) -> None:
        """Trigger _load_more when the user scrolls near the bottom of the grid."""
        if not self._grid_view:
            return
        if not self._scroll_prefetch_due(self._grid_container):
            return
        self._scroll_prefetch_armed_at = time.monotonic()
        self._load_more()

    def on_mouse_scroll_down(self, event: MouseScrollDown) -> None:
        """Force pagination when wheeling beyond what the active scroll can scroll.

        Three collapsed triggers, both views: (1) content already fits the
        viewport so ``max_scroll_y == 0`` and wheel events produce no scroll
        delta, (2) the user has wheeled to ``scroll_y == max_scroll_y`` and
        further wheels produce no delta, (3) list view has the same problem
        as grid view -- the scroll watcher only fires on scroll_y changes,
        so a wheel at max_y is invisible to ``_on_list_scrolled`` /
        ``_on_grid_scrolled``. Re-check here and fetch the next page
        directly. Cooldown prevents a cascade as new rows shift max_scroll_y.
        """
        if not self._active_task_has_more() or self._loading_more:
            return
        container = self._grid_container if self._grid_view else self._list_widget
        max_y = container.max_scroll_y
        if max_y > 0 and container.scroll_y < max_y:
            return
        if self._scroll_prefetch_armed_at:
            elapsed = time.monotonic() - self._scroll_prefetch_armed_at
            if elapsed < self._SCROLL_PREFETCH_COOLDOWN:
                return
        self._scroll_prefetch_armed_at = time.monotonic()
        self._load_more()

    def _scroll_prefetch_due(self, widget: VerticalScroll | ModelList) -> bool:
        # Cooldown blocks a runaway cascade where appending rows shifts
        # max_scroll_y, the watcher refires, and load_more kicks off the
        # next fetch before the user notices.
        if not self._active_task_has_more() or self._loading_more:
            return False
        if self._scroll_prefetch_armed_at:
            elapsed = time.monotonic() - self._scroll_prefetch_armed_at
            if elapsed < self._SCROLL_PREFETCH_COOLDOWN:
                return False
        max_y = widget.max_scroll_y
        if max_y <= 0:
            return False
        return widget.scroll_y / max_y >= self._SCROLL_PREFETCH_RATIO

    def _page_rows(self) -> int:
        """How many cursor steps make up one 'page' in the active view."""
        return _GRID_PAGE_ROWS if self._grid_view else _LIST_PAGE_ROWS

    def action_page_down(self) -> None:
        if self._search_focused:
            return
        if self._grid_view:
            if (grid := self._focused_grid()) is not None:
                for _ in range(self._page_rows()):
                    grid.action_cursor_down()
        else:
            self._nudge_list(self._page_rows())

    def action_page_up(self) -> None:
        if self._search_focused:
            return
        if self._grid_view:
            if (grid := self._focused_grid()) is not None:
                for _ in range(self._page_rows()):
                    grid.action_cursor_up()
        else:
            self._nudge_list(-self._page_rows())

    def action_cursor_down(self) -> None:
        if self._search_focused:
            return
        if self._grid_view:
            grid = self._focused_grid() or self._first_grid_or_none()
            if grid is not None:
                grid.focus()
                grid.action_cursor_down()
        else:
            self._nudge_list(1)

    def action_cursor_up(self) -> None:
        if self._search_focused:
            return
        if self._grid_view:
            grid = self._focused_grid() or self._first_grid_or_none()
            if grid is not None:
                grid.focus()
                grid.action_cursor_up()
        else:
            self._nudge_list(-1)

    def _first_grid_or_none(self) -> ModelGrid | None:
        """Return the first ModelGrid in the active tab's container, or None."""
        from textual.css.query import NoMatches

        try:
            return self._grid_container.query(ModelGrid).first()
        except NoMatches:
            return None

    def action_jump_top(self) -> None:
        if self._search_focused:
            return
        if self._grid_view:
            if (grid := self._focused_grid()) is not None:
                grid.highlight_first()
        else:
            self._focus_list_item(0)

    def action_jump_bottom(self) -> None:
        if self._search_focused:
            return
        if self._grid_view:
            if (grid := self._focused_grid()) is not None:
                grid.highlight_last()
        else:
            count = self._list_widget.option_count
            if count:
                self._focus_list_item(count - 1)
                self._maybe_prefetch_on_nav()
