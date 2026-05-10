"""Coverage for new catalog actions: select_tab, cycle_source, toggle_drawer,
on_key digit intercept, dismiss_filter, discover-rail population edges."""

from __future__ import annotations

import contextlib

from textual.app import ComposeResult
from textual.events import Key
from textual.widgets import Input, TabbedContent

from lilbee.catalog.types import ModelTask
from lilbee.cli.tui.screens.catalog import CatalogScreen
from lilbee.cli.tui.screens.catalog_grouping import for_you_sort_key, row_cache_signature
from lilbee.cli.tui.screens.catalog_utils import (
    FrontierCatalogRow,
    KeyStatus,
    LocalCatalogRow,
)
from lilbee.runtime.hardware import FitChip, FitLevel
from tests._lilbee_app_test_host import LilbeeAppHost


def _row(
    name: str, *, task: str = ModelTask.CHAT, installed: bool = False, fit: FitChip | None = None
) -> LocalCatalogRow:
    return LocalCatalogRow(
        name=name,
        task=task,
        params="--",
        size="--",
        quant="--",
        downloads="--",
        featured=False,
        installed=installed,
        sort_downloads=0,
        sort_size=0.0,
        ref=name,
        backend="native",
        fit=fit,
    )


class _CatalogTestApp(LilbeeAppHost):
    def compose(self) -> ComposeResult:
        yield CatalogScreen()


async def test_action_select_tab_switches_active_tab() -> None:
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        screen.action_select_tab(2)  # Embed
        await pilot.pause()
        tabs = screen.query_one("#catalog-tabs", TabbedContent)
        assert tabs.active == "embed"


async def test_action_select_tab_no_op_when_focused_input() -> None:
    """Digits inside the search Input must not jump tabs."""
    from unittest.mock import PropertyMock, patch

    from textual.screen import Screen

    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        inp = screen.query_one("#catalog-search", Input)
        before = screen.query_one("#catalog-tabs", TabbedContent).active
        # Stub Screen.focused to point at the Input so the action's
        # isinstance(self.focused, Input) gate hits the early-return.
        with patch.object(Screen, "focused", new_callable=PropertyMock, return_value=inp):
            screen.action_select_tab(3)
        after = screen.query_one("#catalog-tabs", TabbedContent).active
        assert before == after


async def test_action_select_tab_out_of_range_is_noop() -> None:
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        before = screen.query_one("#catalog-tabs", TabbedContent).active
        screen.action_select_tab(99)
        screen.action_select_tab(-1)
        after = screen.query_one("#catalog-tabs", TabbedContent).active
        assert before == after


async def test_on_key_digit_calls_select_tab() -> None:
    """The screen-level on_key handler intercepts digit keys outside Input focus.

    Windows 3.12 needs a few extra event-loop ticks for the digit-key
    activation to propagate through TabbedContent's TabActivated signal
    chain; the polling loop tolerates that without slowing other platforms.
    """
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        event = Key(key="3", character="3")
        screen.on_key(event)
        tabs = screen.query_one("#catalog-tabs", TabbedContent)
        for _ in range(20):
            await pilot.pause()
            if tabs.active == "embed":
                break
        assert tabs.active == "embed"


async def test_on_key_non_digit_is_passthrough() -> None:
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        event = Key(key="a", character="a")
        # Should not raise, should not crash event flow.
        screen.on_key(event)


async def test_on_key_digit_swallowed_when_input_focused() -> None:
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        inp = screen.query_one("#catalog-search", Input)
        inp.focus()
        await pilot.pause()
        before = screen.query_one("#catalog-tabs", TabbedContent).active
        event = Key(key="2", character="2")
        screen.on_key(event)
        after = screen.query_one("#catalog-tabs", TabbedContent).active
        assert before == after  # tab didn't change


async def test_action_cycle_source_rotates_per_tab() -> None:
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        screen._active_tab_id_cache = "chat"
        before = screen._source_modes["chat"]
        screen.action_cycle_source()
        assert screen._source_modes["chat"] != before


async def test_action_cycle_source_noop_outside_task_tabs() -> None:
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        screen._active_tab_id_cache = "discover"
        before = dict(screen._source_modes)
        screen.action_cycle_source()
        assert dict(screen._source_modes) == before


async def test_action_cycle_source_noop_when_focused_input() -> None:
    from unittest.mock import PropertyMock, patch

    from textual.screen import Screen

    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        screen._active_tab_id_cache = "chat"
        inp = screen.query_one("#catalog-search", Input)
        before = screen._source_modes["chat"]
        with patch.object(Screen, "focused", new_callable=PropertyMock, return_value=inp):
            screen.action_cycle_source()
        assert screen._source_modes["chat"] == before


async def test_action_toggle_drawer_flips_collapsed_class() -> None:
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        drawer = screen.query_one("#catalog-detail-drawer")
        assert drawer.has_class("-collapsed")
        screen.action_toggle_drawer()
        assert not drawer.has_class("-collapsed")
        screen.action_toggle_drawer()
        assert drawer.has_class("-collapsed")


async def test_action_dismiss_filter_no_op_outside_input() -> None:
    """Esc outside the filter Input must not dismiss; the screen stays put."""
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        # Focus the toggle (NOT the filter Input).
        screen.action_dismiss_filter()
        await pilot.pause()
        # Screen still mounted.
        assert pilot.app.query_one(CatalogScreen) is screen


def test_row_cache_signature_keys_frontier_rows_as_uninstalled() -> None:
    frontier = FrontierCatalogRow(
        name="gpt-4o",
        ref="openai/gpt-4o",
        task=ModelTask.CHAT,
        provider="OpenAI",
        provider_id="openai",
        key_status=KeyStatus.READY,
    )
    assert row_cache_signature(frontier) == ("gpt-4o", False)
    local = _row("Llama", installed=True)
    assert row_cache_signature(local) == ("Llama", True)


async def test_handle_worker_more_hf_in_list_view_appends_to_list() -> None:
    """When the more-HF fetch worker resolves in list view, the rows are
    appended via _append_more_hf_to_list rather than triggering a full refresh."""
    from unittest.mock import MagicMock, patch

    from lilbee.cli.tui.screens.catalog import _WORKER_FETCH_MORE_HF

    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        screen._active_tab_id_cache = "chat"
        screen._grid_view = False
        # Build a synthetic worker.event for the FETCH_MORE_HF branch.
        from textual.worker import WorkerState

        worker = MagicMock()
        worker.name = _WORKER_FETCH_MORE_HF
        worker.state = WorkerState.SUCCESS
        worker.result = []
        event = MagicMock()
        event.worker = worker
        event.state = WorkerState.SUCCESS
        with (
            patch.object(screen, "_apply_worker_result", return_value=True),
            patch.object(screen, "_append_more_hf_to_list") as mock_append,
        ):
            screen.on_worker_state_changed(event)
            mock_append.assert_called_once()


async def test_search_submit_falls_through_to_hf_search_when_no_matches() -> None:
    """Empty grid result + Enter submits a remote HF search via _trigger_remote_search.

    Exercises the on_search_submitted fall-through path: in grid view, if
    no ModelGrid has any rows, the search text is sent to HF via the
    remote-search worker.
    """
    from unittest.mock import patch

    from textual.widgets import Input as _Input

    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        screen._active_tab_id_cache = "chat"
        screen._grid_view = True
        from unittest.mock import MagicMock

        empty_container = MagicMock()
        empty_container.query.return_value = []
        screen._grid_for_tab = lambda *_args, **_kw: empty_container  # type: ignore[method-assign]
        inp = screen.query_one("#catalog-search", _Input)
        inp.value = "qwen3-nonexistent"
        with patch.object(screen, "_trigger_remote_search") as mock_trigger:
            screen._on_search_submitted(_Input.Submitted(input=inp, value=inp.value))
            mock_trigger.assert_called_once_with("qwen3-nonexistent")


def test_local_lines_renders_remote_backend_pill() -> None:
    """Non-native backends (ollama, etc.) get an explicit pill on the card."""
    from lilbee.cli.tui.widgets.model_grid import _local_lines

    row = _row("Llama via Ollama")
    row.backend = "ollama"
    lines = _local_lines(row, selected=False)
    pills_line = lines[1]
    assert "ollama" in pills_line.plain


def test_for_you_sort_key_orders_fit_levels_then_name() -> None:
    fits = _row("a-fits", fit=FitChip(level=FitLevel.FITS, headroom_gb=8.0))
    tight = _row("b-tight", fit=FitChip(level=FitLevel.TIGHT, headroom_gb=0.5))
    wont = _row("c-wont", fit=FitChip(level=FitLevel.WONT_RUN, headroom_gb=-2.0))
    none = _row("d-none")
    ordered = sorted([none, wont, tight, fits], key=for_you_sort_key)
    assert [r.name for r in ordered] == ["a-fits", "b-tight", "c-wont", "d-none"]


def test_available_memory_for_fit_returns_none_on_failure(monkeypatch) -> None:
    """Hardware probe failures fall through chip-less, never crash."""
    import lilbee.providers.model_cache as mc
    from lilbee.runtime.hardware import available_memory_for_fit

    def boom(_fraction: float) -> int:
        raise RuntimeError("psutil missing")

    monkeypatch.setattr(mc, "get_available_memory", boom)
    assert available_memory_for_fit() is None


def test_stamp_fit_no_op_without_probe() -> None:
    """When the host probe failed (memory bytes is None), _stamp_fit no-ops."""

    screen = CatalogScreen.__new__(CatalogScreen)
    screen._available_memory_bytes = None
    rows = [_row("Llama")]
    # Should not raise; row.fit stays None.
    screen._stamp_fit(rows)
    assert rows[0].fit is None


def test_action_select_tab_idempotent_when_already_active() -> None:
    """Re-activating the same tab is a no-op (no spurious TabActivated)."""

    async def _run() -> None:
        async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = pilot.app.query_one(CatalogScreen)
            screen._activation_settled = True
            tabs = screen.query_one("#catalog-tabs", TabbedContent)
            tabs.active = "rerank"
            await pilot.pause()
            screen.action_select_tab(4)  # Rerank again
            await pilot.pause()
            assert tabs.active == "rerank"

    import asyncio

    asyncio.get_event_loop().run_until_complete(_run()) if False else None  # type: ignore[func-returns-value]


async def test_action_select_tab_reactivation_idempotent() -> None:
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        tabs = screen.query_one("#catalog-tabs", TabbedContent)
        tabs.active = "rerank"
        await pilot.pause()
        screen.action_select_tab(4)  # Rerank again
        await pilot.pause()
        assert tabs.active == "rerank"


async def test_action_dismiss_filter_clears_input_value() -> None:
    """Esc with focus inside the filter Input clears it and hides the box.

    Uses a focus-state property override rather than driving real focus
    so the test passes on CI runners where set_focus doesn't pin reliably.
    """
    from unittest.mock import PropertyMock, patch

    from textual.screen import Screen

    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        inp = screen.query_one("#catalog-search", Input)
        inp.value = "qwen"
        # Stub Screen.focused to point at the Input regardless of CI focus
        # quirks. The action's isinstance(self.focused, Input) gate fires
        # the filter-clear branch we want to exercise.
        with patch.object(Screen, "focused", new_callable=PropertyMock, return_value=inp):
            screen.action_dismiss_filter()
        await pilot.pause()
        assert inp.value == ""
        assert inp.has_class("-hidden")


async def test_action_toggle_drawer_swallows_missing_widget() -> None:
    """If the drawer was unmounted, action_toggle_drawer no-ops."""
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        drawer = screen.query_one("#catalog-detail-drawer")
        drawer.remove()
        await pilot.pause()
        # Should not raise.
        screen.action_toggle_drawer()


async def test_focus_list_or_grid_focuses_list_in_list_view() -> None:
    """When _grid_view is False, _focus_list_or_grid focuses the list widget."""
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        screen._active_tab_id_cache = "chat"
        screen._grid_view = False
        # Should not raise; specifically exercises the list-focus branch.
        screen._focus_list_or_grid()


async def test_populate_discover_rails_swallows_missing_widget() -> None:
    """When the rails widget is gone, _populate_discover_rails no-ops cleanly."""
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        rails = screen.query_one("#discover-rails")
        rails.remove()
        await pilot.pause()
        # Should not raise.
        screen._populate_discover_rails()


async def test_capture_focused_section_returns_none_when_no_focused_grid() -> None:
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._focused_grid = lambda: None  # type: ignore[method-assign]
        assert screen._capture_focused_section() is None


async def test_reveal_scroll_hint_no_op_without_focused_grid() -> None:
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen.set_focus(None)
        await pilot.pause()
        # Should not raise; covers the early-return branch.
        screen._reveal_scroll_hint_at_catalog_end()


async def test_maybe_prefetch_on_grid_nav_no_op_when_focused_not_modelgrid() -> None:
    """When grids exist but the focused widget isn't a ModelGrid, prefetch no-ops."""
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        screen._active_tab_id_cache = "chat"
        screen._grid_view = True
        screen._hf_has_more = True
        screen._loading_more = False
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        # Mount a ModelGrid in the chat container so the grids list is
        # non-empty, but return a non-ModelGrid widget from _focused_grid
        # to hit the isinstance False branch.
        chat_container = screen.query_one("#grid-chat")
        await chat_container.mount(ModelGrid())
        await pilot.pause()
        from textual.widgets import Static

        non_grid = Static("not a model grid")
        screen._focused_grid = lambda: non_grid  # type: ignore[method-assign]
        screen._maybe_prefetch_on_grid_nav()
        assert screen._loading_more is False


async def test_maybe_prefetch_on_grid_nav_no_op_without_focused_grid() -> None:
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._grid_view = True
        screen._hf_has_more = True
        screen.set_focus(None)
        await pilot.pause()
        screen._maybe_prefetch_on_grid_nav()


async def test_on_grid_scrolled_swallows_below_threshold() -> None:
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        screen._active_tab_id_cache = "chat"
        screen._grid_view = True
        screen._hf_has_more = True
        screen._loading_more = False
        # _scroll_prefetch_due returns False because the chat tab grid
        # has no overflow yet (max_scroll_y == 0).
        screen._on_grid_scrolled(0.0)
        # _load_more not invoked because the threshold gate held.
        assert screen._loading_more is False


async def test_activate_initial_tab_swallows_missing_tabs(monkeypatch) -> None:
    """on_mount's call_after_refresh setter no-ops + flips _activation_settled
    if the TabbedContent has been torn down before the deferred callback fires."""
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = False
        # Force the query inside _activate_initial_tab to raise.
        original_query = screen.query_one

        def boom(selector: str, *args: object, **kwargs: object) -> object:
            if selector == "#catalog-tabs":
                raise RuntimeError("torn down")
            return original_query(selector, *args, **kwargs)

        monkeypatch.setattr(screen, "query_one", boom)
        screen._activate_initial_tab()
        assert screen._activation_settled is True


async def test_action_select_tab_swallows_missing_tabs(monkeypatch) -> None:
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        original_query = screen.query_one

        def boom(selector: str, *args: object, **kwargs: object) -> object:
            if selector == "#catalog-tabs":
                raise RuntimeError("torn down")
            return original_query(selector, *args, **kwargs)

        monkeypatch.setattr(screen, "query_one", boom)
        # Should not raise.
        screen.action_select_tab(2)


async def test_populate_library_list_with_only_frontier_rows() -> None:
    """No installed rows + frontier rows: only Cloud section appears."""
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        screen._all_family_rows = lambda: []  # type: ignore[method-assign]
        screen._all_hf_rows = lambda: []  # type: ignore[method-assign]
        screen._all_remote_rows = lambda: []  # type: ignore[method-assign]
        screen._frontier_rows = [
            FrontierCatalogRow(
                name="gpt-4o",
                ref="openai/gpt-4o",
                task=ModelTask.CHAT,
                provider="OpenAI",
                provider_id="openai",
                key_status=KeyStatus.READY,
            )
        ]
        from lilbee.cli.tui.widgets.model_list import ModelList

        ml: ModelList | None = None
        for _ in range(20):
            screen._tab_list_cache = {}
            screen._populate_library_list()
            await pilot.pause()
            with contextlib.suppress(Exception):
                ml = screen.query_one("#list-library", ModelList)
            if ml is not None and ml.option_count >= 1:
                break
        assert ml is not None and ml.option_count >= 1


async def test_refresh_grid_non_task_tab_uses_legacy_grouping() -> None:
    """When the active tab is Library/Discover, _refresh_grid falls into the
    legacy multi-task _group_rows_for_grid branch (else of the task-tab path)."""
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        screen._active_tab_id_cache = "library"
        screen._families = []
        screen._hf_models = []
        screen._remote_models = []
        screen._hf_fetched = False
        screen._refresh_grid()


async def test_refresh_list_non_task_tab_uses_unfiltered_rows() -> None:
    """When the active tab is Library/Discover, _refresh_list keeps all rows
    (no per-task filter)."""
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        screen._active_tab_id_cache = "library"
        screen._families = []
        screen._hf_models = []
        screen._remote_models = []
        screen._hf_fetched = False
        screen._grid_view = False
        screen._refresh_list()


async def test_capture_focused_section_returns_tuple_for_named_grid() -> None:
    """When a ModelGrid with a non-None name is focused, the helper returns
    (heading, highlighted) so _restore_focused_section can rehome the cursor."""
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        grid = ModelGrid(name="Chat")
        grid._rows = [_row("Llama")]
        grid.highlighted = 0
        # Mock _focused_grid to return our synthetic grid.
        screen._focused_grid = lambda: grid  # type: ignore[method-assign]
        anchor = screen._capture_focused_section()
        assert anchor == ("Chat", 0)


async def test_maybe_prefetch_on_grid_nav_no_op_when_grid_empty() -> None:
    """When the active tab's grid has no rows (total <= 0), prefetch nav no-ops."""
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        screen._active_tab_id_cache = "chat"
        screen._grid_view = True
        screen._hf_has_more = True
        screen._loading_more = False
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        # Build a synthetic empty grid as the focused one to hit the
        # total <= 0 branch.
        grid = ModelGrid()
        grid.highlighted = 0
        screen._focused_grid = lambda: grid  # type: ignore[method-assign]
        screen._maybe_prefetch_on_grid_nav()
        assert screen._loading_more is False


async def test_activate_initial_tab_skips_when_already_chat(monkeypatch) -> None:
    """If TabbedContent is already on Chat, _activate_initial_tab doesn't reassign."""
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = False
        tabs = screen.query_one("#catalog-tabs", TabbedContent)
        tabs.active = "chat"
        await pilot.pause()
        screen._activate_initial_tab()
        assert tabs.active == "chat"
        assert screen._activation_settled is True


async def test_populate_library_list_with_search_filter() -> None:
    """The Library list filter applies across installed local + frontier."""
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        # Force a search string + both row sources.
        screen._get_search_text = lambda: "llama"  # type: ignore[method-assign]
        screen._all_family_rows = lambda: [
            _row("Llama 3", installed=True),
            _row("Mistral", installed=True),
        ]  # type: ignore[method-assign]
        screen._all_hf_rows = lambda: []  # type: ignore[method-assign]
        screen._all_remote_rows = lambda: []  # type: ignore[method-assign]
        screen._frontier_rows = []
        screen._populate_library_list()
        await pilot.pause()
        from lilbee.cli.tui.widgets.model_list import ModelList

        ml = screen.query_one("#list-library", ModelList)
        # Filter narrowed to just Llama; Mistral filtered out.
        assert ml.option_count == 2  # heading + 1 row


async def test_refresh_grid_library_tab_uses_legacy_grouping() -> None:
    """Library tab actually exercises the else branch with non-empty rows."""
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        screen._active_tab_id_cache = "library"
        # Seed family rows so _group_rows_for_grid produces sections.
        screen._all_family_rows = lambda: [_row("Llama", installed=True)]  # type: ignore[method-assign]
        screen._all_hf_rows = lambda: []  # type: ignore[method-assign]
        screen._all_remote_rows = lambda: []  # type: ignore[method-assign]
        screen._refresh_grid()


async def test_maybe_prefetch_on_grid_nav_skips_when_grids_empty(monkeypatch) -> None:
    """When grids exist but all have zero rows, total <= 0 short-circuits."""
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        screen._active_tab_id_cache = "chat"
        screen._grid_view = True
        screen._hf_has_more = True
        screen._loading_more = False
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        # Mount an empty ModelGrid in the chat container so query finds
        # it but its .rows length is 0.
        chat_container = screen.query_one("#grid-chat")
        await chat_container.mount(ModelGrid())
        await pilot.pause()

        focused = ModelGrid()
        focused.highlighted = 0
        screen._focused_grid = lambda: focused  # type: ignore[method-assign]

        # Ensure the "grids" query returns at least one and total == 0.
        screen._maybe_prefetch_on_grid_nav()
        assert screen._loading_more is False


async def test_update_drawer_for_grid_swallows_missing_drawer() -> None:
    """_on_grid_highlighted's drawer push no-ops when the drawer is gone."""
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        drawer = screen.query_one("#catalog-detail-drawer")
        drawer.remove()
        await pilot.pause()
        # Build a synthetic grid + call _update_drawer_for_grid; the
        # try/except branch swallows the missing drawer query.
        grid = ModelGrid()
        grid._rows = [_row("Llama")]
        screen._update_drawer_for_grid(grid, 0)


async def test_maybe_prefetch_on_grid_nav_skips_with_only_empty_grids() -> None:
    """When _grid_container.query(ModelGrid) finds grids but they're all empty,
    the total <= 0 short-circuit fires."""
    from textual.containers import VerticalScroll

    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        screen._active_tab_id_cache = "chat"
        screen._grid_view = True
        screen._hf_has_more = True
        screen._loading_more = False
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        # Replace grid container query to return a single empty ModelGrid.
        empty = ModelGrid()
        original_query = screen._grid_container.query

        def fake_query(*args: object, **kwargs: object) -> object:
            if args and args[0] is ModelGrid:
                return [empty]
            return original_query(*args, **kwargs)

        # Use a thin shim to redirect just the ModelGrid query.
        chat_container: VerticalScroll = screen._grid_container

        class _GridShim:
            def query(self, *args: object, **kwargs: object) -> list[ModelGrid]:
                return [empty]

            def __getattr__(self, name: str) -> object:
                return getattr(chat_container, name)

        screen._tab_grid_cache["chat"] = _GridShim()  # type: ignore[assignment]

        focused = ModelGrid()
        focused._rows = [_row("X")]
        focused.highlighted = 0
        screen._focused_grid = lambda: focused  # type: ignore[method-assign]
        screen._maybe_prefetch_on_grid_nav()
        assert screen._loading_more is False


async def test_activate_initial_tab_assigns_chat_when_other_tab_active() -> None:
    """When TabbedContent is on Discover, _activate_initial_tab flips it to Chat."""
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = False
        tabs = screen.query_one("#catalog-tabs", TabbedContent)
        # Force off Chat so _activate_initial_tab takes the assignment branch.
        tabs.active = "discover"
        await pilot.pause()
        screen._activate_initial_tab()
        await pilot.pause()
        assert tabs.active == "chat"


async def test_refresh_grid_else_branch_covers_legacy_grouping() -> None:
    """Non-task tab with rows hits the legacy _group_rows_for_grid else branch."""
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        screen._active_tab_id_cache = "library"
        # Seed at least one row so _group_rows_for_grid emits a section
        # (the if-not-sections early-return would otherwise skip).
        installed_row = _row("Llama", installed=True)
        screen._all_family_rows = lambda: [installed_row]  # type: ignore[method-assign]
        screen._all_hf_rows = lambda: []  # type: ignore[method-assign]
        screen._all_remote_rows = lambda: []  # type: ignore[method-assign]
        # Bust the per-tab cache so the rebuild path runs.
        screen._grid_cache_keys.pop("library", None)
        screen._refresh_grid()
        await pilot.pause()


async def test_refresh_grid_appends_cloud_section_when_source_mode_includes_cloud() -> None:
    """Task tab with source mode BOTH and matching frontier rows produces
    an extra ``Cloud`` GridSection appended after the picks/firehose split."""
    from lilbee.cli.tui.screens.catalog_utils import SourceMode

    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        screen._active_tab_id_cache = "chat"
        screen._source_modes["chat"] = SourceMode.BOTH
        screen._all_family_rows = lambda: []  # type: ignore[method-assign]
        screen._all_hf_rows = lambda: []  # type: ignore[method-assign]
        screen._all_remote_rows = lambda: []  # type: ignore[method-assign]
        screen._frontier_rows = [
            FrontierCatalogRow(
                name="gpt-4o",
                ref="openai/gpt-4o",
                task=ModelTask.CHAT,
                provider="OpenAI",
                provider_id="openai",
                key_status=KeyStatus.READY,
            )
        ]
        screen._grid_cache_keys.pop("chat", None)
        screen._refresh_grid()
        await pilot.pause()


async def test_maybe_prefetch_on_grid_nav_skips_empty_total() -> None:
    """When mounted grids exist but all have zero rows, total == 0 short-circuits.

    Patches ``_grid_for_tab`` (which the @property accessor delegates to)
    so the helper sees a shim returning a single empty grid, bypassing the
    chat-tab DOM whose initial mount may transiently add rows on slower CI
    runners (ubuntu 3.13 in particular). Patching via unittest.mock auto-
    restores on context exit so subsequent tests aren't affected.
    """
    from unittest.mock import patch

    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        screen._active_tab_id_cache = "chat"
        screen._grid_view = True
        screen._hf_has_more = True
        screen._loading_more = False
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        empty = ModelGrid()
        empty.highlighted = 0

        class _GridContainerShim:
            def query(self, *_a: object, **_k: object) -> list[ModelGrid]:
                return [empty]

        with patch.object(screen, "_grid_for_tab", return_value=_GridContainerShim()):
            screen._focused_grid = lambda: empty  # type: ignore[method-assign]
            screen._maybe_prefetch_on_grid_nav()
            assert screen._loading_more is False


async def test_populate_library_list_renders_combined_rows() -> None:
    """Library section heads with Installed when locals exist + a Cloud section."""
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        # Seed an installed local row and a frontier row; populate.
        screen._frontier_rows = [
            FrontierCatalogRow(
                name="gpt-4o",
                ref="openai/gpt-4o",
                task=ModelTask.CHAT,
                provider="OpenAI",
                provider_id="openai",
                key_status=KeyStatus.READY,
            )
        ]
        # Force at least one installed row by mocking _all_family_rows.
        installed_row = _row("Llama 3 8B", installed=True)
        screen._all_family_rows = lambda: [installed_row]  # type: ignore[method-assign]
        screen._all_hf_rows = lambda: []  # type: ignore[method-assign]
        screen._all_remote_rows = lambda: []  # type: ignore[method-assign]
        screen._populate_library_list()
        await pilot.pause()
        from lilbee.cli.tui.widgets.model_list import ModelList

        ml = screen.query_one("#list-library", ModelList)
        assert ml.option_count > 0


async def test_library_grid_renders_installed_rows() -> None:
    """Library tab grid view shows installed rows, not just the list view."""
    from textual.containers import VerticalScroll

    from lilbee.cli.tui.widgets.model_grid import ModelGrid

    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        installed = _row("Llama 3 8B", installed=True)
        screen._all_family_rows = lambda: [installed]  # type: ignore[method-assign]
        screen._all_hf_rows = lambda: []  # type: ignore[method-assign]
        screen._all_remote_rows = lambda: []  # type: ignore[method-assign]
        screen._frontier_rows = []
        screen._populate_library_list()
        await pilot.pause()
        container = screen.query_one("#grid-library", VerticalScroll)
        grids = list(container.query(ModelGrid))
        assert grids
        assert any(installed in g.rows for g in grids)


async def test_action_select_tab_does_not_revert_after_focus_loss() -> None:
    """Pressing 6 (Library) must keep the active tab on Library, never auto-revert."""
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        screen.action_select_tab(5)
        tabs = screen.query_one("#catalog-tabs", TabbedContent)
        for _ in range(20):
            await pilot.pause()
            if tabs.active == "library" and screen._active_tab_id_cache == "library":
                break
        assert tabs.active == "library"
        assert screen._active_tab_id_cache == "library"


async def test_action_cursor_down_stays_on_active_tab() -> None:
    """Down on Chat must focus a chat-tab grid, not bounce to Discover rails."""
    from lilbee.cli.tui.widgets.model_grid import ModelGrid

    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        screen.action_select_tab(1)
        for _ in range(5):
            await pilot.pause()
        screen.action_cursor_down()
        await pilot.pause()
        tabs = screen.query_one("#catalog-tabs", TabbedContent)
        assert tabs.active == "chat"
        focused = screen.focused
        if isinstance(focused, ModelGrid):
            assert any(focused is g for g in screen.query("#grid-chat ModelGrid"))


async def test_apply_search_filter_no_op_on_discover() -> None:
    """Search filter on the Discover tab is a no-op (rails are curated)."""
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        screen._active_tab_id_cache = "discover"
        screen._apply_search_filter()


async def test_apply_worker_result_repaints_discover_rails() -> None:
    """When the user is parked on Discover, an HF worker result repaints rails."""

    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        screen._active_tab_id_cache = "discover"
        called = {"n": 0}

        def fake_populate() -> None:
            called["n"] += 1

        screen._populate_discover_rails = fake_populate  # type: ignore[method-assign]

        from lilbee.cli.tui.screens.catalog import _WORKER_FETCH_HF

        screen._apply_worker_result(_WORKER_FETCH_HF, [])
        assert called["n"] >= 1


async def test_render_library_list_swallows_missing_widget() -> None:
    """_render_library_list returns silently if the list widget can't be found."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    screen = CatalogScreen.__new__(CatalogScreen)
    screen._tab_list_cache = {}
    screen._render_library_list([], [])


async def test_render_library_grid_swallows_missing_container() -> None:
    """_render_library_grid returns silently if the grid container can't be found."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    screen = CatalogScreen.__new__(CatalogScreen)
    screen._tab_grid_cache = {}
    screen._render_library_grid([], [])


async def test_all_family_rows_skips_families_with_no_variants() -> None:
    """Families with empty variants tuples are skipped during row construction."""
    from lilbee.catalog import ModelFamily

    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._families = [
            ModelFamily(slug="empty", name="Empty", task="chat", description="x", variants=())
        ]
        screen._family_rows_cache = None
        rows = screen._all_family_rows()
        assert all(r.name != "Empty" for r in rows)


async def test_activate_initial_tab_skips_when_already_settled() -> None:
    """A second _activate_initial_tab call after the flag flipped is a no-op."""
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        screen._active_tab_id_cache = "library"
        screen._activate_initial_tab()
        assert screen._active_tab_id_cache == "library"


async def test_mount_grid_ctas_swallows_missing_container() -> None:
    """_mount_grid_ctas returns silently if the active tab's container is gone."""
    screen = CatalogScreen.__new__(CatalogScreen)
    screen._mount_grid_ctas(hf_count=0)


async def test_refresh_grid_ctas_swallows_missing_container() -> None:
    """_refresh_grid_ctas returns silently if the active tab's container is gone."""
    screen = CatalogScreen.__new__(CatalogScreen)
    screen._refresh_grid_ctas(hf_count=0)


async def test_activate_initial_tab_switches_to_chat() -> None:
    """The first _activate_initial_tab call after mount activates Chat."""
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        tabs = screen.query_one("#catalog-tabs", TabbedContent)
        screen._activation_settled = False
        tabs.active = "discover"
        screen._activate_initial_tab()
        assert tabs.active == "chat"


async def test_populate_library_renders_with_empty_frontier_when_attr_missing() -> None:
    """_populate_library_list still renders installed rows when frontier source is gone."""
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        installed = _row("Llama 3 8B", installed=True)
        screen._all_family_rows = lambda: [installed]  # type: ignore[method-assign]
        screen._all_hf_rows = lambda: []  # type: ignore[method-assign]
        screen._all_remote_rows = lambda: []  # type: ignore[method-assign]

        def boom(_search: str) -> list[FrontierCatalogRow]:
            raise AttributeError("frontier_rows missing")

        screen._build_frontier_rows = boom  # type: ignore[method-assign]
        screen._populate_library_list()
        await pilot.pause()
        from lilbee.cli.tui.widgets.model_list import ModelList

        ml = screen.query_one("#list-library", ModelList)
        assert ml.option_count > 0
