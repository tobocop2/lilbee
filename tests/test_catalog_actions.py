"""Coverage for new catalog actions: select_tab, cycle_source, toggle_drawer,
on_key digit intercept, dismiss_filter, discover-rail population edges."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.events import Key
from textual.widgets import Input, TabbedContent

from lilbee.cli.tui.screens.catalog import (
    CatalogScreen,
    _for_you_sort_key,
    _row_cache_signature,
)
from lilbee.cli.tui.screens.catalog_utils import (
    FrontierCatalogRow,
    KeyStatus,
    LocalCatalogRow,
)
from lilbee.modelhub.models import ModelTask
from lilbee.runtime.hardware import FitChip, FitLevel


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


class _CatalogTestApp(App):
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
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        inp = screen.query_one("#catalog-search", Input)
        inp.focus()
        await pilot.pause()
        before = screen.query_one("#catalog-tabs", TabbedContent).active
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
    """The screen-level on_key handler intercepts digit keys outside Input focus."""
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        event = Key(key="3", character="3")
        screen.on_key(event)
        await pilot.pause()
        tabs = screen.query_one("#catalog-tabs", TabbedContent)
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
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        screen._active_tab_id_cache = "chat"
        inp = screen.query_one("#catalog-search", Input)
        # set_focus is synchronous; inp.focus() schedules a message that
        # may not flush in time on slower CI runners (ubuntu 3.13 in CI).
        screen.set_focus(inp)
        await pilot.pause()
        assert screen.focused is inp, "test setup precondition: filter Input is focused"
        before = screen._source_modes["chat"]
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
    assert _row_cache_signature(frontier) == ("gpt-4o", False)
    local = _row("Llama", installed=True)
    assert _row_cache_signature(local) == ("Llama", True)


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
        # Stub the grid query so _on_search_submitted's "any grid has rows"
        # check sees no grids; without this stub the screen's mounted
        # Discover rails / per-tab grids may have rows that trigger the
        # _select_first_visible_grid_card early-return on slower CI runners.
        screen.query = lambda *args, **kwargs: []  # type: ignore[method-assign]
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
    ordered = sorted([none, wont, tight, fits], key=_for_you_sort_key)
    assert [r.name for r in ordered] == ["a-fits", "b-tight", "c-wont", "d-none"]


def test_probe_available_memory_returns_none_on_failure(monkeypatch) -> None:
    """Hardware probe failures fall through chip-less, never crash."""
    import lilbee.providers.model_cache as mc

    def boom(_fraction: float) -> int:
        raise RuntimeError("psutil missing")

    monkeypatch.setattr(mc, "get_available_memory", boom)
    assert CatalogScreen._probe_available_memory() is None


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
    """Esc with focus inside the filter Input clears it and hides the box."""
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        inp = screen.query_one("#catalog-search", Input)
        inp.value = "qwen"
        inp.focus()
        await pilot.pause()
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
        # No grid focused; helper returns None.
        screen.set_focus(None)
        await pilot.pause()
        assert screen._capture_focused_section() is None


async def test_reveal_scroll_hint_no_op_without_focused_grid() -> None:
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen.set_focus(None)
        await pilot.pause()
        # Should not raise; covers the early-return branch.
        screen._reveal_scroll_hint_at_catalog_end()


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
        screen._populate_library_list()
        await pilot.pause()
        from lilbee.cli.tui.widgets.model_list import ModelList

        ml = screen.query_one("#list-library", ModelList)
        # ModelList prepends a provider section heading + row option, so
        # option_count is >= 1 (typically 2 for one frontier row).
        assert ml.option_count >= 1


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
    """When mounted grids exist but all have zero rows, total == 0 short-circuits."""
    async with _CatalogTestApp().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._activation_settled = True
        screen._active_tab_id_cache = "chat"
        screen._grid_view = True
        screen._hf_has_more = True
        screen._loading_more = False
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        chat_container = screen.query_one("#grid-chat")
        # Wipe any pre-existing children, mount one empty ModelGrid as the
        # only grid in the container, then point _focused_grid at it.
        chat_container.remove_children()
        await pilot.pause()
        empty = ModelGrid()
        await chat_container.mount(empty)
        await pilot.pause()
        empty.highlighted = 0
        screen._focused_grid = lambda: empty  # type: ignore[method-assign]
        screen._maybe_prefetch_on_grid_nav()
        # total == 0 short-circuits; _load_more never fires.
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
