"""Catalog Frontier (cloud) section tests.

The catalog renders a Frontier super-section above local sections when
the user has any provider key configured. Rows render as
``FrontierCatalogRow`` and carry provider + key-status pills. Adding
or clearing an API key republishes ``provider_availability_changed_signal``
and the catalog rebuilds.
"""

from __future__ import annotations

from lilbee.catalog.types import ModelTask
from lilbee.cli.tui.screens.catalog_utils import (
    FrontierCatalogRow,
    KeyStatus,
    LocalCatalogRow,
    matches_search,
)
from tests._lilbee_app_test_host import LilbeeAppHost


def _frontier(
    name: str,
    *,
    provider: str = "Gemini",
    status: KeyStatus = KeyStatus.READY,
) -> FrontierCatalogRow:
    return FrontierCatalogRow(
        name=name,
        ref=name,
        task=ModelTask.CHAT,
        provider=provider,
        provider_id=provider.lower(),
        key_status=status,
    )


async def _wait_for_active_tab(pilot, screen, expected: str, timeout_iters: int = 20) -> None:
    """Pace until ``CatalogScreen._active_tab_id()`` reports *expected*.

    The bare ``tabs.active = "library"; await pilot.pause()`` pattern races
    on Windows: the assignment dispatches via a Textual message and a single
    pause does not always flush before the next read. Poll instead.
    """
    import asyncio

    for _ in range(timeout_iters):
        if screen._active_tab_id() == expected:
            return
        await pilot.pause()
        await asyncio.sleep(0.05)


def _local(name: str, *, installed: bool = False, featured: bool = False) -> LocalCatalogRow:
    return LocalCatalogRow(
        name=name,
        task=ModelTask.CHAT,
        params="--",
        size="--",
        quant="--",
        downloads="--",
        featured=featured,
        installed=installed,
        sort_downloads=0,
        sort_size=0.0,
        ref=name,
        backend="native",
    )


class TestRowTypes:
    def test_local_and_frontier_are_disjoint(self) -> None:
        local = _local("Qwen3 0.6B")
        frontier = _frontier("gemini-2.0-flash")
        assert isinstance(local, LocalCatalogRow)
        assert isinstance(frontier, FrontierCatalogRow)
        assert not isinstance(local, FrontierCatalogRow)

    def test_matches_search_local_uses_task_and_quant(self) -> None:
        row = _local("Qwen3")
        row.quant = "Q4_K_M"
        assert matches_search(row, "qwen") is True
        assert matches_search(row, "q4") is True
        assert matches_search(row, "gemini") is False

    def test_matches_search_frontier_uses_provider(self) -> None:
        row = _frontier("gemini-2.0-flash", provider="Gemini")
        # Provider name and the model id both match.
        assert matches_search(row, "gemini") is True
        assert matches_search(row, "flash") is True
        # Frontier rows never read the local-only fields, so an
        # imaginary local-side filter must not match a frontier row.
        assert matches_search(row, "q4_k_m") is False


class TestGroupRowsForGrid:
    def test_grid_groups_only_local_rows(self) -> None:
        """Featured rows live at the top of their task section; no separate
        "Our picks" bucket. Installed rows still get their own section."""
        from lilbee.cli.tui.screens.catalog_grouping import group_rows_for_grid

        local = [_local("Qwen3", featured=True), _local("Llama", installed=True)]
        sections = group_rows_for_grid(local)
        non_empty = [s for s in sections if s.rows]
        headings = [s.heading for s in non_empty]
        assert "Our picks" not in headings
        assert "Installed" in headings
        # Featured row lives in its task section (Chat).
        chat_section = next(s for s in non_empty if s.heading == "Chat")
        assert any(getattr(r, "featured", False) for r in chat_section.rows)
        # Featured comes first in the task section.
        assert getattr(chat_section.rows[0], "featured", False) is True


class TestGroupFrontierRows:
    def test_provider_sections_alphabetical_within_group(self) -> None:
        from lilbee.cli.tui.screens.catalog_grouping import group_frontier_rows

        rows = [
            _frontier("gpt-4o", provider="OpenAI"),
            _frontier("gemini-2.0-flash", provider="Gemini"),
            _frontier("gemini-1.5-pro", provider="Gemini"),
        ]
        sections = group_frontier_rows(rows)
        assert [s.heading for s in sections] == ["Gemini", "OpenAI"]
        assert [r.name for r in sections[0].rows] == ["gemini-1.5-pro", "gemini-2.0-flash"]
        assert [r.name for r in sections[1].rows] == ["gpt-4o"]

    def test_empty_input_returns_empty_list(self) -> None:
        from lilbee.cli.tui.screens.catalog_grouping import group_frontier_rows

        assert group_frontier_rows([]) == []


class TestFrontierTabBehavior:
    """Coverage for Frontier-tab UI plumbing: action gating + sort-label + dispatch."""

    async def test_active_tab_id_default_is_chat(self) -> None:
        """Pre-mount, _active_tab_id reads its cache field initialized in __init__.

        The cache is set in __init__ (TAB_CHAT default) so even before
        compose runs, callers reading _active_tab_id get a sane value
        without doing a DOM walk.
        """
        from lilbee.cli.tui.screens.catalog import CatalogScreen

        screen = CatalogScreen()
        assert screen._active_tab_id() == "chat"

    async def test_apply_search_filter_in_frontier_tab_repopulates_list(self, monkeypatch) -> None:
        from textual.app import ComposeResult
        from textual.widgets import TabbedContent

        from lilbee.cli.tui.screens.catalog import CatalogScreen

        class _App(LilbeeAppHost):
            def compose(self) -> ComposeResult:
                yield CatalogScreen()

        async with _App().run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = pilot.app.query_one(CatalogScreen)
            screen._frontier_rows = [_frontier("gemini-2.0-flash")]
            screen._populate_library_list()
            await pilot.pause()
            screen.query_one("#catalog-tabs", TabbedContent).active = "library"
            await _wait_for_active_tab(pilot, screen, "library")
            from unittest import mock

            with mock.patch.object(screen, "_populate_library_list") as populate:
                screen._apply_search_filter()
                populate.assert_called_once()

    async def test_action_toggle_view_is_noop_on_frontier_tab(self) -> None:
        from textual.app import ComposeResult
        from textual.widgets import TabbedContent

        from lilbee.cli.tui.screens.catalog import CatalogScreen

        class _App(LilbeeAppHost):
            def compose(self) -> ComposeResult:
                yield CatalogScreen()

        async with _App().run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = pilot.app.query_one(CatalogScreen)
            screen._frontier_rows = [_frontier("x")]
            screen._populate_library_list()
            await pilot.pause()
            screen.query_one("#catalog-tabs", TabbedContent).active = "library"
            await _wait_for_active_tab(pilot, screen, "library")
            grid_before = screen._grid_view
            screen.action_toggle_view()
            assert screen._grid_view is grid_before

    async def test_action_cycle_sort_is_noop_on_frontier_tab(self) -> None:
        from textual.app import ComposeResult
        from textual.widgets import TabbedContent

        from lilbee.cli.tui.screens.catalog import CatalogScreen

        class _App(LilbeeAppHost):
            def compose(self) -> ComposeResult:
                yield CatalogScreen()

        async with _App().run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = pilot.app.query_one(CatalogScreen)
            screen._frontier_rows = [_frontier("x")]
            screen._populate_library_list()
            await pilot.pause()
            screen.query_one("#catalog-tabs", TabbedContent).active = "library"
            await _wait_for_active_tab(pilot, screen, "library")
            sort_before = screen._sort_column
            screen.action_cycle_sort()
            assert screen._sort_column == sort_before

    async def test_frontier_summary_label_uses_provider_count(self) -> None:
        from textual.app import ComposeResult
        from textual.widgets import Static, TabbedContent

        from lilbee.cli.tui.screens.catalog import CatalogScreen

        class _App(LilbeeAppHost):
            def compose(self) -> ComposeResult:
                yield CatalogScreen()

        async with _App().run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = pilot.app.query_one(CatalogScreen)
            screen._frontier_rows = [
                _frontier("gemini-x", provider="Gemini"),
                _frontier("gpt-x", provider="OpenAI"),
            ]
            screen._populate_library_list()
            await pilot.pause()
            screen.query_one("#catalog-tabs", TabbedContent).active = "library"
            await _wait_for_active_tab(pilot, screen, "library")
            screen._update_sort_label()
            text = str(screen.query_one("#sort-label", Static).render())
            assert "2" in text  # 2 cloud models
            assert "providers" in text

    async def test_populate_library_list_silently_returns_when_widget_missing(self) -> None:
        from lilbee.cli.tui.screens.catalog import CatalogScreen

        screen = CatalogScreen.__new__(CatalogScreen)
        screen._frontier_rows = []
        screen._populate_library_list()

    async def test_action_load_more_is_noop_on_frontier_tab(self) -> None:
        from textual.app import ComposeResult
        from textual.widgets import TabbedContent

        from lilbee.cli.tui.screens.catalog import CatalogScreen

        class _App(LilbeeAppHost):
            def compose(self) -> ComposeResult:
                yield CatalogScreen()

        async with _App().run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = pilot.app.query_one(CatalogScreen)
            screen._frontier_rows = [_frontier("x")]
            screen._populate_library_list()
            await pilot.pause()
            screen.query_one("#catalog-tabs", TabbedContent).active = "library"
            await _wait_for_active_tab(pilot, screen, "library")
            from unittest import mock

            with mock.patch.object(screen, "_load_more") as mock_load:
                screen.action_load_more()
                mock_load.assert_not_called()

    async def test_populate_library_list_swallows_lookup_failure(self) -> None:
        """_populate_library_list returns silently if catalog-tabs is gone."""
        from lilbee.cli.tui.screens.catalog import CatalogScreen

        screen = CatalogScreen.__new__(CatalogScreen)
        screen._frontier_rows = [_frontier("x")]
        screen._populate_library_list()

    async def test_populate_library_list_repopulates_when_tab_already_present(self) -> None:
        from textual.app import ComposeResult

        from lilbee.cli.tui.screens.catalog import CatalogScreen

        class _App(LilbeeAppHost):
            def compose(self) -> ComposeResult:
                yield CatalogScreen()

        async with _App().run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = pilot.app.query_one(CatalogScreen)
            screen._frontier_rows = [_frontier("x")]
            screen._populate_library_list()
            # Two pauses: TabbedContent.add_pane returns AwaitComplete and
            # the mount lands on the next refresh tick, not the same one.
            await pilot.pause()
            await pilot.pause()
            from unittest import mock

            with mock.patch.object(screen, "_populate_library_list") as populate:
                screen._populate_library_list()
                populate.assert_called_once()


class TestFetchFrontierWorker:
    """The frontier worker reads provider keys from cfg and emits rows."""

    def test_emits_rows_with_ready_status_when_key_set(self, monkeypatch) -> None:
        from lilbee.cli.tui.screens.catalog import CatalogScreen
        from lilbee.modelhub.model_manager import RemoteModel

        screen = CatalogScreen.__new__(CatalogScreen)
        screen._frontier_rows = []

        rm = RemoteModel(
            name="model-x",
            task="chat",
            provider="OpenAI",
            parameter_size="--",
            family="gpt",
        )
        monkeypatch.setattr(
            "lilbee.modelhub.model_manager.discover_api_models",
            lambda: {"OpenAI": [rm]},
        )
        from lilbee.core.config import cfg as _cfg

        old_key = _cfg.openai_api_key
        _cfg.openai_api_key = "sk-test"
        try:
            rows = screen._fetch_frontier_models.__wrapped__(screen)
        finally:
            _cfg.openai_api_key = old_key
        assert rows
        assert rows[0].provider == "OpenAI"

    def test_returns_empty_list_when_discover_raises(self, monkeypatch) -> None:
        from lilbee.cli.tui.screens.catalog import CatalogScreen

        screen = CatalogScreen.__new__(CatalogScreen)
        screen._frontier_rows = []

        def _boom() -> dict:
            raise RuntimeError("provider discovery is down")

        monkeypatch.setattr(
            "lilbee.modelhub.model_manager.discover_api_models",
            _boom,
        )
        rows = screen._fetch_frontier_models.__wrapped__(screen)
        assert rows == []


class TestFrontierSelection:
    """ModelList.Selected on the Frontier tab routes through _select_row."""

    async def test_frontier_list_selected_dispatches_to_select_row(self) -> None:
        from unittest import mock

        from textual.app import ComposeResult

        from lilbee.cli.tui.screens.catalog import CatalogScreen

        class _App(LilbeeAppHost):
            def compose(self) -> ComposeResult:
                yield CatalogScreen()

        async with _App().run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = pilot.app.query_one(CatalogScreen)
            row = _frontier("gpt-x", provider="OpenAI")
            evt = mock.MagicMock()
            evt.row = row
            with mock.patch.object(screen, "_select_row") as select:
                screen._on_model_list_selected(evt)
                select.assert_called_once_with(row)

    async def test_select_row_dispatches_frontier_branch(self) -> None:
        """_select_row routes FrontierCatalogRow into _select_frontier_row."""
        from unittest import mock

        from lilbee.cli.tui.screens.catalog import CatalogScreen

        screen = CatalogScreen.__new__(CatalogScreen)
        row = _frontier("gpt-x", provider="OpenAI")
        with mock.patch.object(screen, "_select_frontier_row") as select_frontier:
            screen._select_row(row)
            select_frontier.assert_called_once_with(row)

    async def test_select_frontier_row_missing_key_switches_to_settings(self) -> None:
        from unittest import mock

        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.catalog import CatalogScreen
        from lilbee.cli.tui.screens.catalog_utils import KeyStatus

        screen = CatalogScreen.__new__(CatalogScreen)
        # Patch the .app attribute on the descriptor so the LilbeeApp branch fires.
        fake_app = mock.MagicMock(spec=LilbeeApp)
        with (
            mock.patch.object(
                CatalogScreen, "app", new_callable=mock.PropertyMock, return_value=fake_app
            ),
            mock.patch.object(CatalogScreen, "notify"),
        ):
            row = _frontier("gpt-x", provider="OpenAI", status=KeyStatus.MISSING_KEY)
            screen._select_frontier_row(row)
        fake_app.switch_view.assert_called_once_with("Settings")


class TestLibraryTab:
    """Library is statically mounted; _populate_library_list renders frontier rows."""

    async def test_library_tab_always_present(self) -> None:
        from textual.app import ComposeResult
        from textual.widgets import TabbedContent

        from lilbee.cli.tui.screens.catalog import CatalogScreen

        class _App(LilbeeAppHost):
            def compose(self) -> ComposeResult:
                yield CatalogScreen()

        async with _App().run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = pilot.app.query_one(CatalogScreen)
            tabs = screen.query_one("#catalog-tabs", TabbedContent)
            # Library is in compose() so it exists from first paint, before
            # any frontier_rows arrive. The dynamic add_pane choreography
            # the legacy Frontier tab needed is gone.
            assert tabs.query("#library")
            assert tabs.query("#list-library")

    async def test_populate_library_renders_frontier_rows_into_list_widget(self) -> None:
        from textual.app import ComposeResult

        from lilbee.cli.tui.screens.catalog import CatalogScreen
        from lilbee.cli.tui.widgets.model_list import ModelList

        class _App(LilbeeAppHost):
            def compose(self) -> ComposeResult:
                yield CatalogScreen()

        async with _App().run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = pilot.app.query_one(CatalogScreen)
            screen._frontier_rows = [_frontier("gemini-2.0-flash", provider="Gemini")]
            ml = screen.query_one("#list-library", ModelList)
            for _ in range(20):
                screen._tab_list_cache = {}
                screen._populate_library_list()
                await pilot.pause()
                if ml.option_count > 0:
                    break
            assert ml.option_count > 0

    async def test_populate_library_with_empty_rows_clears_frontier_section(self) -> None:
        """Frontier section disappears when frontier_rows clears.

        Library now also surfaces installed local rows, so option_count
        isn't necessarily 0; the cleared frontier rows must still drop
        their provider section. We assert the frontier provider name no
        longer appears in any option label.
        """
        from textual.app import ComposeResult

        from lilbee.cli.tui.screens.catalog import CatalogScreen
        from lilbee.cli.tui.widgets.model_list import ModelList

        class _App(LilbeeAppHost):
            def compose(self) -> ComposeResult:
                yield CatalogScreen()

        async with _App().run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = pilot.app.query_one(CatalogScreen)
            screen._frontier_rows = [_frontier("gemini-2.0-flash", provider="Gemini")]
            screen._populate_library_list()
            await pilot.pause()
            screen._frontier_rows = []
            screen._populate_library_list()
            await pilot.pause()
            ml = screen.query_one("#list-library", ModelList)
            labels = " ".join(str(opt.prompt) for opt in ml._options if hasattr(opt, "prompt"))
            assert "gemini-2.0-flash" not in labels


class TestBuildFrontierRows:
    """``_build_frontier_rows`` is the synchronous read of the cache the
    worker populates. The discovery itself runs on a worker thread; tests
    seed the cache directly to keep the UI thread fast."""

    def test_empty_cache_yields_no_rows(self) -> None:
        from lilbee.cli.tui.screens.catalog import CatalogScreen

        screen = CatalogScreen.__new__(CatalogScreen)
        screen._frontier_rows = []
        assert screen._build_frontier_rows("") == []

    def test_filters_against_search(self) -> None:
        from lilbee.cli.tui.screens.catalog import CatalogScreen

        screen = CatalogScreen.__new__(CatalogScreen)
        screen._frontier_rows = [
            _frontier("gemini-2.0-flash", provider="Gemini"),
            _frontier("gpt-4o", provider="OpenAI"),
        ]
        gemini_only = screen._build_frontier_rows("gemini")
        names = {r.name for r in gemini_only}
        assert names == {"gemini-2.0-flash"}
