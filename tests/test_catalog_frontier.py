"""Catalog Frontier (cloud) section tests.

The catalog renders a Frontier super-section above local sections when
the user has any provider key configured. Rows render as
``FrontierCatalogRow`` and carry provider + key-status pills. Adding
or clearing an API key republishes ``provider_availability_changed_signal``
and the catalog rebuilds.
"""

from __future__ import annotations

from lilbee.cli.tui.screens.catalog_utils import (
    FrontierCatalogRow,
    KeyStatus,
    LocalCatalogRow,
    matches_search,
)
from lilbee.modelhub.models import ModelTask


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
        from lilbee.cli.tui.screens.catalog import _group_rows_for_grid

        local = [_local("Qwen3", featured=True), _local("Llama", installed=True)]
        sections = _group_rows_for_grid(local)
        non_empty = [s for s in sections if s.rows]
        headings = [s.heading for s in non_empty]
        assert "Our picks" in headings
        assert "Installed" in headings


class TestGroupFrontierRows:
    def test_provider_sections_alphabetical_within_group(self) -> None:
        from lilbee.cli.tui.screens.catalog import _group_frontier_rows

        rows = [
            _frontier("gpt-4o", provider="OpenAI"),
            _frontier("gemini-2.0-flash", provider="Gemini"),
            _frontier("gemini-1.5-pro", provider="Gemini"),
        ]
        sections = _group_frontier_rows(rows)
        assert [s.heading for s in sections] == ["Gemini", "OpenAI"]
        assert [r.name for r in sections[0].rows] == ["gemini-1.5-pro", "gemini-2.0-flash"]
        assert [r.name for r in sections[1].rows] == ["gpt-4o"]

    def test_empty_input_returns_empty_list(self) -> None:
        from lilbee.cli.tui.screens.catalog import _group_frontier_rows

        assert _group_frontier_rows([]) == []


class TestSyncFrontierTab:
    """Frontier TabPane is mounted iff at least one frontier row is cached."""

    async def test_tab_absent_until_rows_arrive(self, monkeypatch) -> None:
        from textual.app import App, ComposeResult
        from textual.widgets import TabbedContent

        from lilbee.cli.tui.screens.catalog import CatalogScreen

        class _App(App):
            def compose(self) -> ComposeResult:
                yield CatalogScreen()

        async with _App().run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = pilot.app.query_one(CatalogScreen)
            tabs = screen.query_one("#catalog-tabs", TabbedContent)
            assert not tabs.query("#frontier")
            screen._frontier_rows = [_frontier("gemini-2.0-flash", provider="Gemini")]
            screen._sync_frontier_tab()
            await pilot.pause()
            assert tabs.query("#frontier")

    async def test_tab_removed_when_rows_clear(self) -> None:
        from textual.app import App, ComposeResult
        from textual.widgets import TabbedContent

        from lilbee.cli.tui.screens.catalog import CatalogScreen

        class _App(App):
            def compose(self) -> ComposeResult:
                yield CatalogScreen()

        async with _App().run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = pilot.app.query_one(CatalogScreen)
            screen._frontier_rows = [_frontier("gemini-2.0-flash")]
            screen._sync_frontier_tab()
            await pilot.pause()
            tabs = screen.query_one("#catalog-tabs", TabbedContent)
            assert tabs.query("#frontier")
            screen._frontier_rows = []
            screen._sync_frontier_tab()
            await pilot.pause()
            assert not tabs.query("#frontier")


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
