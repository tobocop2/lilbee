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
        is_curated=True,
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
    def test_frontier_super_section_appears_first(self) -> None:
        from lilbee.cli.tui.screens.catalog import _group_rows_for_grid

        local = [_local("Qwen3", featured=True)]
        frontier = [_frontier("gemini-2.0-flash"), _frontier("gpt-4o", provider="OpenAI")]
        sections = _group_rows_for_grid(local, frontier)

        non_empty = [s for s in sections if s.rows]
        assert non_empty[0].is_frontier is True
        # Both providers get their own subsection, sorted alphabetically.
        provider_headings = [s.heading for s in non_empty if s.is_frontier]
        assert provider_headings[0].endswith("Gemini")
        assert provider_headings[1].endswith("OpenAI")
        # Local sections follow.
        assert non_empty[-1].is_frontier is False


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
