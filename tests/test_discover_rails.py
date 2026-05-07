"""Tests for the Discover-tab rails widget."""

from __future__ import annotations

from textual.app import App, ComposeResult

from lilbee.cli.tui.screens.catalog_utils import LocalCatalogRow
from lilbee.cli.tui.widgets.discover_rails import DiscoverRails
from lilbee.cli.tui.widgets.model_grid import ModelGrid
from lilbee.modelhub.models import ModelTask


def _row(name: str, *, featured: bool = False, installed: bool = False) -> LocalCatalogRow:
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


async def test_three_rails_mount() -> None:
    class _App(App):
        def compose(self) -> ComposeResult:
            yield DiscoverRails(id="discover-rails")

    async with _App().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        rails = pilot.app.query_one(DiscoverRails)
        # All three rail grids exist after compose.
        assert rails.query_one("#discover-grid-for-you", ModelGrid) is not None
        assert rails.query_one("#discover-grid-collection", ModelGrid) is not None
        assert rails.query_one("#discover-grid-fresh", ModelGrid) is not None


async def test_set_rails_pushes_rows_into_each_grid() -> None:
    class _App(App):
        def compose(self) -> ComposeResult:
            yield DiscoverRails(id="discover-rails")

    async with _App().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        rails = pilot.app.query_one(DiscoverRails)
        for_you = [_row("Llama", featured=True)]
        collection = [_row("Mistral", installed=True), _row("bge-large", installed=True)]
        fresh = [_row("DeepSeek"), _row("Hermes"), _row("Qwen")]
        rails.set_rails(for_you=for_you, collection=collection, fresh=fresh)
        await pilot.pause()
        for_you_rows = rails.query_one("#discover-grid-for-you", ModelGrid).rows
        coll_rows = rails.query_one("#discover-grid-collection", ModelGrid).rows
        fresh_rows = rails.query_one("#discover-grid-fresh", ModelGrid).rows
        assert [r.name for r in for_you_rows] == ["Llama"]
        assert [r.name for r in coll_rows] == ["Mistral", "bge-large"]
        assert [r.name for r in fresh_rows] == ["DeepSeek", "Hermes", "Qwen"]


async def test_empty_lists_render_empty_grids() -> None:
    """Rails with zero rows still keep their headings; layout stays stable."""

    class _App(App):
        def compose(self) -> ComposeResult:
            yield DiscoverRails(id="discover-rails")

    async with _App().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        rails = pilot.app.query_one(DiscoverRails)
        rails.set_rails(for_you=[], collection=[], fresh=[])
        await pilot.pause()
        assert rails.query_one("#discover-grid-for-you", ModelGrid).rows == []
        assert rails.query_one("#discover-grid-collection", ModelGrid).rows == []
        assert rails.query_one("#discover-grid-fresh", ModelGrid).rows == []
