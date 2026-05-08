"""Tests for the Discover-tab rails widget."""

from __future__ import annotations

from textual.app import ComposeResult

from lilbee.catalog.types import ModelTask
from lilbee.cli.tui.screens.catalog_utils import LocalCatalogRow
from lilbee.cli.tui.widgets.discover_rails import DiscoverRails
from lilbee.cli.tui.widgets.model_grid import ModelGrid
from tests._lilbee_app_test_host import LilbeeAppHost


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
    class _App(LilbeeAppHost):
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
    class _App(LilbeeAppHost):
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


async def test_action_focus_grid_jumps_to_rail_grid() -> None:
    """Enter on a focused rail heading lands focus on that rail's grid."""

    class _App(LilbeeAppHost):
        def compose(self) -> ComposeResult:
            yield DiscoverRails(id="discover-rails")

    async with _App().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        from lilbee.cli.tui.widgets.discover_rails import _RailHeading

        rails = pilot.app.query_one(DiscoverRails)
        rails.set_rails(for_you=[_row("Llama")], collection=[], fresh=[])
        await pilot.pause()
        heading = rails.query_one("#heading-for-you", _RailHeading)
        heading.focus()
        await pilot.pause()
        heading.action_focus_grid()
        await pilot.pause()
        grid = rails.query_one("#discover-grid-for-you", ModelGrid)
        assert grid.has_focus


async def test_action_focus_grid_swallows_missing_grid() -> None:
    """If the grid id is somehow missing, action_focus_grid no-ops gracefully."""
    from lilbee.cli.tui.widgets.discover_rails import _RailHeading

    class _Bare(LilbeeAppHost):
        def compose(self) -> ComposeResult:
            # Heading without its sibling grid; query_one inside
            # action_focus_grid raises NoMatches and the action no-ops.
            yield _RailHeading("Lonely", rail_id="missing")

    async with _Bare().run_test(size=(80, 20)) as pilot:
        await pilot.pause()
        heading = pilot.app.query_one(_RailHeading)
        # Should not raise.
        heading.action_focus_grid()


async def test_action_focus_grid_handles_unparented_heading() -> None:
    """A heading instantiated outside a parent no-ops on Enter."""
    from lilbee.cli.tui.widgets.discover_rails import _RailHeading

    heading = _RailHeading("Detached", rail_id="for-you")
    # Should not raise.
    heading.action_focus_grid()


async def test_set_rail_swallows_missing_grid() -> None:
    """_set_rail no-ops when the rail's grid id is gone (post-remount race)."""
    from lilbee.cli.tui.widgets.discover_rails import DiscoverRails as DiscoverRailsCls

    class _Empty(LilbeeAppHost):
        def compose(self) -> ComposeResult:
            return iter(())

    async with _Empty().run_test(size=(80, 20)) as pilot:
        await pilot.pause()
        # Build a DiscoverRails outside the mounted DOM, then call _set_rail
        # query_one will raise and the helper should swallow it cleanly.
        rails = DiscoverRailsCls()
        rails._set_rail("for-you", [_row("Llama")])


async def test_empty_lists_render_empty_grids() -> None:
    """Rails with zero rows still keep their headings; layout stays stable."""

    class _App(LilbeeAppHost):
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
