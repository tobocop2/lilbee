"""Catalog screen: `f` keybinding toggles the hide-unsupported filter."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widgets import Static

from lilbee.catalog.types import ModelCompat
from lilbee.cli.tui.screens.catalog import CatalogScreen, _is_unsupported_local_row
from lilbee.cli.tui.screens.catalog_utils import LocalCatalogRow
from tests._lilbee_app_test_host import LilbeeAppHost


def _row(compat: ModelCompat, *, name: str = "x") -> LocalCatalogRow:
    return LocalCatalogRow(
        name=name,
        task="chat",
        params="",
        size="",
        quant="",
        downloads="",
        featured=False,
        installed=False,
        sort_downloads=0,
        sort_size=0,
        ref=f"acme/{name}-GGUF",
        backend="native",
        compat=compat,
    )


def test_is_unsupported_local_row_matches_only_local_unsupported() -> None:
    assert _is_unsupported_local_row(_row(ModelCompat.UNSUPPORTED)) is True
    assert _is_unsupported_local_row(_row(ModelCompat.UNKNOWN)) is False
    assert _is_unsupported_local_row(_row(ModelCompat.SUPPORTED)) is False


def test_is_unsupported_local_row_ignores_frontier_rows() -> None:
    from lilbee.cli.tui.screens.catalog_utils import (
        CatalogRowKind as _Kind,
    )
    from lilbee.cli.tui.screens.catalog_utils import (
        FrontierCatalogRow,
        KeyStatus,
    )

    frontier = FrontierCatalogRow(
        name="gpt-5",
        ref="openai/gpt-5",
        task="chat",
        provider="OpenAI",
        provider_id="openai",
        key_status=KeyStatus.READY,
    )
    assert frontier.kind == _Kind.FRONTIER
    assert _is_unsupported_local_row(frontier) is False


async def test_action_toggle_flips_state_and_notifies() -> None:
    class _App(LilbeeAppHost):
        def compose(self) -> ComposeResult:
            yield Static("host")

    async with _App().run_test(size=(120, 40)) as pilot:
        screen = CatalogScreen()
        await pilot.app.push_screen(screen)
        await pilot.pause()

        assert screen._hide_unsupported is False
        screen.action_toggle_hide_unsupported()
        await pilot.pause()
        assert screen._hide_unsupported is True
        screen.action_toggle_hide_unsupported()
        await pilot.pause()
        assert screen._hide_unsupported is False


async def test_action_toggle_skipped_when_search_focused(
    monkeypatch,
) -> None:
    class _App(LilbeeAppHost):
        def compose(self) -> ComposeResult:
            yield Static("host")

    async with _App().run_test(size=(120, 40)) as pilot:
        screen = CatalogScreen()
        await pilot.app.push_screen(screen)
        await pilot.pause()
        monkeypatch.setattr(type(screen), "_search_focused", property(lambda _self: True))
        screen.action_toggle_hide_unsupported()
        assert screen._hide_unsupported is False
