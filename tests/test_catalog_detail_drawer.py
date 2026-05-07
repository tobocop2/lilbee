"""Tests for the catalog detail drawer (right pane)."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Static

from lilbee.cli.tui.screens.catalog_utils import (
    FrontierCatalogRow,
    KeyStatus,
    LocalCatalogRow,
    SizeVariant,
)
from lilbee.cli.tui.widgets.catalog_detail import (
    _EMPTY_HINT,
    CatalogDetailDrawer,
)
from lilbee.modelhub.models import ModelTask
from lilbee.runtime.hardware import FitChip, FitLevel


def _local_row(
    name: str, *, fit: FitChip | None = None, variants: list[SizeVariant] | None = None
) -> LocalCatalogRow:
    return LocalCatalogRow(
        name=name,
        task=ModelTask.CHAT,
        params="8B",
        size="4.6 GB",
        quant="Q4_K_M",
        downloads="--",
        featured=False,
        installed=False,
        sort_downloads=0,
        sort_size=4.6,
        ref=name,
        backend="native",
        size_variants=variants or [],
        fit=fit,
    )


async def _drawer_in_test_app() -> tuple[CatalogDetailDrawer, App]:
    class _App(App):
        def compose(self) -> ComposeResult:
            yield CatalogDetailDrawer(id="catalog-detail-drawer")

    app = _App()
    return app, app


async def test_initial_state_shows_empty_hint() -> None:
    class _App(App):
        def compose(self) -> ComposeResult:
            yield CatalogDetailDrawer(id="catalog-detail-drawer")

    async with _App().run_test(size=(80, 20)) as pilot:
        await pilot.pause()
        drawer = pilot.app.query_one(CatalogDetailDrawer)
        name = drawer.query_one("#catalog-detail-name", Static)
        assert _EMPTY_HINT in str(name.render())


async def test_update_for_row_renders_local_row_name() -> None:
    class _App(App):
        def compose(self) -> ComposeResult:
            yield CatalogDetailDrawer(id="catalog-detail-drawer")

    async with _App().run_test(size=(80, 20)) as pilot:
        await pilot.pause()
        drawer = pilot.app.query_one(CatalogDetailDrawer)
        drawer.update_for_row(_local_row("Llama 3.1 8B"))
        await pilot.pause()
        rendered = str(drawer.query_one("#catalog-detail-name", Static).render())
        assert "Llama 3.1 8B" in rendered


async def test_update_for_row_with_none_clears_to_empty_hint() -> None:
    class _App(App):
        def compose(self) -> ComposeResult:
            yield CatalogDetailDrawer(id="catalog-detail-drawer")

    async with _App().run_test(size=(80, 20)) as pilot:
        await pilot.pause()
        drawer = pilot.app.query_one(CatalogDetailDrawer)
        drawer.update_for_row(_local_row("Llama"))
        await pilot.pause()
        drawer.update_for_row(None)
        await pilot.pause()
        rendered = str(drawer.query_one("#catalog-detail-name", Static).render())
        assert _EMPTY_HINT in rendered


async def test_update_for_row_lists_size_variants() -> None:
    class _App(App):
        def compose(self) -> ComposeResult:
            yield CatalogDetailDrawer(id="catalog-detail-drawer")

    async with _App().run_test(size=(80, 20)) as pilot:
        await pilot.pause()
        drawer = pilot.app.query_one(CatalogDetailDrawer)
        variants = [
            SizeVariant(label="8B Q4_K_M", quant="Q4_K_M", size_gb=4.6, ref="r/q4"),
            SizeVariant(label="8B Q5_K_M", quant="Q5_K_M", size_gb=5.7, ref="r/q5"),
        ]
        drawer.update_for_row(_local_row("Llama", variants=variants))
        await pilot.pause()
        rendered = str(drawer.query_one("#catalog-detail-sizes", Static).render())
        assert "Q4_K_M" in rendered
        assert "Q5_K_M" in rendered


async def test_update_for_row_renders_fit_chip_text() -> None:
    class _App(App):
        def compose(self) -> ComposeResult:
            yield CatalogDetailDrawer(id="catalog-detail-drawer")

    async with _App().run_test(size=(80, 20)) as pilot:
        await pilot.pause()
        drawer = pilot.app.query_one(CatalogDetailDrawer)
        chip = FitChip(level=FitLevel.FITS, headroom_gb=8.0)
        drawer.update_for_row(_local_row("Llama", fit=chip))
        await pilot.pause()
        rendered = str(drawer.query_one("#catalog-detail-fit", Static).render())
        assert "fits" in rendered.lower() or "+8" in rendered


async def test_frontier_row_shows_provider_in_license_slot() -> None:
    class _App(App):
        def compose(self) -> ComposeResult:
            yield CatalogDetailDrawer(id="catalog-detail-drawer")

    async with _App().run_test(size=(80, 20)) as pilot:
        await pilot.pause()
        drawer = pilot.app.query_one(CatalogDetailDrawer)
        row = FrontierCatalogRow(
            name="gpt-4o",
            ref="openai/gpt-4o",
            task=ModelTask.CHAT,
            provider="OpenAI",
            provider_id="openai",
            key_status=KeyStatus.READY,
        )
        drawer.update_for_row(row)
        await pilot.pause()
        rendered = str(drawer.query_one("#catalog-detail-license", Static).render())
        assert "OpenAI" in rendered
