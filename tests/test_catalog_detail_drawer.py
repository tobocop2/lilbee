"""Tests for the catalog detail drawer (right pane)."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Static

from lilbee.catalog.types import ModelTask
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
from lilbee.runtime.hardware import FitChip, FitLevel
from tests._lilbee_app_test_host import LilbeeAppHost


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
    class _App(LilbeeAppHost):
        def compose(self) -> ComposeResult:
            yield CatalogDetailDrawer(id="catalog-detail-drawer")

    app = _App()
    return app, app


async def test_initial_state_shows_empty_hint() -> None:
    class _App(LilbeeAppHost):
        def compose(self) -> ComposeResult:
            yield CatalogDetailDrawer(id="catalog-detail-drawer")

    async with _App().run_test(size=(80, 20)) as pilot:
        await pilot.pause()
        drawer = pilot.app.query_one(CatalogDetailDrawer)
        name = drawer.query_one("#catalog-detail-name", Static)
        assert _EMPTY_HINT in str(name.render())


async def test_update_for_row_renders_local_row_name() -> None:
    class _App(LilbeeAppHost):
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
    class _App(LilbeeAppHost):
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
    class _App(LilbeeAppHost):
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
    class _App(LilbeeAppHost):
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


async def test_render_fit_pill_handles_tight_branch() -> None:
    from lilbee.cli.tui.widgets.catalog_detail import _render_fit_pill

    chip = FitChip(level=FitLevel.TIGHT, headroom_gb=0.5)
    rendered = _render_fit_pill(chip)
    assert "tight" in rendered.plain.lower()
    assert "+0.5 GB" in rendered.plain


async def test_render_fit_pill_handles_wont_run_branch() -> None:
    from lilbee.cli.tui.widgets.catalog_detail import _render_fit_pill

    chip = FitChip(level=FitLevel.WONT_RUN, headroom_gb=-2.0)
    rendered = _render_fit_pill(chip)
    assert "won't" in rendered.plain
    assert "-2.0 GB" in rendered.plain


async def test_render_sizes_block_marks_each_fit_level() -> None:
    """The drawer's Sizes block annotates each variant with ✓/⚠/✗ glyphs."""
    from lilbee.cli.tui.widgets.catalog_detail import _render_sizes_block

    variants = [
        SizeVariant(
            label="Q4_K_M",
            quant="Q4_K_M",
            size_gb=4.6,
            ref="r/q4",
            fit=FitChip(level=FitLevel.FITS, headroom_gb=8.0),
        ),
        SizeVariant(
            label="Q5_K_M",
            quant="Q5_K_M",
            size_gb=5.7,
            ref="r/q5",
            fit=FitChip(level=FitLevel.TIGHT, headroom_gb=0.5),
        ),
        SizeVariant(
            label="F16",
            quant="F16",
            size_gb=16.0,
            ref="r/f16",
            fit=FitChip(level=FitLevel.WONT_RUN, headroom_gb=-2.0),
        ),
    ]
    rendered = _render_sizes_block(variants)
    assert "✓" in rendered
    assert "⚠" in rendered
    assert "✗" in rendered


async def test_description_falls_back_to_family_when_no_catalog_model() -> None:
    """_description_text reads ModelFamily.description when catalog_model is absent."""
    from lilbee.catalog import ModelFamily, ModelVariant
    from lilbee.cli.tui.widgets.catalog_detail import _description_text

    family = ModelFamily(
        slug="qwen3",
        name="Qwen3",
        task="chat",
        description="Qwen3 long-context chat model.",
        variants=(
            ModelVariant(
                hf_repo="qwen/q",
                filename="m.gguf",
                param_count="0.6B",
                quant="Q4",
                size_mb=400,
                recommended=False,
            ),
        ),
    )
    row = _local_row("Qwen3 0.6B")
    row.family = family
    row.catalog_model = None
    assert "long-context" in _description_text(row)


async def test_description_uses_catalog_model_first() -> None:
    """_description_text prefers catalog_model.description when both fields set."""
    from lilbee.catalog import CatalogModel
    from lilbee.cli.tui.widgets.catalog_detail import _description_text

    cm = CatalogModel(
        hf_repo="meta/llama-3-8b",
        gguf_filename="m.gguf",
        size_gb=4.6,
        min_ram_gb=8.0,
        description="Meta Llama 3 chat model.",
        featured=True,
        downloads=12_300_000,
        task="chat",
    )
    row = _local_row("Llama 3 8B")
    row.catalog_model = cm
    assert _description_text(row).startswith("Meta Llama 3")


async def test_description_returns_empty_when_no_source_provides_one() -> None:
    from lilbee.cli.tui.widgets.catalog_detail import _description_text

    row = _local_row("Plain Row")
    row.catalog_model = None
    row.family = None
    assert _description_text(row) == ""


async def test_license_text_returns_empty_placeholder() -> None:
    """_license_text is a stable seam for future license plumbing; returns ''."""
    from lilbee.cli.tui.widgets.catalog_detail import _license_text

    assert _license_text(_local_row("any")) == ""


async def test_frontier_row_shows_provider_in_license_slot() -> None:
    class _App(LilbeeAppHost):
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
