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
    # Grammatical, with a positive shortfall (not the bare "won't -2.0 GB").
    assert "won't run, short by 2.0 GB" in rendered.plain


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


async def test_a_bracketed_name_and_description_render_as_written() -> None:
    """Names and descriptions come from HuggingFace cards; ``[/x]`` must not parse."""
    from dataclasses import replace

    from lilbee.catalog.models import CatalogModel

    class _App(LilbeeAppHost):
        def compose(self) -> ComposeResult:
            yield CatalogDetailDrawer(id="catalog-detail-drawer")

    card = CatalogModel(
        hf_repo="acme/m",
        gguf_filename="m.gguf",
        size_gb=1.0,
        min_ram_gb=2.0,
        description="See [paper](x) and [/x] notes",
        featured=False,
        downloads=0,
        task="chat",
    )
    row = replace(_local_row("m[red]x"), catalog_model=card)
    async with _App().run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        drawer = pilot.app.query_one(CatalogDetailDrawer)
        drawer.update_for_row(row)
        await pilot.pause()
        name = str(drawer.query_one("#catalog-detail-name", Static).render())
        description = str(drawer.query_one("#catalog-detail-description", Static).render())
    assert "m[red]x" in name
    assert "See [paper](x) and [/x] notes" in description


async def test_a_bracketed_architecture_renders_as_written() -> None:
    """The compatibility sentence quotes the GGUF architecture; ``[/x]`` must not parse."""
    from dataclasses import replace

    from lilbee.catalog.models import CatalogModel
    from lilbee.catalog.types import ModelCompat

    class _App(LilbeeAppHost):
        def compose(self) -> ComposeResult:
            yield CatalogDetailDrawer(id="catalog-detail-drawer")

    card = CatalogModel(
        hf_repo="acme/m",
        gguf_filename="m.gguf",
        size_gb=1.0,
        min_ram_gb=2.0,
        description="",
        featured=False,
        downloads=0,
        task="chat",
        architecture="llama[/x]",
    )
    row = replace(_local_row("m"), catalog_model=card, compat=ModelCompat.UNSUPPORTED)
    async with _App().run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        drawer = pilot.app.query_one(CatalogDetailDrawer)
        drawer.update_for_row(row)
        await pilot.pause()
        compat = str(drawer.query_one("#catalog-detail-compat", Static).render())
    assert "Architecture llama[/x] is not in the supported set." in compat


async def test_a_bracketed_provider_renders_as_written() -> None:
    """A frontier row's provider label fills the license field; ``[red]`` must not restyle it."""
    from lilbee.catalog.types import KeyStatus
    from lilbee.cli.tui.screens.catalog_utils import FrontierCatalogRow

    class _App(LilbeeAppHost):
        def compose(self) -> ComposeResult:
            yield CatalogDetailDrawer(id="catalog-detail-drawer")

    row = FrontierCatalogRow(
        name="m",
        ref="acme/m",
        task=ModelTask.CHAT,
        provider="Ac[red]me",
        provider_id="acme",
        key_status=KeyStatus.READY,
    )
    async with _App().run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        drawer = pilot.app.query_one(CatalogDetailDrawer)
        drawer.update_for_row(row)
        await pilot.pause()
        license_text = str(drawer.query_one("#catalog-detail-license", Static).render())
    assert license_text == "Provider  Ac[red]me"
