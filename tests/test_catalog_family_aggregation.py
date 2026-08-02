"""Tests for family-as-card aggregation in _all_family_rows."""

from __future__ import annotations

from textual.app import ComposeResult

from lilbee.catalog import ModelFamily, ModelVariant
from lilbee.catalog.types import ModelCompat
from tests._lilbee_app_test_host import LilbeeAppHost


def _variant(
    quant: str,
    size_mb: int,
    *,
    hf_repo: str = "demo/qwen",
    compat: ModelCompat = ModelCompat.SUPPORTED,
) -> ModelVariant:
    return ModelVariant(
        hf_repo=hf_repo,
        filename=f"model-{quant}.gguf",
        param_count="8B",
        quant=quant,
        size_mb=size_mb,
        compat=compat,
    )


def _family(*variants: ModelVariant) -> ModelFamily:
    return ModelFamily(
        slug="qwen3",
        name="Qwen3",
        task="chat",
        description="",
        variants=variants,
    )


async def _build_screen_with_family(family: ModelFamily):
    """Spin up a CatalogScreen with a single seeded family.

    We need a real screen instance because _all_family_rows reads
    ``self._families`` and the row builder cache. Using App().run_test
    keeps the lifecycle clean even though we never interact with the
    DOM here.
    """
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    class _App(LilbeeAppHost):
        def compose(self) -> ComposeResult:
            yield CatalogScreen()

    app = _App()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.query_one(CatalogScreen)
        screen._families = [family]
        screen._family_rows_cache = None
        screen._data_version += 1
        return screen._all_family_rows()


async def test_family_with_multiple_variants_collapses_to_one_row() -> None:
    fam = _family(
        _variant("Q4_K_M", size_mb=4_600),
        _variant("Q5_K_M", size_mb=5_700),
        _variant("F16", size_mb=16_000),
    )
    rows = await _build_screen_with_family(fam)
    assert len(rows) == 1


async def test_family_row_carries_all_size_variants() -> None:
    fam = _family(
        _variant("Q4_K_M", size_mb=4_600),
        _variant("Q5_K_M", size_mb=5_700),
        _variant("F16", size_mb=16_000),
    )
    rows = await _build_screen_with_family(fam)
    assert {v.quant for v in rows[0].size_variants} == {"Q4_K_M", "Q5_K_M", "F16"}


async def test_smallest_variant_drives_primary_metadata() -> None:
    """The row shows the cheapest quant of the family, whatever else it holds."""
    fam = _family(
        _variant("Q4_K_M", size_mb=4_600),
        _variant("Q5_K_M", size_mb=5_700),
        _variant("F16", size_mb=16_000),
    )
    rows = await _build_screen_with_family(fam)
    assert rows[0].quant == "Q4_K_M"
    assert rows[0].variant is not None


async def test_empty_variant_family_is_skipped() -> None:
    """Families with no variants don't blow up; they just don't render."""
    fam = ModelFamily(slug="x", name="X", task="chat", description="", variants=())
    rows = await _build_screen_with_family(fam)
    assert rows == []


def test_family_row_carries_the_variants_compat() -> None:
    """Regression: family rows hardcoded SUPPORTED, so an architecture the
    engine cannot load rendered as installable and only failed at download."""
    from lilbee.catalog.types import ModelCompat
    from lilbee.cli.tui.screens.catalog_utils import variant_to_row

    fam = _family(_variant("Q4_K_M", size_mb=4_600, compat=ModelCompat.UNSUPPORTED))
    row = variant_to_row(fam.variants[0], fam, installed=False)
    assert row.compat is ModelCompat.UNSUPPORTED
