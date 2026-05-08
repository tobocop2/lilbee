"""Tests for ``family_to_size_variants`` and ``LocalCatalogRow.size_variants``."""

from __future__ import annotations

from lilbee.catalog import ModelFamily, ModelVariant
from lilbee.cli.tui.screens.catalog_utils import (
    LocalCatalogRow,
    SizeVariant,
    family_to_size_variants,
)


def _variant(quant: str, size_mb: int, *, hf_repo: str = "demo/repo") -> ModelVariant:
    return ModelVariant(
        hf_repo=hf_repo,
        filename=f"model-{quant}.gguf",
        param_count="8B",
        quant=quant,
        size_mb=size_mb,
        recommended=False,
    )


def _family(*variants: ModelVariant) -> ModelFamily:
    return ModelFamily(
        slug="qwen3",
        name="Qwen3",
        task="chat",
        description="",
        variants=variants,
    )


def test_size_variants_sorted_ascending_by_size() -> None:
    family = _family(
        _variant("F16", size_mb=16_000),
        _variant("Q4_K_M", size_mb=4_600),
        _variant("Q5_K_M", size_mb=5_700),
    )

    variants = family_to_size_variants(family)

    assert [v.quant for v in variants] == ["Q4_K_M", "Q5_K_M", "F16"]
    assert [round(v.size_gb, 2) for v in variants] == [
        round(4600 / 1024, 2),
        round(5700 / 1024, 2),
        round(16000 / 1024, 2),
    ]


def test_size_variant_label_includes_params_and_quant() -> None:
    family = _family(_variant("Q4_K_M", size_mb=4_600))

    variants = family_to_size_variants(family)

    assert variants[0].label == "8B Q4_K_M"


def test_size_variants_are_immutable() -> None:
    import dataclasses

    family = _family(_variant("Q4_K_M", size_mb=4_600))
    variant = family_to_size_variants(family)[0]
    try:
        variant.label = "changed"  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("SizeVariant should be frozen")


def test_size_variants_carry_hf_repo_as_ref() -> None:
    family = _family(_variant("Q4_K_M", size_mb=4_600, hf_repo="meta/llama-3.1-8b-gguf"))

    variants = family_to_size_variants(family)

    assert variants[0].ref == "meta/llama-3.1-8b-gguf"


def test_size_variants_default_fit_is_none() -> None:
    family = _family(_variant("Q4_K_M", size_mb=4_600))

    variants = family_to_size_variants(family)

    assert variants[0].fit is None


def test_local_catalog_row_default_size_variants_empty() -> None:
    row = LocalCatalogRow(
        name="Qwen3 8B",
        task="chat",
        params="8B",
        size="4.5 GB",
        quant="Q4_K_M",
        downloads="1.2M",
        featured=False,
        installed=False,
        sort_downloads=1_200_000,
        sort_size=4.5,
    )

    assert row.size_variants == []
    assert row.fit is None


def test_local_catalog_row_accepts_size_variants_and_fit() -> None:
    from lilbee.runtime.hardware import FitChip, FitLevel

    chip = FitChip(level=FitLevel.FITS, headroom_gb=8.0)
    variants = [
        SizeVariant(label="8B Q4_K_M", quant="Q4_K_M", size_gb=4.6, ref="r/q4"),
        SizeVariant(label="8B F16", quant="F16", size_gb=16.0, ref="r/f16"),
    ]

    row = LocalCatalogRow(
        name="Llama 3.1 8B",
        task="chat",
        params="8B",
        size="4.6 GB",
        quant="Q4_K_M",
        downloads="12.3M",
        featured=True,
        installed=True,
        sort_downloads=12_300_000,
        sort_size=4.6,
        size_variants=variants,
        fit=chip,
    )

    assert row.size_variants == variants
    assert row.fit is chip
