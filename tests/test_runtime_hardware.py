"""Tests for ``lilbee.runtime.hardware``."""

from __future__ import annotations

from lilbee.catalog.models import ModelFamily, ModelVariant
from lilbee.runtime.hardware import (
    FitLevel,
    SizeVariantInfo,
    available_memory_for_fit,
    compute_fit,
    family_size_variants,
)

_GB = 1024**3


def test_fits_when_headroom_at_least_one_gb() -> None:
    chip = compute_fit(model_size_bytes=4 * _GB, available_bytes=8 * _GB)
    assert chip.level is FitLevel.FITS
    assert chip.headroom_gb == 4.0


def test_tight_when_headroom_under_one_gb() -> None:
    chip = compute_fit(model_size_bytes=int(7.5 * _GB), available_bytes=8 * _GB)
    assert chip.level is FitLevel.TIGHT
    assert 0 <= chip.headroom_gb < 1


def test_tight_at_exact_zero_headroom() -> None:
    chip = compute_fit(model_size_bytes=8 * _GB, available_bytes=8 * _GB)
    assert chip.level is FitLevel.TIGHT
    assert chip.headroom_gb == 0.0


def test_wont_run_when_oversized() -> None:
    chip = compute_fit(model_size_bytes=10 * _GB, available_bytes=8 * _GB)
    assert chip.level is FitLevel.WONT_RUN
    assert chip.headroom_gb == -2.0


def test_fits_at_exact_one_gb_boundary() -> None:
    chip = compute_fit(model_size_bytes=7 * _GB, available_bytes=8 * _GB)
    assert chip.level is FitLevel.FITS
    assert chip.headroom_gb == 1.0


def test_chip_is_immutable() -> None:
    import dataclasses

    chip = compute_fit(model_size_bytes=4 * _GB, available_bytes=8 * _GB)
    try:
        chip.level = FitLevel.WONT_RUN  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("FitChip should be frozen")


def test_available_memory_for_fit_sums_whole_fleet(monkeypatch) -> None:
    """The fit chip must ask for the whole-fleet total (total=True) so a model
    that tensor-splits across cards isn't wrongly marked 'won't run'."""
    import lilbee.providers.model_cache as mc
    from lilbee.core.config import cfg

    cfg.gpu_memory_fraction = 0.5
    captured: dict[str, object] = {}

    def fake(fraction: float, *, total: bool = False) -> int:
        captured["total"] = total
        # one 16 GB card vs four 16 GB cards summed
        return int((64 if total else 16) * _GB * fraction)

    monkeypatch.setattr(mc, "get_available_memory", fake)
    assert available_memory_for_fit() == 32 * _GB
    assert captured["total"] is True


def test_available_memory_for_fit_returns_none_when_probe_raises(monkeypatch) -> None:
    import lilbee.providers.model_cache as mc

    def boom(_fraction: float, *, total: bool = False) -> int:
        raise RuntimeError("psutil missing")

    monkeypatch.setattr(mc, "get_available_memory", boom)
    assert available_memory_for_fit() is None


def _variant(repo: str, params: str, quant: str, size_mb: int) -> ModelVariant:
    return ModelVariant(
        hf_repo=repo,
        filename="*.gguf",
        param_count=params,
        quant=quant,
        size_mb=size_mb,
        recommended=False,
    )


def test_family_size_variants_orders_by_size_and_builds_label() -> None:
    family = ModelFamily(
        slug="qwen3",
        name="Qwen3",
        task="chat",
        description="",
        variants=(
            _variant("Qwen/Qwen3-8B-GGUF", "8B", "Q4_K_M", 5 * 1024),
            _variant("Qwen/Qwen3-0.6B-GGUF", "0.6B", "Q4_K_M", 512),
        ),
    )
    out = family_size_variants(family)
    assert [v.params for v in out] == ["0.6B", "8B"]
    assert out[0] == SizeVariantInfo(
        size_label="0.6B Q4_K_M", params="0.6B", size_gb=0.5, ref="Qwen/Qwen3-0.6B-GGUF"
    )
    assert out[1].size_label == "8B Q4_K_M"


def test_family_size_variants_handles_missing_param_or_quant() -> None:
    family = ModelFamily(
        slug="anon",
        name="Anon",
        task="chat",
        description="",
        variants=(_variant("anon/repo", "", "", 100),),
    )
    [only] = family_size_variants(family)
    assert only.size_label == "--"
    assert only.params == ""
