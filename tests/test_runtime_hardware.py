"""Tests for ``lilbee.runtime.hardware``."""

from __future__ import annotations

from conftest import make_test_catalog_model
from lilbee.catalog.models import ModelFamily, ModelVariant
from lilbee.runtime.hardware import (
    FitLevel,
    SizeVariantInfo,
    available_memory_for_fit,
    compute_fit,
    family_size_variants,
    fit_for_size,
    make_fit_filter,
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


def test_fit_for_size_classifies_a_footprint_against_the_budget() -> None:
    assert fit_for_size(4.0, 8 * _GB) is FitLevel.FITS
    assert fit_for_size(7.5, 8 * _GB) is FitLevel.TIGHT
    assert fit_for_size(10.0, 8 * _GB) is FitLevel.WONT_RUN


def test_fit_for_size_is_unknown_without_a_budget() -> None:
    assert fit_for_size(4.0, None) is None


def test_fit_for_size_is_unknown_without_a_size() -> None:
    assert fit_for_size(0.0, 8 * _GB) is None


def test_make_fit_filter_returns_no_predicate_without_a_threshold() -> None:
    assert make_fit_filter(None, 8 * _GB) is None


def test_fit_filter_keeps_rows_no_worse_than_the_threshold() -> None:
    keep = make_fit_filter(FitLevel.TIGHT, 8 * _GB)
    assert keep is not None
    assert keep(make_test_catalog_model(size_gb=4.0)) is True
    assert keep(make_test_catalog_model(size_gb=7.5)) is True
    assert keep(make_test_catalog_model(size_gb=10.0)) is False


def test_fit_filter_at_fits_drops_a_tight_row() -> None:
    keep = make_fit_filter(FitLevel.FITS, 8 * _GB)
    assert keep is not None
    assert keep(make_test_catalog_model(size_gb=7.5)) is False


def test_fit_filter_keeps_a_row_whose_size_is_unknown() -> None:
    """An unmeasurable row is kept: the host cannot prove it will not run."""
    keep = make_fit_filter(FitLevel.FITS, 8 * _GB)
    assert keep is not None
    assert keep(make_test_catalog_model(size_gb=0.0)) is True


def test_fit_filter_keeps_every_row_when_the_memory_probe_failed() -> None:
    keep = make_fit_filter(FitLevel.FITS, None)
    assert keep is not None
    assert keep(make_test_catalog_model(size_gb=500.0)) is True


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


def test_expert_offload_headroom_is_capacity_not_free_right_now(monkeypatch) -> None:
    """The fit budget it joins is capacity-based; mixing bases makes it drift.

    Sizing from what is free this instant made a catalog entry fit or not fit
    depending on whatever else the machine was doing, and shrank the budget
    exactly when another model was already resident.
    """
    from lilbee.core.config import cfg
    from lilbee.providers import model_cache
    from lilbee.runtime import hardware

    monkeypatch.setattr(cfg, "cpu_moe", True, raising=False)
    monkeypatch.setattr(cfg, "n_cpu_moe", None, raising=False)
    monkeypatch.setattr(cfg, "gpu_memory_fraction", 0.5, raising=False)
    monkeypatch.setattr(model_cache, "has_nvidia_gpu", lambda: True)
    monkeypatch.setattr(model_cache, "total_system_memory", lambda: 64 * 10**9)
    # A machine busy right now must not shrink a capacity-based budget.
    monkeypatch.setattr(model_cache, "free_system_memory", lambda: 1 * 10**9)

    assert hardware._expert_offload_headroom() == 32 * 10**9


def _count_probes(monkeypatch) -> dict[str, int]:
    """Replace the memory probe with a counter returning 64 GB scaled by the fraction."""
    import lilbee.providers.model_cache as mc

    calls = {"n": 0}

    def fake(fraction: float, *, total: bool = False) -> int:
        calls["n"] += 1
        return int(64 * _GB * fraction)

    monkeypatch.setattr(mc, "get_available_memory", fake)
    return calls


def test_available_memory_for_fit_caches_within_ttl(monkeypatch) -> None:
    """Repeated calls within the TTL share one probe and add the headroom every time."""
    from lilbee.core.config import cfg
    from lilbee.runtime import hardware

    monkeypatch.setattr(cfg, "gpu_memory_fraction", 0.5)
    monkeypatch.setattr(hardware, "_expert_offload_headroom", lambda: 3 * _GB)
    calls = _count_probes(monkeypatch)

    first = available_memory_for_fit()
    second = available_memory_for_fit()

    assert first == second == 32 * _GB + 3 * _GB
    assert calls["n"] == 1


def test_available_memory_for_fit_invalidates_on_fraction_change(monkeypatch) -> None:
    """A config change invalidates the cache and re-probes."""
    from lilbee.core.config import cfg

    monkeypatch.setattr(cfg, "gpu_memory_fraction", 0.5)
    calls = _count_probes(monkeypatch)

    available_memory_for_fit()
    monkeypatch.setattr(cfg, "gpu_memory_fraction", 0.8)
    available_memory_for_fit()

    assert calls["n"] == 2


def test_available_memory_for_fit_does_not_cache_a_failed_probe(monkeypatch) -> None:
    """A probe failure reports None and the next call probes again."""
    import lilbee.providers.model_cache as mc

    calls = {"n": 0}

    def failing(fraction: float, *, total: bool = False) -> int:
        calls["n"] += 1
        raise RuntimeError("nvidia-smi timed out")

    monkeypatch.setattr(mc, "get_available_memory", failing)

    assert available_memory_for_fit() is None
    assert available_memory_for_fit() is None
    assert calls["n"] == 2
