"""Tests for fleet per-device context sizing (fit_split_ctx)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from lilbee.core.config.enums import KvCacheType
from lilbee.providers.fleet import ctx as ctx_mod
from lilbee.providers.fleet.ctx import fit_split_ctx
from lilbee.providers.fleet.vram import GgufVramEstimate
from lilbee.providers.model_cache import _DYNAMIC_CTX_FLOOR, _DYNAMIC_CTX_QUANTUM

_GB = 1024**3


def _peak_estimator(peak_for: Callable[[int], int]):
    """Fake estimate_instance_footprint whose per-device peak is a function of total ctx."""

    def fake(_model_path: Path, **kw: object) -> GgufVramEstimate:
        peak = peak_for(int(kw["ctx"]))  # type: ignore[arg-type]
        return GgufVramEstimate(
            vram_bytes=peak, ram_bytes=0, unified_bytes=0, per_device_vram=(peak,)
        )

    return fake


def _fit(model_path: Path, **overrides: object) -> int:
    kwargs: dict[str, object] = {
        "meta": {"arch": "x"},
        "slots": 4,
        "ratio": (1, 1, 1),
        "per_device_free_bytes": [80 * _GB, 80 * _GB, 80 * _GB],
        "gpu_layers": -1,
        "flash_attn": True,
        "kv_cache_type": KvCacheType.F16,
    }
    kwargs.update(overrides)
    return fit_split_ctx(model_path, **kwargs)  # type: ignore[arg-type]


class TestFitSplitCtx:
    def test_returns_floor_when_bottleneck_nonpositive(self, monkeypatch) -> None:
        # No usable headroom on the busiest card -> the floor, without estimating.
        monkeypatch.setattr(ctx_mod, "estimate_instance_footprint", _peak_estimator(lambda _c: 1))
        assert _fit(Path("/m.gguf"), per_device_free_bytes=[0, 0]) == _DYNAMIC_CTX_FLOOR

    def test_returns_floor_when_even_floor_overflows(self, monkeypatch) -> None:
        # The smallest context already exceeds the budget -> launch at the floor anyway.
        monkeypatch.setattr(ctx_mod, "chat_ctx_ceiling", lambda _m, _p: 131072)
        monkeypatch.setattr(
            ctx_mod, "estimate_instance_footprint", _peak_estimator(lambda _c: 999 * _GB)
        )
        assert _fit(Path("/m.gguf")) == _DYNAMIC_CTX_FLOOR

    def test_returns_quantized_ceiling_when_everything_fits(self, monkeypatch) -> None:
        # Every probe fits -> the largest grid point at or below the ceiling.
        monkeypatch.setattr(ctx_mod, "chat_ctx_ceiling", lambda _m, _p: 32768)
        monkeypatch.setattr(ctx_mod, "estimate_instance_footprint", _peak_estimator(lambda _c: 1))
        result = _fit(Path("/m.gguf"))
        assert (result - _DYNAMIC_CTX_FLOOR) % _DYNAMIC_CTX_QUANTUM == 0
        assert 32768 - _DYNAMIC_CTX_QUANTUM < result <= 32768

    def test_searches_largest_per_slot_within_bottleneck(self, monkeypatch) -> None:
        # peak == total ctx, so the per-slot value is gated at bottleneck / slots.
        bottleneck_free = 36000  # *0.9 usable -> 32400 budget; slots=4 -> per_slot <= 8100
        monkeypatch.setattr(ctx_mod, "chat_ctx_ceiling", lambda _m, _p: 131072)
        monkeypatch.setattr(ctx_mod, "estimate_instance_footprint", _peak_estimator(lambda c: c))
        result = _fit(
            Path("/m.gguf"), slots=4, per_device_free_bytes=[bottleneck_free, bottleneck_free]
        )
        budget = int(bottleneck_free * 0.9)
        assert (result - _DYNAMIC_CTX_FLOOR) % _DYNAMIC_CTX_QUANTUM == 0
        assert result * 4 <= budget
        assert (result + _DYNAMIC_CTX_QUANTUM) * 4 > budget

    def test_each_device_share_is_checked_against_its_own_headroom(self, monkeypatch) -> None:
        # Unequal cards with a matching unequal split: the big card's share must be
        # allowed to exceed the SMALL card's headroom, as long as each share fits
        # its own card. Comparing the max share to the min headroom would
        # under-size this fit.
        big_free, small_free = 40000, 10000

        def fake(_model_path: Path, **kw: object) -> GgufVramEstimate:
            total = int(kw["ctx"])  # type: ignore[arg-type]
            shares = (total * 4 // 5, total // 5)  # 4:1 split across the two cards
            return GgufVramEstimate(
                vram_bytes=total, ram_bytes=0, unified_bytes=0, per_device_vram=shares
            )

        monkeypatch.setattr(ctx_mod, "chat_ctx_ceiling", lambda _m, _p: 131072)
        monkeypatch.setattr(ctx_mod, "estimate_instance_footprint", fake)
        result = _fit(
            Path("/m.gguf"),
            slots=1,
            ratio=(4, 1),
            per_device_free_bytes=[big_free, small_free],
        )
        # Old peak-vs-min sizing would cap total ctx at min_headroom (9000) so the
        # per-slot result would sit below 9000; per-device sizing allows the big
        # card's 0.8 share to use its own 36000 headroom (total up to 45000).
        assert result > int(small_free * 0.9)
        # Invariant: the estimate at the returned ctx fits every device's headroom.
        accepted = fake(Path("/m.gguf"), ctx=result)
        headrooms = [int(big_free * 0.9), int(small_free * 0.9)]
        assert all(
            share <= room for share, room in zip(accepted.per_device_vram, headrooms, strict=True)
        )

    def test_falls_back_to_peak_when_breakdown_is_missing(self, monkeypatch) -> None:
        # An estimate with no usable per-device breakdown gates the peak against
        # the tightest card (the conservative direction).
        monkeypatch.setattr(ctx_mod, "chat_ctx_ceiling", lambda _m, _p: 131072)
        monkeypatch.setattr(ctx_mod, "estimate_instance_footprint", _peak_estimator(lambda c: c))
        result = _fit(Path("/m.gguf"), slots=1, per_device_free_bytes=[40000, 10000])
        assert result <= int(10000 * 0.9)
