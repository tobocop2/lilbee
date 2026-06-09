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
