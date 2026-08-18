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
    """Fake estimate_instance_footprint whose per-device peak is a function of total ctx.

    Takes the per-slot context and multiplies, as the real estimator does.
    """

    def fake(_model_path: Path, **kw: object) -> GgufVramEstimate:
        peak = peak_for(int(kw["ctx"]) * int(kw["slots"]))  # type: ignore[arg-type]
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
        "kv_cache_type_v": KvCacheType.F16,
        # Large by default so the fit-logic tests are gated by the monkeypatched
        # chat_ctx_ceiling; the ceiling-cap test lowers it.
        "ctx_ceiling": 1_000_000,
        "expert_offload": (),
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

    def test_caps_at_ctx_ceiling_below_the_model_max(self, monkeypatch) -> None:
        # Even when the model + cards could hold the full trained context, the split
        # never exceeds the caller's planned working context (bb-ev9): a 235B whose
        # trained ceiling is 262144, planned for a 24576 ctx_ceiling, is capped there
        # so it can't over-commit VRAM and OOM under load.
        monkeypatch.setattr(ctx_mod, "chat_ctx_ceiling", lambda _m, _p: 262144)
        monkeypatch.setattr(ctx_mod, "estimate_instance_footprint", _peak_estimator(lambda _c: 1))
        result = _fit(Path("/m.gguf"), ctx_ceiling=24576)
        assert 24576 - _DYNAMIC_CTX_QUANTUM < result <= 24576

    def test_falls_back_to_peak_when_breakdown_is_missing(self, monkeypatch) -> None:
        # An estimate with no usable per-device breakdown gates the peak against
        # the tightest card (the conservative direction).
        monkeypatch.setattr(ctx_mod, "chat_ctx_ceiling", lambda _m, _p: 131072)
        monkeypatch.setattr(ctx_mod, "estimate_instance_footprint", _peak_estimator(lambda c: c))
        result = _fit(Path("/m.gguf"), slots=1, per_device_free_bytes=[40000, 10000])
        assert result <= int(10000 * 0.9)

    def test_a_tight_group_is_sized_per_device_not_as_one_card(self, monkeypatch) -> None:
        # A tight placement launches with no ratio so the engine runs its own fit
        # pass, and the estimator reads the ratio as its device count. Passing the
        # emptiness on made it size the whole model as a single card, which no
        # card's headroom holds, so every context down to the floor was rejected
        # and a 70B on two 24 GiB cards was refused a window it measurably serves.
        seen: list[tuple[int, ...]] = []

        def fake(_model_path: Path, **kw: object) -> GgufVramEstimate:
            ratio: tuple[int, ...] = kw["tensor_split"]  # type: ignore[assignment]
            seen.append(ratio)
            # Weights split across the cards, plus KV proportional to the context.
            per_device = tuple(20 * _GB + int(kw["ctx"]) * int(kw["slots"]) for _ in ratio)
            return GgufVramEstimate(
                vram_bytes=sum(per_device),
                ram_bytes=0,
                unified_bytes=0,
                per_device_vram=per_device,
            )

        monkeypatch.setattr(ctx_mod, "chat_ctx_ceiling", lambda _m, _p: 32768)
        monkeypatch.setattr(ctx_mod, "estimate_instance_footprint", fake)
        result = _fit(
            Path("/m.gguf"), slots=1, ratio=(), per_device_free_bytes=[24 * _GB, 24 * _GB]
        )
        assert seen and all(len(r) == 2 for r in seen)
        assert result > _DYNAMIC_CTX_FLOOR

    def test_a_single_card_keeps_its_empty_ratio(self, monkeypatch) -> None:
        # One card is not a split; deriving a ratio there would make the estimator
        # report a breakdown the launch never asks for.
        seen: list[tuple[int, ...]] = []

        def fake(_model_path: Path, **kw: object) -> GgufVramEstimate:
            seen.append(kw["tensor_split"])  # type: ignore[arg-type]
            return GgufVramEstimate(
                vram_bytes=1, ram_bytes=0, unified_bytes=0, per_device_vram=(1,)
            )

        monkeypatch.setattr(ctx_mod, "chat_ctx_ceiling", lambda _m, _p: 8192)
        monkeypatch.setattr(ctx_mod, "estimate_instance_footprint", fake)
        _fit(Path("/m.gguf"), slots=1, ratio=(), per_device_free_bytes=[24 * _GB])
        assert seen and all(r == () for r in seen)


def _single_estimator(vram_for, unified_for=None, seen=None):
    seen = [] if seen is None else seen
    """Fake estimate_instance_footprint for the single-device fit.

    *vram_for* maps total ctx (per-slot x slots) to discrete-GPU bytes;
    *unified_for* does the same for the unified figure, defaulting to zero so a
    test that charges VRAM cannot pass by reading the wrong field. Every probe's
    kwargs land in *seen*.
    """

    def fake(_model_path: Path, **kw: object) -> GgufVramEstimate:
        seen.append(kw)
        total = int(kw["ctx"]) * int(kw["slots"])  # type: ignore[arg-type]
        return GgufVramEstimate(
            vram_bytes=vram_for(total),
            ram_bytes=0,
            unified_bytes=unified_for(total) if unified_for else 0,
        )

    return fake


def _fit_single(model_path: Path, **overrides: object) -> int:
    kwargs: dict[str, object] = {
        "meta": {"arch": "x"},
        "slots": 1,
        "available_bytes": 40 * _GB,
        "gpu_layers": -1,
        "flash_attn": True,
        "kv_cache_type": KvCacheType.F16,
        "kv_cache_type_v": KvCacheType.F16,
        "unified": False,
        "ctx_ceiling": 1_000_000,
        "expert_offload": (),
    }
    kwargs.update(overrides)
    return ctx_mod.fit_single_ctx(model_path, **kwargs)  # type: ignore[arg-type]


class TestFitSingleCtx:
    def test_returns_floor_when_there_is_no_budget(self, monkeypatch) -> None:
        monkeypatch.setattr(ctx_mod, "estimate_instance_footprint", _single_estimator(lambda _c: 1))
        assert _fit_single(Path("/m.gguf"), available_bytes=0) == _DYNAMIC_CTX_FLOOR

    def test_returns_floor_when_even_the_floor_overflows(self, monkeypatch) -> None:
        monkeypatch.setattr(ctx_mod, "chat_ctx_ceiling", lambda _m, _p: 131072)
        monkeypatch.setattr(
            ctx_mod, "estimate_instance_footprint", _single_estimator(lambda _c: 999 * _GB)
        )
        assert _fit_single(Path("/m.gguf")) == _DYNAMIC_CTX_FLOOR

    def test_returns_the_quantized_ceiling_when_everything_fits(self, monkeypatch) -> None:
        monkeypatch.setattr(ctx_mod, "chat_ctx_ceiling", lambda _m, _p: 32768)
        monkeypatch.setattr(ctx_mod, "estimate_instance_footprint", _single_estimator(lambda _c: 1))
        result = _fit_single(Path("/m.gguf"))
        assert (result - _DYNAMIC_CTX_FLOOR) % _DYNAMIC_CTX_QUANTUM == 0
        assert 32768 - _DYNAMIC_CTX_QUANTUM < result <= 32768

    def test_bisects_to_the_largest_window_the_budget_backs(self, monkeypatch) -> None:
        # 1 GiB of weights plus 1 MiB per token: 20 GiB backs just under 19456 tokens.
        monkeypatch.setattr(ctx_mod, "chat_ctx_ceiling", lambda _m, _p: 131072)
        monkeypatch.setattr(
            ctx_mod,
            "estimate_instance_footprint",
            _single_estimator(lambda c: _GB + c * 1024**2),
        )
        result = _fit_single(Path("/m.gguf"), available_bytes=20 * _GB)
        assert result == 19456

    def test_a_unified_host_is_charged_its_unified_footprint(self, monkeypatch) -> None:
        # VRAM reads as free all the way up; only the unified figure bounds the fit,
        # so a fit that read the wrong field would return the ceiling.
        monkeypatch.setattr(ctx_mod, "chat_ctx_ceiling", lambda _m, _p: 131072)
        monkeypatch.setattr(
            ctx_mod,
            "estimate_instance_footprint",
            _single_estimator(lambda _c: 0, unified_for=lambda c: _GB + c * 1024**2),
        )
        result = _fit_single(Path("/m.gguf"), available_bytes=20 * _GB, unified=True)
        assert result == 19456


class TestTheFitChargesWhatTheLaunchRuns:
    """A probe that prices tensors the launch will not hold is not the launch."""

    def test_the_single_fit_passes_the_expert_offload_through(self, monkeypatch) -> None:
        # cpu_moe moves the experts to system memory. Charging them to VRAM
        # shrinks the window for exactly the giant MoE models that need it.
        seen: list[dict] = []
        monkeypatch.setattr(ctx_mod, "chat_ctx_ceiling", lambda _m, _p: 4096)
        monkeypatch.setattr(
            ctx_mod, "estimate_instance_footprint", _single_estimator(lambda _c: 1, seen=seen)
        )
        _fit_single(Path("/m.gguf"), expert_offload=("blk.*ffn.*exps.*=CPU",))
        assert seen, "the fit never called the estimator"
        assert all(kw["expert_offload"] == ("blk.*ffn.*exps.*=CPU",) for kw in seen)

    def test_the_split_fit_passes_the_expert_offload_through(self, monkeypatch) -> None:
        seen: list[dict] = []

        def fake(_model_path: Path, **kw: object) -> GgufVramEstimate:
            seen.append(kw)
            return GgufVramEstimate(
                vram_bytes=1, ram_bytes=0, unified_bytes=0, per_device_vram=(1,)
            )

        monkeypatch.setattr(ctx_mod, "chat_ctx_ceiling", lambda _m, _p: 4096)
        monkeypatch.setattr(ctx_mod, "estimate_instance_footprint", fake)
        _fit(Path("/m.gguf"), expert_offload=("blk.*ffn.*exps.*=CPU",))
        assert seen, "the fit never called the estimator"
        assert all(kw["expert_offload"] == ("blk.*ffn.*exps.*=CPU",) for kw in seen)
