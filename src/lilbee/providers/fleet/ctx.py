"""Context sizing for a chat model, from gguf-parser estimates.

:func:`fit_single_ctx` sizes one device against the budget the planner scaled;
:func:`fit_split_ctx` sizes a tensor split against the busiest card's headroom
rather than the summed pool. Both bisect the same quantized grid.
``engine_params.resolve_chat_fit`` is the entry point for the single-device fit
and holds the header-math fallback; ``planning.fit_chat_ctx`` searches the offload
around this window search. See docs/architecture.md (VRAM estimation).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from lilbee.core.config.enums import KvCacheType
from lilbee.providers.engine_params import chat_ctx_ceiling
from lilbee.providers.fleet.placement import _vram_proportional_split
from lilbee.providers.fleet.vram import estimate_instance_footprint, usable_vram_fraction
from lilbee.providers.model_cache import _DYNAMIC_CTX_FLOOR, _DYNAMIC_CTX_QUANTUM

# Extra VRAM held back on the busiest card on top of the gguf-parser estimate.
# In --split-mode layer, gguf-parser ignores --main-gpu and so under-models the
# compute graph + logits that real llama.cpp concentrates on the main device.
# Measured on 2x4090 (70B Q4_K_M, ctx 5888): the main device lands 0.26 GiB over
# its estimate, an order of magnitude inside the usable_vram_fraction already held
# back per card, so the reserve stays 0.
_MAIN_GPU_SKEW_RESERVE_BYTES = 0


def _largest_fitting_ctx(upper: int, fits: Callable[[int], bool]) -> int:
    """The largest quantized per-slot n_ctx at or below *upper* that *fits*.

    The grid is ``_DYNAMIC_CTX_FLOOR`` plus whole ``_DYNAMIC_CTX_QUANTUM`` steps,
    so every probe is a context the engine launches with cleanly. Returns the
    floor when even the floor overflows, which leaves the caller to refuse a
    window too small to serve.
    """
    if not fits(_DYNAMIC_CTX_FLOOR):
        return _DYNAMIC_CTX_FLOOR
    steps = max(0, (upper - _DYNAMIC_CTX_FLOOR) // _DYNAMIC_CTX_QUANTUM)
    lo, hi, best = 0, steps, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if fits(_DYNAMIC_CTX_FLOOR + mid * _DYNAMIC_CTX_QUANTUM):
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return _DYNAMIC_CTX_FLOOR + best * _DYNAMIC_CTX_QUANTUM


def fit_single_ctx(
    model_path: Path,
    *,
    meta: dict[str, str] | None,
    slots: int,
    available_bytes: int,
    gpu_layers: int,
    flash_attn: bool,
    kv_cache_type: KvCacheType,
    kv_cache_type_v: KvCacheType,
    unified: bool,
    ctx_ceiling: int,
    expert_offload: tuple[str, ...],
) -> int:
    """Largest quantized n_ctx whose gguf-parser estimate fits *available_bytes*.

    The estimator prices the cache each layer of this architecture actually
    holds, so a linear-attention, sliding-window or MLA model is granted the
    window it can serve rather than one budgeted for dense attention everywhere.

    *available_bytes* is the budget the caller already scaled
    (``planning.plan_sizing_budget``), so no further fraction applies here.
    *unified* charges the shared-memory figure, which is the whole resident
    footprint on a host whose GPU memory is the system's memory.
    *expert_offload* names the tensors the launch moves to system memory, so a
    mixture-of-experts model is not charged VRAM for experts it will not hold.
    """
    if available_bytes <= 0:
        return _DYNAMIC_CTX_FLOOR
    upper = min(chat_ctx_ceiling(meta, model_path), ctx_ceiling)

    def _fits(per_slot: int) -> bool:
        est = estimate_instance_footprint(
            model_path,
            ctx=per_slot,
            slots=slots,
            gpu_layers=gpu_layers,
            flash_attn=flash_attn,
            kv_cache_type=kv_cache_type,
            kv_cache_type_v=kv_cache_type_v,
            expert_offload=expert_offload,
        )
        return est.footprint(unified=unified) <= available_bytes

    return _largest_fitting_ctx(upper, _fits)


def fit_split_ctx(
    model_path: Path,
    *,
    meta: dict[str, str] | None,
    slots: int,
    ratio: tuple[int, ...],
    per_device_free_bytes: Sequence[int],
    gpu_layers: int,
    flash_attn: bool,
    kv_cache_type: KvCacheType,
    kv_cache_type_v: KvCacheType,
    ctx_ceiling: int,
    expert_offload: tuple[str, ...],
) -> int:
    """Largest quantized per-slot n_ctx that fits every card, capped at *ctx_ceiling*.

    Binary-searches the gguf-parser estimate at the launch tensor-split *ratio*:
    each probe passes a per-slot value, which the estimator charges across the
    slot count as the server does, and accepts it when every device's own share
    stays under that device's usable headroom. *ctx_ceiling* is the working context the
    caller planned for (``planning._placement_estimate_ctx``: a ``cfg.num_ctx`` pin,
    else ``cfg.chat_n_ctx_target``); the search never exceeds it, nor the model's
    trained context. Note the ceiling bounds the PER-SLOT window, not the total:
    placement reserves KV for one full window, while a split whose cards hold
    several may serve up to ``_CHAT_SLOTS`` of them. What keeps that honest is
    the per-device check below against real free bytes, not the placement
    reserve. Falls to the floor when even the floor overflows; plan_launches
    refuses a chat launch left at a window below the minimum grounded prompt.

    An empty *ratio* is the tight placement, which launches without one so the
    engine runs its own fit pass. It is also the estimator's only device-count
    signal, so size against a headroom-proportional one here: without it
    gguf-parser reports the whole model as a single card and nothing fits.
    """
    headrooms = [
        int(free * usable_vram_fraction()) - _MAIN_GPU_SKEW_RESERVE_BYTES
        for free in per_device_free_bytes
    ]
    if min(headrooms) <= 0:
        return _DYNAMIC_CTX_FLOOR
    if not ratio and len(per_device_free_bytes) > 1:
        by_position = {i: float(free) for i, free in enumerate(per_device_free_bytes)}
        ratio = _vram_proportional_split(list(by_position), by_position)
    # Bound the per-slot search by the planned working context, not just the model's
    # trained max: filling VRAM to that max OOM'd large tensor-split models under
    # load (a 235B took the full 262144-token ctx and crashed). The caller passes the
    # target placement sized its reserve against, so no single sequence exceeds the
    # plan; the total across slots can, and is held instead by the per-device
    # headroom test, which measures each card's real free bytes at launch.
    upper = min(chat_ctx_ceiling(meta, model_path), ctx_ceiling)

    def _peak_fits(per_slot: int) -> bool:
        est = estimate_instance_footprint(
            model_path,
            ctx=per_slot,
            slots=slots,
            gpu_layers=gpu_layers,
            flash_attn=flash_attn,
            kv_cache_type=kv_cache_type,
            kv_cache_type_v=kv_cache_type_v,
            tensor_split=ratio,
            expert_offload=expert_offload,
        )
        shares = est.per_device_vram
        if len(shares) != len(headrooms):
            # No usable per-device breakdown: fall back to peak vs the tightest card.
            return est.peak_footprint(unified=False) <= min(headrooms)
        return all(share <= room for share, room in zip(shares, headrooms, strict=True))

    return _largest_fitting_ctx(upper, _peak_fits)
