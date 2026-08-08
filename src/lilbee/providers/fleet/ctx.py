"""Per-device context sizing for a tensor-split chat model in the fleet.

Single-GPU chat ctx lives in ``engine_params.resolve_chat_ctx``; this is its
multi-GPU counterpart, sizing the per-slot context against the busiest card's
headroom rather than the summed pool. See docs/architecture.md (VRAM estimation).
"""

from __future__ import annotations

from collections.abc import Sequence
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
        )
        shares = est.per_device_vram
        if len(shares) != len(headrooms):
            # No usable per-device breakdown: fall back to peak vs the tightest card.
            return est.peak_footprint(unified=False) <= min(headrooms)
        return all(share <= room for share, room in zip(shares, headrooms, strict=True))

    if not _peak_fits(_DYNAMIC_CTX_FLOOR):
        return _DYNAMIC_CTX_FLOOR
    steps = max(0, (upper - _DYNAMIC_CTX_FLOOR) // _DYNAMIC_CTX_QUANTUM)
    lo, hi, best = 0, steps, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if _peak_fits(_DYNAMIC_CTX_FLOOR + mid * _DYNAMIC_CTX_QUANTUM):
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return _DYNAMIC_CTX_FLOOR + best * _DYNAMIC_CTX_QUANTUM
