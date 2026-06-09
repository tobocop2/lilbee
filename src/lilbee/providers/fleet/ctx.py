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
from lilbee.providers.fleet.vram import USABLE_VRAM_FRACTION, estimate_instance_footprint
from lilbee.providers.model_cache import _DYNAMIC_CTX_FLOOR, _DYNAMIC_CTX_QUANTUM

# Extra VRAM held back on the busiest card on top of the gguf-parser estimate.
# In --split-mode layer, gguf-parser ignores --main-gpu and so under-models the
# compute graph + logits that real llama.cpp concentrates on the main device.
# Default 0 (trust the estimate); raise once measured on a multi-GPU pod (bb-ds8).
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
) -> int:
    """Largest quantized per-slot n_ctx whose per-device peak fits the busiest card.

    Binary-searches the gguf-parser estimate at the launch tensor-split *ratio*:
    the server serves ``--ctx-size = per_slot x slots``, so each probe estimates
    that total and accepts the per-slot value when the peak device's footprint
    stays under its usable headroom. Falls to the floor when even that overflows.
    """
    bottleneck = (
        min(int(free * USABLE_VRAM_FRACTION) for free in per_device_free_bytes)
        - _MAIN_GPU_SKEW_RESERVE_BYTES
    )
    if bottleneck <= 0:
        return _DYNAMIC_CTX_FLOOR
    upper = chat_ctx_ceiling(meta, model_path)

    def _peak_fits(per_slot: int) -> bool:
        est = estimate_instance_footprint(
            model_path,
            ctx=per_slot * slots,
            slots=slots,
            gpu_layers=gpu_layers,
            flash_attn=flash_attn,
            kv_cache_type=kv_cache_type,
            tensor_split=ratio,
        )
        return est.peak_footprint(unified=False) <= bottleneck

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
