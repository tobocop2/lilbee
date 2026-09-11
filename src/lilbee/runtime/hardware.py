"""Hardware-fit signaling and per-row size-variant grouping for the catalog."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from cachetools import TTLCache
from pydantic import BaseModel

from lilbee.catalog.models import CatalogModel, ModelFamily
from lilbee.core.config import cfg

_BYTES_PER_GB = 1024**3
_FITS_HEADROOM_BYTES = 1 * _BYTES_PER_GB

# Cache the budget keyed on gpu_memory_fraction so repeated catalog requests
# share one probe; the probe is an nvidia-smi subprocess without pynvml.
_available_memory_cache: TTLCache[float, int] = TTLCache[float, int](maxsize=8, ttl=60.0)


class FitLevel(StrEnum):
    FITS = "fits"
    TIGHT = "tight"
    WONT_RUN = "wont_run"


# Fit levels in rank order, best first.
FIT_RANK: dict[FitLevel, int] = {
    FitLevel.FITS: 0,
    FitLevel.TIGHT: 1,
    FitLevel.WONT_RUN: 2,
}


@dataclass(frozen=True)
class FitChip:
    level: FitLevel
    headroom_gb: float


def compute_fit(model_size_bytes: int, available_bytes: int) -> FitChip:
    """Classify how a model footprint fits the available memory budget.

    Headroom_gb is positive when the model fits and negative when it
    won't. The 1 GB band between FITS and TIGHT leaves room for the
    inference runtime, KV cache, and OS overhead beyond the raw weight
    file.
    """
    headroom_bytes = available_bytes - model_size_bytes
    headroom_gb = headroom_bytes / _BYTES_PER_GB
    if headroom_bytes >= _FITS_HEADROOM_BYTES:
        level = FitLevel.FITS
    elif headroom_bytes >= 0:
        level = FitLevel.TIGHT
    else:
        level = FitLevel.WONT_RUN
    return FitChip(level=level, headroom_gb=headroom_gb)


def fit_for_size(size_gb: float, available_bytes: int | None) -> FitLevel | None:
    """Fit level for a *size_gb* footprint, or None when it cannot be measured."""
    if available_bytes is None or size_gb <= 0:
        return None
    return compute_fit(int(size_gb * _BYTES_PER_GB), available_bytes).level


def make_fit_filter(
    worst: FitLevel | None, available_bytes: int | None
) -> Callable[[CatalogModel], bool] | None:
    """Row predicate for a *worst* acceptable fit, or None when no fit was asked for.

    A row whose fit cannot be measured is kept.
    """
    if worst is None:
        return None
    worst_rank = FIT_RANK[worst]

    def keep(model: CatalogModel) -> bool:
        level = fit_for_size(model.size_gb, available_bytes)
        return level is None or FIT_RANK[level] <= worst_rank

    return keep


def available_memory_for_fit() -> int | None:
    """Bytes available to a model after ``cfg.gpu_memory_fraction``, or None on probe failure.

    Sums every GPU's memory (``total=True``) because lilbee tensor-splits a model
    too large for one card across the whole fleet; sizing the fit chip against a
    single card would wrongly mark a runnable split model "won't run". The actual
    per-card placement is decided precisely by the fleet planner at load time.

    Single entry point so the TUI and the HTTP catalog handler classify fit
    against the same number; otherwise the same model would chip differently in
    each surface.

    Result is cached briefly keyed on gpu_memory_fraction: the underlying probe
    is expensive (an nvidia-smi subprocess with a 5s timeout when pynvml is
    absent) and the catalog stamps a fit chip on every page, so an uncached
    probe repeats that cost per request.
    """
    try:
        from lilbee.providers.model_cache import get_available_memory

        fraction = cfg.gpu_memory_fraction
        cached = _available_memory_cache.get(fraction)
        if cached is not None:
            return cached
        budget = get_available_memory(fraction, total=True)
    except Exception:
        return None
    _available_memory_cache[fraction] = budget
    return budget + _expert_offload_headroom()


def _expert_offload_headroom() -> int:
    """System memory the fit budget may borrow when expert offload is configured.

    A sparse model's experts live in system RAM under offload, so a host whose
    budget is discrete VRAM can run a model larger than that VRAM and must not
    be told otherwise. Zero unless the budget really is device memory: every
    other path (Apple unified memory, a non-NVIDIA or CPU-only host) already
    reports system RAM, and adding it twice would invent capacity. Zero too for a
    non-positive ``n_cpu_moe``, which offloads nothing. The chip is per-family and
    this budget is global, so it reads optimistically for a dense model pulled on
    an offload-enabled host (a sparse model gains the room, a dense one still
    fails to place); the planner sizes the real placement at load time.

    Scaled from installed RAM, not from what is free this instant, to match the
    capacity basis of the VRAM budget it is added to. Mixing the two made a
    catalog entry fit or not fit depending on whatever else the machine happened
    to be doing when the page was drawn, and shrank the budget exactly when
    another model was already resident.
    """
    from lilbee.providers.model_cache import has_nvidia_gpu, total_system_memory

    if not (cfg.cpu_moe or (cfg.n_cpu_moe is not None and cfg.n_cpu_moe >= 1)):
        return 0
    try:
        if not has_nvidia_gpu():
            return 0
        return int(total_system_memory() * cfg.gpu_memory_fraction)
    except Exception:
        return 0


class SizeVariantInfo(BaseModel):
    """One size/quant of a model family, serialised for HTTP responses."""

    size_label: str
    params: str
    size_gb: float
    ref: str


def family_size_variants(family: ModelFamily) -> list[SizeVariantInfo]:
    """Build the per-row size-variant strip for a featured ModelFamily, smallest first."""
    variants = sorted(family.variants, key=lambda v: v.size_mb)
    return [
        SizeVariantInfo(
            size_label=_size_variant_label(v.param_count, v.quant),
            params=v.param_count,
            size_gb=v.size_mb / 1024,
            ref=v.hf_repo,
        )
        for v in variants
    ]


def _size_variant_label(param_count: str, quant: str) -> str:
    """Render the compact label for one size variant (``8B Q4_K_M``)."""
    pieces = [p for p in (param_count, quant) if p]
    return " ".join(pieces) if pieces else "--"
