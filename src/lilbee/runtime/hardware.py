"""Hardware-fit signaling and per-row size-variant grouping for the catalog."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel

from lilbee.catalog.models import ModelFamily
from lilbee.core.config import cfg

_BYTES_PER_GB = 1024**3
_FITS_HEADROOM_BYTES = 1 * _BYTES_PER_GB


class FitLevel(StrEnum):
    FITS = "fits"
    TIGHT = "tight"
    WONT_RUN = "wont_run"


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


def available_memory_for_fit() -> int | None:
    """Bytes available to a model after ``cfg.gpu_memory_fraction``, or None on probe failure.

    Sums every GPU's memory (``total=True``) because lilbee tensor-splits a model
    too large for one card across the whole fleet; sizing the fit chip against a
    single card would wrongly mark a runnable split model "won't run". The actual
    per-card placement is decided precisely by the fleet planner at load time.

    Single entry point so the TUI and the HTTP catalog handler classify fit
    against the same number; otherwise the same model would chip differently in
    each surface.
    """
    try:
        from lilbee.providers.model_cache import get_available_memory

        budget = get_available_memory(cfg.gpu_memory_fraction, total=True)
    except Exception:
        return None
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
