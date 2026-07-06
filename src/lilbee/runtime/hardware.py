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

        return get_available_memory(cfg.gpu_memory_fraction, total=True)
    except Exception:
        return None


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
