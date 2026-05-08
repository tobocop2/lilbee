"""Hardware-fit signaling for the catalog UI.

`compute_fit` reports whether a model's footprint fits the host's
available memory and how much headroom or shortfall remains. The
catalog renders this as a per-card chip (`fits +N GB`, `tight +N GB`,
`won't run -N GB`) so users can tell at a glance whether a model
will run before they download it.

Hardware probing lives in `lilbee.providers.model_cache.get_available_memory`
(macOS unified memory, NVIDIA GPU, system RAM fallback). This module
only owns the fit semantics so it stays free of provider/runtime
layering concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

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

    `headroom_gb` is positive when the model fits and negative when it
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
