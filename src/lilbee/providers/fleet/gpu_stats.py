"""Live per-GPU activity stats for the placement view.

devices.py enumerates GPUs once (structural: index, name, total VRAM). This reads
the moving numbers, compute utilization and current free memory, so a client can
animate a per-card load bar.

Vendor-specific probing lives in gpu_backends/; this module groups devices by
backend, dispatches to resolve_backend(), and merges live samples into GpuStat
entries. Adding a new vendor means one file in gpu_backends/ and one registry
line; this file is not touched.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from lilbee.providers.fleet.gpu_backends import (
    IntelUtilHint,
    UtilSample,
    resolve_backend,
    util_backend_name,
)
from lilbee.providers.fleet.gpu_backends import (
    intel_util_hint as _detect_intel_util_hint,
)


class DeviceLike(Protocol):
    """Structural view of a probed GPU (FleetDevice or app-layer GpuInfo)."""

    @property
    def index(self) -> int: ...
    @property
    def backend(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def total_bytes(self) -> int: ...
    @property
    def free_bytes(self) -> int: ...


@dataclass(frozen=True)
class GpuStat:
    """A live snapshot of one GPU, keyed to its structural index."""

    index: int
    utilization_pct: int | None
    free_bytes: int
    total_bytes: int
    temperature_c: int | None = None


def _safe_sample(
    backend_name: str,
    indices: frozenset[int],
) -> dict[int, UtilSample]:
    """Dispatch to the vendor backend; return {} on any failure."""
    backend = resolve_backend(backend_name)
    if backend is None:
        return {}
    try:
        return backend.sample(indices)
    except Exception:  # backends must never crash the sampler
        return {}


# One in-flight probe per device set, shared by every concurrent caller. The
# probe shells out to a vendor SMI tool (five-second timeout) and the Intel paths
# sleep and scan /proc on top, so N open placement views would otherwise mean N
# concurrent subprocesses every tick against the same hardware. A sample is
# reused for slightly under one tick, which is fresh enough for a live view and
# turns the per-client cost back into a per-machine one.
_SHARED_SAMPLE_TTL_S = 0.9
_shared_lock = threading.Lock()
_shared_sample: dict[tuple[int, ...], tuple[float, dict[int, GpuStat]]] = {}


def probe_gpu_stats_shared(devices: Sequence[DeviceLike]) -> dict[int, GpuStat]:
    """``probe_gpu_stats``, coalesced across concurrent callers.

    Callers arriving while a sample is fresh reuse it; the rest serialise on the
    lock so exactly one probe runs per device set per interval, rather than one
    per subscriber. Kept separate from ``probe_gpu_stats`` so one-shot callers
    still get an uncached reading.
    """
    key = tuple(sorted(d.index for d in devices))
    with _shared_lock:
        cached = _shared_sample.get(key)
        if cached is not None and time.monotonic() - cached[0] < _SHARED_SAMPLE_TTL_S:
            return cached[1]
        stats = probe_gpu_stats(devices)
        _shared_sample[key] = (time.monotonic(), stats)
        return stats


# Which visible-devices variable masks each util backend's index space. The
# engine numbers its devices after the mask, the SMI tools before it, so under
# CUDA_VISIBLE_DEVICES=2,3 the fleet holds devices 0 and 1 while nvidia-smi keeps
# reporting 0..3. Merging those two spaces by ordinal attributes another
# tenant's cards' utilization, temperature and free memory to this fleet.
_BACKEND_VISIBLE_VARS: dict[str, tuple[str, ...]] = {
    "CUDA": ("CUDA_VISIBLE_DEVICES",),
    # Innermost mask first. ROCr filters, then HIP re-indexes within the
    # survivors, so walking outward from the fleet's index means undoing HIP
    # before ROCr.
    "ROCm": ("HIP_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "ROCR_VISIBLE_DEVICES"),
    "HIP": ("HIP_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "ROCR_VISIBLE_DEVICES"),
}


def _physical_index(backend_name: str, fleet_index: int) -> int | None:
    """The index the vendor's SMI tool uses for the fleet's device *fleet_index*.

    Resolved by walking the visible-devices masks outward from the fleet's own
    index, undoing each one in the reverse of the order the runtime applied it.
    ``None`` when a mask cannot be undone, which is the whole point of the
    return type.

    A mask naming devices by UUID or MIG instance cannot be inverted by
    arithmetic, and neither can an index the mask does not reach. Returning the
    fleet index there was a guess, and on a masked host it is reliably the wrong
    card: another tenant's utilization, temperature and free VRAM then drive both
    the GPU panel and the adaptive ingest throttle. No sample is the honest
    answer, and the caller already has a device-reported baseline to fall back on.

    An unset mask, or a backend with no mask, leaves the index alone: that is
    every unmasked host. Intel is absent because ONEAPI_DEVICE_SELECTOR is a
    selector grammar rather than an index list, and xpu-smi is called with the
    fleet index directly.
    """
    index = fleet_index
    for var in _BACKEND_VISIBLE_VARS.get(backend_name, ()):
        raw = os.environ.get(var)
        if not raw:
            continue
        entries = [e.strip() for e in raw.split(",")]
        if not all(e.isdigit() for e in entries) or index >= len(entries):
            return None
        index = int(entries[index])
    return index


def probe_gpu_stats(devices: Sequence[DeviceLike]) -> dict[int, GpuStat]:
    """Live stats keyed by device index. Empty when no devices are given.

    Groups devices by vendor backend, dispatches once per group, and merges
    live util/temp samples into GpuStat entries. Any index the backend can't
    cover falls back to structural VRAM with util=None and temperature_c=None.
    """
    # Structural fallbacks: util=None, temp=None, VRAM from the probe.
    stats: dict[int, GpuStat] = {
        d.index: GpuStat(d.index, None, d.free_bytes, d.total_bytes) for d in devices
    }

    # Group by the util backend, not the raw inference backend: a Vulkan-exposed
    # consumer GPU routes to its vendor's util source by name.
    by_backend: dict[str, list[DeviceLike]] = {}
    for d in devices:
        by_backend.setdefault(util_backend_name(d.backend, d.name), []).append(d)

    for backend_name, group in by_backend.items():
        # Ask the SMI tool about its own indices, and merge its answers back into
        # the engine's; the two spaces differ whenever a visible-devices mask is set.
        # A device whose mask cannot be inverted is left out entirely rather than
        # merged under a guessed index: it keeps the structural reading the probe
        # already gave it, which is right, instead of another card's live one.
        fleet_by_physical = {
            physical: d.index
            for d in group
            if (physical := _physical_index(backend_name, d.index)) is not None
        }
        for physical, sample in _safe_sample(backend_name, frozenset(fleet_by_physical)).items():
            index = fleet_by_physical.get(physical)
            if index is None or index not in stats:
                continue
            base = stats[index]
            # Keep structural VRAM when the backend returns the 0/0 sentinel
            # (amd-smi metric mode and xpu-smi stats both omit total VRAM); a
            # backend that does report memory takes precedence.
            free = sample.free_bytes if sample.free_bytes or sample.total_bytes else base.free_bytes
            total = sample.total_bytes if sample.total_bytes else base.total_bytes
            stats[index] = GpuStat(
                index=index,
                utilization_pct=sample.utilization_pct,
                free_bytes=free,
                total_bytes=total,
                temperature_c=sample.temperature_c,
            )

    return {i: stats[i] for i in sorted(stats)}


def intel_util_hint(
    devices: Sequence[DeviceLike], stats: dict[int, GpuStat]
) -> IntelUtilHint | None:
    """The fix that would unblock an Intel GPU's missing util reading, else None.

    Fires only when an Intel device's util is actually missing; a surface turns
    the hint into a localized message (grant when intel_gpu_top is installed but
    blocked, install when it is absent, e.g. kernels too old for fdinfo).
    """
    for d in devices:
        if util_backend_name(d.backend, d.name) != "SYCL":
            continue
        stat = stats.get(d.index)
        if stat is not None and stat.utilization_pct is None:
            return _detect_intel_util_hint()
    return None


def probe_intel_util_hint(devices: Sequence[DeviceLike]) -> IntelUtilHint | None:
    """Probe live stats and evaluate the Intel util hint in one call."""
    return intel_util_hint(devices, probe_gpu_stats(devices))
