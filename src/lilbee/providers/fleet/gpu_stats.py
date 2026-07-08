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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from lilbee.providers.fleet.gpu_backends import (
    UtilSample,
    intel_gpu_top_grant_binary,
    resolve_backend,
    util_backend_name,
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
        indices = frozenset(d.index for d in group)
        for index, sample in _safe_sample(backend_name, indices).items():
            if index not in stats:
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


def intel_grant_binary(devices: Sequence[DeviceLike], stats: dict[int, GpuStat]) -> str | None:
    """The intel_gpu_top path a grant would unblock when an Intel GPU's util is
    missing only for that reason, else None.

    A surface turns the binary into the localized grant hint. Modern kernels read
    util with no grant, so this stays silent there, and the None-util gate clears
    it once a grant makes util read.
    """
    for d in devices:
        if util_backend_name(d.backend, d.name) != "SYCL":
            continue
        stat = stats.get(d.index)
        if stat is not None and stat.utilization_pct is None:
            binary = intel_gpu_top_grant_binary()
            if binary is not None:
                return binary
    return None
