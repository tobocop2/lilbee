"""Live per-GPU activity stats for the placement view.

devices.py enumerates GPUs once (structural: index, name, total VRAM). This reads
the moving numbers, compute utilization and current free memory, so a client can
animate a per-card load bar. CUDA cards report utilization via nvidia-smi; other
backends report memory only (utilization stays None).
"""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

_CUDA_BACKEND = "CUDA"
_MIB = 1024 * 1024
_SMI_TIMEOUT_S = 5.0
_SMI_QUERY = "index,utilization.gpu,memory.used,memory.total"
_SMI_FIELDS = 4
# A nvidia-smi spawn costs tens of milliseconds. Cache it just under the stream's
# tick so concurrent placement views coalesce to one probe per window instead of
# one each. nvidia-smi indexes by PCI bus order, matching the CUDA enumeration.
_SMI_CACHE_TTL_S = 0.9
_NVIDIA_SMI = shutil.which("nvidia-smi") or "nvidia-smi"


class _DeviceLike(Protocol):
    """Structural view of a probed GPU (FleetDevice or app-layer GpuInfo)."""

    @property
    def index(self) -> int: ...
    @property
    def backend(self) -> str: ...
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


class _SmiCache:
    """Shares one nvidia-smi probe across streams that ask within the TTL."""

    def __init__(self) -> None:
        self._at = 0.0
        self._value: dict[int, GpuStat] = {}
        self._primed = False

    def stats(self) -> dict[int, GpuStat]:
        now = time.monotonic()
        if not self._primed or now - self._at >= _SMI_CACHE_TTL_S:
            self._value = _nvidia_smi_stats()
            self._at = now
            self._primed = True
        return self._value

    def reset(self) -> None:
        self._primed = False


_smi_cache = _SmiCache()


def probe_gpu_stats(devices: Sequence[_DeviceLike]) -> dict[int, GpuStat]:
    """Live stats keyed by device index. Empty when no probe is available.

    CUDA devices are read from nvidia-smi (utilization + live memory). Any device
    the probe can't cover falls back to its structural totals with a None
    utilization, so the caller always has an entry per known GPU.
    """
    stats: dict[int, GpuStat] = {
        d.index: GpuStat(d.index, None, d.free_bytes, d.total_bytes) for d in devices
    }
    if any(d.backend == _CUDA_BACKEND for d in devices):
        stats.update(_smi_cache.stats())
    return {i: stats[i] for i in sorted(stats)}


def _nvidia_smi_stats() -> dict[int, GpuStat]:
    """Parse nvidia-smi utilization + memory, or {} when it can't run."""
    out = _nvidia_smi_output()
    stats: dict[int, GpuStat] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != _SMI_FIELDS:
            continue
        try:
            index, util, used_mib, total_mib = (int(p) for p in parts)
        except ValueError:
            continue
        total = total_mib * _MIB
        stats[index] = GpuStat(index, util, max(total - used_mib * _MIB, 0), total)
    return stats


def _nvidia_smi_output() -> str:
    """nvidia-smi query stdout, or "" when it can't run."""
    try:
        proc = subprocess.run(  # noqa: S603 - fixed query against the resolved nvidia-smi
            [_NVIDIA_SMI, f"--query-gpu={_SMI_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=_SMI_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout if proc.returncode == 0 else ""
