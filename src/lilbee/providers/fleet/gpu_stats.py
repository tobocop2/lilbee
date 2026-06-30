"""Live per-GPU activity stats for the placement view.

devices.py enumerates GPUs once (structural: index, name, total VRAM). This reads
the moving numbers, compute utilization and current free memory, so a client can
animate a per-card load bar.

Each vendor backend implements _UtilBackend: given device indices, it returns
{index: GpuStat} (or a partial dict when some indices can't be read). VRAM
(free/total) stays cross-vendor from the existing device probe and is used as the
fallback for any index the backend can't cover.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from lilbee.providers.fleet.devices import _MIB

# --- backend name constants --------------------------------------------------

_BACKEND_CUDA = "CUDA"
_BACKEND_ROCM = "ROCm"
_BACKEND_HIP = "HIP"
_BACKEND_SYCL = "SYCL"
_BACKEND_METAL = "MTL"  # llama-server --list-devices emits "MTL0:", "MTL1:", ...

# --- tool name constants -----------------------------------------------------

_TOOL_NVIDIA_SMI = "nvidia-smi"
_TOOL_AMD_SMI = "amd-smi"
_TOOL_ROCM_SMI = "rocm-smi"
_TOOL_XPU_SMI = "xpu-smi"

# --- query string constants --------------------------------------------------

_NVIDIA_SMI_QUERY = "index,utilization.gpu,memory.used,memory.total"
_NVIDIA_SMI_FIELDS = 4

# amd-smi metric output columns used when --csv is available:
# device,gfx_activity,vram_used,vram_total,temp_edge
_AMD_SMI_ARGS = ["metric", "--usage", "--temperature", "--json"]
_ROCM_SMI_ARGS = ["--showuse", "--showmeminfo", "vram", "--showtemp", "--json"]

# xpu-smi stats columns (JSON output):  "device_id", "gpu_utilization", "gpu_memory_used",
# "gpu_memory_total", "gpu_temperature"
_XPU_SMI_ARGS = ["stats", "--json"]

# --- cache / timing constants -----------------------------------------------

_SMI_TIMEOUT_S = 5.0
# A CLI spawn costs tens of milliseconds. Cache just under the stream tick so
# concurrent placement views coalesce to one probe per window.
_SMI_CACHE_TTL_S = 0.9


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
    temperature_c: int | None = None


@runtime_checkable
class _UtilBackend(Protocol):
    """One vendor's live-util probe: takes a set of device indices, returns partials."""

    def query(self, indices: frozenset[int]) -> dict[int, GpuStat]: ...


# --- CUDA backend (nvidia-smi) -----------------------------------------------


class _SmiCache:
    """Shares one nvidia-smi probe across streams that ask within the TTL."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._at = 0.0
        self._value: dict[int, GpuStat] = {}
        self._primed = False

    def stats(self) -> dict[int, GpuStat]:
        with self._lock:
            now = time.monotonic()
            if not self._primed or now - self._at >= _SMI_CACHE_TTL_S:
                self._value = _nvidia_smi_stats()
                self._at = now
                self._primed = True
            return self._value

    def reset(self) -> None:
        with self._lock:
            self._primed = False


_smi_cache = _SmiCache()


class _CudaBackend:
    """CUDA util via nvidia-smi (cached)."""

    def query(self, indices: frozenset[int]) -> dict[int, GpuStat]:
        return {i: s for i, s in _smi_cache.stats().items() if i in indices}


def _nvidia_smi_stats() -> dict[int, GpuStat]:
    """Parse nvidia-smi utilization + memory, or {} when it can't run."""
    out = _nvidia_smi_output()
    stats: dict[int, GpuStat] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != _NVIDIA_SMI_FIELDS:
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
    binary = shutil.which(_TOOL_NVIDIA_SMI) or _TOOL_NVIDIA_SMI
    try:
        proc = subprocess.run(  # noqa: S603 - fixed query against the resolved nvidia-smi
            [binary, f"--query-gpu={_NVIDIA_SMI_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=_SMI_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout if proc.returncode == 0 else ""


# --- ROCm / HIP backend (amd-smi with rocm-smi fallback) --------------------


class _RocmBackend:
    """ROCm/HIP util via amd-smi (preferred) or rocm-smi (fallback)."""

    def query(self, indices: frozenset[int]) -> dict[int, GpuStat]:
        result = _amd_smi_stats(indices)
        if not result:
            result = _rocm_smi_stats(indices)
        return result


def _amd_smi_output() -> str:
    """amd-smi JSON output, or "" when it can't run."""
    binary = shutil.which(_TOOL_AMD_SMI)
    if binary is None:
        return ""
    try:
        proc = subprocess.run(  # noqa: S603 - fixed args against the resolved amd-smi
            [binary, *_AMD_SMI_ARGS],
            capture_output=True,
            text=True,
            timeout=_SMI_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout if proc.returncode == 0 else ""


def _amd_smi_stats(indices: frozenset[int]) -> dict[int, GpuStat]:
    """Parse amd-smi JSON for util + temp, or {} on failure."""
    import json

    raw = _amd_smi_output()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}

    # amd-smi metric --usage --temperature --json emits a list of GPU objects.
    # Each has a "gpu" key (integer index) and nested "gfx_activity"/"temp" info.
    stats: dict[int, GpuStat] = {}
    items = data if isinstance(data, list) else data.get("gpu", [])
    for item in items:
        try:
            index = int(item.get("gpu", item.get("id", -1)))
        except (ValueError, TypeError):
            continue
        if index not in indices:
            continue
        util = _extract_int(item, ("gfx_activity", "gfx_busy_percent", "gpu_activity"))
        temp = _extract_int(item, ("temperature_c", "temp_edge", "edge"))
        # VRAM: amd-smi doesn't always include memory in metric mode; use 0 as sentinel.
        stats[index] = GpuStat(index, util, 0, 0, temp)
    return stats


def _rocm_smi_output() -> str:
    """rocm-smi JSON output, or "" when it can't run."""
    binary = shutil.which(_TOOL_ROCM_SMI)
    if binary is None:
        return ""
    try:
        proc = subprocess.run(  # noqa: S603 - fixed args against the resolved rocm-smi
            [binary, *_ROCM_SMI_ARGS],
            capture_output=True,
            text=True,
            timeout=_SMI_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout if proc.returncode == 0 else ""


def _rocm_smi_stats(indices: frozenset[int]) -> dict[int, GpuStat]:
    """Parse rocm-smi JSON for util + temp, or {} on failure."""
    import json

    raw = _rocm_smi_output()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}

    # rocm-smi --json emits {"card0": {...}, "card1": {...}} or {"GPU[0]": {...}}.
    stats: dict[int, GpuStat] = {}
    for key, val in data.items():
        if not isinstance(val, dict):
            continue
        index = _parse_device_index(key)
        if index is None or index not in indices:
            continue
        util = _extract_int(val, ("GPU use (%)", "GPU_UTIL", "gfx_activity"))
        temp = _extract_int(val, ("Temperature (Sensor edge) (C)", "temp_edge", "temp"))
        stats[index] = GpuStat(index, util, 0, 0, temp)
    return stats


# --- SYCL / Intel backend (xpu-smi) -----------------------------------------


class _SyclBackend:
    """SYCL/Intel GPU util via xpu-smi."""

    def query(self, indices: frozenset[int]) -> dict[int, GpuStat]:
        return _xpu_smi_stats(indices)


def _xpu_smi_output() -> str:
    """xpu-smi JSON output, or "" when it can't run."""
    binary = shutil.which(_TOOL_XPU_SMI)
    if binary is None:
        return ""
    try:
        proc = subprocess.run(  # noqa: S603 - fixed args against the resolved xpu-smi
            [binary, *_XPU_SMI_ARGS],
            capture_output=True,
            text=True,
            timeout=_SMI_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout if proc.returncode == 0 else ""


def _xpu_smi_stats(indices: frozenset[int]) -> dict[int, GpuStat]:
    """Parse xpu-smi JSON for util + temp, or {} on failure."""
    import json

    raw = _xpu_smi_output()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}

    # xpu-smi stats --json emits a list of device objects.
    stats: dict[int, GpuStat] = {}
    items = data if isinstance(data, list) else data.get("device_list", [])
    for item in items:
        try:
            index = int(item.get("device_id", -1))
        except (ValueError, TypeError):
            continue
        if index not in indices:
            continue
        util = _extract_int(item, ("gpu_utilization", "eu_active", "xe_eu_active"))
        temp = _extract_int(item, ("gpu_temperature", "temperature"))
        used_mib = _extract_int(item, ("gpu_memory_used_in_mb", "mem_used"))
        total_mib = _extract_int(item, ("gpu_memory_size_in_mb", "mem_total"))
        free = max((total_mib or 0) - (used_mib or 0), 0) * _MIB if total_mib else 0
        total = (total_mib or 0) * _MIB
        stats[index] = GpuStat(index, util, free, total, temp)
    return stats


# --- Apple / Metal backend ---------------------------------------------------


class _MetalBackend:
    """Apple Metal GPU util stub.

    Util and temperature require either IOReport (private macOS framework,
    no public headers, ctypes access is fragile across OS versions) or
    powermetrics (needs sudo). Neither is safe to ship without more empirical
    validation on real Apple Silicon. VRAM is reported by the structural probe.
    # TODO(apple-ioreport): implement via ctypes IOReport when validated.
    """

    def query(self, indices: frozenset[int]) -> dict[int, GpuStat]:
        _ = indices  # stub: no tool to query; returns {} so structural fallback applies
        return {}


# --- backend registry --------------------------------------------------------

# Maps the backend string (as emitted by llama-server --list-devices) to the
# backend instance that can probe util for that vendor's devices.
_BACKENDS: dict[str, _UtilBackend] = {
    _BACKEND_CUDA: _CudaBackend(),
    _BACKEND_ROCM: _RocmBackend(),
    _BACKEND_HIP: _RocmBackend(),
    _BACKEND_SYCL: _SyclBackend(),
    _BACKEND_METAL: _MetalBackend(),
}


# --- parse helpers -----------------------------------------------------------


def _extract_int(obj: object, keys: tuple[str, ...]) -> int | None:
    """Return the first key found as an int, or None."""
    if not isinstance(obj, dict):
        return None
    for key in keys:
        val = obj.get(key)
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                pass
    return None


def _parse_device_index(key: str) -> int | None:
    """Extract a zero-based GPU index from keys like 'card0', 'GPU[0]', '0'."""
    import re

    m = re.search(r"\d+", key)
    return int(m.group()) if m else None


# --- public API --------------------------------------------------------------


def _safe_query(backend: _UtilBackend, indices: frozenset[int]) -> dict[int, GpuStat]:
    """Call backend.query, returning {} if any exception is raised."""
    try:
        return backend.query(indices)
    except Exception:  # backends must never crash the sampler
        return {}


def probe_gpu_stats(devices: Sequence[_DeviceLike]) -> dict[int, GpuStat]:
    """Live stats keyed by device index. Empty when no probe is available.

    Dispatches to the right vendor backend by device.backend. Any index the
    backend can't cover falls back to structural totals with util=None and
    temperature_c=None, so the caller always has an entry per known GPU.
    """
    # Start with structural fallbacks (util=None, temp=None, VRAM from probe).
    stats: dict[int, GpuStat] = {
        d.index: GpuStat(d.index, None, d.free_bytes, d.total_bytes) for d in devices
    }

    # Group by backend and dispatch once per vendor.
    by_backend: dict[str, list[_DeviceLike]] = {}
    for d in devices:
        by_backend.setdefault(d.backend, []).append(d)

    for backend_name, group in by_backend.items():
        backend = _BACKENDS.get(backend_name)
        if backend is None:
            continue
        indices = frozenset(d.index for d in group)
        partial = _safe_query(backend, indices)
        for index, live in partial.items():
            if index not in stats:
                continue
            base = stats[index]
            # Prefer live VRAM when the backend provides it (xpu-smi does);
            # fall back to structural when the backend returns 0/0 (amd-smi
            # metric mode omits VRAM so we keep the structural values).
            free = live.free_bytes if live.free_bytes or live.total_bytes else base.free_bytes
            total = live.total_bytes if live.total_bytes else base.total_bytes
            stats[index] = GpuStat(
                index,
                live.utilization_pct,
                free,
                total,
                live.temperature_c,
            )

    return {i: stats[i] for i in sorted(stats)}
