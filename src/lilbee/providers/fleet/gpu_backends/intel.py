"""Intel GPU utilization from the first available source.

Intel exposes GPU activity three different ways depending on the GPU class,
kernel, and privileges. We try them in order and take the first that reports:

1. ``xpu-smi`` (Intel XPU Manager) -- Data Center GPU Max/Flex and Arc only.
   Reads a ``device_level`` array of {metrics_type, value} objects keyed by the
   xpum_stats_type_enum names. One device per call, no root.
2. DRM ``fdinfo`` -- the kernel's per-client engine-busy counters. Covers
   consumer iGPUs with no root and no extra tool, but only on kernels new enough
   to publish i915 engine stats (~6.2+).
3. ``intel_gpu_top`` (Intel GPU Tools) -- reads the i915 PMU, so it covers
   essentially every consumer iGPU on kernel 4.16+, but needs CAP_PERFMON (or
   root); it falls through cleanly when the permission isn't granted.

The fdinfo and intel_gpu_top sources report one device (the consumer case is a
single iGPU); their reading is keyed to the lowest requested index. VRAM is left
as the 0/0 structural sentinel since none of these report an iGPU's total memory.
"""

from __future__ import annotations

import json
import shutil
import subprocess

from lilbee.providers.fleet.gpu_backends import fdinfo
from lilbee.providers.fleet.gpu_backends.base import UtilSample, extract_int, run_smi

_TOOL_XPU_SMI = "xpu-smi"
_TOOL_IGT = "intel_gpu_top"
_TIMEOUT_S = 5.0

# xpu-smi device_level metrics_type names (xpum_stats_type_enum).
_METRIC_UTIL = "XPUM_STATS_GPU_UTILIZATION"
_METRIC_TEMP = "XPUM_STATS_GPU_CORE_TEMPERATURE"

# intel_gpu_top streams JSON samples forever; run it for one short window and
# read the partial output. A couple of 200ms periods lands a fresh per-interval
# reading while keeping the (synchronous) probe from stalling the caller long.
_IGT_SAMPLE_MS = 200
_IGT_CAPTURE_S = 0.6

# The i915 DRM driver name, for the fdinfo reader.
_I915 = "i915"


class IntelBackend:
    """Intel util from xpu-smi, then DRM fdinfo, then intel_gpu_top."""

    def sample(self, indices: frozenset[int]) -> dict[int, UtilSample]:
        for source in (_xpu_smi_samples, _fdinfo_samples, _intel_gpu_top_samples):
            result = source(indices)
            if result:
                return result
        return {}


def _xpu_smi_output(index: int) -> str:
    """xpu-smi stats JSON stdout for one device, or "" when it can't run."""
    return run_smi(_TOOL_XPU_SMI, ["stats", "-d", str(index), "-j"], _TIMEOUT_S)


def _device_level(data: object) -> list[object]:
    """Return the device_level metric array from parsed stats JSON, or []."""
    # stats -d <id> -j emits a single device object with a "device_level" array.
    # Tolerate a bare list or a {"device_list": [...]} wrapper defensively.
    if isinstance(data, dict):
        top_level = data.get("device_level")
        if isinstance(top_level, list):
            return top_level
        wrapped = data.get("device_list")
        if isinstance(wrapped, list) and wrapped and isinstance(wrapped[0], dict):
            inner = wrapped[0].get("device_level")
            return inner if isinstance(inner, list) else []
    if isinstance(data, list) and data and isinstance(data[0], dict):
        inner = data[0].get("device_level")
        return inner if isinstance(inner, list) else []
    return []


def _metric_int(entries: list[object], metrics_type: str) -> int | None:
    """First device_level entry with this metrics_type, coerced to int, or None."""
    for entry in entries:
        if isinstance(entry, dict) and entry.get("metrics_type") == metrics_type:
            return extract_int(entry, ("value",))
    return None


def _parse_xpu_smi(raw: str, index: int) -> UtilSample | None:
    """Parse one device's xpu-smi stats JSON into a UtilSample, or None on failure."""
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    entries = _device_level(data)
    if not entries:
        return None
    return UtilSample(
        index=index,
        utilization_pct=_metric_int(entries, _METRIC_UTIL),
        temperature_c=_metric_int(entries, _METRIC_TEMP),
        # stats reports memory-used but not total; leave the 0/0 sentinel so the
        # orchestrator keeps structural VRAM.
        free_bytes=0,
        total_bytes=0,
    )


def _xpu_smi_samples(indices: frozenset[int]) -> dict[int, UtilSample]:
    samples: dict[int, UtilSample] = {}
    for index in sorted(indices):
        sample = _parse_xpu_smi(_xpu_smi_output(index), index)
        if sample is not None:
            samples[index] = sample
    return samples


def _fdinfo_samples(indices: frozenset[int]) -> dict[int, UtilSample]:
    if not indices:
        return {}
    util = fdinfo.read_drm_util(_I915)
    if util is None:
        return {}
    return _single(indices, util)


def _intel_gpu_top_output() -> str:
    """intel_gpu_top -J partial stdout over one short window, or "" on failure.

    intel_gpu_top streams until killed, so it always hits the timeout on success;
    a permission failure exits fast with empty stdout. Both yield the right thing.
    """
    binary = shutil.which(_TOOL_IGT)
    if binary is None:
        return ""
    try:
        proc = subprocess.run(  # noqa: S603 - fixed args, resolved binary
            [binary, "-J", "-s", str(_IGT_SAMPLE_MS)],
            capture_output=True,
            text=True,
            timeout=_IGT_CAPTURE_S,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout
        if out is None:
            return ""
        return out if isinstance(out, str) else out.decode(errors="replace")
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout


def _last_json_object(raw: str) -> dict[str, object] | None:
    """Parse the last complete sample from intel_gpu_top's streamed JSON array."""
    raw = raw.strip()
    if not raw:
        return None
    # The stream is an unclosed '[ {..}, {..},' -- close it and take the last item.
    if raw.startswith("[") and not raw.endswith("]"):
        raw = raw.rstrip().rstrip(",") + "]"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(data, list):
        last = data[-1] if data else None
        return last if isinstance(last, dict) else None
    return data if isinstance(data, dict) else None


def _igt_max_busy(raw: str) -> int | None:
    """Busiest engine's busy percent from an intel_gpu_top sample, or None."""
    obj = _last_json_object(raw)
    if obj is None:
        return None
    engines = obj.get("engines")
    if not isinstance(engines, dict):
        return None
    busies = [
        float(eng["busy"])
        for eng in engines.values()
        if isinstance(eng, dict) and isinstance(eng.get("busy"), (int, float))
    ]
    if not busies:
        return None
    return round(max(busies))


def _intel_gpu_top_samples(indices: frozenset[int]) -> dict[int, UtilSample]:
    if not indices:
        return {}
    util = _igt_max_busy(_intel_gpu_top_output())
    if util is None:
        return {}
    return _single(indices, util)


def _single(indices: frozenset[int], util: int) -> dict[int, UtilSample]:
    """One device-level reading keyed to the lowest requested index (iGPU case)."""
    idx = min(indices)
    return {
        idx: UtilSample(idx, utilization_pct=util, temperature_c=None, free_bytes=0, total_bytes=0)
    }
