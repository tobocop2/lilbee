"""SYCL/Intel GPU utilization via xpu-smi.

`xpu-smi stats -d <id> -j` reports device metrics as a ``device_level`` array of
``{"metrics_type": "XPUM_STATS_...", "value": N}`` objects (one metric per entry),
not the flat ``gpu_utilization`` keys an earlier parser assumed. The metrics_type
strings are the names of xpum_stats_type_enum; utilization is a percentage and
memory-used is in bytes. Total VRAM is not reported by stats, so VRAM is left as
the 0/0 sentinel and the orchestrator falls back to structural VRAM.

stats reports one device per call, so we run it once per requested index.
"""

from __future__ import annotations

import json

from lilbee.providers.fleet.gpu_backends.base import UtilSample, extract_int, run_smi

_TOOL = "xpu-smi"
_TIMEOUT_S = 5.0

# xpum_stats_type_enum names, as emitted verbatim in the device_level "metrics_type".
_METRIC_UTIL = "XPUM_STATS_GPU_UTILIZATION"
_METRIC_TEMP = "XPUM_STATS_GPU_CORE_TEMPERATURE"


class IntelBackend:
    """SYCL util via xpu-smi, one `stats -d <id>` call per device."""

    def sample(self, indices: frozenset[int]) -> dict[int, UtilSample]:
        samples: dict[int, UtilSample] = {}
        for index in sorted(indices):
            sample = _parse_xpu_smi(_xpu_smi_output(index), index)
            if sample is not None:
                samples[index] = sample
        return samples


def _xpu_smi_output(index: int) -> str:
    """xpu-smi stats JSON stdout for one device, or "" when it can't run."""
    return run_smi(_TOOL, ["stats", "-d", str(index), "-j"], _TIMEOUT_S)


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
