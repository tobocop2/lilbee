"""SYCL/Intel GPU utilization via xpu-smi."""

from __future__ import annotations

import json

from lilbee.providers.fleet.devices import MIB
from lilbee.providers.fleet.gpu_backends.base import UtilSample, extract_int, run_smi

_TOOL = "xpu-smi"
_ARGS = ("stats", "--json")
_TIMEOUT_S = 5.0


class IntelBackend:
    """SYCL util via xpu-smi."""

    def sample(self, indices: frozenset[int]) -> dict[int, UtilSample]:
        return _xpu_smi_samples(indices)


def _xpu_smi_output() -> str:
    """xpu-smi JSON stdout, or "" when it can't run."""
    return run_smi(_TOOL, list(_ARGS), _TIMEOUT_S)


def _parse_xpu_smi(raw: str, indices: frozenset[int]) -> dict[int, UtilSample]:
    """Parse xpu-smi JSON into UtilSample per index, or {} on failure."""
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    # xpu-smi stats --json emits a list of device objects or {"device_list": [...]}.
    items: list[object] = data if isinstance(data, list) else data.get("device_list", [])
    samples: dict[int, UtilSample] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("device_id", -1))
        except (ValueError, TypeError):
            continue
        if index not in indices:
            continue
        util = extract_int(item, ("gpu_utilization", "eu_active", "xe_eu_active"))
        temp = extract_int(item, ("gpu_temperature", "temperature"))
        used_mib = extract_int(item, ("gpu_memory_used_in_mb", "mem_used"))
        total_mib = extract_int(item, ("gpu_memory_size_in_mb", "mem_total"))
        free = max((total_mib or 0) - (used_mib or 0), 0) * MIB if total_mib else 0
        total = (total_mib or 0) * MIB
        samples[index] = UtilSample(
            index=index,
            utilization_pct=util,
            temperature_c=temp,
            free_bytes=free,
            total_bytes=total,
        )
    return samples


def _xpu_smi_samples(indices: frozenset[int]) -> dict[int, UtilSample]:
    return _parse_xpu_smi(_xpu_smi_output(), indices)
