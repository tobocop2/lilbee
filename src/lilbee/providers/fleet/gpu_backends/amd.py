"""ROCm/HIP GPU utilization via amd-smi (preferred) or rocm-smi (fallback)."""

from __future__ import annotations

import json

from lilbee.providers.fleet.gpu_backends.base import (
    UtilSample,
    extract_int,
    parse_device_index,
    run_smi,
)

_TOOL_AMD_SMI = "amd-smi"
_TOOL_ROCM_SMI = "rocm-smi"

_AMD_SMI_ARGS = ("metric", "--usage", "--temperature", "--json")
_ROCM_SMI_ARGS = ("--showuse", "--showmeminfo", "vram", "--showtemp", "--json")

# rocm-smi key names for VRAM (byte values).
_ROCM_VRAM_TOTAL_KEY = "VRAM Total Memory (B)"
_ROCM_VRAM_USED_KEY = "VRAM Total Used Memory (B)"

_TIMEOUT_S = 5.0


class AmdBackend:
    """ROCm/HIP util via amd-smi, falling back to rocm-smi."""

    def sample(self, indices: frozenset[int]) -> dict[int, UtilSample]:
        result = _amd_smi_samples(indices)
        if not result:
            result = _rocm_smi_samples(indices)
        return result


def _amd_smi_output() -> str:
    """amd-smi JSON stdout, or "" when it can't run."""
    return run_smi(_TOOL_AMD_SMI, list(_AMD_SMI_ARGS), _TIMEOUT_S)


def _rocm_smi_output() -> str:
    """rocm-smi JSON stdout, or "" when it can't run."""
    return run_smi(_TOOL_ROCM_SMI, list(_ROCM_SMI_ARGS), _TIMEOUT_S)


def _nested_int(obj: dict[str, object], key: str, nested_key: str = "value") -> int | None:
    """Extract an int from obj[key] when that value is a nested {"value": N} dict."""
    val = obj.get(key)
    if isinstance(val, dict):
        return extract_int(val, (nested_key,))
    return None


def _parse_amd_smi(raw: str, indices: frozenset[int]) -> dict[int, UtilSample]:
    """Parse amd-smi JSON into UtilSample per index, or {} on failure.

    Tolerates two shapes emitted by different amd-smi versions:
    - Flat:   {"gfx_activity": 72, "temperature_c": 61}
    - Nested: {"gfx_activity": {"value": 72, "unit": "%"}, "temperature": {"edge": 61}}
    """
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    # amd-smi metric --json emits a list of GPU objects or {"gpu": [...]}.
    items: list[object] = data if isinstance(data, list) else data.get("gpu", [])
    samples: dict[int, UtilSample] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_index = item.get("gpu") if "gpu" in item else item.get("id", -1)
        try:
            index = int(raw_index)  # type: ignore[arg-type]
        except (ValueError, TypeError):
            continue
        if index not in indices:
            continue
        # Util: try flat keys first, then nested {"value": N} form.
        util = extract_int(item, ("gfx_activity", "gfx_busy_percent", "gpu_activity"))
        if util is None:
            util = (
                _nested_int(item, "gfx_activity")
                or _nested_int(item, "gfx_busy_percent")
                or _nested_int(item, "gpu_activity")
            )
        # Temp: flat keys, then nested sensor dict {"edge": N} under "temperature".
        temp = extract_int(item, ("temperature_c", "temp_edge", "edge"))
        if temp is None:
            temp_block = item.get("temperature")
            if isinstance(temp_block, dict):
                temp = extract_int(temp_block, ("edge", "junction", "mem"))
        # VRAM not reliably present in metric mode; leave 0 so the orchestrator
        # keeps structural VRAM.
        samples[index] = UtilSample(
            index=index,
            utilization_pct=util,
            temperature_c=temp,
            free_bytes=0,
            total_bytes=0,
        )
    return samples


def _parse_rocm_smi(raw: str, indices: frozenset[int]) -> dict[int, UtilSample]:
    """Parse rocm-smi JSON into UtilSample per index, or {} on failure.

    VRAM is parsed when the rocm-smi VRAM keys are present (byte values);
    absent keys leave free_bytes/total_bytes as 0 (structural-fallback sentinel).
    """
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    # rocm-smi --json emits {"card0": {...}, "card1": {...}} or {"GPU[0]": {...}}.
    if not isinstance(data, dict):
        return {}
    samples: dict[int, UtilSample] = {}
    for key, val in data.items():
        if not isinstance(val, dict):
            continue
        index = parse_device_index(key)
        if index is None or index not in indices:
            continue
        util = extract_int(val, ("GPU use (%)", "GPU_UTIL", "gfx_activity"))
        temp = extract_int(val, ("Temperature (Sensor edge) (C)", "temp_edge", "temp"))
        # VRAM keys carry byte strings; parse when present.
        total_b = extract_int(val, (_ROCM_VRAM_TOTAL_KEY,))
        used_b = extract_int(val, (_ROCM_VRAM_USED_KEY,))
        if total_b is not None:
            free_b = max(total_b - (used_b or 0), 0)
        else:
            total_b = 0
            free_b = 0
        samples[index] = UtilSample(
            index=index,
            utilization_pct=util,
            temperature_c=temp,
            free_bytes=free_b,
            total_bytes=total_b,
        )
    return samples


def _amd_smi_samples(indices: frozenset[int]) -> dict[int, UtilSample]:
    return _parse_amd_smi(_amd_smi_output(), indices)


def _rocm_smi_samples(indices: frozenset[int]) -> dict[int, UtilSample]:
    return _parse_rocm_smi(_rocm_smi_output(), indices)
