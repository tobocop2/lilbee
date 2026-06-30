"""ROCm/HIP GPU utilization via amd-smi (preferred) or rocm-smi (fallback)."""

from __future__ import annotations

import json
import shutil
import subprocess

from lilbee.providers.fleet.gpu_backends.base import UtilSample, extract_int, parse_device_index

_TOOL_AMD_SMI = "amd-smi"
_TOOL_ROCM_SMI = "rocm-smi"

_AMD_SMI_ARGS = ("metric", "--usage", "--temperature", "--json")
_ROCM_SMI_ARGS = ("--showuse", "--showmeminfo", "vram", "--showtemp", "--json")

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
    binary = shutil.which(_TOOL_AMD_SMI)
    if binary is None:
        return ""
    try:
        proc = subprocess.run(  # noqa: S603 - fixed args against the resolved amd-smi
            [binary, *_AMD_SMI_ARGS],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout if proc.returncode == 0 else ""


def _rocm_smi_output() -> str:
    """rocm-smi JSON stdout, or "" when it can't run."""
    binary = shutil.which(_TOOL_ROCM_SMI)
    if binary is None:
        return ""
    try:
        proc = subprocess.run(  # noqa: S603 - fixed args against the resolved rocm-smi
            [binary, *_ROCM_SMI_ARGS],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout if proc.returncode == 0 else ""


def _parse_amd_smi(raw: str, indices: frozenset[int]) -> dict[int, UtilSample]:
    """Parse amd-smi JSON into UtilSample per index, or {} on failure."""
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
        util = extract_int(item, ("gfx_activity", "gfx_busy_percent", "gpu_activity"))
        temp = extract_int(item, ("temperature_c", "temp_edge", "edge"))
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
    """Parse rocm-smi JSON into UtilSample per index, or {} on failure."""
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
        samples[index] = UtilSample(
            index=index,
            utilization_pct=util,
            temperature_c=temp,
            free_bytes=0,
            total_bytes=0,
        )
    return samples


def _amd_smi_samples(indices: frozenset[int]) -> dict[int, UtilSample]:
    return _parse_amd_smi(_amd_smi_output(), indices)


def _rocm_smi_samples(indices: frozenset[int]) -> dict[int, UtilSample]:
    return _parse_rocm_smi(_rocm_smi_output(), indices)
