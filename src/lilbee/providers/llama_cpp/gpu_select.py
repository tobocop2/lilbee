"""Best-GPU autodetection for the Vulkan / CUDA / ROCm backends.

On a host with multiple GPUs (typical dual-GPU laptop: discrete NVIDIA
plus integrated AMD/Intel), Vulkan device ordering is driver- and
OS-dependent. llama.cpp's Vulkan backend doesn't sort by device type,
so a model can land on the integrated GPU and stall against shared
system memory.

This module probes the host's adapter list (via ``vulkaninfo --summary``
or NVIDIA's ``pynvml`` / ``nvidia-smi`` binding) and returns a string
suitable for ``GGML_VK_VISIBLE_DEVICES`` / ``CUDA_VISIBLE_DEVICES``,
pinning inference to the best-available discrete GPU. The detection is
best-effort and silent on failure: when no probe succeeds, the caller
keeps the default Vulkan-loader ordering.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Probe timeout for the external GPU enumeration tools. 3 s is more than
# enough for vulkaninfo / nvidia-smi on a healthy system and short enough
# that a hung driver doesn't slow the first inference call.
_PROBE_TIMEOUT_S = 3.0

# vulkaninfo summary device-type tokens. Score order picks discrete > integrated
# > virtual / CPU (CPU softpipe is never what a user wants for inference).
_DEVICE_TYPE_RANK: dict[str, int] = {
    "PHYSICAL_DEVICE_TYPE_DISCRETE_GPU": 3,
    "PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU": 2,
    "PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU": 1,
    "PHYSICAL_DEVICE_TYPE_CPU": 0,
    "PHYSICAL_DEVICE_TYPE_OTHER": 0,
}

_VULKANINFO_GPU_HEADER_RE = re.compile(r"^GPU(\d+):")
_VULKANINFO_KV_RE = re.compile(r"^\s*(\w+)\s*=\s*(\S+)")


@dataclass(frozen=True)
class VulkanDevice:
    """One Vulkan adapter as reported by vulkaninfo."""

    index: int
    device_type: str
    device_name: str


def autoselect_best_gpu_index() -> str | None:
    """Return a comma-separated index list pinning inference to the best GPU.

    Returns ``None`` when no probe succeeds (vulkaninfo missing,
    parsing failed, only one device, etc.) so the caller knows to
    leave device visibility untouched. The result format matches
    ``GGML_VK_VISIBLE_DEVICES`` (``"0"`` or ``"0,1"``).
    """
    devices = _list_vulkan_devices()
    if devices is None:
        return None
    best = _pick_best_device(devices)
    if best is None:
        return None
    # Only emit a pin when there's a real choice to make: if every visible
    # device is the same type, the default ordering is already correct and
    # forcing the index would only hide a user's manual override on rebuild.
    if len(devices) == 1:
        return None
    return str(best.index)


def _list_vulkan_devices() -> list[VulkanDevice] | None:
    """Run ``vulkaninfo --summary`` and parse the device list.

    Returns ``None`` on any failure (binary missing, non-zero exit,
    timeout, unparseable output). Empty list is a distinct outcome
    ("ran fine, found no devices") and propagates back.
    """
    if shutil.which("vulkaninfo") is None:
        return None
    try:
        result = subprocess.run(
            ["vulkaninfo", "--summary"],  # noqa: S607 -- vulkaninfo on PATH is the contract
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return _parse_vulkaninfo_summary(result.stdout)


def _parse_vulkaninfo_summary(output: str) -> list[VulkanDevice]:
    """Extract one :class:`VulkanDevice` per ``GPU<N>:`` block in the summary.

    The summary format groups device fields under a ``GPU<index>:``
    header, with ``key = value`` pairs indented underneath. We only
    pull ``deviceType`` and ``deviceName`` (the rest is informational).
    Blocks missing ``deviceType`` are skipped: an unparseable device
    can't be ranked.
    """
    devices: list[VulkanDevice] = []
    current_index: int | None = None
    current_fields: dict[str, str] = {}

    def flush() -> None:
        if current_index is None:
            return
        dtype = current_fields.get("deviceType")
        if not dtype:
            return
        devices.append(
            VulkanDevice(
                index=current_index,
                device_type=dtype,
                device_name=current_fields.get("deviceName", ""),
            )
        )

    for line in output.splitlines():
        header = _VULKANINFO_GPU_HEADER_RE.match(line)
        if header is not None:
            flush()
            current_index = int(header.group(1))
            current_fields = {}
            continue
        kv = _VULKANINFO_KV_RE.match(line)
        if kv is not None and current_index is not None:
            current_fields[kv.group(1)] = kv.group(2)
    flush()
    return devices


def _pick_best_device(devices: list[VulkanDevice]) -> VulkanDevice | None:
    """Return the highest-ranked device, preferring lower indexes on ties.

    Sort is stable so the original enumeration order acts as the
    tie-breaker; this matches the user's expectation that "GPU0" wins
    when two adapters are the same type.
    """
    if not devices:
        return None
    ranked = sorted(
        devices,
        key=lambda d: (-_DEVICE_TYPE_RANK.get(d.device_type, 0), d.index),
    )
    best = ranked[0]
    if _DEVICE_TYPE_RANK.get(best.device_type, 0) <= 0:
        return None
    return best
