"""Enumerate and pin GPUs using the llama-server binary's own device view.

The hazard this avoids: a device index from one API (Vulkan) is meaningless to
another (CUDA); the same ordinal can be a different physical card. So both
enumeration and pinning go through the binary's native backend index space,
obtained from ``llama-server --list-devices``. The Vulkan VRAM probe is only a
fallback when the binary can't enumerate. See docs/architecture.md.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

_LIST_DEVICES_TIMEOUT_S = 60.0
_MIB = 1024 * 1024
# "  CUDA0: NVIDIA GeForce RTX 3090 (24268 MiB, 23500 MiB free)"
_DEVICE_RE = re.compile(
    r"^\s*([A-Za-z]+)(\d+):\s*(.+?)\s*\((\d+)\s*MiB(?:,\s*(\d+)\s*MiB\s*free)?\)\s*$"
)
# Pin priority when a build reports more than one GPU backend: a real GPU
# backend always wins over Vulkan, which wins over CPU.
_BACKEND_RANK = {"CUDA": 3, "ROCm": 3, "HIP": 3, "SYCL": 2, "Vulkan": 1}


@dataclass(frozen=True)
class FleetDevice:
    """One GPU as the binary's backend enumerates it (native index space)."""

    backend: str
    index: int
    name: str
    total_bytes: int
    free_bytes: int


def _probe_env() -> dict[str, str]:
    """Env for the probe: stable PCI ordering so CUDA indices match what we pin."""
    return {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}


def probe_devices(binary: Path) -> list[FleetDevice]:
    """Parse ``<binary> --list-devices``; ``[]`` when unavailable/unparseable.

    Filtered to a single GPU backend (the highest-ranked one present) so device
    indices are unambiguous when a build exposes several backends.
    """
    try:
        proc = subprocess.run(  # noqa: S603 - binary is the resolved llama-server
            [str(binary), "--list-devices"],
            capture_output=True,
            text=True,
            timeout=_LIST_DEVICES_TIMEOUT_S,
            env=_probe_env(),
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return _select_backend(_parse_devices(proc.stdout + proc.stderr))


def _parse_devices(text: str) -> list[FleetDevice]:
    devices: list[FleetDevice] = []
    for line in text.splitlines():
        match = _DEVICE_RE.match(line)
        if match is None:
            continue
        backend, index, name, total_mib, free_mib = match.groups()
        total = int(total_mib) * _MIB
        free = int(free_mib) * _MIB if free_mib else total
        devices.append(FleetDevice(backend, int(index), name.strip(), total, free))
    return devices


def _select_backend(devices: list[FleetDevice]) -> list[FleetDevice]:
    """Keep one GPU backend's devices (highest rank, ties broken by name).

    Returns a single backend so pinning is unambiguous: ``visible_env`` keys off
    one backend, and mixing index spaces is the very hazard this module avoids.
    """
    ranked = [d for d in devices if d.backend in _BACKEND_RANK]
    if not ranked:
        return []
    backend = max(ranked, key=lambda d: (_BACKEND_RANK[d.backend], d.backend)).backend
    return [d for d in ranked if d.backend == backend]


def visible_env(devices: tuple[FleetDevice, ...]) -> dict[str, str]:
    """Env that pins a child to *devices* via the right var for their backend.

    Indices are the backend-native ones from ``probe_devices`` paired with the
    matching visible-devices variable, so no cross-API index translation occurs.
    """
    if not devices:
        return {}
    backend = devices[0].backend
    ids = ",".join(str(d.index) for d in devices)
    if backend == "CUDA":
        return {"CUDA_VISIBLE_DEVICES": ids, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    if backend in ("ROCm", "HIP"):
        return {"ROCR_VISIBLE_DEVICES": ids, "HIP_VISIBLE_DEVICES": ids}
    if backend == "Vulkan":
        return {"GGML_VK_VISIBLE_DEVICES": ids}
    if backend == "SYCL":
        return {"ONEAPI_DEVICE_SELECTOR": "level_zero:" + ",".join(str(d.index) for d in devices)}
    return {}
