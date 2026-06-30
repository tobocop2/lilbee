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
MIB = 1024 * 1024
# Per-backend visible-devices env vars (the probe inherits them; the children
# re-emit them, composed through any parent restriction).
_CUDA_VISIBLE_VAR = "CUDA_VISIBLE_DEVICES"
_CUDA_ORDER_VAR = "CUDA_DEVICE_ORDER"
_PCI_BUS_ID_ORDER = "PCI_BUS_ID"
_ROCR_VISIBLE_VAR = "ROCR_VISIBLE_DEVICES"
_HIP_VISIBLE_VAR = "HIP_VISIBLE_DEVICES"
_VK_VISIBLE_VAR = "GGML_VK_VISIBLE_DEVICES"
_ONEAPI_SELECTOR_VAR = "ONEAPI_DEVICE_SELECTOR"
_LEVEL_ZERO_PREFIX = "level_zero:"
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
    """Env for the probe: stable PCI ordering so CUDA indices match what we pin.

    A preset ``CUDA_DEVICE_ORDER`` is respected; ``visible_env`` re-emits the same
    order var, so the probe and the spawned servers see one device ordering.
    """
    env = dict(os.environ)
    env.setdefault(_CUDA_ORDER_VAR, _PCI_BUS_ID_ORDER)
    return env


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
        total = int(total_mib) * MIB
        free = int(free_mib) * MIB if free_mib else total
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


def _compose_visible(indices: list[int], parent_value: str | None) -> str:
    """Visible-devices value naming the same physical devices the probe saw.

    When the parent env already restricts the var, the probe's indices are
    relative to that comma-separated list (integer or UUID entries), so each
    index maps through it; the child's value then names the same physical
    devices instead of being re-interpreted as absolute.
    """
    if parent_value is None:
        return ",".join(str(i) for i in indices)
    entries = [entry.strip() for entry in parent_value.split(",") if entry.strip()]
    out: list[str] = []
    for i in indices:
        if i >= len(entries):
            # The probe enumerates devices under the parent restriction, so every
            # index must map into it. An out-of-range index is an invariant
            # violation; emitting a bare ``str(i)`` would pin an absolute integer
            # into a possibly UUID-namespaced list, silently selecting the wrong
            # GPU. Fail loudly instead.
            raise ValueError(
                f"device index {i} is outside the parent visible-devices list "
                f"{parent_value!r}; cannot compose a child pin without selecting the wrong GPU"
            )
        out.append(entries[i])
    return ",".join(out)


def visible_env(devices: tuple[FleetDevice, ...]) -> dict[str, str]:
    """Env that pins a child to *devices* via the right var for their backend.

    Indices are the backend-native ones from ``probe_devices``, composed through
    any parent visible-devices restriction so the child names the same physical
    devices the probe enumerated; no cross-API index translation occurs.
    """
    if not devices:
        return {}
    backend = devices[0].backend
    indices = [d.index for d in devices]
    if backend == "CUDA":
        return {
            _CUDA_VISIBLE_VAR: _compose_visible(indices, os.environ.get(_CUDA_VISIBLE_VAR)),
            _CUDA_ORDER_VAR: os.environ.get(_CUDA_ORDER_VAR, _PCI_BUS_ID_ORDER),
        }
    if backend in ("ROCm", "HIP"):
        return _amd_visible_env(indices)
    if backend == "Vulkan":
        return {_VK_VISIBLE_VAR: _compose_visible(indices, os.environ.get(_VK_VISIBLE_VAR))}
    if backend == "SYCL":
        return {_ONEAPI_SELECTOR_VAR: _compose_sycl(indices, os.environ.get(_ONEAPI_SELECTOR_VAR))}
    return {}


def _amd_visible_env(indices: list[int]) -> dict[str, str]:
    """Pin an AMD ROCm/HIP child to the probe's *indices* with one visibility var.

    ``ROCR_VISIBLE_DEVICES`` and ``HIP_VISIBLE_DEVICES`` are applied sequentially
    by the runtime: ROCR filters first, then HIP re-indexes within the survivors.
    The probe enumerated a single index space already filtered by whichever var
    the parent set, so emitting BOTH (each composed against its own parent) would
    double-filter and select the wrong cards. Emit only the var the parent used,
    composed against that parent value, and leave the other inherited untouched;
    default to HIP when the parent restricted neither. The child inherits the
    parent env, so an unset override keeps any inherited sibling var in force.
    """
    parent_rocr = os.environ.get(_ROCR_VISIBLE_VAR)
    parent_hip = os.environ.get(_HIP_VISIBLE_VAR)
    if parent_rocr is not None and parent_hip is None:
        return {_ROCR_VISIBLE_VAR: _compose_visible(indices, parent_rocr)}
    return {_HIP_VISIBLE_VAR: _compose_visible(indices, parent_hip)}


def _compose_sycl(indices: list[int], parent_value: str | None) -> str:
    """``ONEAPI_DEVICE_SELECTOR`` value naming the same devices the probe saw.

    A parent selector shaped ``level_zero:i,j`` makes the probe's indices
    relative to its post-colon list, so each index maps through that list like
    :func:`_compose_visible`; any other shape (or none) emits absolute indices.
    """
    if parent_value is not None and parent_value.startswith(_LEVEL_ZERO_PREFIX):
        parent_list = parent_value[len(_LEVEL_ZERO_PREFIX) :]
        return _LEVEL_ZERO_PREFIX + _compose_visible(indices, parent_list)
    return _LEVEL_ZERO_PREFIX + ",".join(str(i) for i in indices)
