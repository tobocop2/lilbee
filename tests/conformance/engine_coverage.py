"""Every engine cell lilbee ships: compared, deferred or excluded, with the reason."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from lilbee.providers.fleet.engines.base import EngineName


class Coverage(StrEnum):
    """What the conformance suite holds for one engine cell and device."""

    COMPARED = "compared"
    DEFERRED = "deferred"
    EXCLUDED = "excluded"


class Device(StrEnum):
    """The hardware a cell runs its work on."""

    CPU = "cpu"
    CUDA = "cuda"
    METAL = "metal"
    INTEL_GPU = "intel-gpu"
    NPU = "npu"


# An engine served by a preset, not by the fleet; it has no EngineName member and no pack.
INFINITY = "infinity"
PRESET_CELL = "preset"

_CPU_HOST = "lilbee-eng-g-1 (16-core CPU)"
_CUDA_HOST = "one rented L4"
_MAC_HOST = "the Mac"
_WINDOWS_HOST = "a Windows host"
_WINDOWS_CUDA_HOST = "a Windows host with an NVIDIA GPU"
_NO_NPU = "owner: no NPU hardware, so NPU cells are excluded"
_NO_RECORD = "pack ships; no comparison record in git yet"
_MAC_LAST = "pack ships; Mac runs come last"
_INSTALLED = "installs with lilbee engine install; the pack table has no pack for it yet"

LINUX_CPU = "linux-x86_64-cpu"


@dataclass(frozen=True)
class CoverageRow:
    """One engine cell and device with its status and the reason for it."""

    engine: str
    cell: str
    device: Device
    status: Coverage
    reason: str
    host: str = ""
    owner_decision: str = ""
    record: str = ""


def _deferred(engine: str, cell: str, device: Device, host: str, reason: str) -> CoverageRow:
    return CoverageRow(engine, cell, device, Coverage.DEFERRED, reason, host=host)


def _excluded(engine: str, cell: str, device: Device, decision: str) -> CoverageRow:
    return CoverageRow(
        engine, cell, device, Coverage.EXCLUDED, "no hardware to run on", owner_decision=decision
    )


COVERAGE_ROWS: tuple[CoverageRow, ...] = (
    _deferred(EngineName.LLAMA, LINUX_CPU, Device.CPU, _CPU_HOST, _NO_RECORD),
    _deferred(EngineName.LLAMA, "linux-x86_64-cuda", Device.CUDA, _CUDA_HOST, _NO_RECORD),
    _deferred(EngineName.LLAMA, "macos-arm64-metal", Device.METAL, _MAC_HOST, _MAC_LAST),
    _deferred(EngineName.LLAMA, "windows-x86_64-cpu", Device.CPU, _WINDOWS_HOST, _NO_RECORD),
    _deferred(EngineName.LLAMA, "windows-x86_64-cuda", Device.CUDA, _WINDOWS_CUDA_HOST, _NO_RECORD),
    _deferred(EngineName.IK_LLAMA, LINUX_CPU, Device.CPU, _CPU_HOST, _NO_RECORD),
    _deferred(EngineName.MISTRAL, LINUX_CPU, Device.CPU, _CPU_HOST, _NO_RECORD),
    _deferred(EngineName.MISTRAL, "linux-x86_64-cuda128-sm89", Device.CUDA, _CUDA_HOST, _NO_RECORD),
    _deferred(EngineName.TEI, LINUX_CPU, Device.CPU, _CPU_HOST, _NO_RECORD),
    _deferred(EngineName.TEI, "linux-x86_64-cuda-sm89", Device.CUDA, _CUDA_HOST, _NO_RECORD),
    _deferred(
        EngineName.MLX,
        "macos-arm64-metal",
        Device.METAL,
        _MAC_HOST,
        f"{_INSTALLED}; Mac runs come last",
    ),
    _deferred(
        EngineName.OPENVINO,
        "linux-x86_64-openvino",
        Device.CPU,
        _CPU_HOST,
        f"{_INSTALLED}; no comparison record in git yet",
    ),
    _deferred(
        EngineName.OPENVINO,
        "linux-x86_64-openvino",
        Device.INTEL_GPU,
        "the ThinkPad",
        f"{_INSTALLED}; one hardware recording, no comparison record in git yet",
    ),
    _excluded(EngineName.OPENVINO, "linux-x86_64-openvino", Device.NPU, _NO_NPU),
    _excluded(EngineName.LEMONADE, "linux-x86_64-npu", Device.NPU, _NO_NPU),
    _excluded(EngineName.LEMONADE, "windows-x86_64-npu", Device.NPU, _NO_NPU),
    _deferred(
        INFINITY,
        PRESET_CELL,
        Device.CPU,
        _CPU_HOST,
        "recorded in fixtures/presets and docs/preset-results.md; no llama.cpp comparison yet",
    ),
)
