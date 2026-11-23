"""Every engine cell lilbee ships: compared, deferred or excluded, with the reason."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from lilbee.providers.fleet.engines.base import EngineName
from tests.conformance.compare_record import Step
from tests.conformance.compare_verdicts import Oracle


class Coverage(StrEnum):
    """What the conformance suite holds for one engine cell and device."""

    COMPARED = "compared"
    DEFERRED = "deferred"
    EXCLUDED = "excluded"


class Route(StrEnum):
    """How the engine of a cell reaches a user, read from the adapter and the install path."""

    PACK = "pack"
    """The pack table has a pack; `engine install` fetches it."""
    USER_INSTALLED = "user-installed"
    """The user installs the engine; lilbee finds its binary. `engine install` refuses."""
    UNPUBLISHED = "unpublished"
    """The adapter records no pack digest; `engine install` refuses."""
    PRESET = "preset"
    """A preset serves it; the fleet has no such engine."""


class Device(StrEnum):
    """The hardware a cell runs its work on."""

    CPU = "cpu"
    CUDA = "cuda"
    METAL = "metal"
    VULKAN = "vulkan"
    ROCM = "rocm"
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
_AMD_HOST = "a Linux host with an AMD GPU"
_WINDOWS_AMD_HOST = "a Windows host with an AMD or Intel GPU"
_NO_NPU = "owner: no NPU hardware, so NPU cells are excluded"
_NO_RECORD = "no comparison record in git yet"
_MAC_LAST = "Mac runs come last"

LINUX_CPU = "linux-x86_64-cpu"


@dataclass(frozen=True)
class CoverageRow:
    """One engine cell and device with its status and the reason for it."""

    engine: str
    cell: str
    device: Device
    status: Coverage
    route: Route
    reason: str
    host: str = ""
    owner_decision: str = ""
    record: str = ""
    oracle: Oracle = Oracle.LLAMA
    steps: tuple[Step, ...] = ()
    """The steps a COMPARED row claims; its record must hold each one at the verdict it needs."""


def _deferred(
    engine: str, cell: str, device: Device, route: Route, host: str, reason: str
) -> CoverageRow:
    return CoverageRow(engine, cell, device, Coverage.DEFERRED, route, reason, host=host)


def _excluded(engine: str, cell: str, device: Device, route: Route, decision: str) -> CoverageRow:
    return CoverageRow(
        engine,
        cell,
        device,
        Coverage.EXCLUDED,
        route,
        "no hardware to run on",
        owner_decision=decision,
    )


def _compared(
    engine: str,
    cell: str,
    device: Device,
    version: str,
    steps: tuple[Step, ...],
    reason: str,
    oracle: Oracle = Oracle.LLAMA,
) -> CoverageRow:
    return CoverageRow(
        engine,
        cell,
        device,
        Coverage.COMPARED,
        Route.PACK,
        reason,
        record=f"tests/conformance/fixtures/compare/{engine}-{cell}/{version}.json",
        oracle=oracle,
        steps=steps,
    )


_LLAMA_VERSION = "1.12032.1"
_IK_VERSION = "0+git13cffc006b82"
_MISTRAL_VERSION = "0.9.5"
_TEI_VERSION = "1.9.4"
_ORACLE_ITSELF = "the oracle itself: the absolute checks and the null block"
_IK_STEPS = tuple(step for step in Step if step not in (Step.VISION, Step.RERANK))
_IK_REASON = (
    "chat and embed against llama.cpp, with --no-context-shift so overflow passes; "
    "json_object passed in this run; it fails in some runs on ik_llama.cpp "
    "and that defect is still open"
)
_MISTRAL_STEPS = tuple(step for step in Step if step is not Step.RERANK)
_TEI_STEPS = (Step.EMBED, Step.RERANK)

COVERAGE_ROWS: tuple[CoverageRow, ...] = (
    _compared(
        EngineName.LLAMA,
        LINUX_CPU,
        Device.CPU,
        _LLAMA_VERSION,
        tuple(Step),
        _ORACLE_ITSELF,
        Oracle.ABSOLUTE_ONLY,
    ),
    _deferred(
        EngineName.LLAMA, "linux-x86_64-cuda", Device.CUDA, Route.PACK, _CUDA_HOST, _NO_RECORD
    ),
    _deferred(
        EngineName.LLAMA, "macos-arm64-metal", Device.METAL, Route.PACK, _MAC_HOST, _MAC_LAST
    ),
    _deferred(
        EngineName.LLAMA, "windows-x86_64-cpu", Device.CPU, Route.PACK, _WINDOWS_HOST, _NO_RECORD
    ),
    _deferred(
        EngineName.LLAMA,
        "windows-x86_64-cuda",
        Device.CUDA,
        Route.PACK,
        _WINDOWS_CUDA_HOST,
        _NO_RECORD,
    ),
    _deferred(
        EngineName.LLAMA, "linux-x86_64-vulkan", Device.VULKAN, Route.PACK, _AMD_HOST, _NO_RECORD
    ),
    _deferred(
        EngineName.LLAMA, "linux-x86_64-rocm", Device.ROCM, Route.PACK, _AMD_HOST, _NO_RECORD
    ),
    _deferred(
        EngineName.LLAMA,
        "windows-x86_64-vulkan",
        Device.VULKAN,
        Route.PACK,
        _WINDOWS_AMD_HOST,
        _NO_RECORD,
    ),
    _compared(EngineName.IK_LLAMA, LINUX_CPU, Device.CPU, _IK_VERSION, _IK_STEPS, _IK_REASON),
    _deferred(
        EngineName.IK_LLAMA, "linux-x86_64-cuda", Device.CUDA, Route.PACK, _CUDA_HOST, _NO_RECORD
    ),
    _compared(
        EngineName.MISTRAL,
        LINUX_CPU,
        Device.CPU,
        _MISTRAL_VERSION,
        _MISTRAL_STEPS,
        "chat on GGUF, embed and vision on safetensors, against llama.cpp",
    ),
    _deferred(
        EngineName.MISTRAL, "macos-arm64-metal", Device.METAL, Route.PACK, _MAC_HOST, _MAC_LAST
    ),
    _compared(
        EngineName.TEI,
        LINUX_CPU,
        Device.CPU,
        _TEI_VERSION,
        _TEI_STEPS,
        "embed and rerank on safetensors, against llama.cpp",
    ),
    *(
        _deferred(
            EngineName.TEI,
            f"linux-x86_64-cuda-sm{cc}",
            Device.CUDA,
            Route.PACK,
            _CUDA_HOST,
            _NO_RECORD,
        )
        for cc in (75, 80, 86, 89, 90, 100, 120)
    ),
    _deferred(EngineName.TEI, "macos-arm64-metal", Device.METAL, Route.PACK, _MAC_HOST, _MAC_LAST),
    _deferred(
        EngineName.MLX,
        "macos-arm64-metal",
        Device.METAL,
        Route.UNPUBLISHED,
        _MAC_HOST,
        _MAC_LAST,
    ),
    _deferred(
        EngineName.OPENVINO,
        "linux-x86_64-openvino",
        Device.CPU,
        Route.UNPUBLISHED,
        _CPU_HOST,
        _NO_RECORD,
    ),
    _deferred(
        EngineName.OPENVINO,
        "linux-x86_64-openvino",
        Device.INTEL_GPU,
        Route.UNPUBLISHED,
        "the ThinkPad",
        "one hardware recording, no comparison record in git yet",
    ),
    _excluded(EngineName.OPENVINO, "linux-x86_64-openvino", Device.NPU, Route.UNPUBLISHED, _NO_NPU),
    _deferred(
        EngineName.OVMS,
        "linux-x86_64-openvino",
        Device.CPU,
        Route.USER_INSTALLED,
        _CPU_HOST,
        _NO_RECORD,
    ),
    _excluded(
        EngineName.OVMS, "linux-x86_64-openvino", Device.INTEL_GPU, Route.USER_INSTALLED, _NO_NPU
    ),
    _excluded(EngineName.OVMS, "linux-x86_64-openvino", Device.NPU, Route.USER_INSTALLED, _NO_NPU),
    _excluded(EngineName.LEMONADE, "linux-x86_64-npu", Device.NPU, Route.USER_INSTALLED, _NO_NPU),
    _excluded(EngineName.LEMONADE, "windows-x86_64-npu", Device.NPU, Route.USER_INSTALLED, _NO_NPU),
    _deferred(
        INFINITY,
        PRESET_CELL,
        Device.CPU,
        Route.PRESET,
        _CPU_HOST,
        "recorded in fixtures/presets and docs/preset-results.md; no llama.cpp comparison yet",
    ),
)
