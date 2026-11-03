"""What the host can run: CPU instruction sets, NVIDIA GPUs, OS and architecture.

Each answer carries the reason an engine can show when the capability is absent.
This module is the one reader of the NVML bindings; it reads no device text itself.
"""

from __future__ import annotations

import platform
import re
from collections.abc import Callable, Collection, Sequence
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import ModuleType
from typing import Any, TypeVar

T = TypeVar("T")

NO_NVIDIA_DRIVER = "no NVIDIA driver"
NVIDIA_UNREADABLE = "the NVIDIA driver answers could not be read"
MACOS_UNREADABLE = "The macOS version of this computer could not be read."
_NVML_CUDA_MAJOR_DIVISOR = 1000
_NVML_CUDA_MINOR_DIVISOR = 10
_SYSFS_ACCEL_DIR = Path("/sys/class/accel")
NPU_UNREAD = "The NPU list of this computer could not be read."


class HostOS(StrEnum):
    """Operating systems an engine archive targets."""

    LINUX = "linux"
    MACOS = "macos"
    WINDOWS = "windows"


class HostArch(StrEnum):
    """CPU architectures an engine archive targets."""

    X86_64 = "x86_64"
    ARM64 = "arm64"


class NpuVendor(StrEnum):
    """Makers of the NPUs the host can report."""

    INTEL = "intel"
    AMD = "amd"
    QUALCOMM = "qualcomm"


# PCI vendor id of an accelerator, as sysfs writes it. Vendor ids are public PCI data. A
# Qualcomm id on Linux names any Qualcomm accelerator, a data-centre card among them.
_NPU_VENDOR_BY_PCI = {
    "0x8086": NpuVendor.INTEL,
    "0x1022": NpuVendor.AMD,
    "0x17cb": NpuVendor.QUALCOMM,
}
_NPU_DISPLAY = {
    NpuVendor.INTEL: "Intel NPU",
    NpuVendor.AMD: "AMD NPU",
    NpuVendor.QUALCOMM: "Qualcomm accelerator",
}
# AMD device id to generation. Only ids this table names have a generation; any other
# AMD id reads as an NPU of unknown generation. Unverified against the amdxdna driver.
_AMD_GENERATION_BY_DEVICE = {"0x1502": "XDNA", "0x17f0": "XDNA2"}

_OS_BY_SYSTEM = {"Linux": HostOS.LINUX, "Darwin": HostOS.MACOS, "Windows": HostOS.WINDOWS}
_ARCH_BY_MACHINE = {
    "x86_64": HostArch.X86_64,
    "AMD64": HostArch.X86_64,
    "arm64": HostArch.ARM64,
    "aarch64": HostArch.ARM64,
    "ARM64": HostArch.ARM64,
}


@dataclass(frozen=True)
class Feature:
    """One capability; ``reason`` says why it is absent or unknown.

    When ``known`` is false the host could not be read, and ``present`` says nothing.
    """

    present: bool
    reason: str = ""
    known: bool = True


@dataclass(frozen=True)
class NvidiaGpu:
    """One NVIDIA GPU and its CUDA compute capability."""

    name: str
    compute: tuple[int, int]


@dataclass(frozen=True)
class Npu:
    """One neural processing unit; ``generation`` is ``None`` when the host does not say."""

    vendor: NpuVendor
    name: str
    generation: str | None = None


@dataclass(frozen=True)
class HostCaps:
    """The capabilities of one host."""

    os: HostOS | None
    arch: HostArch | None
    avx2: Feature
    neon: Feature
    gpus: tuple[NvidiaGpu, ...]
    nvidia_driver_cuda: tuple[int, int] | None
    nvidia_reason: str
    nvidia_known: bool
    macos_version: tuple[int, ...] | None = None
    """The macOS version as numbers, or ``None`` off macOS and when it could not be read."""
    macos_reason: str = ""
    macos_known: bool = True
    """False only on macOS when the version could not be read; then ``macos_reason`` says so."""
    npus: tuple[Npu, ...] = ()
    npu_known: bool = False
    """False when the NPU list could not be read, as on Windows; ``npus`` then says nothing."""
    npu_reason: str = NPU_UNREAD


def _decoded(value: str | bytes) -> str:
    """NVML bindings return ``str`` on recent versions and ``bytes`` on older ones."""
    return value.decode() if isinstance(value, bytes) else value


def _open_nvml() -> ModuleType | None:
    """The initialised NVML bindings, or ``None`` when the library or driver is missing.

    Any other failure to initialise raises: the driver is present but its state is unknown.
    """
    try:
        import pynvml  # type: ignore[import-untyped]
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
    except (pynvml.NVMLError_LibraryNotFound, pynvml.NVMLError_DriverNotLoaded):
        return None
    module: ModuleType = pynvml
    return module


def _close_nvml(nvml: ModuleType) -> None:
    with suppress(Exception):  # a failed shutdown must not discard a finished read
        nvml.nvmlShutdown()


def nvml_device_totals() -> list[tuple[str, int]] | None:
    """``[(uuid, total_bytes), ...]`` in driver order, or ``None`` on any NVML failure."""
    try:
        nvml = _open_nvml()
    except Exception:
        return None
    if nvml is None:
        return None
    try:
        found = []
        for i in range(nvml.nvmlDeviceGetCount()):
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
            uuid = _decoded(nvml.nvmlDeviceGetUUID(handle))
            found.append((uuid, int(nvml.nvmlDeviceGetMemoryInfo(handle).total)))
    except Exception:
        return None
    finally:
        _close_nvml(nvml)
    return found or None


@dataclass(frozen=True)
class CardMemory:
    """The memory of one NVIDIA card, as the driver reports it now."""

    uuid: str
    name: str
    total_bytes: int
    free_bytes: int


def read_card_memory() -> list[CardMemory]:
    """Each NVIDIA card in driver order, in this process; empty on any NVML failure."""
    try:
        nvml = _open_nvml()
    except Exception:
        return []
    if nvml is None:
        return []
    try:
        cards = []
        for i in range(nvml.nvmlDeviceGetCount()):
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
            memory = nvml.nvmlDeviceGetMemoryInfo(handle)
            cards.append(
                CardMemory(
                    uuid=_decoded(nvml.nvmlDeviceGetUUID(handle)),
                    name=_decoded(nvml.nvmlDeviceGetName(handle)),
                    total_bytes=int(memory.total),
                    free_bytes=int(memory.free),
                )
            )
    except Exception:
        return []
    finally:
        _close_nvml(nvml)
    return cards


def _read_or_none(read: Callable[[], T]) -> T | None:
    """The value ``read`` returns, or ``None`` when the driver or binding fails the call."""
    try:
        return read()
    except Exception:
        return None


def _read_gpus(nvml: ModuleType) -> list[dict[str, Any]]:
    gpus = []
    for i in range(nvml.nvmlDeviceGetCount()):
        handle = nvml.nvmlDeviceGetHandleByIndex(i)
        name = _decoded(nvml.nvmlDeviceGetName(handle))
        gpus.append({"name": name, "cc": list(nvml.nvmlDeviceGetCudaComputeCapability(handle))})
    return gpus


def read_nvml() -> dict[str, Any] | None:
    """Driver version, driver CUDA version and GPUs from NVML.

    ``None`` means no NVML library or driver. A field that could not be read is ``None``.
    """
    try:
        nvml = _open_nvml()
    except Exception:
        return {"driver": None, "cuda_driver": None, "gpus": None}
    if nvml is None:
        return None
    try:
        return {
            "driver": _read_or_none(lambda: _decoded(nvml.nvmlSystemGetDriverVersion())),
            "cuda_driver": _read_or_none(lambda: int(nvml.nvmlSystemGetCudaDriverVersion())),
            "gpus": _read_or_none(lambda: _read_gpus(nvml)),
        }
    finally:
        _close_nvml(nvml)


def _cuda_version(code: object) -> tuple[int, int] | None:
    """NVML encodes CUDA 12.4 as 12040; anything else is unreadable."""
    if not isinstance(code, int) or isinstance(code, bool) or code <= 0:
        return None
    return code // _NVML_CUDA_MAJOR_DIVISOR, (
        code % _NVML_CUDA_MAJOR_DIVISOR
    ) // _NVML_CUDA_MINOR_DIVISOR


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _gpus(raw: Any) -> tuple[NvidiaGpu, ...] | None:
    """The GPUs in an NVML answer, or ``None`` when the answer is not readable."""
    try:
        gpus = tuple(NvidiaGpu(g["name"], (g["cc"][0], g["cc"][1])) for g in raw)
    except (TypeError, KeyError, IndexError):
        return None
    if not all(isinstance(g.name, str) and all(map(_is_int, g.compute)) for g in gpus):
        return None
    return gpus


def _avx2(arch: HostArch | None, cpu_info: dict[str, Any]) -> Feature:
    if arch is None:
        return Feature(False, "the CPU architecture could not be read", known=False)
    if arch is not HostArch.X86_64:
        return Feature(False, "not an x86 CPU")
    flags = cpu_info.get("flags")
    brand = cpu_info.get("brand_raw") or "this CPU"
    if not isinstance(flags, Collection) or isinstance(flags, str):
        return Feature(False, f"the CPU flags of {brand} could not be read", known=False)
    if "avx2" in flags:
        return Feature(True)
    return Feature(False, f"{brand} lacks AVX2")


def _neon(arch: HostArch | None) -> Feature:
    """NEON is part of every 64-bit ARM CPU; Apple Silicon reports no flags to read."""
    if arch is None:
        return Feature(False, "the CPU architecture could not be read", known=False)
    if arch is HostArch.ARM64:
        return Feature(True)
    return Feature(False, "not an ARM CPU")


_NUMBER = re.compile(r"\d+")


def version_numbers(version: str) -> tuple[int, ...]:
    """The numbers of a dotted version string; empty when it holds none."""
    return tuple(int(part) for part in _NUMBER.findall(version))


def _macos_version(host_os: HostOS | None, raw: str) -> tuple[int, ...] | None:
    """The numbers of a macOS version string, or ``None`` off macOS or when unparseable."""
    if host_os is not HostOS.MACOS:
        return None
    return version_numbers(raw) or None


def _npu(vendor_id: str, device_id: str) -> Npu | None:
    vendor = _NPU_VENDOR_BY_PCI.get(vendor_id.lower())
    if vendor is None:
        return None
    generation = (
        _AMD_GENERATION_BY_DEVICE.get(device_id.lower()) if vendor is NpuVendor.AMD else None
    )
    return Npu(vendor, _NPU_DISPLAY[vendor], generation)


def _npus(
    host_os: HostOS | None, accel: Sequence[tuple[str, str]] | None
) -> tuple[tuple[Npu, ...], bool, str]:
    """The NPUs, whether the list was read, and why not. Only Linux lists accelerators."""
    if host_os is not HostOS.LINUX:
        where = host_os.value if host_os else "this OS"
        return (), False, f"lilbee cannot read the NPU list on {where}."
    if accel is None:
        return (), False, NPU_UNREAD
    found = (_npu(vendor, device) for vendor, device in accel)
    return tuple(npu for npu in found if npu is not None), True, ""


def collect(
    *,
    system: str,
    machine: str,
    cpu_info: dict[str, Any],
    nvml: dict[str, Any] | None,
    mac_ver: str = "",
    accel_devices: Sequence[tuple[str, str]] | None = None,
) -> HostCaps:
    """Build the host answer from raw readings; a value that cannot be read is unknown."""
    host_os = _OS_BY_SYSTEM.get(system)
    arch = _ARCH_BY_MACHINE.get(machine)
    gpus = _gpus(nvml.get("gpus")) if nvml else ()
    cuda = _cuda_version(nvml.get("cuda_driver")) if nvml else None
    nvidia_known = gpus is not None and (cuda is not None or not nvml)
    macos_version = _macos_version(host_os, mac_ver)
    macos_known = host_os is not HostOS.MACOS or macos_version is not None
    npus, npu_known, npu_reason = _npus(host_os, accel_devices)
    return HostCaps(
        os=host_os,
        arch=arch,
        avx2=_avx2(arch, cpu_info),
        neon=_neon(arch),
        gpus=gpus or (),
        nvidia_driver_cuda=cuda,
        nvidia_reason=_nvidia_reason(nvml, nvidia_known),
        nvidia_known=nvidia_known,
        macos_version=macos_version,
        macos_reason="" if macos_known else MACOS_UNREADABLE,
        macos_known=macos_known,
        npus=npus,
        npu_known=npu_known,
        npu_reason=npu_reason,
    )


def _nvidia_reason(nvml: dict[str, Any] | None, known: bool) -> str:
    if not nvml:
        return NO_NVIDIA_DRIVER
    return "" if known else NVIDIA_UNREADABLE


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None


def _accel_devices() -> list[tuple[str, str]]:
    """``(vendor, device)`` PCI ids of each device in the Linux accelerator class."""
    try:
        folders = sorted(_SYSFS_ACCEL_DIR.glob("accel*/device"))
    except OSError:
        return []
    found = []
    for folder in folders:
        vendor, device = _read_text(folder / "vendor"), _read_text(folder / "device")
        if vendor is not None:
            found.append((vendor, device or ""))
    return found


def _cpu_info() -> dict[str, Any]:
    import cpuinfo

    info: dict[str, Any] = cpuinfo.get_cpu_info()
    return info


def detect() -> HostCaps:
    """Read the live host."""
    return collect(
        system=platform.system(),
        machine=platform.machine(),
        cpu_info=_read_or_none(_cpu_info) or {},
        nvml=read_nvml(),
        mac_ver=platform.mac_ver()[0],
        accel_devices=_accel_devices() if platform.system() == "Linux" else None,
    )
