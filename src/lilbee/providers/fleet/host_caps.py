"""What the host can run: CPU instruction sets, NVIDIA GPUs, NPUs, OS and architecture.

Each answer carries the reason an engine can show when the capability is absent.
This module is the one reader of the NVML bindings; it reads no device text itself.
"""

from __future__ import annotations

import platform
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import ModuleType
from typing import Any

NO_NVIDIA_DRIVER = "no NVIDIA driver"
_NVML_CUDA_MAJOR_DIVISOR = 1000
_NVML_CUDA_MINOR_DIVISOR = 10
_SYSFS_ACCEL_DIR = Path("/sys/class/accel")


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


# PCI vendor id of an accelerator, as sysfs writes it, to the NPU name reported.
_ACCEL_NPU_NAMES = {
    "0x8086": "Intel NPU",
    "0x1022": "AMD NPU",
    "0x17cb": "Qualcomm NPU",
}

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
    """One yes-or-no capability; ``reason`` says why it is absent."""

    present: bool
    reason: str = ""


@dataclass(frozen=True)
class NvidiaGpu:
    """One NVIDIA GPU and its CUDA compute capability."""

    name: str
    compute: tuple[int, int]


@dataclass(frozen=True)
class Npu:
    """One neural processing unit."""

    vendor: NpuVendor
    name: str


@dataclass(frozen=True)
class HostCaps:
    """The capabilities of one host."""

    os: HostOS
    arch: HostArch
    macos_version: tuple[int, ...] | None
    avx2: Feature
    neon: Feature
    gpus: tuple[NvidiaGpu, ...]
    nvidia_driver_cuda: tuple[int, int] | None
    nvidia_reason: str
    npus: tuple[Npu, ...]


def _decoded(value: str | bytes) -> str:
    """NVML bindings return ``str`` on recent versions and ``bytes`` on older ones."""
    return value.decode() if isinstance(value, bytes) else value


@contextmanager
def _nvml() -> Iterator[ModuleType | None]:
    """The initialised NVML bindings, or ``None`` when the library or driver is missing.

    An NVML error inside the block ends the block with the caller's answer unset.
    """
    try:
        import pynvml  # type: ignore[import-untyped]
    except ImportError:
        yield None
        return
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        yield None
        return
    try:
        yield pynvml
    except pynvml.NVMLError:
        # A mid-read driver error leaves the caller's answer unset, which reads as "no driver".
        pass
    finally:
        pynvml.nvmlShutdown()


def nvml_device_totals() -> list[tuple[str, int]] | None:
    """``[(uuid, total_bytes), ...]`` in driver order, or ``None`` without a usable NVML."""
    totals: list[tuple[str, int]] = []
    with _nvml() as nvml:
        if nvml is not None:
            found = []
            for i in range(nvml.nvmlDeviceGetCount()):
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                uuid = _decoded(nvml.nvmlDeviceGetUUID(handle))
                found.append((uuid, int(nvml.nvmlDeviceGetMemoryInfo(handle).total)))
            totals = found
    return totals or None


def read_nvml() -> dict[str, Any] | None:
    """Driver version, driver CUDA version and GPUs from NVML, or ``None`` without it."""
    answer: dict[str, Any] | None = None
    with _nvml() as nvml:
        if nvml is not None:
            gpus = []
            for i in range(nvml.nvmlDeviceGetCount()):
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                name = _decoded(nvml.nvmlDeviceGetName(handle))
                gpus.append(
                    {"name": name, "cc": list(nvml.nvmlDeviceGetCudaComputeCapability(handle))}
                )
            answer = {
                "driver": _decoded(nvml.nvmlSystemGetDriverVersion()),
                "cuda_driver": int(nvml.nvmlSystemGetCudaDriverVersion()),
                "gpus": gpus,
            }
    return answer


def _cuda_version(code: int) -> tuple[int, int]:
    """NVML encodes CUDA 12.4 as 12040."""
    return code // _NVML_CUDA_MAJOR_DIVISOR, (
        code % _NVML_CUDA_MAJOR_DIVISOR
    ) // _NVML_CUDA_MINOR_DIVISOR


def _avx2(arch: HostArch, cpu_info: dict[str, Any]) -> Feature:
    if arch is not HostArch.X86_64:
        return Feature(False, "not an x86 CPU")
    flags = cpu_info.get("flags")
    brand = cpu_info.get("brand_raw") or "this CPU"
    if flags is None:
        return Feature(False, f"the CPU flags of {brand} could not be read")
    if "avx2" in flags:
        return Feature(True)
    return Feature(False, f"{brand} lacks AVX2")


def _neon(arch: HostArch) -> Feature:
    """NEON is part of every 64-bit ARM CPU; Apple Silicon reports no flags to read."""
    if arch is HostArch.ARM64:
        return Feature(True)
    return Feature(False, "not an ARM CPU")


def _npu(name: str) -> Npu | None:
    lowered = name.lower()
    for vendor in NpuVendor:
        if vendor.value in lowered:
            return Npu(vendor, name)
    return None


def _macos_version(mac_ver: str) -> tuple[int, ...] | None:
    return tuple(int(part) for part in mac_ver.split(".")) if mac_ver else None


def collect(
    *,
    system: str,
    machine: str,
    mac_ver: str,
    cpu_info: dict[str, Any],
    nvml: dict[str, Any] | None,
    npu_devices: Sequence[str],
) -> HostCaps:
    """Build the host answer from raw readings."""
    host_os = _OS_BY_SYSTEM[system]
    arch = _ARCH_BY_MACHINE[machine]
    gpus = (
        tuple(NvidiaGpu(g["name"], (g["cc"][0], g["cc"][1])) for g in nvml["gpus"]) if nvml else ()
    )
    return HostCaps(
        os=host_os,
        arch=arch,
        macos_version=_macos_version(mac_ver) if host_os is HostOS.MACOS else None,
        avx2=_avx2(arch, cpu_info),
        neon=_neon(arch),
        gpus=gpus,
        nvidia_driver_cuda=_cuda_version(nvml["cuda_driver"]) if nvml else None,
        nvidia_reason="" if nvml else NO_NVIDIA_DRIVER,
        npus=tuple(npu for npu in map(_npu, npu_devices) if npu is not None),
    )


def _accel_device_names() -> list[str]:
    """Names of the NPUs the Linux accelerator class lists, from the PCI vendor of each."""
    names = []
    for vendor_file in sorted(_SYSFS_ACCEL_DIR.glob("accel*/device/vendor")):
        name = _ACCEL_NPU_NAMES.get(vendor_file.read_text(encoding="utf-8").strip().lower())
        if name is not None:
            names.append(name)
    return names


def detect() -> HostCaps:
    """Read the live host."""
    import cpuinfo

    return collect(
        system=platform.system(),
        machine=platform.machine(),
        mac_ver=platform.mac_ver()[0],
        cpu_info=cpuinfo.get_cpu_info(),
        nvml=read_nvml(),
        npu_devices=_accel_device_names(),
    )
