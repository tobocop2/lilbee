"""Best-GPU autodetection for the Vulkan backend.

Vulkan enumerates discrete and integrated adapters without sorting, so a model
can land on the integrated GPU. This module probes the loader via ``ctypes``,
ranks adapters by type (discrete > integrated > virtual > CPU), and returns the
index to pin via ``GGML_VK_VISIBLE_DEVICES``. CUDA/ROCm are out of scope: each
sees only its vendor's devices, so neither hits the dual-GPU mis-pick.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import fnmatch
import logging
import ntpath
import os
import sys
from ctypes import POINTER, byref, c_char, c_char_p, c_uint8, c_uint32, c_uint64, c_void_p
from dataclasses import dataclass
from enum import IntEnum, StrEnum

from lilbee.providers.fleet.vulkan_icd_discovery import (
    iter_vulkan_manifest_paths,
)

log = logging.getLogger(__name__)

# vk.h constants. Mirrored here so we don't drag a vulkan-headers
# dependency in for four magic numbers. See the upstream definitions in
# https://github.com/KhronosGroup/Vulkan-Headers/blob/main/include/vulkan/vulkan_core.h
# (VkStructureType enum and the VK_API_VERSION_1_0 / VK_SUCCESS macros).
_VK_STRUCTURE_TYPE_APPLICATION_INFO = 0
_VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
_VK_SUCCESS = 0
_VK_API_VERSION_1_0 = (1 << 22) | (0 << 12) | 0


class VkDeviceType(IntEnum):
    """``VkPhysicalDeviceType`` enum from vulkan_core.h.

    Values match the C ABI verbatim; the loader writes one of these
    into the ``deviceType`` field of ``VkPhysicalDeviceProperties``.
    """

    OTHER = 0
    INTEGRATED_GPU = 1
    DISCRETE_GPU = 2
    VIRTUAL_GPU = 3
    CPU = 4


# Preference order for picking the best adapter; higher is better.
# Software rendering (CPU) is never the right pick, so it ranks 0
# and ``_pick_best_device`` rejects it.
_DEVICE_TYPE_RANK: dict[VkDeviceType, int] = {
    VkDeviceType.DISCRETE_GPU: 4,
    VkDeviceType.INTEGRATED_GPU: 3,
    VkDeviceType.VIRTUAL_GPU: 2,
    VkDeviceType.OTHER: 1,
    VkDeviceType.CPU: 0,
}


def _rank_for(device_type: int) -> int:
    """Lookup the rank for a ``deviceType`` value, ``0`` if the driver returns an unknown one."""
    try:
        return _DEVICE_TYPE_RANK[VkDeviceType(device_type)]
    except ValueError:
        return 0


# vk.h sizes for the inline char arrays inside VkPhysicalDeviceProperties.
# Both constants are part of the Vulkan 1.0 ABI and frozen forever; see
# VK_MAX_PHYSICAL_DEVICE_NAME_SIZE and VK_UUID_SIZE in
# https://github.com/KhronosGroup/Vulkan-Headers/blob/main/include/vulkan/vulkan_core.h
_VK_MAX_PHYSICAL_DEVICE_NAME_SIZE = 256
_VK_UUID_SIZE = 16


@dataclass(frozen=True)
class VulkanDevice:
    """One Vulkan adapter as reported by the loader."""

    index: int
    device_type: int
    device_name: str
    vendor_id: int
    vram_bytes: int = 0


class PCIVendorID(IntEnum):
    """PCI-SIG vendor IDs for the GPU vendors that ship Vulkan ICDs.

    Values are the canonical PCI vendor IDs that
    ``VkPhysicalDeviceProperties.vendorID`` surfaces. They are issued by
    PCI-SIG and frozen per company; see the public PCI vendor-ID
    registry at https://pcisig.com/membership/member-companies (also
    mirrored at https://devicehunt.com/all-pci-vendors). Only the
    vendors we have explicit ICD-disable globs for are enumerated;
    unknown vendors fall through the dispatch as no-op.
    """

    NVIDIA = 0x10DE  # NVIDIA Corporation
    AMD = 0x1002  # Advanced Micro Devices, Inc. [AMD/ATI]
    INTEL = 0x8086  # Intel Corporation


# Vulkan loader manifest filename globs, per vendor. The loader matches these
# against the JSON manifest filename in its known-drivers list (see
# https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderInterfaceArchitecture.md).
# Each vendor ships under multiple names across drivers/OSes; list every form
# we may encounter so disabling one vendor's drivers doesn't half-disable them.
_VENDOR_ICD_GLOBS: dict[PCIVendorID, tuple[str, ...]] = {
    # nv-vk*.json (Windows), nvidia_*.json (Linux). Both match nv*.
    PCIVendorID.NVIDIA: ("nv*",),
    # amdvlk64.json (Windows AMDVLK), amd_icd*.json (Linux AMDVLK),
    # amd-vulkan*.json (legacy AMDVLK builds), radeon_icd.*.json
    # (Mesa RADV on Linux). Adding amd_icd* explicitly because no
    # other glob covers the Linux AMDVLK manifest.
    PCIVendorID.AMD: ("amdvlk*", "amd_icd*", "amd-vulkan*", "radeon*"),
    # intel_icd.*.json (Mesa Intel ANV on Linux), igvk*.json (Windows).
    PCIVendorID.INTEL: ("intel*", "igvk*"),
}


class VulkanIcdEnvVar(StrEnum):
    """Every documented Vulkan loader env var that influences ICD selection.

    Names are the verbatim loader env vars from the Khronos
    LoaderInterfaceArchitecture spec; the StrEnum lets each member be
    used directly as a ``str`` argument to ``os.environ.get`` /
    ``os.environ.setdefault`` without ``.value`` plumbing. Any value
    being non-empty in the environment is treated as a user override
    and suppresses the dual-vendor auto-pin.
    """

    DRIVER_FILES = "VK_DRIVER_FILES"
    ICD_FILENAMES = "VK_ICD_FILENAMES"
    ADD_DRIVER_FILES = "VK_ADD_DRIVER_FILES"
    LOADER_DRIVERS_DISABLE = "VK_LOADER_DRIVERS_DISABLE"
    LOADER_DRIVERS_SELECT = "VK_LOADER_DRIVERS_SELECT"


# Field layouts from the Vulkan 1.0 spec. ctypes maps the C structs
# verbatim so the loader populates them directly; only the prefix
# fields we read are commented (the trailing fields are kept for ABI
# alignment, not consumed).


class _VkApplicationInfo(ctypes.Structure):
    _fields_ = [
        ("sType", c_uint32),
        ("pNext", c_void_p),
        ("pApplicationName", c_char_p),
        ("applicationVersion", c_uint32),
        ("pEngineName", c_char_p),
        ("engineVersion", c_uint32),
        ("apiVersion", c_uint32),
    ]


class _VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", c_uint32),
        ("pNext", c_void_p),
        ("flags", c_uint32),
        ("pApplicationInfo", POINTER(_VkApplicationInfo)),
        ("enabledLayerCount", c_uint32),
        ("ppEnabledLayerNames", POINTER(c_char_p)),
        ("enabledExtensionCount", c_uint32),
        ("ppEnabledExtensionNames", POINTER(c_char_p)),
    ]


class _VkPhysicalDeviceLimits(ctypes.Structure):
    # Opaque to us; we only need the parent struct's *layout* to match
    # the driver-populated bytes so the loader can write a vendorID and
    # deviceType into the prefix fields we actually read.
    #
    # Size = sum of every field in VkPhysicalDeviceLimits in
    # https://github.com/KhronosGroup/Vulkan-Headers/blob/main/include/vulkan/vulkan_core.h
    # (104 ULONG32s, plus alignment padding, totals 504 bytes for the
    # Vulkan 1.0 ABI). The number is part of the frozen Vulkan 1.0 layout
    # so it doesn't drift across driver versions.
    _fields_ = [("_opaque", c_uint8 * 504)]


class _VkPhysicalDeviceSparseProperties(ctypes.Structure):
    # 5 ULONG32 booleans, also part of the Vulkan 1.0 ABI; see same header.
    _fields_ = [("_opaque", c_uint32 * 5)]


class _VkPhysicalDeviceProperties(ctypes.Structure):
    _fields_ = [
        ("apiVersion", c_uint32),
        ("driverVersion", c_uint32),
        ("vendorID", c_uint32),
        ("deviceID", c_uint32),
        ("deviceType", c_uint32),
        ("deviceName", c_char * _VK_MAX_PHYSICAL_DEVICE_NAME_SIZE),
        ("pipelineCacheUUID", c_uint8 * _VK_UUID_SIZE),
        ("limits", _VkPhysicalDeviceLimits),
        ("sparseProperties", _VkPhysicalDeviceSparseProperties),
    ]


# VkPhysicalDeviceMemoryProperties layout (Vulkan 1.0 ABI, frozen). Array
# bounds and the device-local heap flag are from vulkan_core.h. The
# device-local heap size is the cross-vendor VRAM signal (the same heap
# nvidia-smi/rocm-smi report) used for placement bin-packing.
_VK_MAX_MEMORY_TYPES = 32
_VK_MAX_MEMORY_HEAPS = 16
_VK_MEMORY_HEAP_DEVICE_LOCAL_BIT = 0x00000001


class _VkMemoryType(ctypes.Structure):
    _fields_ = [("propertyFlags", c_uint32), ("heapIndex", c_uint32)]


class _VkMemoryHeap(ctypes.Structure):
    _fields_ = [("size", c_uint64), ("flags", c_uint32)]


class _VkPhysicalDeviceMemoryProperties(ctypes.Structure):
    _fields_ = [
        ("memoryTypeCount", c_uint32),
        ("memoryTypes", _VkMemoryType * _VK_MAX_MEMORY_TYPES),
        ("memoryHeapCount", c_uint32),
        ("memoryHeaps", _VkMemoryHeap * _VK_MAX_MEMORY_HEAPS),
    ]


def autoselect_best_gpu_index() -> str | None:
    """Return the Vulkan device index of the best-available adapter, or ``None``.

    Returns ``None`` when the Vulkan loader is unavailable, the probe
    fails, or only one adapter is visible (no decision to make). The
    string format matches ``GGML_VK_VISIBLE_DEVICES`` (``"0"`` /
    ``"1"`` etc.). CUDA / HIP / ROCm enumeration are out of scope:
    those backends are single-vendor and the env vars don't mean the
    same thing as the Vulkan loader's enumeration order.
    """
    devices = _enumerate_vulkan_devices()
    if devices is None:
        return None
    best = _pick_best_device(devices)
    if best is None:
        return None
    # Only emit a pin when there's a real choice between adapter types:
    # if every visible device has the same rank, the loader's default
    # ordering is already correct and forcing the index would hide a
    # user's manual override on rebuild.
    ranks = {_rank_for(d.device_type) for d in devices}
    if len(ranks) <= 1:
        return None
    return str(best.index)


def enumerate_gpu_vram() -> list[tuple[int, int]] | None:
    """Return ``[(device_index, device_local_vram_bytes), ...]`` or ``None``.

    Cross-vendor via the Vulkan probe (NVIDIA/AMD/Intel). ``None`` when the
    loader/probe is unavailable (macOS Metal, no Vulkan driver), so the
    placement planner can degrade to count-only or in-process.
    """
    devices = _enumerate_vulkan_devices()
    if devices is None:
        return None
    return [(d.index, d.vram_bytes) for d in devices]


def _enumerate_vulkan_devices() -> list[VulkanDevice] | None:
    """Open libvulkan, create a throwaway instance, enumerate adapters.

    Returns ``None`` if the loader can't be found or any Vulkan call
    fails; empty list ("loader present, no adapters") is a distinct
    outcome and propagates back. The bootstrap calls this twice
    (autoselect plus the dual-vendor ICD pin) at process startup; the
    Vulkan probe is ms-scale, no caching needed.
    """
    lib = _load_vulkan_loader()
    if lib is None:
        return None
    try:
        return _list_devices_with_instance(lib)
    except OSError:
        # ctypes argument / call-site errors land here; treat as
        # "probe failed" rather than crashing the host process.
        return None


def _load_vulkan_loader() -> ctypes.CDLL | None:
    """Locate and load the Vulkan loader for the current platform.

    Returns ``None`` when the loader isn't installed, which is the
    expected outcome on stock macOS (we ship a Metal wheel there) and
    on hosts without a Vulkan-capable driver.
    """
    candidates: tuple[str, ...]
    if sys.platform == "win32":
        candidates = ("vulkan-1.dll",)
    elif sys.platform == "darwin":
        # MoltenVK exposes a different ABI than libvulkan; lilbee's
        # macOS wheel uses Metal directly, so skipping the probe on
        # Darwin is correct.
        return None
    else:
        candidates = ("libvulkan.so.1", "libvulkan.so")

    for name in candidates:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    # ctypes.util.find_library is a last-resort fallback for distros
    # where the soname isn't directly loadable.
    resolved = ctypes.util.find_library("vulkan")
    if resolved is not None:
        try:
            return ctypes.CDLL(resolved)
        except OSError:
            return None
    return None


def _list_devices_with_instance(lib: ctypes.CDLL) -> list[VulkanDevice]:
    """Create a temporary VkInstance, enumerate physical devices, destroy.

    Mirrors what ``vulkaninfo --summary`` does internally. The
    instance is short-lived (created and destroyed in the same call)
    so the probe leaves no driver state behind.
    """
    (
        create_instance,
        destroy_instance,
        enum_physical,
        get_properties,
        get_memory,
    ) = _resolve_vk_symbols(lib)

    app_info = _VkApplicationInfo(
        sType=_VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pNext=None,
        pApplicationName=b"lilbee-gpu-probe",
        applicationVersion=0,
        pEngineName=b"lilbee",
        engineVersion=0,
        apiVersion=_VK_API_VERSION_1_0,
    )
    create_info = _VkInstanceCreateInfo(
        sType=_VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pNext=None,
        flags=0,
        pApplicationInfo=ctypes.pointer(app_info),
        enabledLayerCount=0,
        ppEnabledLayerNames=None,
        enabledExtensionCount=0,
        ppEnabledExtensionNames=None,
    )
    instance = c_void_p()
    result = create_instance(byref(create_info), None, byref(instance))
    if result != _VK_SUCCESS or not instance.value:
        return []

    try:
        count = c_uint32(0)
        result = enum_physical(instance, byref(count), None)
        if result != _VK_SUCCESS or count.value == 0:
            return []
        handles = (c_void_p * count.value)()
        result = enum_physical(instance, byref(count), handles)
        if result != _VK_SUCCESS:
            return []
        devices: list[VulkanDevice] = []
        for i in range(count.value):
            props = _VkPhysicalDeviceProperties()
            get_properties(handles[i], byref(props))
            mem = _VkPhysicalDeviceMemoryProperties()
            get_memory(handles[i], byref(mem))
            devices.append(
                VulkanDevice(
                    index=i,
                    device_type=int(props.deviceType),
                    device_name=props.deviceName.decode("utf-8", errors="replace"),
                    vendor_id=int(props.vendorID),
                    vram_bytes=_device_local_vram(mem),
                )
            )
        return devices
    finally:
        destroy_instance(instance, None)


def _resolve_vk_symbols(
    lib: ctypes.CDLL,
) -> tuple[
    ctypes._FuncPointer,
    ctypes._FuncPointer,
    ctypes._FuncPointer,
    ctypes._FuncPointer,
    ctypes._FuncPointer,
]:
    """Look up the five Vulkan symbols this probe needs and stamp argtypes.

    All argtypes / restypes are set here so ctypes uses the same
    calling convention as the C ABI; missing this on Windows produces
    silent stack corruption.
    """
    create_instance = lib.vkCreateInstance
    create_instance.argtypes = [
        POINTER(_VkInstanceCreateInfo),
        c_void_p,
        POINTER(c_void_p),
    ]
    create_instance.restype = c_uint32

    destroy_instance = lib.vkDestroyInstance
    destroy_instance.argtypes = [c_void_p, c_void_p]
    destroy_instance.restype = None

    enum_physical = lib.vkEnumeratePhysicalDevices
    enum_physical.argtypes = [c_void_p, POINTER(c_uint32), POINTER(c_void_p)]
    enum_physical.restype = c_uint32

    get_properties = lib.vkGetPhysicalDeviceProperties
    get_properties.argtypes = [c_void_p, POINTER(_VkPhysicalDeviceProperties)]
    get_properties.restype = None

    get_memory = lib.vkGetPhysicalDeviceMemoryProperties
    get_memory.argtypes = [c_void_p, POINTER(_VkPhysicalDeviceMemoryProperties)]
    get_memory.restype = None

    return create_instance, destroy_instance, enum_physical, get_properties, get_memory


def _device_local_vram(mem_props: _VkPhysicalDeviceMemoryProperties) -> int:
    """Sum the device-local heap sizes (bytes), the cross-vendor VRAM signal."""
    total = 0
    for i in range(mem_props.memoryHeapCount):
        heap = mem_props.memoryHeaps[i]
        if heap.flags & _VK_MEMORY_HEAP_DEVICE_LOCAL_BIT:
            total += int(heap.size)
    return total


def _pick_best_device(devices: list[VulkanDevice]) -> VulkanDevice | None:
    """Return the highest-ranked device, preferring lower indexes on ties.

    Sort is stable so the loader's enumeration order acts as the
    tie-breaker; this matches user expectation that "device 0" wins
    when two adapters are the same type.
    """
    if not devices:
        return None
    ranked = sorted(devices, key=lambda d: (-_rank_for(d.device_type), d.index))
    best = ranked[0]
    if _rank_for(best.device_type) <= 0:
        return None
    return best


# Single-vendor boxes don't need a pin -- only that vendor's ICD loads,
# no cross-vendor collision possible.
_MIN_VENDORS_FOR_CONFLICT = 2

# Pin priority on dual-vendor hosts. NVIDIA wins because the documented
# crash signature is AMDVLK alongside NVIDIA (Khronos forum,
# SHARK-Studio#1636) and NVIDIA is the more common dGPU on those boxes.
# AMD-then-Intel covers AMD-discrete + Intel-iGPU laptops.
_PREFERRED_VENDOR_ORDER: tuple[PCIVendorID, ...] = (
    PCIVendorID.NVIDIA,
    PCIVendorID.AMD,
    PCIVendorID.INTEL,
)


def _icds_to_disable(best: PCIVendorID, all_vendors: set[PCIVendorID]) -> list[str]:
    """Return the manifest globs for every known vendor except *best*."""
    globs: list[str] = []
    for vendor in sorted(all_vendors, key=int):
        if vendor is best:
            continue
        globs.extend(_VENDOR_ICD_GLOBS[vendor])
    return globs


def _classify_manifest_vendor(manifest_filename: str) -> PCIVendorID | None:
    """Map a manifest filename to its GPU vendor via ``_VENDOR_ICD_GLOBS``."""
    name = manifest_filename.lower()
    for vendor, globs in _VENDOR_ICD_GLOBS.items():
        for glob in globs:
            if fnmatch.fnmatchcase(name, glob.lower()):
                return vendor
    return None


def _vulkan_vendors_present() -> set[PCIVendorID]:
    """Vendors with at least one installed Vulkan ICD on this host."""
    vendors: set[PCIVendorID] = set()
    for manifest_path in iter_vulkan_manifest_paths():
        # ntpath.basename splits on both '\\' and '/', so it handles
        # Windows-registry paths and Linux Path.__str__() output uniformly.
        filename = ntpath.basename(manifest_path)
        vendor = _classify_manifest_vendor(filename)
        if vendor is not None:
            vendors.add(vendor)
    return vendors


def _select_best_vendor(vendors: set[PCIVendorID]) -> PCIVendorID | None:
    """First match against ``_PREFERRED_VENDOR_ORDER``, or ``None`` if empty."""
    for vendor in _PREFERRED_VENDOR_ORDER:
        if vendor in vendors:
            return vendor
    return None


def _platform_supports_icd_pin() -> bool:
    """True on Windows + Linux, where dual-vendor ICD crashes are documented."""
    return sys.platform == "win32" or sys.platform.startswith("linux")


# References for the dual-vendor ICD mitigation below:
#   - Khronos Vulkan-Loader env var spec (VK_LOADER_DRIVERS_DISABLE / VK_DRIVER_FILES):
#     https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderInterfaceArchitecture.md
#   - ICD manifest filename conventions and Windows registry discovery order:
#     https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderDriverInterface.md
#   - "Failure in one ICD causes total failure of vkEnumeratePhysicalDevices":
#     https://github.com/KhronosGroup/Vulkan-Loader/issues/1467
#   - Khronos forum: amdvlk64.dll crashes in vkCreateInstance on mixed-vendor hosts:
#     https://community.khronos.org/t/crash-in-amdvlk64-dll-during-vkcreateinstance/105022
#   - SHARK-Studio #1636 (the same crash hits another Python ML inference tool):
#     https://github.com/nod-ai/SHARK-Studio/issues/1636
#   - Steam overlay multi-VkDevice crash on Linux (ValveSoftware/steam-for-linux#9120):
#     https://github.com/ValveSoftware/steam-for-linux/issues/9120
#   - Mesa RADV pipeline-creation heap corruption (ggml-org/llama.cpp#22128):
#     https://github.com/ggml-org/llama.cpp/issues/22128
#   - NVIDIA help article 5182, dual-vendor Vulkan apps on notebooks:
#     https://nvidia.custhelp.com/app/answers/detail/a_id/5182/
#   - Heroic Games Launcher ICD-selection issue (same mitigation pattern in prod):
#     https://github.com/Heroic-Games-Launcher/HeroicGamesLauncher/issues/3796
#   - Blender Vulkan backend startup failure on dual-vendor hosts:
#     https://projects.blender.org/blender/blender/issues/129917
def disable_conflicting_vulkan_icds() -> str | None:
    """Manifest-filename glob list of non-preferred ICDs to disable, or ``None``.

    Preferred-vendor order is NVIDIA > AMD > Intel. Returns ``None`` when the
    user has pinned a GPU, when fewer than two vendors are present, or when the
    platform has no documented dual-vendor crash class. Discovery reads manifests
    from disk (registry on Windows, XDG on Linux); enumerating via
    ``vkCreateInstance`` would pre-load every vendor's ICD before the disable lands.
    """
    from lilbee.core.config import cfg

    if not _platform_supports_icd_pin():
        return None
    if any(os.environ.get(env_var) for env_var in VulkanIcdEnvVar):
        return None
    if cfg.gpu_devices:
        return None
    vendors = _vulkan_vendors_present()
    if len(vendors) < _MIN_VENDORS_FOR_CONFLICT:
        return None
    best = _select_best_vendor(vendors)
    if best is None:  # pragma: no cover - invariant: vendors is non-empty here
        return None
    return ",".join(_icds_to_disable(best, vendors))
