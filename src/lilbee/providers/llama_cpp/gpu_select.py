"""Best-GPU autodetection for the Vulkan backend.

On a host with multiple GPUs (typical dual-GPU laptop: discrete NVIDIA
plus integrated AMD/Intel), Vulkan device ordering is driver- and
OS-dependent. llama.cpp's Vulkan backend enumerates all discrete AND
integrated adapters in the order Vulkan's ICD loader returns them
(see ``ggml-vulkan.cpp::ggml_vk_instance_init``: both
``eDiscreteGpu`` and ``eIntegratedGpu`` are added without sorting),
so a model can land on the integrated GPU and stall against shared
system memory.

This module probes the Vulkan loader directly via ``ctypes`` to
enumerate adapters, ranks them by ``VkPhysicalDeviceType`` (discrete
> integrated > virtual > CPU), and returns the index that should be
pinned via ``GGML_VK_VISIBLE_DEVICES``. Going through ``ctypes``
instead of a subprocess avoids any dependency on the Vulkan SDK
(``vulkaninfo`` isn't installed on stock Windows or macOS), so the
autodetect works on every machine that already has a Vulkan driver.

CUDA and ROCm enumeration are deliberately out of scope: CUDA only
sees NVIDIA devices and HIP/ROCm only sees AMD devices, so neither
backend exhibits the dual-GPU mis-pick problem. The Vulkan probe
result is applied to ``GGML_VK_VISIBLE_DEVICES`` alone; applying it
to ``CUDA_VISIBLE_DEVICES`` would risk hiding the only CUDA device
on a CUDA wheel + dual-GPU host.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import fnmatch
import logging
import ntpath
import os
import sys
from ctypes import POINTER, byref, c_char, c_char_p, c_uint8, c_uint32, c_void_p
from dataclasses import dataclass
from enum import IntEnum, StrEnum

from lilbee.providers.llama_cpp.vulkan_icd_registry import (
    iter_windows_vulkan_manifest_paths,
)

log = logging.getLogger(__name__)

# vk.h constants. Mirrored here so we don't drag a vulkan-headers
# dependency in for four magic numbers.
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
_VK_MAX_PHYSICAL_DEVICE_NAME_SIZE = 256
_VK_UUID_SIZE = 16


@dataclass(frozen=True)
class VulkanDevice:
    """One Vulkan adapter as reported by the loader."""

    index: int
    device_type: int
    device_name: str
    vendor_id: int


class PCIVendorID(IntEnum):
    """PCI-SIG vendor IDs for the GPU vendors that ship Vulkan ICDs.

    Values are the canonical PCI vendor IDs that ``VkPhysicalDeviceProperties.vendorID``
    surfaces. Only the vendors we have explicit ICD-disable globs for are
    enumerated; unknown vendors fall through the dispatch as no-op.
    """

    NVIDIA = 0x10DE
    AMD = 0x1002
    INTEL = 0x8086


# Vulkan loader manifest filename globs, per vendor. The loader matches these
# against the JSON manifest filename in its known-drivers list (see
# https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderInterfaceArchitecture.md).
# Each vendor ships under multiple names across drivers/OSes; list every form
# we may encounter so disabling one vendor's drivers doesn't half-disable them.
_VENDOR_ICD_GLOBS: dict[PCIVendorID, tuple[str, ...]] = {
    PCIVendorID.NVIDIA: ("nv*",),
    PCIVendorID.AMD: ("amdvlk*", "amd-vulkan*", "radeon*"),
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
    # Opaque to us; size pulled from vulkan_core.h so the parent struct
    # layout matches the driver-populated bytes.
    _fields_ = [("_opaque", c_uint8 * 504)]


class _VkPhysicalDeviceSparseProperties(ctypes.Structure):
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
    create_instance, destroy_instance, enum_physical, get_properties = _resolve_vk_symbols(lib)

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
            devices.append(
                VulkanDevice(
                    index=i,
                    device_type=int(props.deviceType),
                    device_name=props.deviceName.decode("utf-8", errors="replace"),
                    vendor_id=int(props.vendorID),
                )
            )
        return devices
    finally:
        destroy_instance(instance, None)


def _resolve_vk_symbols(
    lib: ctypes.CDLL,
) -> tuple[ctypes._FuncPointer, ctypes._FuncPointer, ctypes._FuncPointer, ctypes._FuncPointer]:
    """Look up the four Vulkan symbols this probe needs and stamp argtypes.

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

    return create_instance, destroy_instance, enum_physical, get_properties


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


_MIN_VENDORS_FOR_CONFLICT = 2
"""Number of distinct GPU vendors that must be present before pinning kicks in.

A single-vendor box (e.g., laptop with one AMD iGPU only) is not at risk:
only that vendor's ICD is loaded, no cross-vendor heap collision is possible.
"""


_PREFERRED_VENDOR_ORDER: tuple[PCIVendorID, ...] = (
    PCIVendorID.NVIDIA,
    PCIVendorID.AMD,
    PCIVendorID.INTEL,
)
"""Vendor pin priority for dual-vendor Windows boxes.

The documented heap-corruption signature is AMDVLK loaded alongside
NVIDIA in the same process (lilbee QA b473, Khronos forum, SHARK
Studio #1636). NVIDIA-first keeps the discrete card that's
overwhelmingly the user-visible "main GPU" on these boxes and disables
the AMD ICD that's behind the crashes. AMD-then-Intel is the natural
fall-through for AMD-discrete + Intel-iGPU laptops; Intel-only never
needs disabling.
"""


def _icds_to_disable(best: PCIVendorID, all_vendors: set[PCIVendorID]) -> list[str]:
    """Return the manifest globs for every known vendor except *best*."""
    globs: list[str] = []
    for vendor in sorted(all_vendors, key=int):
        if vendor is best:
            continue
        globs.extend(_VENDOR_ICD_GLOBS[vendor])
    return globs


def _classify_manifest_vendor(manifest_filename: str) -> PCIVendorID | None:
    """Map a Vulkan ICD manifest filename to its GPU vendor.

    Matches the bare filename (no directory) against ``_VENDOR_ICD_GLOBS``
    using case-insensitive ``fnmatch``. Returns ``None`` for filenames
    that don't match any vendor we know how to disable.
    """
    name = manifest_filename.lower()
    for vendor, globs in _VENDOR_ICD_GLOBS.items():
        for glob in globs:
            if fnmatch.fnmatchcase(name, glob.lower()):
                return vendor
    return None


def _windows_vulkan_vendors_present() -> set[PCIVendorID]:
    """Set of GPU vendors with at least one installed Vulkan ICD on this host.

    Pure-registry walk via :mod:`vulkan_icd_registry`, no Vulkan-loader
    involvement. The detection runs before the disable env var is set,
    so any ``vkCreateInstance`` call we triggered here would defeat the
    fix's purpose by pre-loading the very ICD we're trying to disable.
    Manifest filenames that don't match a known vendor glob are dropped:
    we have no glob to disable them with so listing them is moot.
    """
    vendors: set[PCIVendorID] = set()
    for manifest_path in iter_windows_vulkan_manifest_paths():
        # ntpath.basename (rather than os.path.basename) splits on '\\'
        # even when the test process runs on Linux/macOS; the registry
        # always reports Windows-style paths.
        filename = ntpath.basename(manifest_path)
        vendor = _classify_manifest_vendor(filename)
        if vendor is not None:
            vendors.add(vendor)
    return vendors


def _select_best_vendor(vendors: set[PCIVendorID]) -> PCIVendorID | None:
    """Pick the vendor to keep when several are present, by ``_PREFERRED_VENDOR_ORDER``."""
    for vendor in _PREFERRED_VENDOR_ORDER:
        if vendor in vendors:
            return vendor
    return None


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
#   - NVIDIA help article 5182, dual-vendor Vulkan apps on notebooks:
#     https://nvidia.custhelp.com/app/answers/detail/a_id/5182/
#   - Heroic Games Launcher ICD-selection issue (same mitigation pattern in prod):
#     https://github.com/Heroic-Games-Launcher/HeroicGamesLauncher/issues/3796
#   - Blender Vulkan backend startup failure on dual-vendor hosts:
#     https://projects.blender.org/blender/blender/issues/129917
def disable_conflicting_vulkan_icds() -> str | None:
    """Compute a ``VK_LOADER_DRIVERS_DISABLE`` value for dual-vendor Windows boxes.

    Returns a comma-separated glob list naming the ICD manifest filenames
    of every vendor *except* the preferred one (NVIDIA > AMD > Intel), or
    ``None`` when there's no conflict, the user has set any Vulkan
    ICD-selection env var, or registry enumeration finds at most one
    known vendor. The caller writes the returned value to
    ``VK_LOADER_DRIVERS_DISABLE`` so the Vulkan loader skips those ICDs
    at the next ``vkCreateInstance``.

    Detection reads installed ICD manifest paths from the Windows
    registry directly (legacy Khronos software key plus per-adapter PnP
    keys) so no Vulkan call runs while the disable env var is still
    unset. The earlier ``vkCreateInstance``-based probe pre-loaded every
    vendor's ICD into the process before the disable arrived, which
    defeated the fix on hosts where AMDVLK self-pins its DLL handle and
    survived ``FreeLibrary`` (lilbee QA b473 minidumps confirmed
    ``amdvlk64.dll`` still resident in 5 of 9 crash dumps).

    Windows-only: macOS uses Metal; Linux has the same loader mechanism
    but no reported crashes, so we don't pin there until a report drives
    it.
    """
    if sys.platform != "win32":
        return None
    if any(os.environ.get(env_var) for env_var in VulkanIcdEnvVar):
        return None
    vendors = _windows_vulkan_vendors_present()
    if len(vendors) < _MIN_VENDORS_FOR_CONFLICT:
        return None
    best = _select_best_vendor(vendors)
    if best is None:
        return None
    # ``vendors`` has >=2 members from the preferred set and ``best`` is one
    # of them, so ``_icds_to_disable`` returns a non-empty list here.
    return ",".join(_icds_to_disable(best, vendors))
