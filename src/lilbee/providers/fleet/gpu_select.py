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
from collections import Counter
from ctypes import POINTER, byref, c_char, c_char_p, c_uint8, c_uint32, c_uint64, c_void_p
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from functools import lru_cache

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
# 1.1 is asked for first, purely to make vkGetPhysicalDeviceProperties2 (and the
# device UUID it carries) core rather than an extension; a loader that refuses
# it gets the 1.0 request back and the probe simply has no UUIDs to dedup by.
_VK_API_VERSION_1_1 = (1 << 22) | (1 << 12) | 0
_VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 = 1000059000
_VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 = 1000059001
_VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES = 1000071004
_VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES = 1000083000
_VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2 = 1000059006
_VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT = 1000237000
# The device extension that turns heap sizes into a live budget. Without it the
# only figure available is the heap's capacity, which never moves.
_VK_EXT_MEMORY_BUDGET = b"VK_EXT_memory_budget"
_VK_MAX_EXTENSION_NAME_SIZE = 256


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


# The device types ggml's Vulkan backend will actually run on. Anything else --
# a software rasterizer, a paravirtual adapter, an unknown type -- is not a
# device the engine would choose, so planning against one guarantees a mismatch.
USABLE_VULKAN_TYPES = frozenset({VkDeviceType.DISCRETE_GPU, VkDeviceType.INTEGRATED_GPU})


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
    # VkPhysicalDeviceIDProperties::deviceUUID, empty when the loader could not
    # be asked for it. The spec requires it to be immutable for a given device
    # across instances, processes, driver APIs, driver versions and reboots, so
    # two entries sharing one is one piece of silicon behind two drivers.
    device_uuid: bytes = b""
    # storageBuffer16BitAccess, the single feature ggml's Vulkan backend requires
    # of a device before it will use it. Read from
    # VkPhysicalDevice16BitStorageFeatures rather than the
    # VkPhysicalDeviceVulkan11Features ggml itself uses: same bit, but the latter
    # arrived in Vulkan 1.2 and this probe asks for a 1.1 instance.
    # ``None`` when the loader could not be asked, which is not a refusal.
    storage_buffer_16bit: bool | None = None
    # Device-local memory not already committed, from VK_EXT_memory_budget.
    # ``None`` when the device does not expose that extension, which is the
    # difference between "nothing else is using this card" and "cannot tell".
    free_bytes: int | None = None


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
    # 504 bytes, per VkPhysicalDeviceLimits in
    # https://github.com/KhronosGroup/Vulkan-Headers/blob/main/include/vulkan/vulkan_core.h
    # The size is part of the frozen Vulkan 1.0 layout, so it does not drift
    # across driver versions.
    #
    # Declared as uint64 rather than bytes for its ALIGNMENT, not its size. The
    # real struct mixes uint32, uint64 (VkDeviceSize), size_t and float, so its C
    # alignment is 8. A c_uint8 array aligns to 1, which let ctypes seat this
    # field at offset 292 in the parent instead of the 296 the ABI pads it to,
    # making the mirror 816 bytes against the driver's 824. The driver fills the
    # caller's buffer using its own layout, so every probe wrote sparseProperties
    # four bytes past the end of a Python-heap allocation -- absorbed by allocator
    # slack, which is what kept it silent.
    _fields_ = [("_opaque", c_uint64 * 63)]


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


class _VkPhysicalDevice16BitStorageFeatures(ctypes.Structure):
    # Promoted to core in Vulkan 1.1 from VK_KHR_16bit_storage. Chained onto
    # VkPhysicalDeviceFeatures2; only the first flag is read.
    _fields_ = [
        ("sType", c_uint32),
        ("pNext", c_void_p),
        ("storageBuffer16BitAccess", c_uint32),
        ("uniformAndStorageBuffer16BitAccess", c_uint32),
        ("storagePushConstant16", c_uint32),
        ("storageInputOutput16", c_uint32),
    ]


class _VkPhysicalDeviceFeatures2(ctypes.Structure):
    # VkPhysicalDeviceFeatures is a flat run of VkBool32s whose count grows with
    # no version of the spec but is easy to miscount, and the driver writes the
    # whole thing into this buffer. Declared larger than the real struct so a
    # miscount cannot become a heap overrun the way the limits mirror once did;
    # the field sits last, so the extra words shift nothing the driver reads.
    _fields_ = [
        ("sType", c_uint32),
        ("pNext", c_void_p),
        ("features", c_uint32 * 128),
    ]


class _VkExtensionProperties(ctypes.Structure):
    _fields_ = [
        ("extensionName", c_char * _VK_MAX_EXTENSION_NAME_SIZE),
        ("specVersion", c_uint32),
    ]


class _VkPhysicalDeviceIDProperties(ctypes.Structure):
    # VkPhysicalDeviceIDProperties, promoted to core in Vulkan 1.1. Chained onto
    # VkPhysicalDeviceProperties2 via pNext; the driver fills every field, so the
    # trailing ones are declared for layout even though only deviceUUID is read.
    _fields_ = [
        ("sType", c_uint32),
        ("pNext", c_void_p),
        ("deviceUUID", c_uint8 * _VK_UUID_SIZE),
        ("driverUUID", c_uint8 * _VK_UUID_SIZE),
        ("deviceLUID", c_uint8 * 8),
        ("deviceNodeMask", c_uint32),
        ("deviceLUIDValid", c_uint32),
    ]


class _VkPhysicalDeviceProperties2(ctypes.Structure):
    _fields_ = [
        ("sType", c_uint32),
        ("pNext", c_void_p),
        ("properties", _VkPhysicalDeviceProperties),
    ]


class _VkPhysicalDeviceMemoryProperties(ctypes.Structure):
    _fields_ = [
        ("memoryTypeCount", c_uint32),
        ("memoryTypes", _VkMemoryType * _VK_MAX_MEMORY_TYPES),
        ("memoryHeapCount", c_uint32),
        ("memoryHeaps", _VkMemoryHeap * _VK_MAX_MEMORY_HEAPS),
    ]


class _VkPhysicalDeviceMemoryProperties2(ctypes.Structure):
    _fields_ = [
        ("sType", c_uint32),
        ("pNext", c_void_p),
        ("memoryProperties", _VkPhysicalDeviceMemoryProperties),
    ]


class _VkPhysicalDeviceMemoryBudgetPropertiesEXT(ctypes.Structure):
    # heapBudget is what this process may still allocate from each heap and
    # heapUsage what it already has; the difference across the device-local heaps
    # is the only cross-vendor figure that moves when another process takes VRAM.
    _fields_ = [
        ("sType", c_uint32),
        ("pNext", c_void_p),
        ("heapBudget", c_uint64 * _VK_MAX_MEMORY_HEAPS),
        ("heapUsage", c_uint64 * _VK_MAX_MEMORY_HEAPS),
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


def enumerate_gpu_vram() -> list[tuple[int, int, int]] | None:
    """Return ``[(device_index, device_local_vram_bytes, free_bytes), ...]`` or ``None``.

    Cross-vendor via the Vulkan probe (NVIDIA/AMD/Intel). ``None`` when the
    loader/probe is unavailable (macOS Metal, no Vulkan driver), so the
    placement planner can degrade to count-only or in-process.

    Only discrete and integrated adapters are returned, the same rule ggml's
    Vulkan backend applies when it picks a device, so this cannot offer
    placement something the engine would refuse to run on. Matching that rule
    is the point: where the two disagree about which devices exist, placement
    sizes against a device llama-server never uses.

    Two kinds are excluded. Mesa's llvmpipe is a software rasterizer that
    advertises itself through Vulkan and reports system RAM as its device
    memory, so beside integrated graphics it appears at an identical size and
    is indistinguishable by VRAM alone; planning against it splits the model
    across a real GPU and a CPU renderer. Paravirtual adapters (virgl, VMware,
    VirtIO-GPU) report as ``VIRTUAL_GPU`` and are typically compute-incapable
    or proxies that fail on allocation.

    The device type is the only signal separating any of these, and the caller
    has no access to it: this returns sizes, and the ``--list-devices`` text it
    feeds carries no names on the fallback path.
    """
    devices = _enumerate_vulkan_devices()
    if devices is None:
        return None
    return [
        (d.index, d.vram_bytes, d.free_bytes if d.free_bytes is not None else d.vram_bytes)
        for d in devices
        if d.device_type in USABLE_VULKAN_TYPES
    ]


@lru_cache(maxsize=1)
def integrated_vulkan_indices() -> frozenset[int]:
    """Loader indices of adapters whose memory is the host's.

    Empty when the loader is unavailable or the probe fails, which reads as
    "assume dedicated" and preserves the behaviour discrete hosts already have.

    Cached because the device parser asks per device line: without it an
    N-device host paid N loader loads and N instance creations to answer the
    same question. Which adapters are integrated is a property of the machine,
    so one answer per process is right.
    """
    devices = _enumerate_vulkan_devices()
    if not devices:
        return frozenset()
    return frozenset(d.index for d in devices if d.device_type == VkDeviceType.INTEGRATED_GPU)


def vulkan_free_bytes_by_name() -> dict[str, int]:
    """Device-local memory still free, keyed by the name the loader reports.

    Deliberately not cached: free memory is a live number, and freezing it for
    the process lifetime would hand every later probe the first reading taken.
    Callers sample it once per parse rather than per device line.

    Only devices whose driver exposes ``VK_EXT_memory_budget`` appear; the rest
    have no live figure to offer and are absent rather than guessed at. A name
    two adapters share is also absent: two identical cards have their own free
    figures, and nothing in the engine's text says which line is which. Guessing
    would report one card's headroom for the other.
    """
    devices = _enumerate_vulkan_devices()
    if not devices:
        return {}
    seen = Counter(d.device_name for d in devices)
    return {
        d.device_name: d.free_bytes
        for d in devices
        if d.free_bytes is not None and seen[d.device_name] == 1
    }


@lru_cache(maxsize=1)
def vulkan_device_types_by_name() -> dict[str, VkDeviceType]:
    """Adapter type keyed by the name the loader reports, empty when unavailable.

    Keyed by name rather than index because the engine's ``--list-devices``
    ordinals are assigned after ggml has filtered and deduplicated the loader's
    list, so ``Vulkan0`` is only the loader's device 0 when nothing ahead of it
    was dropped. The name is the one field both views print verbatim from
    ``VkPhysicalDeviceProperties``, so it correlates the two without either
    side having to replicate the other's filtering.

    Two adapters of the same model share a name, which is harmless: they share
    a type too, and the type is all this answers.
    """
    devices = _enumerate_vulkan_devices()
    if not devices:
        return {}
    return {
        d.device_name: device_type
        for d in devices
        if (device_type := _known_device_type(d.device_type)) is not None
    }


def discrete_gpu_from_vendor(vendor_id: int) -> bool | None:
    """Whether the loader reports a discrete adapter from *vendor_id*.

    ``None`` when the loader cannot be reached, which is a different answer from
    "no": a caller deciding whether to fail loud must not read silence as proof
    that a card is absent, nor as proof that one is present.
    """
    devices = _enumerate_vulkan_devices()
    if not devices:
        return None
    return any(
        d.vendor_id == vendor_id and d.device_type == VkDeviceType.DISCRETE_GPU for d in devices
    )


def host_has_no_discrete_gpu() -> bool:
    """Whether the Vulkan loader can see adapters and none of them is discrete.

    The vendor-neutral answer to a question CUDA and ROCm cannot be asked
    through text: their ``--list-devices`` lines carry no device type, so an AMD
    APU and a Jetson enumerate exactly like a discrete card while reporting
    system RAM as their memory. Every such part also ships a Vulkan driver, and
    a machine whose loader reports adapters but no discrete one has no discrete
    GPU for CUDA or ROCm to be enumerating.

    The verdict rests on an integrated adapter actually being there. Software
    rasterizers report through the loader on any host with mesa installed, even
    with no vendor ICD present at all, which is ordinary on headless CUDA boxes
    and in containers; concluding from a list that holds only those would mark a
    real discrete card as sharing the host's memory and shrink its budget.

    False when the loader is unreachable, when any discrete adapter exists, or
    when nothing but rasterizers answered, so a host with a real card is never
    talked into the shared-memory budget. A host holding both a discrete card
    and an APU also answers False, which leaves the APU sized as dedicated;
    correlating individual devices across two backends' naming needs more than
    the type.
    """
    types = set(vulkan_device_types_by_name().values())
    if VkDeviceType.DISCRETE_GPU in types:
        return False
    return VkDeviceType.INTEGRATED_GPU in types


def _known_device_type(value: int) -> VkDeviceType | None:
    """The enum member for a raw ``deviceType``, ``None`` for a value vk.h doesn't define."""
    try:
        return VkDeviceType(value)
    except ValueError:
        return None


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
        devices = _list_devices_with_instance(lib)
        return _deduplicate_by_uuid(_drop_devices_the_engine_refuses(devices))
    except OSError:
        # ctypes argument / call-site errors land here; treat as
        # "probe failed" rather than crashing the host process.
        return None


def _drop_devices_the_engine_refuses(devices: list[VulkanDevice]) -> list[VulkanDevice]:
    """Drop adapters ggml's Vulkan backend would exclude from its device pool.

    ``ggml_vk_device_is_supported`` gates on exactly one feature,
    ``storageBuffer16BitAccess``, and excludes devices without it silently, with
    no error anywhere. Some Adreno parts are the documented case. Keeping such a
    device means placement sizes a fleet against VRAM the engine will never
    touch, and the engine quietly runs on the CPU or another adapter instead.

    Only a definite ``False`` drops a device: a loader too old to be asked
    reports ``None``, and that is not a refusal.
    """
    return [d for d in devices if d.storage_buffer_16bit is not False]


def _deduplicate_by_uuid(devices: list[VulkanDevice]) -> list[VulkanDevice]:
    """Collapse adapters that share a ``deviceUUID`` into one, keeping the first.

    Two ICDs able to drive the same card (RADV beside AMDVLK is the case ggml's
    own dedup names) enumerate it twice. ggml counts it once, so without this
    lilbee plans a two-GPU fleet on one piece of silicon and tensor-splits a
    model across a card and itself.

    ggml breaks the same tie with a driver-priority table, picking which
    driver's entry survives. Lowest index is enough here because nothing lilbee
    reads off a device tells the two entries apart: the type, the name and the
    device-local heap size describe the silicon, not the driver, and no caller
    pins by the raw enumeration index any more.

    Devices with no UUID are all kept, since "the loader would not say" is not
    evidence that two adapters are one.
    """
    seen: set[bytes] = set()
    unique: list[VulkanDevice] = []
    for device in devices:
        if device.device_uuid and device.device_uuid in seen:
            log.debug(
                "Vulkan device %d (%s) is device %s under a second driver; ignoring the duplicate",
                device.index,
                device.device_name,
                next(d.index for d in unique if d.device_uuid == device.device_uuid),
            )
            continue
        if device.device_uuid:
            seen.add(device.device_uuid)
        unique.append(device)
    return unique


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


def _create_probe_instance(create_instance: ctypes._FuncPointer) -> tuple[c_void_p | None, int]:
    """Create the throwaway instance, asking for 1.1 and settling for 1.0.

    Returns the instance and the API version it was created with. 1.1 makes
    ``vkGetPhysicalDeviceProperties2`` core, which is where the device UUID
    lives; a 1.0-only loader rejects the request outright, so the 1.0 retry is
    what keeps the probe working there at all rather than silently reporting no
    adapters.
    """
    for api_version in (_VK_API_VERSION_1_1, _VK_API_VERSION_1_0):
        app_info = _VkApplicationInfo(
            sType=_VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pNext=None,
            pApplicationName=b"lilbee-gpu-probe",
            applicationVersion=0,
            pEngineName=b"lilbee",
            engineVersion=0,
            apiVersion=api_version,
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
        if result == _VK_SUCCESS and instance.value:
            return instance, api_version
    return None, 0


def _resolve_properties2(lib: ctypes.CDLL) -> ctypes._FuncPointer | None:
    """``vkGetPhysicalDeviceProperties2`` with argtypes stamped, ``None`` if absent."""
    try:
        get_properties2 = lib.vkGetPhysicalDeviceProperties2
    except AttributeError:
        return None
    get_properties2.argtypes = [c_void_p, POINTER(_VkPhysicalDeviceProperties2)]
    get_properties2.restype = None
    return get_properties2


def _resolve_memory_budget(
    lib: ctypes.CDLL,
) -> tuple[ctypes._FuncPointer, ctypes._FuncPointer] | None:
    """``(vkGetPhysicalDeviceMemoryProperties2, vkEnumerateDeviceExtensionProperties)``."""
    try:
        get_memory2 = lib.vkGetPhysicalDeviceMemoryProperties2
        enum_extensions = lib.vkEnumerateDeviceExtensionProperties
    except AttributeError:
        return None
    get_memory2.argtypes = [c_void_p, POINTER(_VkPhysicalDeviceMemoryProperties2)]
    get_memory2.restype = None
    enum_extensions.argtypes = [
        c_void_p,
        c_char_p,
        POINTER(c_uint32),
        POINTER(_VkExtensionProperties),
    ]
    enum_extensions.restype = c_uint32
    return get_memory2, enum_extensions


def _supports_memory_budget(handle: c_void_p, enum_extensions: ctypes._FuncPointer) -> bool:
    """Whether the device advertises ``VK_EXT_memory_budget``.

    Asked rather than assumed: chaining the budget struct onto a device that
    does not support it leaves it zeroed, and zero budget is indistinguishable
    from a full card.
    """
    count = c_uint32(0)
    if enum_extensions(handle, None, byref(count), None) != _VK_SUCCESS or count.value == 0:
        return False
    props = (_VkExtensionProperties * count.value)()
    if enum_extensions(handle, None, byref(count), props) != _VK_SUCCESS:
        return False
    return any(props[i].extensionName == _VK_EXT_MEMORY_BUDGET for i in range(count.value))


def _free_device_local_bytes(
    handle: c_void_p, memory_budget: tuple[ctypes._FuncPointer, ctypes._FuncPointer] | None
) -> int | None:
    """Device-local memory still available, or ``None`` when it cannot be asked.

    ``None`` rather than the heap size on purpose. Reporting capacity as free is
    how a desktop holding gigabytes of compositor and browser VRAM was planned
    as an empty card.
    """
    if memory_budget is None:
        return None
    get_memory2, enum_extensions = memory_budget
    if not _supports_memory_budget(handle, enum_extensions):
        return None
    budget = _VkPhysicalDeviceMemoryBudgetPropertiesEXT(
        sType=_VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT, pNext=None
    )
    props2 = _VkPhysicalDeviceMemoryProperties2(
        sType=_VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
        pNext=ctypes.cast(ctypes.pointer(budget), c_void_p),
    )
    get_memory2(handle, byref(props2))
    mem = props2.memoryProperties
    free = 0
    for i in range(mem.memoryHeapCount):
        if mem.memoryHeaps[i].flags & _VK_MEMORY_HEAP_DEVICE_LOCAL_BIT:
            free += max(0, int(budget.heapBudget[i]) - int(budget.heapUsage[i]))
    return free


def _resolve_features2(lib: ctypes.CDLL) -> ctypes._FuncPointer | None:
    """``vkGetPhysicalDeviceFeatures2`` with argtypes stamped, ``None`` if absent."""
    try:
        get_features2 = lib.vkGetPhysicalDeviceFeatures2
    except AttributeError:
        return None
    get_features2.argtypes = [c_void_p, POINTER(_VkPhysicalDeviceFeatures2)]
    get_features2.restype = None
    return get_features2


def _storage_buffer_16bit(
    handle: c_void_p, get_features2: ctypes._FuncPointer | None
) -> bool | None:
    """Whether the adapter supports ``storageBuffer16BitAccess``, ``None`` if unasked.

    The one feature ggml's Vulkan backend requires before it will put a device
    in its pool, and it drops devices that lack it silently. Some Adreno parts
    expose ``uniformAndStorageBuffer16BitAccess`` without it, so the two are not
    interchangeable and only the first flag answers the question.
    """
    if get_features2 is None:
        return None
    storage = _VkPhysicalDevice16BitStorageFeatures(
        sType=_VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES, pNext=None
    )
    features2 = _VkPhysicalDeviceFeatures2(
        sType=_VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        pNext=ctypes.cast(ctypes.pointer(storage), c_void_p),
    )
    get_features2(handle, byref(features2))
    return bool(storage.storageBuffer16BitAccess)


def _device_uuid(handle: c_void_p, get_properties2: ctypes._FuncPointer | None) -> bytes:
    """The adapter's ``deviceUUID``, empty when it cannot be asked for.

    An all-zero UUID is returned as empty too: it is what a driver leaves behind
    when it ignores the chained struct, and treating it as a real identity would
    collapse every such adapter into one.
    """
    if get_properties2 is None:
        return b""
    id_props = _VkPhysicalDeviceIDProperties(
        sType=_VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES, pNext=None
    )
    props2 = _VkPhysicalDeviceProperties2(
        sType=_VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        pNext=ctypes.cast(ctypes.pointer(id_props), c_void_p),
    )
    get_properties2(handle, byref(props2))
    uuid = bytes(id_props.deviceUUID)
    return b"" if not any(uuid) else uuid


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

    instance, api_version = _create_probe_instance(create_instance)
    if instance is None:
        return []
    core_1_1 = api_version >= _VK_API_VERSION_1_1
    get_properties2 = _resolve_properties2(lib) if core_1_1 else None
    get_features2 = _resolve_features2(lib) if core_1_1 else None
    memory_budget = _resolve_memory_budget(lib) if core_1_1 else None

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
            # Indexing the array yields a bare address; the queries below take a
            # handle, so make it one rather than widen every signature to int.
            handle = c_void_p(handles[i])
            props = _VkPhysicalDeviceProperties()
            get_properties(handle, byref(props))
            mem = _VkPhysicalDeviceMemoryProperties()
            get_memory(handle, byref(mem))
            devices.append(
                VulkanDevice(
                    index=i,
                    device_type=int(props.deviceType),
                    device_name=props.deviceName.decode("utf-8", errors="replace"),
                    vendor_id=int(props.vendorID),
                    vram_bytes=_device_local_vram(mem),
                    device_uuid=_device_uuid(handle, get_properties2),
                    storage_buffer_16bit=_storage_buffer_16bit(handle, get_features2),
                    free_bytes=_free_device_local_bytes(handle, memory_budget),
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
