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
import logging
import sys
from ctypes import POINTER, byref, c_char, c_char_p, c_uint8, c_uint32, c_void_p
from dataclasses import dataclass
from enum import IntEnum

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
    outcome and propagates back.
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
