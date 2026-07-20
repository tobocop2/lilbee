"""Vulkan ABI mirroring in gpu_select."""

from __future__ import annotations


def test_vulkan_properties_struct_matches_the_driver_abi() -> None:
    """The driver fills this buffer using its own layout, not ours.

    VkPhysicalDeviceLimits contains VkDeviceSize and size_t members, so its C
    alignment is 8. Mirroring it as a byte array gave it alignment 1, seating it
    at offset 292 instead of the 296 the ABI pads it to and making the whole
    struct 816 bytes against the driver's 824. vkGetPhysicalDeviceProperties
    then wrote sparseProperties four bytes past the end of a Python-heap
    allocation on every probe, on the default startup and placement path.
    Allocator slack absorbed it, so nothing ever crashed.
    """
    import ctypes

    from lilbee.providers.fleet import gpu_select

    props = gpu_select._VkPhysicalDeviceProperties
    assert ctypes.alignment(gpu_select._VkPhysicalDeviceLimits) == 8
    assert props.limits.offset == 296
    assert props.sparseProperties.offset == 800
    assert ctypes.sizeof(props) == 824
