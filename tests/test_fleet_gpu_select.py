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


def test_enumerate_gpu_vram_omits_software_rasterizers(monkeypatch) -> None:
    """The exact shape seen on an Intel Iris Xe laptop with mesa installed.

    llvmpipe reports system RAM as device memory, so beside an iGPU that shares
    the same RAM the two are identical by size and only the device type tells
    them apart. This enumeration is the fallback the placement path uses when
    the engine's --list-devices reports nothing, and it drops names, so a
    name-based filter downstream cannot see the rasterizer at all.
    """
    from lilbee.providers.fleet import gpu_select

    fifteen_gib = 15 * 1024**3
    monkeypatch.setattr(
        gpu_select,
        "_enumerate_vulkan_devices",
        lambda: [
            gpu_select.VulkanDevice(
                0, gpu_select.VkDeviceType.INTEGRATED_GPU, "Intel Iris Xe", 0x8086, fifteen_gib
            ),
            gpu_select.VulkanDevice(
                1, gpu_select.VkDeviceType.CPU, "llvmpipe (LLVM 22.1.8)", 0x10005, fifteen_gib
            ),
        ],
    )
    assert gpu_select.enumerate_gpu_vram() == [(0, fifteen_gib)]
