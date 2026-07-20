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


def test_rasterizer_first_keeps_the_real_gpu_at_its_own_index(monkeypatch) -> None:
    """Loader order is the index space the pin uses, so gaps must survive."""
    from lilbee.providers.fleet import gpu_select

    fifteen_gib = 15 * 1024**3
    monkeypatch.setattr(
        gpu_select,
        "_enumerate_vulkan_devices",
        lambda: [
            gpu_select.VulkanDevice(
                0, gpu_select.VkDeviceType.CPU, "llvmpipe", 0x10005, fifteen_gib
            ),
            gpu_select.VulkanDevice(
                1, gpu_select.VkDeviceType.INTEGRATED_GPU, "Intel Iris Xe", 0x8086, fifteen_gib
            ),
        ],
    )
    # Not renumbered to 0: GGML_VK_VISIBLE_DEVICES names the loader's index.
    assert gpu_select.enumerate_gpu_vram() == [(1, fifteen_gib)]
    assert gpu_select.autoselect_best_gpu_index() == "1"


def test_paravirtual_adapters_are_not_offered_to_placement() -> None:
    """ggml's Vulkan backend runs on discrete and integrated devices only.

    virgl, VMware and VirtIO-GPU report as VIRTUAL_GPU and are typically
    compute-incapable. Offering one to placement guarantees a disagreement with
    the engine about which devices exist, which is the shape of the
    device-count mismatches other launchers have hit on hybrid hosts.
    """
    from unittest import mock

    from lilbee.providers.fleet import gpu_select

    gib = 1024**3
    devices = [
        gpu_select.VulkanDevice(0, gpu_select.VkDeviceType.VIRTUAL_GPU, "virgl", 0x1AF4, 4 * gib),
        gpu_select.VulkanDevice(
            1, gpu_select.VkDeviceType.DISCRETE_GPU, "RTX 4090", 0x10DE, 24 * gib
        ),
    ]
    with mock.patch.object(gpu_select, "_enumerate_vulkan_devices", lambda: devices):
        assert gpu_select.enumerate_gpu_vram() == [(1, 24 * gib)]


def test_integrated_index_probe_runs_once_per_process(monkeypatch) -> None:
    """The device parser asks per device line; the answer is a machine property."""
    from lilbee.providers.fleet import gpu_select

    calls: list[int] = []

    def _counting():
        calls.append(1)
        return [gpu_select.VulkanDevice(0, gpu_select.VkDeviceType.INTEGRATED_GPU, "x", 0, 0)]

    gpu_select.integrated_vulkan_indices.cache_clear()
    monkeypatch.setattr(gpu_select, "_enumerate_vulkan_devices", _counting)
    try:
        for _ in range(4):
            assert gpu_select.integrated_vulkan_indices() == frozenset({0})
        assert len(calls) == 1, f"probed the Vulkan loader {len(calls)} times"
    finally:
        gpu_select.integrated_vulkan_indices.cache_clear()


def _device(index: int, uuid: bytes, name: str = "AMD Radeon RX 7900 XTX"):
    from lilbee.providers.fleet import gpu_select

    return gpu_select.VulkanDevice(
        index=index,
        device_type=gpu_select.VkDeviceType.DISCRETE_GPU,
        device_name=name,
        vendor_id=0x1002,
        vram_bytes=24 * 1024**3,
        device_uuid=uuid,
    )


def test_one_card_behind_two_drivers_counts_once() -> None:
    """RADV and AMDVLK installed together enumerate the same card twice.

    ggml deduplicates on deviceUUID and counts it once, so without this lilbee
    plans a two-GPU fleet on one piece of silicon and tensor-splits a model
    across a card and itself.
    """
    from lilbee.providers.fleet.gpu_select import _deduplicate_by_uuid

    uuid = bytes(range(16))
    radv = _device(0, uuid, "AMD Radeon RX 7900 XTX (RADV NAVI31)")
    amdvlk = _device(1, uuid)

    assert [d.index for d in _deduplicate_by_uuid([radv, amdvlk])] == [0]


def test_two_identical_cards_are_still_two_cards() -> None:
    """Same model, same name, different silicon: the UUID is what separates them."""
    from lilbee.providers.fleet.gpu_select import _deduplicate_by_uuid

    first = _device(0, bytes([1] * 16))
    second = _device(1, bytes([2] * 16))

    assert [d.index for d in _deduplicate_by_uuid([first, second])] == [0, 1]


def test_devices_without_a_uuid_are_all_kept() -> None:
    """A 1.0-only loader says nothing about identity; silence is not a match."""
    from lilbee.providers.fleet.gpu_select import _deduplicate_by_uuid

    devices = [_device(0, b""), _device(1, b"")]

    assert [d.index for d in _deduplicate_by_uuid(devices)] == [0, 1]


def test_a_driver_that_leaves_the_chained_struct_alone_reports_no_uuid() -> None:
    """All zeros is what an ignored pNext looks like, not an identity every card shares."""
    from lilbee.providers.fleet.gpu_select import _device_uuid

    assert _device_uuid(None, lambda *_a: None) == b""


def test_the_uuid_is_read_back_through_the_pnext_chain() -> None:
    """The driver writes into the struct lilbee chained on, so follow the real pointer."""
    import ctypes

    from lilbee.providers.fleet import gpu_select

    written = bytes(range(1, 17))

    def _fake_get_properties2(_handle, props2_ref) -> None:
        id_props = ctypes.cast(
            props2_ref._obj.pNext, ctypes.POINTER(gpu_select._VkPhysicalDeviceIDProperties)
        )
        assert id_props[0].sType == gpu_select._VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES
        id_props[0].deviceUUID = (ctypes.c_uint8 * 16)(*written)

    assert gpu_select._device_uuid(None, _fake_get_properties2) == written


def test_a_loader_that_refuses_vulkan_1_1_still_enumerates() -> None:
    """Asking for 1.1 buys the device UUID; being refused must not cost the probe."""
    from lilbee.providers.fleet import gpu_select

    attempts: list[int] = []

    def _create(create_info_ref, _alloc, instance_ref):
        api_version = create_info_ref._obj.pApplicationInfo[0].apiVersion
        attempts.append(api_version)
        if api_version == gpu_select._VK_API_VERSION_1_1:
            return 1  # VK_ERROR_INCOMPATIBLE_DRIVER
        instance_ref._obj.value = 0xDEAD
        return gpu_select._VK_SUCCESS

    instance, api_version = gpu_select._create_probe_instance(_create)

    assert attempts == [gpu_select._VK_API_VERSION_1_1, gpu_select._VK_API_VERSION_1_0]
    assert instance is not None
    assert api_version == gpu_select._VK_API_VERSION_1_0


def test_the_probe_itself_returns_deduplicated_devices(monkeypatch) -> None:
    """The dedup has to sit on the path every caller uses, not beside it."""
    from lilbee.providers.fleet import gpu_select

    uuid = bytes(range(16))
    monkeypatch.setattr(gpu_select, "_load_vulkan_loader", lambda: object())
    monkeypatch.setattr(
        gpu_select, "_list_devices_with_instance", lambda _lib: [_device(0, uuid), _device(1, uuid)]
    )

    devices = gpu_select._enumerate_vulkan_devices()

    assert devices is not None
    assert [d.index for d in devices] == [0]
