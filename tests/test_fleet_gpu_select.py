"""Vulkan ABI mirroring in gpu_select."""

from __future__ import annotations

from dataclasses import replace


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
    assert gpu_select.enumerate_gpu_vram() == [(0, fifteen_gib, fifteen_gib)]


def test_rasterizer_first_keeps_the_real_gpu_at_its_own_index(monkeypatch) -> None:
    """Filtering a device must not renumber the ones that survive.

    These are loader ordinals, and the fallback that consumes them pairs each
    with the memory it reported. Renumbering after a drop would pair the iGPU's
    index with the rasterizer's size, or point at a device that is not there.
    """
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

    assert gpu_select.enumerate_gpu_vram() == [(1, fifteen_gib, fifteen_gib)]


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
        assert gpu_select.enumerate_gpu_vram() == [(1, 24 * gib, 24 * gib)]


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

    devices = gpu_select.enumerate_in_process()

    assert devices is not None
    assert [d.index for d in devices] == [0]


def test_a_device_the_engine_would_refuse_is_not_enumerated(monkeypatch) -> None:
    """ggml gates its Vulkan pool on storageBuffer16BitAccess and drops failures silently.

    Some Adreno parts expose uniformAndStorageBuffer16BitAccess without it, so
    lilbee would size a fleet against VRAM the engine never touches while the
    engine quietly ran on the CPU.
    """
    from lilbee.providers.fleet import gpu_select

    supported = replace(_device(0, b"\x01" * 16), storage_buffer_16bit=True)
    refused = replace(_device(1, b"\x02" * 16), storage_buffer_16bit=False)
    monkeypatch.setattr(gpu_select, "_load_vulkan_loader", lambda: object())
    monkeypatch.setattr(
        gpu_select, "_list_devices_with_instance", lambda _lib: [supported, refused]
    )

    devices = gpu_select.enumerate_in_process()

    assert devices is not None
    assert [d.index for d in devices] == [0]


def test_a_loader_that_cannot_report_features_drops_nothing(monkeypatch) -> None:
    """None means unasked, not refused; a 1.0 loader must not blind the probe."""
    from lilbee.providers.fleet import gpu_select

    monkeypatch.setattr(gpu_select, "_load_vulkan_loader", lambda: object())
    monkeypatch.setattr(
        gpu_select, "_list_devices_with_instance", lambda _lib: [_device(0, b""), _device(1, b"")]
    )

    devices = gpu_select.enumerate_in_process()

    assert devices is not None
    assert [d.index for d in devices] == [0, 1]


def test_the_feature_flag_is_read_back_through_the_pnext_chain() -> None:
    """The driver writes into the chained struct, so follow the real pointer."""
    import ctypes

    from lilbee.providers.fleet import gpu_select

    def _fake_get_features2(_handle, features2_ref) -> None:
        storage = ctypes.cast(
            features2_ref._obj.pNext,
            ctypes.POINTER(gpu_select._VkPhysicalDevice16BitStorageFeatures),
        )
        assert (
            storage[0].sType == gpu_select._VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES
        )
        # An Adreno part: the uniform variant only.
        storage[0].uniformAndStorageBuffer16BitAccess = 1
        storage[0].storageBuffer16BitAccess = 0

    assert gpu_select._storage_buffer_16bit(None, _fake_get_features2) is False


def test_features_are_not_guessed_when_the_loader_cannot_be_asked() -> None:
    from lilbee.providers.fleet.gpu_select import _storage_buffer_16bit

    assert _storage_buffer_16bit(None, None) is None


def test_free_memory_is_read_back_through_the_budget_chain() -> None:
    """budget minus usage on the device-local heaps, per VK_EXT_memory_budget."""
    import ctypes

    from lilbee.providers.fleet import gpu_select

    def _fake_get_memory2(_handle, props2_ref) -> None:
        props2 = props2_ref._obj
        mem = props2.memoryProperties
        mem.memoryHeapCount = 2
        mem.memoryHeaps[0].flags = gpu_select._VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
        mem.memoryHeaps[1].flags = 0  # host-visible system memory, not VRAM
        budget = ctypes.cast(
            props2.pNext, ctypes.POINTER(gpu_select._VkPhysicalDeviceMemoryBudgetPropertiesEXT)
        )
        budget[0].heapBudget[0] = 24 * 1024**3
        budget[0].heapUsage[0] = 3 * 1024**3
        budget[0].heapBudget[1] = 64 * 1024**3
        budget[0].heapUsage[1] = 0

    def _fake_enum_extensions(_handle, _layer, count_ref, props_ref) -> int:
        if props_ref is None:
            count_ref._obj.value = 1
        else:
            props_ref[0].extensionName = gpu_select._VK_EXT_MEMORY_BUDGET
        return gpu_select._VK_SUCCESS

    free = gpu_select._free_device_local_bytes(None, (_fake_get_memory2, _fake_enum_extensions))

    assert free == 21 * 1024**3


def test_a_device_without_the_budget_extension_reports_unknown(monkeypatch) -> None:
    """Reporting the heap size as free is how a loaded desktop looked empty."""
    from lilbee.providers.fleet import gpu_select

    monkeypatch.setattr(gpu_select, "_supports_memory_budget", lambda *_a: False)

    assert gpu_select._free_device_local_bytes(None, (lambda *_a: None, lambda *_a: None)) is None


def test_a_loader_too_old_for_the_budget_query_reports_unknown() -> None:
    from lilbee.providers.fleet.gpu_select import _free_device_local_bytes

    assert _free_device_local_bytes(None, None) is None


def test_free_memory_is_sampled_fresh_not_frozen_for_the_process(monkeypatch) -> None:
    """Free VRAM is a live number; a process-lifetime cache hands every later
    probe the first reading ever taken, while the read path re-probes every two
    seconds expecting a current one."""
    from lilbee.providers.fleet import gpu_select

    readings = iter(
        [
            [replace(_device(0, b"\x01" * 16, "RTX 4090"), free_bytes=20 * 1024**3)],
            [replace(_device(0, b"\x01" * 16, "RTX 4090"), free_bytes=4 * 1024**3)],
        ]
    )
    monkeypatch.setattr(gpu_select, "_enumerate_vulkan_devices", lambda: next(readings))

    first = gpu_select.vulkan_free_bytes_by_name()
    second = gpu_select.vulkan_free_bytes_by_name()

    assert first == {"RTX 4090": 20 * 1024**3}
    assert second == {"RTX 4090": 4 * 1024**3}


def test_two_identical_cards_report_no_free_figure_rather_than_one_of_theirs(
    monkeypatch,
) -> None:
    """The map is keyed by name, and two identical cards share one.

    They do not share a free figure, and nothing in the engine's text says which
    line is which, so reporting either one attributes one card's headroom to the
    other. Absent means the heap size stands in, which is merely imprecise.
    """
    from lilbee.providers.fleet import gpu_select

    monkeypatch.setattr(
        gpu_select,
        "_enumerate_vulkan_devices",
        lambda: [
            replace(_device(0, b"\x01" * 16, "RTX 4090"), free_bytes=20 * 1024**3),
            replace(_device(1, b"\x02" * 16, "RTX 4090"), free_bytes=2 * 1024**3),
        ],
    )

    assert gpu_select.vulkan_free_bytes_by_name() == {}


def test_distinct_cards_still_each_report_their_own(monkeypatch) -> None:
    from lilbee.providers.fleet import gpu_select

    monkeypatch.setattr(
        gpu_select,
        "_enumerate_vulkan_devices",
        lambda: [
            replace(_device(0, b"\x01" * 16, "RTX 4090"), free_bytes=20 * 1024**3),
            replace(_device(1, b"\x02" * 16, "RX 7900 XTX"), free_bytes=2 * 1024**3),
        ],
    )

    assert gpu_select.vulkan_free_bytes_by_name() == {
        "RTX 4090": 20 * 1024**3,
        "RX 7900 XTX": 2 * 1024**3,
    }


class TestTheProbeDegradesOnAnIncompleteLoader:
    """Every Properties2-family entry point is resolved defensively.

    A loader old enough to lack them is a real configuration, and the probe has
    to fall back to what Vulkan 1.0 offers rather than raise out of a bootstrap.
    """

    class _LoaderWithout:
        """A loader that raises AttributeError for any symbol asked of it."""

        def __getattr__(self, name: str):
            raise AttributeError(name)

    def test_properties2_absent_yields_no_uuid_reader(self) -> None:
        from lilbee.providers.fleet.gpu_select import _resolve_properties2

        assert _resolve_properties2(self._LoaderWithout()) is None

    def test_features2_absent_yields_no_feature_reader(self) -> None:
        from lilbee.providers.fleet.gpu_select import _resolve_features2

        assert _resolve_features2(self._LoaderWithout()) is None

    def test_memory_budget_absent_yields_no_budget_reader(self) -> None:
        from lilbee.providers.fleet.gpu_select import _resolve_memory_budget

        assert _resolve_memory_budget(self._LoaderWithout()) is None


class TestDeviceTypesTheHeadersDoNotDefine:
    def test_an_unknown_device_type_is_not_invented(self) -> None:
        """A driver returning a value vk.h has no name for must not become one."""
        from lilbee.providers.fleet.gpu_select import _known_device_type

        assert _known_device_type(99) is None

    def test_a_known_device_type_round_trips(self) -> None:
        from lilbee.providers.fleet.gpu_select import VkDeviceType, _known_device_type

        assert _known_device_type(2) is VkDeviceType.DISCRETE_GPU


class TestTheMemoryBudgetExtensionIsAskedFor:
    """Chaining the budget struct onto a device without the extension leaves it
    zeroed, and a zero budget cannot be told from a full card."""

    def _enum(self, *, first_rc=0, second_rc=0, count=1, name=b"VK_EXT_memory_budget"):
        from lilbee.providers.fleet import gpu_select

        def _enum_extensions(_handle, _layer, count_ref, props_ref) -> int:
            if props_ref is None:
                count_ref._obj.value = count
                return first_rc
            for i in range(count):
                props_ref[i].extensionName = name
            return second_rc

        return gpu_select, _enum_extensions

    def test_the_extension_being_present_is_detected(self) -> None:
        gpu_select, enum_extensions = self._enum()

        assert gpu_select._supports_memory_budget(None, enum_extensions) is True

    def test_a_device_listing_other_extensions_only(self) -> None:
        gpu_select, enum_extensions = self._enum(name=b"VK_KHR_swapchain")

        assert gpu_select._supports_memory_budget(None, enum_extensions) is False

    def test_a_device_listing_no_extensions(self) -> None:
        gpu_select, enum_extensions = self._enum(count=0)

        assert gpu_select._supports_memory_budget(None, enum_extensions) is False

    def test_a_failed_count_query(self) -> None:
        gpu_select, enum_extensions = self._enum(first_rc=1)

        assert gpu_select._supports_memory_budget(None, enum_extensions) is False

    def test_a_failed_second_query(self) -> None:
        gpu_select, enum_extensions = self._enum(second_rc=1)

        assert gpu_select._supports_memory_budget(None, enum_extensions) is False


class _FakeSymbol:
    """Stands in for a ctypes function pointer: accepts argtypes/restype."""

    argtypes: object = None
    restype: object = None


class _LoaderWith:
    """A loader exposing every symbol the probe may resolve."""

    def __init__(self) -> None:
        self._symbols: dict[str, _FakeSymbol] = {}

    def __getattr__(self, name: str) -> _FakeSymbol:
        return self._symbols.setdefault(name, _FakeSymbol())


class TestResolvedSymbolsCarryTheirCallingConvention:
    """argtypes and restype are stamped on every resolved symbol.

    Skipping it lets ctypes guess the calling convention, which on Windows is
    silent stack corruption rather than an error.
    """

    def test_properties2_is_stamped(self) -> None:
        from lilbee.providers.fleet.gpu_select import _resolve_properties2

        resolved = _resolve_properties2(_LoaderWith())

        assert resolved is not None
        assert resolved.argtypes is not None
        assert resolved.restype is None

    def test_features2_is_stamped(self) -> None:
        from lilbee.providers.fleet.gpu_select import _resolve_features2

        resolved = _resolve_features2(_LoaderWith())

        assert resolved is not None
        assert resolved.argtypes is not None

    def test_the_memory_budget_pair_is_stamped(self) -> None:
        from lilbee.providers.fleet.gpu_select import _resolve_memory_budget

        resolved = _resolve_memory_budget(_LoaderWith())

        assert resolved is not None
        get_memory2, enum_extensions = resolved
        assert get_memory2.argtypes is not None
        assert enum_extensions.restype is not None


class TestProvingADiscreteCardFromAVendor:
    """The ROCm fail-loud guard asks this before refusing to start, so the three
    answers have to stay distinct: yes, no, and cannot tell."""

    def test_a_discrete_card_of_that_vendor_is_proof(self, monkeypatch) -> None:
        from lilbee.providers.fleet import gpu_select

        monkeypatch.setattr(
            gpu_select,
            "_enumerate_vulkan_devices",
            lambda: [
                gpu_select.VulkanDevice(0, gpu_select.VkDeviceType.DISCRETE_GPU, "RX", 0x1002, 0)
            ],
        )

        assert gpu_select.discrete_gpu_from_vendor(0x1002) is True

    def test_a_discrete_card_of_another_vendor_is_not(self, monkeypatch) -> None:
        from lilbee.providers.fleet import gpu_select

        monkeypatch.setattr(
            gpu_select,
            "_enumerate_vulkan_devices",
            lambda: [
                gpu_select.VulkanDevice(0, gpu_select.VkDeviceType.DISCRETE_GPU, "RTX", 0x10DE, 0)
            ],
        )

        assert gpu_select.discrete_gpu_from_vendor(0x1002) is False

    def test_an_unreachable_loader_cannot_tell(self, monkeypatch) -> None:
        from lilbee.providers.fleet import gpu_select

        monkeypatch.setattr(gpu_select, "_enumerate_vulkan_devices", lambda: None)

        assert gpu_select.discrete_gpu_from_vendor(0x1002) is None
