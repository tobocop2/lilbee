"""An integrated GPU is not a small dedicated one."""

from __future__ import annotations

from lilbee.providers.fleet.devices import FleetDevice
from lilbee.providers.fleet.planning import _device_capacity

_GB = 1024**3


# NOTE on the "ranking is memory-only" half of this: it does not reproduce at the
# backend level. _backend_preference compares _BACKEND_RANK first, and every
# discrete backend outranks Vulkan, so a Vulkan iGPU never wins against a CUDA or
# ROCm card however large its heap. Where the large shared heap really does win is
# inside the bin-pack, which picks by remaining capacity, and that is what the
# capacity fix below removes it from.


class TestAnIntegratedDeviceIsNotPackedBesideADiscreteOne:
    """Its memory is the host's. Bin-packing it as though it had its own pool
    promises the machine's RAM twice: once to the iGPU's budget and once to
    everything else running on the box."""

    def test_the_igpu_is_dropped_when_a_real_card_exists(self) -> None:
        devices = [
            FleetDevice("Vulkan", 0, "Radeon 780M", 32 * _GB, 32 * _GB, unified=True),
            FleetDevice("Vulkan", 1, "RX 7900", 24 * _GB, 24 * _GB),
        ]
        assert set(_device_capacity(devices, False)) == {1}

    def test_an_igpu_only_host_keeps_it(self) -> None:
        # Nothing else to serve from; the shared-memory budget governs instead.
        devices = [FleetDevice("Vulkan", 0, "Radeon 780M", 32 * _GB, 32 * _GB, unified=True)]
        assert set(_device_capacity(devices, False)) == {0}

    def test_an_all_discrete_host_is_unchanged(self) -> None:
        devices = [
            FleetDevice("CUDA", 0, "A", 24 * _GB, 24 * _GB),
            FleetDevice("CUDA", 1, "B", 24 * _GB, 24 * _GB),
        ]
        assert set(_device_capacity(devices, False)) == {0, 1}


class TestATinyCarveoutIsNotADedicatedCard:
    """An APU with a small BIOS VRAM carveout reports a handful of MiB as its
    total. Planned as a dedicated device that size, every role is refused, when
    the machine in fact has the whole system's RAM to share."""

    def test_a_carveout_sized_device_is_treated_as_shared(self) -> None:
        from lilbee.providers.fleet.devices import _parse_devices

        parsed = _parse_devices("  ROCm0: AMD Radeon Graphics (512 MiB, 512 MiB free)")
        assert parsed[0].unified is True

    def test_a_real_card_of_the_same_backend_is_not(self, monkeypatch) -> None:
        from lilbee.providers.fleet import devices as devices_mod

        monkeypatch.setattr(devices_mod, "_is_unified", lambda _b, _n: False)
        parsed = devices_mod._parse_devices(
            "  ROCm0: AMD Radeon RX 7900 (24000 MiB, 24000 MiB free)"
        )
        assert parsed[0].unified is False
