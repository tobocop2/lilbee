"""lilbee's own type verdict must not take away the engine's only GPU."""

from __future__ import annotations

from lilbee.providers.fleet import devices as devices_mod
from lilbee.providers.fleet.devices import FleetDevice
from lilbee.providers.fleet.gpu_select import VkDeviceType

_GB = 1024**3


def _vulkan(index: int, name: str) -> FleetDevice:
    return FleetDevice("Vulkan", index, name, 8 * _GB, 8 * _GB)


class TestATypeVerdictNeverEmptiesTheFleet:
    """The engine listed the device, so ggml is willing to run on it.

    lilbee then asks the loader for a type and refuses anything outside discrete
    and integrated. On real hardware that the loader types OTHER, and on a
    passthrough adapter it types VIRTUAL_GPU, that refusal produced --device none
    and a CPU fleet on a machine with a working GPU.
    """

    def test_an_unclassified_adapter_is_kept(self, monkeypatch) -> None:
        # OTHER is the loader declining to say, which some real drivers do.
        monkeypatch.setattr(devices_mod, "_vulkan_device_type", lambda _n: VkDeviceType.OTHER)
        kept = devices_mod._select_backend([_vulkan(0, "Some Adapter")])
        assert [d.index for d in kept] == [0]

    def test_a_paravirtual_adapter_is_still_refused(self, monkeypatch) -> None:
        # VIRTUAL_GPU is a positive claim, and the CPU-shaped plan that follows
        # is deliberate: a VM's paravirtual adapter is worse than the CPU path.
        monkeypatch.setattr(devices_mod, "_vulkan_device_type", lambda _n: VkDeviceType.VIRTUAL_GPU)
        assert devices_mod._select_backend([_vulkan(0, "Virtio-GPU Venus")]) == []

    def test_a_paravirtual_adapter_is_dropped_beside_a_real_one(self, monkeypatch) -> None:
        types = {"Paravirtual": VkDeviceType.VIRTUAL_GPU, "Real Card": VkDeviceType.DISCRETE_GPU}
        monkeypatch.setattr(devices_mod, "_vulkan_device_type", lambda name: types[name])
        kept = devices_mod._select_backend([_vulkan(0, "Paravirtual"), _vulkan(1, "Real Card")])
        assert [d.name for d in kept] == ["Real Card"]

    def test_a_software_rasterizer_is_still_refused_outright(self, monkeypatch) -> None:
        # A CPU rasterizer really is worse than the CPU path, so this one keeps
        # its veto even when it leaves nothing.
        monkeypatch.setattr(devices_mod, "_vulkan_device_type", lambda _n: VkDeviceType.CPU)
        assert devices_mod._select_backend([_vulkan(0, "llvmpipe (LLVM 15.0.7)")]) == []
