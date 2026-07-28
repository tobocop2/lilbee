"""A discrete card the Vulkan loader cannot see is still a discrete card.

Captured from a hybrid ThinkPad: an Intel CometLake iGPU and an NVIDIA GTX
1650 Ti. On this configuration, which is most gaming and workstation laptops
sold in the last decade, the loader enumerates only the integrated adapter
because the discrete one is powered down until something asks for it through
prime-run. lilbee concluded from that list that the host had no discrete GPU,
and then classified the 4 GB card as sharing system memory.
"""

from __future__ import annotations

from lilbee.providers.fleet import gpu_select
from lilbee.providers.fleet.gpu_select import PCIVendorID, VkDeviceType

_INTEL_ONLY = {"Intel(R) UHD Graphics (CML GT2)": VkDeviceType.INTEGRATED_GPU}


class TestAnOptimusHostIsNotAnIntegratedOnlyHost:
    def test_pci_evidence_of_a_discrete_vendor_overrides_the_loader(self, monkeypatch) -> None:
        monkeypatch.setattr(gpu_select, "vulkan_device_types_by_name", lambda: _INTEL_ONLY)
        monkeypatch.setattr(
            gpu_select,
            "installed_gpu_vendor_ids",
            lambda: frozenset({PCIVendorID.NVIDIA, PCIVendorID.INTEL}),
        )
        assert not gpu_select.host_has_no_discrete_gpu()

    def test_a_genuinely_integrated_only_host_still_says_so(self, monkeypatch) -> None:
        # An Intel laptop with no discrete card at all: the shared-memory budget
        # is correct here and must not be lost to the new check.
        monkeypatch.setattr(gpu_select, "vulkan_device_types_by_name", lambda: _INTEL_ONLY)
        monkeypatch.setattr(
            gpu_select, "installed_gpu_vendor_ids", lambda: frozenset({PCIVendorID.INTEL})
        )
        assert gpu_select.host_has_no_discrete_gpu()

    def test_unreadable_pci_does_not_invent_a_discrete_card(self, monkeypatch) -> None:
        # macOS, an ARM SoC, a container with no /sys: empty means "cannot tell",
        # which the loader's own answer should then decide.
        monkeypatch.setattr(gpu_select, "vulkan_device_types_by_name", lambda: _INTEL_ONLY)
        monkeypatch.setattr(gpu_select, "installed_gpu_vendor_ids", frozenset)
        assert gpu_select.host_has_no_discrete_gpu()

    def test_a_visible_discrete_adapter_is_unchanged(self, monkeypatch) -> None:
        monkeypatch.setattr(
            gpu_select,
            "vulkan_device_types_by_name",
            lambda: {"NVIDIA GeForce RTX 4090": VkDeviceType.DISCRETE_GPU},
        )
        monkeypatch.setattr(
            gpu_select, "installed_gpu_vendor_ids", lambda: frozenset({PCIVendorID.NVIDIA})
        )
        assert not gpu_select.host_has_no_discrete_gpu()
