"""Disabling a driver is only safe when the hardware is actually known."""

from __future__ import annotations

import logging

import pytest

from lilbee.providers.fleet import gpu_select
from lilbee.providers.fleet.gpu_select import PCIVendorID


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    from lilbee.core.config import cfg
    from lilbee.providers.fleet.gpu_select import VulkanIcdEnvVar

    monkeypatch.setattr(cfg, "gpu_devices", None)
    for var in VulkanIcdEnvVar:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(gpu_select, "_platform_supports_icd_pin", lambda: True)


class TestUnknownHardwareDisablesNothing:
    """An empty device tree is "cannot tell", not "no GPU".

    WSL2, a masked /sys/bus/pci and an ARM SoC all read that way, and falling
    back to the manifests alone let a static vendor order disable the only
    driver that works on the machine.
    """

    def test_no_icd_is_disabled_when_the_device_tree_is_unreadable(
        self, monkeypatch, caplog
    ) -> None:
        monkeypatch.setattr(
            gpu_select, "_vulkan_vendors_present", lambda: {PCIVendorID.NVIDIA, PCIVendorID.INTEL}
        )
        monkeypatch.setattr(gpu_select, "installed_gpu_vendor_ids", frozenset)
        with caplog.at_level(logging.DEBUG, logger="lilbee.providers.fleet.gpu_select"):
            assert gpu_select.disable_conflicting_vulkan_icds() is None
        assert "could be confirmed present" in caplog.text
        assert "unknown" in caplog.text

    def test_a_readable_device_tree_still_disables_the_loser(self, monkeypatch) -> None:
        monkeypatch.setattr(
            gpu_select, "_vulkan_vendors_present", lambda: {PCIVendorID.NVIDIA, PCIVendorID.INTEL}
        )
        monkeypatch.setattr(
            gpu_select,
            "installed_gpu_vendor_ids",
            lambda: frozenset({int(PCIVendorID.NVIDIA), int(PCIVendorID.INTEL)}),
        )
        assert gpu_select.disable_conflicting_vulkan_icds() is not None

    def test_a_manifest_for_absent_hardware_is_the_one_disabled(self, monkeypatch) -> None:
        # Mesa ships radeon_icd beside intel_icd on every Linux desktop. With only
        # an Intel card present, the AMD ICD is the one that can only cost: it
        # drives nothing here and still loads into vkCreateInstance.
        monkeypatch.setattr(
            gpu_select, "_vulkan_vendors_present", lambda: {PCIVendorID.AMD, PCIVendorID.INTEL}
        )
        monkeypatch.setattr(
            gpu_select,
            "installed_gpu_vendor_ids",
            lambda: frozenset({int(PCIVendorID.INTEL)}),
        )
        result = gpu_select.disable_conflicting_vulkan_icds()
        assert result is not None
        assert "radeon*" in result
        assert "intel*" not in result
