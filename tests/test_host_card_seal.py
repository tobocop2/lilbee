"""The conftest seal answers "no card" for every read of the host's cards."""

from __future__ import annotations

import sys

from lilbee.providers.fleet import devices, gpu_hardware, planning
from lilbee.providers.fleet.gpu_backends import nvidia


class TestHostWithoutCards:
    def test_nvidia_card_table_is_empty(self) -> None:
        assert nvidia.card_rows() == 0
        assert planning.nvidia_card_rows() == 0

    def test_pci_device_tree_lists_no_card(self) -> None:
        assert gpu_hardware.installed_gpu_counts() == {}
        assert gpu_hardware.installed_gpu_vendor_ids() == frozenset()
        assert gpu_hardware.undriven_gpu_vendor_ids() == frozenset()

    def test_windows_registry_lists_no_adapter(self) -> None:
        assert list(gpu_hardware._windows_adapters()) == []

    def test_nvlink_topology_makes_no_claim(self) -> None:
        assert devices.host_lacks_nvlink() is False
        assert planning.host_lacks_nvlink() is False

    def test_another_probe_still_runs_through_the_real_runner(self) -> None:
        stdout, code = devices.run_bounded(
            [sys.executable, "-c", "print('ran')"], timeout_s=30.0, kill_wait_s=1.0, label="py"
        )
        assert (stdout.strip(), code) == ("ran", 0)
