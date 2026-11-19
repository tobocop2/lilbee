"""The conftest seal answers "no card" for every read of the host's cards.

Each test plants a card where the real reader would find it, so it fails without the seal on a
host that has no card of its own.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

from lilbee.providers.fleet import devices, gpu_hardware, planning
from lilbee.providers.fleet.gpu_backends import nvidia

_FAKE_SMI = """\
#!/bin/sh
case "$1" in
topo)
  printf '\\tGPU0\\tGPU1\\tCPU Affinity\\n'
  printf 'GPU0\\t X \\tPHB\\t0-31\\n'
  printf 'GPU1\\tPHB\\t X \\t0-31\\n'
  ;;
*) echo "0, GPU-fake-0, 24576, 24000, NVIDIA Fake Card" ;;
esac
"""

_posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="the stand-in tool is a shell script"
)


@pytest.fixture
def smi_on_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Put an nvidia-smi on PATH that lists one card and a PCIe-only topology."""
    tool = tmp_path / "nvidia-smi"
    tool.write_text(_FAKE_SMI, encoding="utf-8")
    tool.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")


@_posix_only
class TestHostWithoutCards:
    def test_nvidia_card_table_is_empty(self, smi_on_path: None) -> None:
        assert nvidia.card_rows() == 0
        assert planning.nvidia_card_rows() == 0

    def test_nvlink_topology_makes_no_claim(self, smi_on_path: None) -> None:
        assert devices.host_lacks_nvlink() is False
        assert planning.host_lacks_nvlink() is False

    def test_another_probe_still_runs_through_the_real_runner(self) -> None:
        stdout, code = devices.run_bounded(
            [sys.executable, "-c", "print('ran')"], timeout_s=30.0, kill_wait_s=1.0, label="py"
        )
        assert (stdout.strip(), code) == ("ran", 0)


class TestSealedReaders:
    def test_pci_device_directory_is_not_the_host_bus(self) -> None:
        assert gpu_hardware._SYSFS_PCI_DEVICE_DIR.name == "no-pci-bus"
        assert not gpu_hardware._SYSFS_PCI_DEVICE_DIR.exists()
        assert gpu_hardware.installed_gpu_counts() == {}
        assert gpu_hardware.installed_gpu_vendor_ids() == frozenset()
        assert gpu_hardware.undriven_gpu_vendor_ids() == frozenset()

    def test_windows_registry_lists_no_adapter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        winreg = mock.MagicMock(name="winreg")
        winreg.HKEY_LOCAL_MACHINE = 0
        winreg.OpenKey.side_effect = [mock.MagicMock(name="class_root"), mock.MagicMock()]
        winreg.EnumKey.side_effect = ["0000", OSError("end")]
        winreg.QueryValueEx.return_value = (r"PCI\VEN_10DE&DEV_1F95", 1)
        monkeypatch.setitem(sys.modules, "winreg", winreg)

        assert list(gpu_hardware._windows_adapters()) == []
