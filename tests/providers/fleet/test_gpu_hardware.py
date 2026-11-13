"""``installed_gpu_vendor_ids`` reads GPUs out of the OS device tree, not Vulkan."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

from lilbee.providers.fleet import gpu_hardware

NVIDIA = 0x10DE
AMD = 0x1002
INTEL = 0x8086
VGA_CONTROLLER_CLASS = "0x030000"
# What an NVIDIA Optimus dGPU and every datacenter card report: the display base
# class with the 3D-controller subclass, never the VGA one.
THREE_D_CONTROLLER_CLASS = "0x030200"
DISPLAY_CONTROLLER_CLASS = "0x038000"
NETWORK_CONTROLLER_CLASS = "0x020000"


def _write_pci_device(
    root: Path, slot: str, *, class_code: str, vendor: str, driver_bound: bool = True
) -> None:
    """Write one sysfs PCI device directory.

    Real slot names are ``0000:00:02.0``; the colons are illegal in Windows paths
    and the probe never parses the name, so the fixture uses a portable stand-in.
    sysfs links ``driver`` to the bound kernel driver; a directory stands in for it.
    """
    device = root / slot
    device.mkdir(parents=True)
    (device / "class").write_text(f"{class_code}\n")
    (device / "vendor").write_text(f"{vendor}\n")
    if driver_bound:
        (device / "driver").mkdir()


class TestLinuxSysfs:
    def test_reports_display_controller_vendors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An Intel iGPU beside a discrete NVIDIA card reports both vendors."""
        _write_pci_device(
            tmp_path, "0000-00-02.0", class_code=VGA_CONTROLLER_CLASS, vendor="0x8086"
        )
        _write_pci_device(
            tmp_path, "0000-01-00.0", class_code=VGA_CONTROLLER_CLASS, vendor="0x10de"
        )
        monkeypatch.setattr(gpu_hardware.sys, "platform", "linux")
        monkeypatch.setattr(gpu_hardware, "_SYSFS_PCI_DEVICE_DIR", tmp_path)

        assert gpu_hardware.installed_gpu_vendor_ids() == frozenset({INTEL, NVIDIA})

    def test_a_card_with_no_kernel_driver_is_not_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An NVIDIA card whose module is not loaded cannot run anything.

        Measured on a hybrid ThinkPad with the nvidia module unloaded: sysfs
        lists the card with no ``driver`` link, and reporting it hid the Intel
        GPU that does work.
        """
        _write_pci_device(
            tmp_path, "0000-00-02.0", class_code=VGA_CONTROLLER_CLASS, vendor="0x8086"
        )
        _write_pci_device(
            tmp_path,
            "0000-01-00.0",
            class_code=VGA_CONTROLLER_CLASS,
            vendor="0x10de",
            driver_bound=False,
        )
        monkeypatch.setattr(gpu_hardware.sys, "platform", "linux")
        monkeypatch.setattr(gpu_hardware, "_SYSFS_PCI_DEVICE_DIR", tmp_path)

        assert gpu_hardware.installed_gpu_vendor_ids() == frozenset({INTEL})
        assert gpu_hardware.undriven_gpu_vendor_ids() == frozenset({NVIDIA})

    def test_undriven_cards_are_not_read_off_linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only sysfs records a missing driver; elsewhere the answer is empty."""
        monkeypatch.setattr(gpu_hardware.sys, "platform", "win32")

        assert gpu_hardware.undriven_gpu_vendor_ids() == frozenset()

    def test_non_display_devices_are_ignored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An Intel network card is not an Intel GPU."""
        _write_pci_device(
            tmp_path, "0000-00-1f.6", class_code=NETWORK_CONTROLLER_CLASS, vendor="0x8086"
        )
        monkeypatch.setattr(gpu_hardware.sys, "platform", "linux")
        monkeypatch.setattr(gpu_hardware, "_SYSFS_PCI_DEVICE_DIR", tmp_path)

        assert gpu_hardware.installed_gpu_vendor_ids() == frozenset()

    @pytest.mark.parametrize(
        "class_code",
        [
            pytest.param(VGA_CONTROLLER_CLASS, id="vga"),
            pytest.param(THREE_D_CONTROLLER_CLASS, id="3d-controller"),
            pytest.param(DISPLAY_CONTROLLER_CLASS, id="other-display"),
        ],
    )
    def test_every_display_subclass_counts(
        self, class_code: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole 0x03 base class is a GPU, not just the VGA subclass.

        An Optimus laptop's discrete card and every datacenter card enumerate as
        3D controllers, so narrowing this to VGA would report them as absent and
        disable the driver they need.
        """
        _write_pci_device(tmp_path, "0000-01-00.0", class_code=class_code, vendor="0x1002")
        monkeypatch.setattr(gpu_hardware.sys, "platform", "linux")
        monkeypatch.setattr(gpu_hardware, "_SYSFS_PCI_DEVICE_DIR", tmp_path)

        assert gpu_hardware.installed_gpu_vendor_ids() == frozenset({AMD})

    def test_unreadable_attributes_are_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A device missing class or vendor drops out instead of failing the probe."""
        (tmp_path / "0000-00-02.0").mkdir()  # no class file at all
        no_vendor = tmp_path / "0000-01-00.0"
        no_vendor.mkdir()
        (no_vendor / "class").write_text(VGA_CONTROLLER_CLASS)
        _write_pci_device(
            tmp_path, "0000-02-00.0", class_code=VGA_CONTROLLER_CLASS, vendor="0x10de"
        )
        monkeypatch.setattr(gpu_hardware.sys, "platform", "linux")
        monkeypatch.setattr(gpu_hardware, "_SYSFS_PCI_DEVICE_DIR", tmp_path)

        assert gpu_hardware.installed_gpu_vendor_ids() == frozenset({NVIDIA})

    def test_garbage_attribute_value_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-hex class code cannot raise out of the probe."""
        _write_pci_device(tmp_path, "0000-00-02.0", class_code="not-a-number", vendor="0x8086")
        monkeypatch.setattr(gpu_hardware.sys, "platform", "linux")
        monkeypatch.setattr(gpu_hardware, "_SYSFS_PCI_DEVICE_DIR", tmp_path)

        assert gpu_hardware.installed_gpu_vendor_ids() == frozenset()

    def test_missing_sysfs_reports_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No /sys (a stripped container) is "cannot tell", not a crash."""
        monkeypatch.setattr(gpu_hardware.sys, "platform", "linux")
        monkeypatch.setattr(gpu_hardware, "_SYSFS_PCI_DEVICE_DIR", Path("/nonexistent/pci"))

        assert gpu_hardware.installed_gpu_vendor_ids() == frozenset()


class TestWindowsRegistry:
    def test_reads_vendors_from_display_adapter_hardware_ids(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MatchingDeviceId (REG_SZ) and HardwareID (REG_MULTI_SZ) both carry VEN_xxxx."""
        winreg = mock.MagicMock(name="winreg")
        winreg.HKEY_LOCAL_MACHINE = 0
        class_root = mock.MagicMock(name="class_root")
        subkey0 = mock.MagicMock(name="0000")
        subkey1 = mock.MagicMock(name="0001")
        winreg.OpenKey.side_effect = [class_root, subkey0, subkey1]
        winreg.EnumKey.side_effect = ["0000", "0001", OSError("end")]

        def _query(key: mock.MagicMock, name: str) -> tuple[object, int]:
            if key is subkey0 and name == "MatchingDeviceId":
                return (r"PCI\VEN_10DE&DEV_1F95", 1)
            if key is subkey1 and name == "HardwareID":
                return ([r"PCI\VEN_8086&DEV_9A49&SUBSYS_22C317AA", ""], 7)
            raise OSError("missing value")

        winreg.QueryValueEx.side_effect = _query
        monkeypatch.setitem(sys.modules, "winreg", winreg)
        monkeypatch.setattr(gpu_hardware.sys, "platform", "win32")

        assert gpu_hardware.installed_gpu_vendor_ids() == frozenset({NVIDIA, INTEL})

    def test_non_pci_hardware_id_yields_nothing(self) -> None:
        """A hardware ID with no VEN_ field (a virtual adapter) names no vendor."""
        assert list(gpu_hardware._vendor_ids_in(r"ROOT\BasicDisplay")) == []

    def test_non_string_registry_value_is_skipped(self) -> None:
        """A REG_BINARY or REG_DWORD value cannot raise out of the parse."""
        assert list(gpu_hardware._vendor_ids_in(b"\x00")) == []
        assert list(gpu_hardware._vendor_ids_in([b"\x00", r"PCI\VEN_1002&DEV_73FF"])) == [AMD]

    def test_unreadable_class_root_reports_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A missing display-adapter class key is "cannot tell", not a crash."""
        winreg = mock.MagicMock(name="winreg")
        winreg.HKEY_LOCAL_MACHINE = 0
        winreg.OpenKey.side_effect = OSError("class GUID has no key")
        monkeypatch.setitem(sys.modules, "winreg", winreg)
        monkeypatch.setattr(gpu_hardware.sys, "platform", "win32")

        assert gpu_hardware.installed_gpu_vendor_ids() == frozenset()


def test_darwin_reports_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """macOS has no PCI device tree to read and runs Metal anyway."""
    monkeypatch.setattr(gpu_hardware.sys, "platform", "darwin")

    assert gpu_hardware.installed_gpu_vendor_ids() == frozenset()
    assert gpu_hardware.installed_gpu_counts() == {}


class TestCardCounts:
    def test_linux_counts_each_driven_controller_by_vendor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two NVIDIA cards and an AMD card count 2 and 1; a card with no driver counts none."""
        for slot, vendor, bound in [
            ("0000-01-00.0", "0x10de", True),
            ("0000-02-00.0", "0x10de", True),
            ("0000-03-00.0", "0x1002", True),
            ("0000-04-00.0", "0x8086", False),
        ]:
            _write_pci_device(
                tmp_path, slot, class_code=VGA_CONTROLLER_CLASS, vendor=vendor, driver_bound=bound
            )
        monkeypatch.setattr(gpu_hardware.sys, "platform", "linux")
        monkeypatch.setattr(gpu_hardware, "_SYSFS_PCI_DEVICE_DIR", tmp_path)

        assert gpu_hardware.installed_gpu_counts() == {NVIDIA: 2, AMD: 1}

    def test_windows_counts_one_card_for_each_adapter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An adapter naming its vendor in both values is one card; two adapters are two."""
        winreg = mock.MagicMock(name="winreg")
        winreg.HKEY_LOCAL_MACHINE = 0
        class_root = mock.MagicMock(name="class_root")
        subkeys = [mock.MagicMock(name=f"000{i}") for i in range(3)]
        winreg.OpenKey.side_effect = [class_root, *subkeys]
        winreg.EnumKey.side_effect = ["0000", "0001", "0002", OSError("end")]
        ids = {
            id(subkeys[0]): {
                "MatchingDeviceId": r"PCI\VEN_10DE&DEV_1F95",
                "HardwareID": [r"PCI\VEN_10DE&DEV_1F95&SUBSYS_1", r"PCI\VEN_10DE&DEV_1F95"],
            },
            id(subkeys[1]): {"HardwareID": [r"PCI\VEN_10DE&DEV_2204"]},
            id(subkeys[2]): {"MatchingDeviceId": r"ROOT\BasicDisplay"},
        }

        def _query(key: mock.MagicMock, name: str) -> tuple[object, int]:
            if name in ids[id(key)]:
                return (ids[id(key)][name], 1)
            raise OSError("missing value")

        winreg.QueryValueEx.side_effect = _query
        monkeypatch.setitem(sys.modules, "winreg", winreg)
        monkeypatch.setattr(gpu_hardware.sys, "platform", "win32")

        assert gpu_hardware.installed_gpu_counts() == {NVIDIA: 2}
