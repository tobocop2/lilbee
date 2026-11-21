"""Which GPU vendors are physically installed, read from the OS device tree.

The Vulkan loader cannot answer this. Creating an instance loads every installed
ICD, which is the crash the ICD disable exists to prevent, so the answer has to
come from the OS: Linux reads the PCI display controllers out of sysfs, Windows
reads the PnP display-adapter class out of the registry.

An installed ICD manifest is not evidence of hardware. Mesa ships every vendor's
driver together on Linux (a Flatpak runtime always carries all of them), and a
Windows manifest outlives the card it arrived with.

Read directly rather than through pyudev or WMI: this is one attribute read per
device on either platform, and neither dependency earns its place for that.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lilbee.providers.fleet.vulkan_icd_discovery import (
    PNP_DISPLAY_ADAPTER_CLASS_GUID,
    iter_pnp_class_subkeys,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

# PCI base class 0x03 is "display controller"; sysfs writes the full 24-bit
# class code (0x030000 for VGA), so the base class is its high byte.
_PCI_DISPLAY_CONTROLLER_CLASS = 0x03
_PCI_CLASS_CODE_BASE_SHIFT = 16
_SYSFS_PCI_DEVICE_DIR = Path("/sys/bus/pci/devices")
_SYSFS_CLASS_FILE = "class"
_SYSFS_VENDOR_FILE = "vendor"
# The link to the kernel driver bound to the device. A card with none, such as
# an NVIDIA card whose module is not loaded, cannot run a Vulkan driver.
_SYSFS_DRIVER_LINK = "driver"

# Windows PnP device-instance IDs carry the PCI vendor in their hardware IDs
# ("PCI\\VEN_10DE&DEV_1F95&..."). Both value names are set by the class
# installer; REG_SZ and REG_MULTI_SZ are both allowed.
_PNP_HARDWARE_ID_VALUE_NAMES = ("MatchingDeviceId", "HardwareID")
_PCI_VENDOR_ID_PATTERN = re.compile(r"ven_([0-9a-f]{4})", re.IGNORECASE)


def installed_gpu_vendor_ids() -> frozenset[int]:
    """PCI vendor IDs of this host's display controllers that a kernel driver runs.

    Empty means the device tree holds no such controller or could not be read
    at all (macOS, an ARM SoC whose GPU is not on the PCI bus, a container with
    no ``/sys``, or a Linux host where no display controller has a bound
    driver). That is "cannot tell", not "there is no GPU", and callers must not
    read it as proof that a vendor is absent.
    """
    return frozenset(installed_gpu_counts())


def installed_gpu_counts() -> Counter[int]:
    """How many display controllers of each PCI vendor ID this host holds, driver bound.

    The same reading as ``installed_gpu_vendor_ids``, with the card count kept.
    Windows counts one card for each display adapter, whatever IDs it carries.
    """
    if sys.platform == "win32":
        return Counter(vendor for adapter in _windows_adapters() for vendor in adapter)
    if sys.platform.startswith("linux"):
        return Counter(vendor for vendor, bound in _linux_display_controllers() if bound)
    return Counter()


def undriven_gpu_vendor_ids() -> frozenset[int]:
    """PCI vendor IDs of display controllers with no bound kernel driver.

    Such a card cannot run anything, so it is not in ``installed_gpu_vendor_ids``;
    this names it so a warning can say why the engine found no GPU. Only Linux
    sysfs records the binding, so every other platform answers empty.
    """
    if not sys.platform.startswith("linux"):
        return frozenset()
    return frozenset(vendor for vendor, bound in _linux_display_controllers() if not bound)


def _linux_display_controllers() -> Iterator[tuple[int, bool]]:
    """Yield each readable display controller's vendor ID and whether a driver is bound."""
    try:
        devices = sorted(_SYSFS_PCI_DEVICE_DIR.iterdir())
    except OSError:
        return
    for device in devices:
        class_code = _read_sysfs_hex(device / _SYSFS_CLASS_FILE)
        if class_code is None:
            continue
        if class_code >> _PCI_CLASS_CODE_BASE_SHIFT != _PCI_DISPLAY_CONTROLLER_CLASS:
            continue
        vendor_id = _read_sysfs_hex(device / _SYSFS_VENDOR_FILE)
        if vendor_id is None:
            continue
        try:
            bound = (device / _SYSFS_DRIVER_LINK).exists()
        except OSError:
            continue
        yield vendor_id, bound


def _read_sysfs_hex(path: Path) -> int | None:
    """The ``0x``-prefixed integer in a sysfs attribute, ``None`` when unreadable."""
    try:
        return int(path.read_text(encoding="utf-8").strip(), 16)
    except (OSError, ValueError):
        return None


def _windows_adapters() -> Iterator[frozenset[int]]:
    """Yield the vendor IDs of each registered display adapter."""
    try:
        import winreg
    except ImportError:  # pragma: no cover - winreg ships with CPython on Windows
        return
    yield from _iter_windows_adapters(winreg)


def _iter_windows_adapters(winreg: Any) -> Iterator[frozenset[int]]:
    """Yield the vendor IDs in the hardware IDs of each display-adapter class subkey."""
    for subkey in iter_pnp_class_subkeys(winreg, PNP_DISPLAY_ADAPTER_CLASS_GUID):
        vendors: set[int] = set()
        for value_name in _PNP_HARDWARE_ID_VALUE_NAMES:
            try:
                value, _value_type = winreg.QueryValueEx(subkey, value_name)
            except OSError:
                continue
            vendors.update(_vendor_ids_in(value))
        yield frozenset(vendors)


def _vendor_ids_in(value: object) -> Iterator[int]:
    """Yield the PCI vendor IDs named in one registry hardware-ID value."""
    # Registry values are untyped: REG_SZ arrives as str, REG_MULTI_SZ as list[str].
    entries = value if isinstance(value, list) else [value]
    for entry in entries:
        if not isinstance(entry, str):
            continue
        match = _PCI_VENDOR_ID_PATTERN.search(entry)
        if match is not None:
            yield int(match.group(1), 16)
