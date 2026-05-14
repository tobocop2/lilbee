"""Windows-registry walk for installed Vulkan ICD manifests.

The Vulkan loader scans four registry locations at every ``vkCreateInstance``
to discover installable client drivers (ICDs). We mirror the same walk so
``disable_conflicting_vulkan_icds`` can identify which vendors are
installed *without* calling ``vkCreateInstance`` itself. Calling the
loader to enumerate vendors would defeat the disable's purpose: it
pre-loads every vendor's ICD into the process before the disable env
var arrives, and at least one ICD on Eric's b473 QA box (AMDVLK)
self-pins its DLL handle and stays resident after ``FreeLibrary``.

The four locations, per
https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderDriverInterface.md
("Driver Manifest File Usage" -> "Driver Discovery on Windows"):

1. ``HKLM\\SOFTWARE\\Khronos\\Vulkan\\Drivers``: legacy key for
   software-rasterizer ICDs. Value names are the manifest paths,
   value type is REG_DWORD where 0 = enabled.
2. ``HKLM\\SOFTWARE\\WOW6432Node\\Khronos\\Vulkan\\Drivers``: 32-bit
   mirror of (1) for 32-bit ICDs registered on 64-bit Windows.
3. PnP per-adapter keys under the Display Adapter device-class GUID
   (``{4d36e968-...}``), values ``VulkanDriverName`` (REG_SZ or
   REG_MULTI_SZ) and ``VulkanDriverNameWow``.
4. PnP per-adapter keys under the Software Component device-class
   GUID (``{5c4c3332-...}``), same value names.

Windows-only: on POSIX hosts ``winreg`` is unavailable and the
top-level iterator yields nothing.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator


# Windows PnP device-class GUIDs that publish Vulkan ICD manifest paths.
# These are stable Microsoft-defined class GUIDs that long predate Vulkan;
# the loader spec names them as the authoritative locations.
_PNP_DISPLAY_ADAPTER_CLASS_GUID = "{4d36e968-e325-11ce-bfc1-08002be10318}"
_PNP_SOFTWARE_COMPONENT_CLASS_GUID = "{5c4c3332-344d-483c-8739-259e934c9cc8}"

_KHRONOS_DRIVERS_KEYS = (
    r"SOFTWARE\Khronos\Vulkan\Drivers",
    r"SOFTWARE\WOW6432Node\Khronos\Vulkan\Drivers",
)

_PNP_VULKAN_VALUE_NAMES = ("VulkanDriverName", "VulkanDriverNameWow")


def iter_windows_vulkan_manifest_paths() -> Iterator[str]:
    """Yield every Vulkan ICD manifest path the Windows Vulkan loader would discover.

    Each yielded string is the absolute path to a ``.json`` manifest as
    written by the driver installer. The bare filename is what
    ``VK_LOADER_DRIVERS_DISABLE`` matches against, so callers pass the
    filename through their own vendor classifier.

    Returns an empty iterator on non-Windows platforms; on Windows where
    ``winreg`` is somehow missing (vanishingly rare, ships with CPython)
    it also yields nothing rather than raising.
    """
    if sys.platform != "win32":
        return
    try:
        import winreg
    except ImportError:  # pragma: no cover - winreg ships with CPython on Windows
        return
    yield from _iter_khronos_software_manifests(winreg)
    yield from _iter_pnp_class_manifests(winreg, _PNP_DISPLAY_ADAPTER_CLASS_GUID)
    yield from _iter_pnp_class_manifests(winreg, _PNP_SOFTWARE_COMPONENT_CLASS_GUID)


def _iter_khronos_software_manifests(winreg: Any) -> Iterator[str]:
    """Yield manifest paths from the legacy ``Khronos\\Vulkan\\Drivers`` keys.

    Each value name is the manifest path and the DWORD value is the
    enabled flag (0 = enabled per the Khronos spec; any non-zero value
    means the installer flagged it disabled, so skip it).
    """
    hklm = winreg.HKEY_LOCAL_MACHINE
    for sub_key in _KHRONOS_DRIVERS_KEYS:
        try:
            key = winreg.OpenKey(hklm, sub_key)
        except OSError:
            continue
        try:
            i = 0
            while True:
                try:
                    name, value, _value_type = winreg.EnumValue(key, i)
                except OSError:
                    break
                i += 1
                if value == 0 and name:
                    yield name
        finally:
            winreg.CloseKey(key)


def _iter_pnp_class_manifests(winreg: Any, class_guid: str) -> Iterator[str]:
    """Yield manifest paths from PnP keys under one device-class GUID.

    Walks ``HKLM\\SYSTEM\\CurrentControlSet\\Control\\Class\\{GUID}\\NNNN``,
    reading each subkey's ``VulkanDriverName`` and ``VulkanDriverNameWow``
    values. Both ``REG_SZ`` (single path) and ``REG_MULTI_SZ`` (path list)
    are honoured because the loader spec allows both.
    """
    hklm = winreg.HKEY_LOCAL_MACHINE
    class_root_path = rf"SYSTEM\CurrentControlSet\Control\Class\{class_guid}"
    try:
        class_root = winreg.OpenKey(hklm, class_root_path)
    except OSError:
        return
    try:
        i = 0
        while True:
            try:
                subkey_name = winreg.EnumKey(class_root, i)
            except OSError:
                break
            i += 1
            try:
                subkey = winreg.OpenKey(class_root, subkey_name)
            except OSError:
                continue
            try:
                yield from _read_vulkan_driver_name_values(winreg, subkey)
            finally:
                winreg.CloseKey(subkey)
    finally:
        winreg.CloseKey(class_root)


def _read_vulkan_driver_name_values(winreg: Any, subkey: Any) -> Iterator[str]:
    """Yield manifest paths from one PnP subkey's ``VulkanDriverName*`` values.

    ``QueryValueEx`` returns a ``list[str]`` for ``REG_MULTI_SZ`` and a
    bare ``str`` for ``REG_SZ``. We normalize to a stream of non-empty
    strings; empty entries inside REG_MULTI_SZ lists are filtered out
    because some installers pad the list.
    """
    for value_name in _PNP_VULKAN_VALUE_NAMES:
        try:
            value, _value_type = winreg.QueryValueEx(subkey, value_name)
        except OSError:
            continue
        if isinstance(value, str):
            if value:
                yield value
        elif isinstance(value, list):
            for entry in value:
                if isinstance(entry, str) and entry:
                    yield entry
