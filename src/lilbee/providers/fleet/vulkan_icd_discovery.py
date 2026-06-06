"""Cross-platform discovery of installed Vulkan ICD manifests.

Mirrors the Vulkan loader's own discovery so callers can identify which
vendors are installed without calling ``vkCreateInstance`` (which would
pre-load every vendor's ICD into the process before any disable env
var can take effect). Windows reads the registry; Linux walks the XDG
``vulkan/icd.d`` hierarchy; macOS yields nothing. See
https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderDriverInterface.md
for the loader-side spec.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

log = logging.getLogger(__name__)


def iter_vulkan_manifest_paths() -> Iterator[str]:
    """Yield absolute ``.json`` manifest paths the Vulkan loader would discover.

    Returns an empty iterator on macOS (Metal-only build, no Vulkan loader).
    """
    if sys.platform == "win32":
        yield from _iter_windows_vulkan_manifest_paths()
    elif sys.platform.startswith("linux"):
        yield from _iter_linux_vulkan_manifest_paths()
    else:
        yield from ()


# Microsoft-defined PnP device-setup class GUIDs that host Vulkan ICD manifests.
# Both GUIDs and the Khronos software-driver key are documented in
# https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderDriverInterface.md#driver-discovery-on-windows
# (the GUIDs themselves are the public Windows
# https://learn.microsoft.com/en-us/windows-hardware/drivers/install/system-defined-device-setup-classes-available-to-vendors).
_PNP_DISPLAY_ADAPTER_CLASS_GUID = "{4d36e968-e325-11ce-bfc1-08002be10318}"
_PNP_SOFTWARE_COMPONENT_CLASS_GUID = "{5c4c3332-344d-483c-8739-259e934c9cc8}"

# Legacy software-driver paths (HKLM + WOW6432Node mirror for 32-bit ICDs).
# Each value name is a manifest path; the DWORD value is 0=enabled.
_KHRONOS_DRIVERS_KEYS = (
    r"SOFTWARE\Khronos\Vulkan\Drivers",
    r"SOFTWARE\WOW6432Node\Khronos\Vulkan\Drivers",
)

_PNP_VULKAN_VALUE_NAMES = ("VulkanDriverName", "VulkanDriverNameWow")


def _iter_windows_vulkan_manifest_paths() -> Iterator[str]:
    """Yield manifest paths from the four Windows ICD-discovery locations."""
    try:
        import winreg
    except ImportError:  # pragma: no cover - winreg ships with CPython on Windows
        return
    yield from _iter_khronos_software_manifests(winreg)
    yield from _iter_pnp_class_manifests(winreg, _PNP_DISPLAY_ADAPTER_CLASS_GUID)
    yield from _iter_pnp_class_manifests(winreg, _PNP_SOFTWARE_COMPONENT_CLASS_GUID)


def _iter_khronos_software_manifests(winreg: Any) -> Iterator[str]:
    """Yield enabled-flag (DWORD=0) manifest paths from the Khronos software keys."""
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
    """Yield non-empty paths from one PnP subkey's ``VulkanDriverName*`` values.

    Handles both REG_SZ (single string) and REG_MULTI_SZ (list of strings)
    that the loader spec allows.
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


# Linux ICD-discovery search-path config. Defaults follow the XDG basedir
# spec (https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html);
# SYSCONFDIR / EXTRASYSCONFDIR are loader build-time constants that expand
# to /usr/local/etc and /etc on the distros lilbee ships against. The
# Flatpak export trees aren't in the Khronos spec but the loader picks them
# up via XDG_DATA_DIRS inside a Flatpak runtime; we walk them defensively
# in case lilbee runs outside the sandbox.
_VULKAN_ICD_SUBPATH = "vulkan/icd.d"
_LINUX_FIXED_ETC_ICD_DIRS: tuple[str, ...] = (
    "/usr/local/etc/vulkan/icd.d",
    "/etc/vulkan/icd.d",
)
_LINUX_FLATPAK_ICD_DIRS: tuple[str, ...] = (
    "~/.local/share/flatpak/exports/share/vulkan/icd.d",
    "/var/lib/flatpak/exports/share/vulkan/icd.d",
)


def _iter_linux_vulkan_manifest_paths() -> Iterator[str]:
    """Glob ``*.json`` across the Linux ICD-discovery directories, deduping."""
    seen_dirs: set[Path] = set()
    seen_files: set[Path] = set()
    for directory in _linux_vulkan_icd_directories():
        try:
            resolved = directory.expanduser()
        except RuntimeError:
            # PosixPath.expanduser raises when HOME is unset; skip.
            continue
        if resolved in seen_dirs:
            continue
        seen_dirs.add(resolved)
        if not resolved.is_dir():
            continue
        try:
            entries = sorted(resolved.glob("*.json"))
        except OSError:
            log.debug("Vulkan ICD dir %s could not be read", resolved, exc_info=True)
            continue
        for entry in entries:
            if entry in seen_files or not entry.is_file():
                continue
            seen_files.add(entry)
            yield str(entry)


def _linux_vulkan_icd_directories() -> Iterator[Path]:
    """Yield each Linux ICD search directory in loader-spec order."""
    yield from _xdg_dirs("XDG_CONFIG_HOME", "~/.config", _VULKAN_ICD_SUBPATH)
    yield from _xdg_dirs("XDG_CONFIG_DIRS", "/etc/xdg", _VULKAN_ICD_SUBPATH)
    for fixed in _LINUX_FIXED_ETC_ICD_DIRS:
        yield Path(fixed)
    yield from _xdg_dirs("XDG_DATA_HOME", "~/.local/share", _VULKAN_ICD_SUBPATH)
    yield from _xdg_dirs("XDG_DATA_DIRS", "/usr/local/share:/usr/share", _VULKAN_ICD_SUBPATH)
    for flatpak in _LINUX_FLATPAK_ICD_DIRS:
        yield Path(flatpak)


def _xdg_dirs(env_var: str, default: str, subpath: str) -> Iterator[Path]:
    """Split *env_var* (or *default*) on ``:``, append *subpath* to each.

    Empty components are dropped (the "extra slash in XDG_DATA_DIRS" loader
    quirk, Vulkan-Loader#2331) and appends *subpath* to each remaining
    entry. Falls back to *default* when the env var is unset.
    """
    raw = os.environ.get(env_var) or default
    for component in raw.split(":"):
        stripped = component.strip()
        if not stripped:
            continue
        yield Path(stripped) / subpath
