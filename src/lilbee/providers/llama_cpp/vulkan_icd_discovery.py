"""Cross-platform discovery of installed Vulkan ICD manifests.

The Vulkan loader scans platform-specific locations at every
``vkCreateInstance`` to discover installable client drivers (ICDs).
``disable_conflicting_vulkan_icds`` mirrors that walk so it can identify
which vendors are installed *without* calling ``vkCreateInstance``
itself. Doing the live probe pre-loads every vendor's ICD into the
process before the disable env var arrives; at least one ICD on the b473
QA box (AMDVLK on Windows) self-pins its DLL handle and survives the
loader's ``FreeLibrary``, so the disable lands too late.

The single public entry point is
:func:`iter_vulkan_manifest_paths`, which dispatches by platform:

* **Windows** scans the registry: legacy ``HKLM\\SOFTWARE\\Khronos\\Vulkan\\Drivers``
  (plus the WOW6432Node mirror) and PnP driver-store keys under the
  Display Adapter and Software Component device-class GUIDs. Manifest
  paths live in the ``VulkanDriverName`` / ``VulkanDriverNameWow``
  REG_SZ / REG_MULTI_SZ values.
* **Linux** walks the XDG directory hierarchy documented at
  https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderDriverInterface.md
  ("Driver Discovery on Linux"): ``$XDG_CONFIG_HOME``, ``$XDG_CONFIG_DIRS``,
  ``SYSCONFDIR``, ``EXTRASYSCONFDIR``, ``$XDG_DATA_HOME``, ``$XDG_DATA_DIRS``,
  each with ``/vulkan/icd.d`` appended, plus the Flatpak export
  directories that ship Vulkan ICDs into sandboxed runtimes. Each
  directory is globbed for ``*.json`` manifests.
* **macOS** yields nothing: lilbee's macOS wheel uses Metal directly
  and skips the Vulkan loader.

Manifest filenames are stable across platforms (``amd_icd64.json``,
``nv-vk64.json``, ``radeon_icd.x86_64.json``, ``intel_icd.x86_64.json``,
``nvidia_icd.json``, etc.), and ``VK_LOADER_DRIVERS_DISABLE`` matches
on the *filename* not the library path, so callers only need the
filename out of each yielded absolute path.
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
    """Yield every Vulkan ICD manifest path the loader would discover here.

    Each yielded string is the absolute path to a ``.json`` manifest.
    The bare filename is what ``VK_LOADER_DRIVERS_DISABLE`` matches
    against, so callers pass it through their vendor classifier.

    Returns an empty iterator on platforms with no Vulkan-loader
    discovery (macOS) or where the platform's enumeration mechanism
    isn't available.
    """
    if sys.platform == "win32":
        yield from _iter_windows_vulkan_manifest_paths()
    elif sys.platform.startswith("linux"):
        yield from _iter_linux_vulkan_manifest_paths()
    else:
        # darwin / unknown: the macOS wheel uses Metal directly so the
        # Vulkan loader isn't present; the empty yield-from keeps the
        # function a generator without yielding anything.
        yield from ()


# Windows PnP device-class GUIDs that publish Vulkan ICD manifest paths.
# These are stable Microsoft-defined class GUIDs that long predate Vulkan;
# the Khronos loader spec names them as the authoritative locations.
#
# {4d36e968-e325-11ce-bfc1-08002be10318} = "Display adapters" device setup class.
#   https://learn.microsoft.com/en-us/windows-hardware/drivers/install/system-defined-device-setup-classes-available-to-vendors
# {5c4c3332-344d-483c-8739-259e934c9cc8} = "SoftwareComponent" device setup class.
#   https://learn.microsoft.com/en-us/windows-hardware/drivers/install/system-defined-device-setup-classes-available-to-vendors
# Both registry paths plus the Khronos software-driver key are listed in:
#   https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderDriverInterface.md#driver-discovery-on-windows
_PNP_DISPLAY_ADAPTER_CLASS_GUID = "{4d36e968-e325-11ce-bfc1-08002be10318}"
_PNP_SOFTWARE_COMPONENT_CLASS_GUID = "{5c4c3332-344d-483c-8739-259e934c9cc8}"

# Legacy Khronos-managed registry locations. The WOW6432Node mirror is
# scanned for 32-bit ICDs on 64-bit Windows, per the same Khronos spec.
# Each value name is the absolute path to a ``.json`` manifest, and the
# REG_DWORD value is the enabled flag (0 = enabled, non-zero = disabled).
_KHRONOS_DRIVERS_KEYS = (
    r"SOFTWARE\Khronos\Vulkan\Drivers",
    r"SOFTWARE\WOW6432Node\Khronos\Vulkan\Drivers",
)

# Per-adapter REG_SZ / REG_MULTI_SZ value names the Khronos loader spec
# reads off each device-class subkey:
#   https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderDriverInterface.md#driver-discovery-on-windows
_PNP_VULKAN_VALUE_NAMES = ("VulkanDriverName", "VulkanDriverNameWow")


def _iter_windows_vulkan_manifest_paths() -> Iterator[str]:
    """Yield manifest paths the Windows Vulkan loader would discover.

    Walks the four locations the LoaderDriverInterface spec mandates: legacy
    Khronos software-driver path, its WOW6432Node mirror, and the PnP
    display-adapter and software-component class GUID subkeys.
    """
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


# Suffix appended to every XDG-derived path in the Khronos loader's Linux
# discovery, per
#   https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderDriverInterface.md#driver-discovery-on-linux
_VULKAN_ICD_SUBPATH = "vulkan/icd.d"

# Fixed system paths the Khronos loader hard-codes alongside the XDG
# variables. ``SYSCONFDIR`` / ``EXTRASYSCONFDIR`` are loader build-time
# constants; on the distros lilbee ships against (Ubuntu, Fedora, Arch,
# Debian) they expand to ``/usr/local/etc`` and ``/etc`` respectively
# (see the same LoaderDriverInterface.md "Driver Discovery on Linux"
# table). The XDG basedir defaults are pulled from
#   https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html
_LINUX_FIXED_ETC_ICD_DIRS: tuple[str, ...] = (
    "/usr/local/etc/vulkan/icd.d",
    "/etc/vulkan/icd.d",
)

# Flatpak per-runtime Vulkan ICD trees. Inside a Flatpak sandbox the
# loader picks these up via the runtime's XDG_DATA_DIRS, but a host-side
# launch of lilbee may not inherit that env, so we walk the conventional
# export paths defensively. The Flatpak Vulkan extension convention is
# documented at
#   https://docs.flatpak.org/en/latest/sandbox-permissions.html#device-access
_LINUX_FLATPAK_ICD_DIRS: tuple[str, ...] = (
    "~/.local/share/flatpak/exports/share/vulkan/icd.d",
    "/var/lib/flatpak/exports/share/vulkan/icd.d",
)


def _iter_linux_vulkan_manifest_paths() -> Iterator[str]:
    """Yield manifest paths the Linux Vulkan loader would discover.

    Walks the search path documented in the Khronos LoaderDriverInterface
    spec (``$XDG_CONFIG_HOME``, ``$XDG_CONFIG_DIRS``, ``SYSCONFDIR``,
    ``EXTRASYSCONFDIR``, ``$XDG_DATA_HOME``, ``$XDG_DATA_DIRS``) plus the
    standard Flatpak export trees, each with ``vulkan/icd.d`` appended,
    and globs ``*.json`` in every directory. Yields absolute paths;
    duplicate directories (e.g. when ``XDG_DATA_DIRS`` contains a colon-
    separated path that already matches a fallback) are de-duplicated so
    a single manifest isn't yielded twice.
    """
    seen_dirs: set[Path] = set()
    seen_files: set[Path] = set()
    for directory in _linux_vulkan_icd_directories():
        try:
            resolved = directory.expanduser()
        except RuntimeError:
            # PosixPath.expanduser raises when HOME is unset; skip rather
            # than abort the whole walk.
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
    """Yield each Linux ICD search directory in loader-spec order.

    The XDG defaults follow basedir spec
    (https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html):
    ``$XDG_CONFIG_HOME`` -> ``~/.config``, ``$XDG_CONFIG_DIRS`` -> ``/etc/xdg``,
    ``$XDG_DATA_HOME`` -> ``~/.local/share``,
    ``$XDG_DATA_DIRS`` -> ``/usr/local/share:/usr/share``.
    Empty path components in the colon-separated env vars are dropped
    (matches the loader's own behaviour for the "extra slash" edge case
    reported in Vulkan-Loader#2331).
    """
    yield from _xdg_dirs("XDG_CONFIG_HOME", "~/.config", _VULKAN_ICD_SUBPATH)
    yield from _xdg_dirs("XDG_CONFIG_DIRS", "/etc/xdg", _VULKAN_ICD_SUBPATH)
    for fixed in _LINUX_FIXED_ETC_ICD_DIRS:
        yield Path(fixed)
    yield from _xdg_dirs("XDG_DATA_HOME", "~/.local/share", _VULKAN_ICD_SUBPATH)
    yield from _xdg_dirs("XDG_DATA_DIRS", "/usr/local/share:/usr/share", _VULKAN_ICD_SUBPATH)
    for flatpak in _LINUX_FLATPAK_ICD_DIRS:
        yield Path(flatpak)


def _xdg_dirs(env_var: str, default: str, subpath: str) -> Iterator[Path]:
    """Expand a colon-separated XDG path env var into per-directory ``Path``s.

    Drops empty components (the "extra slash in XDG_DATA_DIRS" loader
    quirk, Vulkan-Loader#2331) and appends *subpath* to each remaining
    entry. Falls back to *default* when the env var is unset.
    """
    raw = os.environ.get(env_var) or default
    for component in raw.split(":"):
        stripped = component.strip()
        if not stripped:
            continue
        yield Path(stripped) / subpath
