"""The engine digest table lookups and the host checks."""

from __future__ import annotations

import platform
import re
from collections.abc import Mapping
from typing import Any, TypeVar

from lilbee.providers.fleet.engine_digests import MANIFESTS
from lilbee.providers.fleet.engine_store import (
    PinLock,
    PlatformAsset,
    UnsupportedPlatformError,
)
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.host_caps import HostCaps, HostOS

_NUMBER = re.compile(r"\d+")
_FLOOR_LABELS = {
    HostOS.MACOS: ("macOS", "Update macOS"),
    HostOS.LINUX: ("glibc", "Update the system"),
    HostOS.WINDOWS: ("Windows", "Update Windows"),
}
_OWN_ENGINE_HINT = ", or set LILBEE_LLAMA_SERVER_PATH to an engine you built"

_T = TypeVar("_T")


class UnsupportedOsError(Exception):
    """The operating system of this computer is older than the floor of the build."""


def _join(os_name: str, arch: str, backend: str) -> str:
    return f"{os_name}-{arch}-{backend}"


def cell_key(host: HostCaps, backend: str | None) -> str | None:
    """The table key of the cell *host* runs, or ``None`` when a part of it is unknown."""
    if host.os is None or host.arch is None or backend is None:
        return None
    return _join(host.os.value, host.arch.value, backend)


def pack_platforms(
    engine: str, version: str, host: HostCaps, backend: str | None
) -> dict[str, Mapping[str, Any]]:
    """The table row of *engine* at exactly *version* for *host*, by cell key; empty without one."""
    key = cell_key(host, backend)
    if key is None:
        return {}
    row = MANIFESTS.get(engine, {}).get(version, {}).get(key)
    return {} if row is None else {key: row}


def pack_digest(engine: str, version: str, host: HostCaps, backend: str | None) -> str | None:
    """The sha256 of the wheel of *engine* at exactly *version* for *host*, or ``None``."""
    rows = pack_platforms(engine, version, host, backend)
    return next((row["sha256"] for row in rows.values()), None)


def find_cell(engine: str, cells: Mapping[str, _T], os_name: str, arch: str, backend: str) -> _T:
    """The entry of *cells* for this host and backend; no other backend is chosen."""
    found = cells.get(_join(os_name, arch, backend))
    if found is not None:
        return found
    prefix = _join(os_name, arch, "")
    here = sorted(key.removeprefix(prefix) for key in cells if key.startswith(prefix))
    raise UnsupportedPlatformError(
        f"Engine {engine} has no {backend} build for {os_name}-{arch}. "
        f"Builds for this computer: {', '.join(here) or 'none'}."
    )


def _numbers(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in _NUMBER.findall(version))


def _computer_has(host_os: HostOS, label: str, host_version: str) -> str:
    if host_version:
        return f"{label} {host_version}"
    return "no glibc" if host_os is HostOS.LINUX else f"an unreadable {label} version"


def require_floor(
    engine: str,
    version: str,
    backend: str,
    asset: PlatformAsset,
    host_os: HostOS,
    host_version: str,
) -> None:
    """Raise ``UnsupportedOsError`` when the host is older than the floor of *asset*."""
    if asset.min_os is None or _numbers(host_version) >= _numbers(asset.min_os):
        return
    label, update = _FLOOR_LABELS[host_os]
    raise UnsupportedOsError(
        f"Engine {engine} {version} ({backend}) needs {label} {asset.min_os} or later. "
        f"This computer has {_computer_has(host_os, label, host_version)}. "
        f"{update}{_OWN_ENGINE_HINT if engine == EngineName.LLAMA else ''}."
    )


def host_os_version(host_os: HostOS) -> str:
    """The macOS version, the glibc version, or the Windows build; empty when unreadable."""
    if host_os is HostOS.MACOS:
        return platform.mac_ver()[0]
    if host_os is HostOS.LINUX:
        libc, version = platform.libc_ver()
        return version if libc == "glibc" else ""
    return platform.version()


def host_cell(lock: PinLock, host: HostCaps) -> str:
    """The cell key of *lock* for *host*; empty for a pin with no cells.

    A pin with cells refuses a computer with no cell or a low OS.
    """
    if not lock.platforms:
        return ""
    if host.os is None or host.arch is None:
        raise UnsupportedPlatformError(
            f"Engine {lock.engine} has no {lock.backend} build for this computer."
        )
    asset = find_cell(lock.engine, lock.platforms, host.os, host.arch, lock.backend)
    require_floor(lock.engine, lock.version, lock.backend, asset, host.os, host_os_version(host.os))
    return _join(host.os, host.arch, lock.backend)
