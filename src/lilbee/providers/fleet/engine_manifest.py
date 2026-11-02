"""The engine manifest: pinned pack digests, the release-manifest parser, the host checks."""

from __future__ import annotations

import platform
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeVar
from urllib.parse import unquote, urlsplit

from lilbee.providers.fleet.engine_store import (
    PinLock,
    PlatformAsset,
    UnsupportedPlatformError,
    licence_permitted,
)
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.host_caps import HostCaps, HostOS

SCHEMA = 1
RELEASE_URL_PREFIX = "https://github.com/tobocop2/lilbee-engines/releases/download/"
_WHEEL_SUFFIX = ".whl"
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")
_VERSION = re.compile(r"\d+(?:\.\d+)*")
_NUMBER = re.compile(r"\d+")
_FLOOR_LABELS = {
    HostOS.MACOS: ("macOS", "Update macOS"),
    HostOS.LINUX: ("glibc", "Update the system"),
    HostOS.WINDOWS: ("Windows", "Update Windows"),
}
_OWN_ENGINE_HINT = ", or set LILBEE_LLAMA_SERVER_PATH to an engine you built"

_T = TypeVar("_T")

# The release manifests of the engine packs, one per engine, copied as published. An engine
# with no published pack has no entry. Each manifest keeps the layout of the release's
# manifest.json: schema, engine, release and cells, where cells maps <os>-<arch>-<backend>
# to the row of one wheel. This table is the only trust root for a pack digest: a manifest
# fetched at run time supplies the url, the size and the floor, never the digest.
# Hand-owned until the generator of design 6 writes it.
MANIFESTS: Mapping[str, Mapping[str, Any]] = {}


class ManifestError(Exception):
    """The manifest breaks its schema."""


class UnsupportedOsError(Exception):
    """The operating system of this computer is older than the floor of the build."""


@dataclass(frozen=True)
class ManifestCell:
    """One build of the manifest: its host, backend, upstream version and wheel."""

    os: str
    arch: str
    backend: str
    version: str
    asset: PlatformAsset


@dataclass(frozen=True)
class Manifest:
    """The pinned builds of one engine release, keyed by ``<os>-<arch>-<backend>``."""

    engine: str
    release: str
    licences: tuple[str, ...]
    cells: Mapping[str, ManifestCell]

    def select(self, os_name: str, arch: str, backend: str, os_version: str) -> PlatformAsset:
        """The wheel for this host, or a refusal that names the problem and the fix."""
        cell = find_cell(self.engine, self.cells, os_name, arch, backend)
        require_floor(self.engine, cell.version, backend, cell.asset, HostOS(os_name), os_version)
        return cell.asset


def _join(os_name: str, arch: str, backend: str) -> str:
    return f"{os_name}-{arch}-{backend}"


def cell_key(host: HostCaps, backend: str | None) -> str | None:
    """The manifest key of the cell *host* runs, or ``None`` when a part of it is unknown."""
    if host.os is None or host.arch is None or backend is None:
        return None
    return _join(host.os.value, host.arch.value, backend)


def _pinned_digest(engine: str, key: str) -> str | None:
    manifest = MANIFESTS.get(engine)
    if manifest is None:
        return None
    if manifest["schema"] != SCHEMA:
        raise ValueError(f"The manifest of engine {engine} has schema {manifest['schema']}.")
    cell = manifest["cells"].get(key)
    return cell["sha256"] if cell is not None else None


def pack_digest(engine: str, host: HostCaps, backend: str | None) -> str | None:
    """The pinned sha256 of the wheel of *engine* for *host*, or ``None`` without one."""
    key = cell_key(host, backend)
    return None if key is None else _pinned_digest(engine, key)


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


def _text(row: Mapping[str, Any], name: str) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value:  # untyped JSON
        raise ManifestError(f"The manifest field {name!r} is missing or is not text.")
    return value


def _url(row: Mapping[str, Any]) -> str:
    url = _text(row, "url")
    path = urlsplit(url).path
    climbs = any(unquote(part) == ".." for part in path.split("/"))
    if (
        not url.startswith(RELEASE_URL_PREFIX)
        or not path.endswith(_WHEEL_SUFFIX)
        or "?" in url
        or "#" in url
        or climbs
    ):
        raise ManifestError(f"The manifest url {url!r} is not a wheel of an engine release.")
    return url


def _digest(row: Mapping[str, Any]) -> str:
    digest = _text(row, "sha256")
    if not _SHA256.fullmatch(digest):
        raise ManifestError(f"The manifest sha256 {digest!r} is not 64 hex digits.")
    return digest


def _size(row: Mapping[str, Any]) -> int:
    size = row.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:  # untyped JSON
        raise ManifestError(f"The manifest size {size!r} is not a positive number.")
    return size


def _floor(row: Mapping[str, Any]) -> str | None:
    floor = row.get("min_os", "")
    if floor is None:
        return None
    if not isinstance(floor, str) or not _VERSION.fullmatch(floor):  # untyped JSON
        raise ManifestError(f"The manifest min_os {floor!r} is not a version or null.")
    return floor


def _cell(key: str, row: object) -> ManifestCell:
    if not isinstance(row, dict):  # untyped JSON
        raise ManifestError(f"The manifest cell {key!r} is not an object.")
    os_name, arch, backend = _text(row, "os"), _text(row, "arch"), _text(row, "backend")
    if key != _join(os_name, arch, backend):
        raise ManifestError(f"The manifest cell {key!r} does not match its os, arch and backend.")
    asset = PlatformAsset(_url(row), _digest(row), _size(row), _floor(row))
    return ManifestCell(os_name, arch, backend, _text(row, "version"), asset)


def _licences(doc: Mapping[str, Any]) -> tuple[str, ...]:
    licences = doc.get("licenses")
    if not isinstance(licences, list) or not licences:  # untyped JSON
        raise ManifestError("The manifest lists no licence.")
    for expression in licences:
        if not isinstance(expression, str) or not licence_permitted(expression):
            raise ManifestError(
                f"The manifest licence {expression!r} is not on the permissive list."
            )
    return tuple(licences)


def parse_manifest(doc: Mapping[str, Any]) -> Manifest:
    """The manifest in *doc*; raises ``ManifestError`` for any breach of the schema."""
    if doc.get("schema") != SCHEMA:
        raise ManifestError(f"The manifest schema {doc.get('schema')!r} is not {SCHEMA}.")
    cells = doc.get("cells")
    if not isinstance(cells, dict) or not cells:  # untyped JSON
        raise ManifestError("The manifest lists no cells.")
    engine = _text(doc, "engine")
    parsed = {key: _cell(key, row) for key, row in cells.items()}
    for key, cell in parsed.items():
        pinned = _pinned_digest(engine, key)
        if pinned is None or pinned.lower() != cell.asset.sha256.lower():
            raise ManifestError(
                f"The manifest sha256 of cell {key!r} of engine {engine} is not its pinned digest."
            )
    return Manifest(
        engine=engine,
        release=_text(doc, "release"),
        licences=_licences(doc),
        cells=parsed,
    )
