"""The llama-swap router, installed from its pinned upstream release beside the engine packs."""

from __future__ import annotations

import shutil
import stat
import tarfile
import uuid
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from lilbee.catalog import runnable
from lilbee.catalog.download import RetryCallback, disk_shortfall, download_url, url_filename
from lilbee.catalog.download_progress import ProgressCallback
from lilbee.core.config import cfg
from lilbee.core.security import FileLockUnavailableError, file_lock
from lilbee.providers.fleet.build_use import make_in_use_file
from lilbee.providers.fleet.engine_store import (
    LOCK_DIR,
    PACK_DIR,
    STAGE_PREFIX,
    ChecksumError,
    EngineDiskError,
    PackUrlError,
    copy_with_digest,
    engines_root,
    file_digest,
    require_pinned_digest,
)
from lilbee.providers.fleet.first_use import RefusedInstalls
from lilbee.providers.fleet.host_caps import HostCaps
from lilbee.runtime.cancellation import CancelSignal

ROUTER_DIRNAME = "router"
ROUTER_VERSION = "v272"
LICENCE_FILE = "LICENSE.md"
_RELEASE_URL = "https://github.com/mostlygeek/llama-swap/releases/download/"
_BINARY_NAME = "llama-swap"
_BINARY_NAMES = (_BINARY_NAME, f"{_BINARY_NAME}.exe")
_LICENCE_MARK = "MIT License"
_INSTALL_LOCK_S = 1800.0


class ArchiveKind(StrEnum):
    """The archive format of a release asset."""

    TAR_GZ = "tar.gz"
    ZIP = "zip"


class RouterNotPublishedError(Exception):
    """The pinned release has no router for this computer."""


class RouterArchiveError(Exception):
    """The release asset does not hold the router and its licence notice."""


class RouterDownloadError(Exception):
    """The release asset could not be downloaded."""


@dataclass(frozen=True)
class RouterAsset:
    """One downloadable release asset of the router, for one computer."""

    filename: str
    sha256: str
    size: int
    archive: ArchiveKind
    binary: str


ROUTER_ASSETS: Mapping[str, RouterAsset] = {
    "linux-x86_64": RouterAsset(
        filename="llama-swap_272_linux_amd64.tar.gz",
        sha256="3f21b4c13f55ea5576badcd6cc1d114ccdcf74a5d56346d3ac1e777d28e0fdf1",
        size=24279896,
        archive=ArchiveKind.TAR_GZ,
        binary=_BINARY_NAME,
    ),
    "linux-arm64": RouterAsset(
        filename="llama-swap_272_linux_arm64.tar.gz",
        sha256="f7fecf8eb526d232c30b302637680e4556188c963736cb5ea10e9f47633acc48",
        size=22684827,
        archive=ArchiveKind.TAR_GZ,
        binary=_BINARY_NAME,
    ),
    "macos-arm64": RouterAsset(
        filename="llama-swap_272_darwin_arm64.tar.gz",
        sha256="ee16e99873d76e0560000742d2524d7b97932c77c9b57a402283015839ef9d25",
        size=23126118,
        archive=ArchiveKind.TAR_GZ,
        binary=_BINARY_NAME,
    ),
    "macos-x86_64": RouterAsset(
        filename="llama-swap_272_darwin_amd64.tar.gz",
        sha256="959885a60348076322629924fe06c90af37b66e5b8d69bafe3e33aa045032d44",
        size=24122979,
        archive=ArchiveKind.TAR_GZ,
        binary=_BINARY_NAME,
    ),
    "windows-x86_64": RouterAsset(
        filename="llama-swap_272_windows_amd64.zip",
        sha256="a1784767d662503bdc85649f939747109229306bd818e77a417c52da28c0084c",
        size=24174334,
        archive=ArchiveKind.ZIP,
        binary=f"{_BINARY_NAME}.exe",
    ),
}

ROUTER_REFUSALS = (
    RouterNotPublishedError,
    RouterArchiveError,
    RouterDownloadError,
    ChecksumError,
    EngineDiskError,
    FileLockUnavailableError,
    PackUrlError,
    OSError,
)
"""The errors that stop a router install for a reason the user can act on."""

_refusals = RefusedInstalls()


def _root() -> Path:
    return engines_root(cfg.models_dir)


def router_dir() -> Path:
    """The folder of the pinned router build, whether or not it is installed."""
    return _root() / ROUTER_DIRNAME / ROUTER_VERSION


def installed_router() -> Path | None:
    """The router of the pinned build, or ``None``; never installs."""
    for name in _BINARY_NAMES:
        found = router_dir() / name
        if found.is_file():
            return found
    return None


def router_asset(host: HostCaps) -> tuple[str, RouterAsset]:
    """The cell key and release asset of the router for *host*."""
    cell = f"{host.os}-{host.arch}" if host.os is not None and host.arch is not None else ""
    asset = ROUTER_ASSETS.get(cell)
    if asset is None:
        raise RouterNotPublishedError(
            f"The llama-swap router {ROUTER_VERSION} has no build for this computer. "
            "Put a llama-swap binary on PATH."
        )
    return cell, asset


def _unpack_tar(archive: Path, names: tuple[str, ...], dest: Path) -> None:
    with tarfile.open(archive) as tar:
        for name in names:
            try:
                member = tar.getmember(name)
            except KeyError:
                raise RouterArchiveError(f"The router archive holds no file {name!r}.") from None
            source = tar.extractfile(member) if member.isfile() else None
            if source is None:
                raise RouterArchiveError(f"The router archive entry {name!r} is not a file.")
            with source, (dest / name).open("wb") as target:
                shutil.copyfileobj(source, target)


def _unpack_zip(archive: Path, names: tuple[str, ...], dest: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        for name in names:
            try:
                info = bundle.getinfo(name)
            except KeyError:
                raise RouterArchiveError(f"The router archive holds no file {name!r}.") from None
            with bundle.open(info) as source, (dest / name).open("wb") as target:
                shutil.copyfileobj(source, target)


_UNPACKERS: Mapping[ArchiveKind, Callable[[Path, tuple[str, ...], Path], None]] = {
    ArchiveKind.TAR_GZ: _unpack_tar,
    ArchiveKind.ZIP: _unpack_zip,
}


def _check_digest(asset: RouterAsset, digest: str, source: Path | None) -> None:
    require_pinned_digest(f"the llama-swap router {ROUTER_VERSION}", digest, asset.sha256, source)


def _unpack(asset: RouterAsset, archive: Path, stage: Path) -> None:
    """Write the router and its licence notice from *archive* into *stage*."""
    _UNPACKERS[asset.archive](archive, (asset.binary, LICENCE_FILE), stage)
    if _LICENCE_MARK not in (stage / LICENCE_FILE).read_text(encoding="utf-8", errors="replace"):
        raise RouterArchiveError(f"The router archive holds no {_LICENCE_MARK} notice.")
    binary = stage / asset.binary
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)


def _fetch(
    asset: RouterAsset,
    stage: Path,
    on_progress: ProgressCallback | None,
    on_retry: RetryCallback | None,
    cancel: CancelSignal | None,
) -> tuple[Path, str, Path | None]:
    """A private archive of *asset* in *stage*, its sha256 and the packs file it came from.

    The packs copy is used when it exists, else the archive is downloaded and the file is None.
    """
    pack = _root() / PACK_DIR / asset.filename
    if pack.is_file():
        private = stage / pack.name
        return private, copy_with_digest(pack, private), pack
    url = f"{_RELEASE_URL}{ROUTER_VERSION}/{asset.filename}"
    shortfall = disk_shortfall(stage, url_filename(url), asset.size)
    if shortfall is not None:
        raise EngineDiskError(shortfall)
    try:
        downloaded = download_url(
            url, stage, asset.size, on_progress=on_progress, on_retry=on_retry, cancel=cancel
        )
    except (OSError, RuntimeError) as exc:
        raise RouterDownloadError(
            f"The llama-swap router {ROUTER_VERSION} is not installed and cannot be downloaded. "
            f"Copy {asset.filename} into {pack.parent} and run the install again."
        ) from exc
    return downloaded, file_digest(downloaded), None


def install_router(
    *,
    on_progress: ProgressCallback | None = None,
    on_retry: RetryCallback | None = None,
    cancel: CancelSignal | None = None,
) -> Path:
    """Install the pinned router once and return its binary; a second caller finds it installed.

    A copy of the archive in the packs folder is used with no download. The archive is copied
    into the stage folder, hashed as it is copied, and read only from that copy.
    """
    _, asset = router_asset(runnable.session_host())
    root = _root()
    with file_lock(root / LOCK_DIR / ROUTER_DIRNAME, _INSTALL_LOCK_S):
        found = installed_router()
        if found is not None:
            return found
        final = router_dir()
        stage = final.parent / f"{STAGE_PREFIX}{uuid.uuid4().hex}"
        stage.mkdir(parents=True)
        try:
            archive, digest, source = _fetch(asset, stage, on_progress, on_retry, cancel)
            _check_digest(asset, digest, source)
            _unpack(asset, archive, stage)
            archive.unlink()
            make_in_use_file(stage)
            shutil.rmtree(final, ignore_errors=True)
            stage.rename(final)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    _refusals.forget(_refusal_key())
    return final / asset.binary


def _refusal_key() -> tuple[str, str, str]:
    return (str(_root()), ROUTER_VERSION, ROUTER_DIRNAME)


def router_on_first_use() -> Path:
    """The router, installed when absent; a refused install is remembered for the process.

    Raises one of ``ROUTER_REFUSALS``; each later lookup raises the same refusal and fetches
    nothing. ``install_router`` always tries again.
    """
    found = installed_router()
    if found is not None:
        return found
    key = _refusal_key()
    _refusals.raise_if_refused(key)
    try:
        return install_router()
    except ROUTER_REFUSALS as exc:
        _refusals.remember(key, exc)
        raise
