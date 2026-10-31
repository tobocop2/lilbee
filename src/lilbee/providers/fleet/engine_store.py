"""Engine builds installed side by side on disk, one directory per build, a record written last."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import stat
import sys
import uuid
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from email import message_from_string
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath

from lilbee.catalog.download import disk_shortfall, retry_transfer
from lilbee.core.security import file_lock

log = logging.getLogger(__name__)

RECORD_NAME = "ENGINE-RECORD.json"
PERMISSIVE_LICENCES = frozenset({"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause"})
_LOCK_DIR = ".locks"
_LOCK_TIMEOUT_S = 300.0
_STAGE_PREFIX = ".stage-"
_DOWNLOAD_PREFIX = ".download-"
_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
_TOOL_PATH = re.compile(r"[^/]+/bin/[^/]+")
_LICENCE_JOINER = re.compile(r"\s+(?:OR|AND)\s+")
_PACK_DIR = "packs"
_PACKAGE_INDEX = "https://lilbee.sh"
_PACKAGE_PREFIX = "lilbee-engine-"


class ChecksumError(Exception):
    """The wheel does not match the sha256 in the pin."""


class LicenceRefusedError(Exception):
    """A licence of the engine is not on the permissive list."""


class UnsafeWheelError(Exception):
    """The wheel holds an entry that would land outside its install directory."""


class EngineConflictError(Exception):
    """A build of the same engine, version and backend exists with a different wheel."""


class MissingPackError(Exception):
    """The build is not installed, has no pack on disk, and cannot be downloaded."""


class UnsupportedPlatformError(Exception):
    """The pin lists no wheel for this platform."""


class EngineDiskError(Exception):
    """The volume of the engines root has too little free space for the wheel."""


class InstallTool(StrEnum):
    """The installer that runs the package route."""

    UV = "uv"
    PIP = "pip"


@dataclass(frozen=True)
class PlatformAsset:
    """One downloadable wheel of a pin, for one platform."""

    repo: str
    revision: str
    filename: str
    sha256: str
    size: int
    min_os: str | None


@dataclass(frozen=True)
class PinLock:
    """The pin of one engine build: name, version, backend, licences and wheel digest."""

    engine: str
    version: str
    backend: str
    platforms: Mapping[str, PlatformAsset]
    licences: tuple[str, ...]
    sha256: str

    def __post_init__(self) -> None:
        for part in (self.engine, self.version, self.backend):
            if not _NAME_PATTERN.fullmatch(part):
                raise ValueError(f"{part!r} cannot name an engine directory")


@dataclass(frozen=True)
class InstalledEngine:
    """One complete build, as its record describes it."""

    engine: str
    version: str
    backend: str
    sha256: str
    licences: tuple[str, ...]
    installed_at: str
    path: Path
    binary: Path | None


def licence_permitted(expression: str) -> bool:
    """True when every OR/AND term of *expression* is on the permissive list."""
    terms = [term for term in _LICENCE_JOINER.split(expression.strip()) if term]
    return bool(terms) and all(term in PERMISSIVE_LICENCES for term in terms)


def _check_licences(lock: PinLock) -> None:
    if not lock.licences:
        raise LicenceRefusedError(f"Engine {lock.engine} {lock.version} lists no licence.")
    for expression in lock.licences:
        if not licence_permitted(expression):
            raise LicenceRefusedError(
                f"Engine {lock.engine} {lock.version} uses the licence {expression!r}, "
                f"which is not on the permissive list."
            )


def _check_wheel(lock: PinLock, wheel: Path) -> None:
    with wheel.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != lock.sha256.lower():
        raise ChecksumError(
            f"The wheel for engine {lock.engine} {lock.version} does not match its pin."
        )
    with zipfile.ZipFile(wheel) as archive:
        _refuse_escaping_entries(lock, archive)
        _refuse_linked_binary(lock, archive)
        names = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        metadata = message_from_string(archive.read(names[0]).decode("utf-8")) if names else None
    expression = (metadata.get("License-Expression") if metadata else None) or ""
    if not licence_permitted(expression):
        raise LicenceRefusedError(
            f"The wheel for engine {lock.engine} {lock.version} declares the licence "
            f"{expression!r}, which is not on the permissive list."
        )


def _leaves_directory(name: str) -> bool:
    """True when the zip entry *name* is absolute, has a drive, or climbs out with ``..``."""
    flat = name.replace("\\", "/")
    return (
        PurePosixPath(flat).is_absolute()
        or bool(PureWindowsPath(flat).drive)
        or ".." in PurePosixPath(flat).parts
    )


def _refuse_escaping_entries(lock: PinLock, archive: zipfile.ZipFile) -> None:
    for name in archive.namelist():
        if _leaves_directory(name):
            raise UnsafeWheelError(
                f"The wheel for engine {lock.engine} {lock.version} holds the entry {name!r}, "
                f"which points outside the engine directory."
            )


def _refuse_linked_binary(lock: PinLock, archive: zipfile.ZipFile) -> None:
    name = _binary_of(archive)
    if name is not None and stat.S_ISLNK(archive.getinfo(name).external_attr >> 16):
        raise UnsafeWheelError(
            f"The wheel for engine {lock.engine} {lock.version} holds the tool {name!r} "
            f"as a symbolic link."
        )


def _binary_of(archive: zipfile.ZipFile) -> str | None:
    """The first ``<package>/bin/<tool>`` file of the wheel, as a POSIX path."""
    tools = sorted(n for n in archive.namelist() if _TOOL_PATH.fullmatch(n))
    return tools[0] if tools else None


def _binary_usable(binary: Path, directory: Path) -> bool:
    if not binary.resolve().is_relative_to(directory.resolve()) or not binary.is_file():
        return False
    return sys.platform == "win32" or bool(binary.stat().st_mode & stat.S_IXUSR)


def _unpack(wheel: Path, stage: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(stage)
        for info in archive.infolist():
            mode = info.external_attr >> 16
            if stat.S_ISREG(mode) and mode & stat.S_IXUSR:
                target = stage / info.filename
                target.chmod(target.stat().st_mode | stat.S_IXUSR)


def _read_record(directory: Path) -> InstalledEngine | None:
    """The record of *directory* when it is a complete build, else None."""
    try:
        data = json.loads((directory / RECORD_NAME).read_text(encoding="utf-8"))
        found = InstalledEngine(
            engine=data["engine"],
            version=data["version"],
            backend=data["backend"],
            sha256=data["sha256"],
            licences=tuple(data["licences"]),
            installed_at=data["installed_at"],
            path=directory,
            binary=directory / data["binary"] if data["binary"] is not None else None,
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if (
        directory.name != f"{found.version}-{found.backend}"
        or directory.parent.name != found.engine
    ):
        return None
    if found.binary is not None and not _binary_usable(found.binary, directory):
        return None
    return found


class EngineStore:
    """The engines root: ``<root>/<engine>/<version>-<backend>/``, complete only with its record."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def build_dir(self, lock: PinLock) -> Path:
        return self.root / lock.engine / f"{lock.version}-{lock.backend}"

    def installed(self) -> list[InstalledEngine]:
        """Every complete build; a directory without a valid record is not listed."""
        if not self.root.is_dir():
            return []
        builds = (
            _read_record(build)
            for engine in sorted(self.root.iterdir())
            if engine.is_dir() and not engine.name.startswith(".")
            for build in sorted(engine.iterdir())
            if build.is_dir() and not build.name.startswith(".")
        )
        return [found for found in builds if found is not None]

    def pack_path(self, lock: PinLock) -> Path:
        """Where a hand-copied wheel of *lock* is looked up."""
        return self.root / _PACK_DIR / f"{lock.engine}-{lock.version}-{lock.backend}.whl"

    def _complete_build(self, lock: PinLock) -> Path | None:
        """The directory of the complete build of *lock*, None when absent or damaged."""
        existing = _read_record(self.build_dir(lock))
        if existing is None:
            return None
        if existing.sha256.lower() != lock.sha256.lower():
            raise EngineConflictError(
                f"Engine {lock.engine} {lock.version} ({lock.backend}) is installed "
                f"from a different wheel than its pin. Remove that build first."
            )
        return existing.path

    def ensure(self, lock: PinLock, fetch: Callable[[PinLock, Path], Path]) -> Path:
        """Return the build of *lock*: installed, else from a pack, else from *fetch*."""
        _check_licences(lock)
        found = self._complete_build(lock)
        if found is not None:
            return found
        pack = self.pack_path(lock)
        if pack.is_file():
            return self.install_from_wheel(lock, pack)
        scratch = self.root / f"{_DOWNLOAD_PREFIX}{uuid.uuid4().hex}"
        try:
            try:
                wheel = fetch(lock, scratch)
            except (OSError, RuntimeError) as exc:
                raise MissingPackError(
                    f"Engine {lock.engine} {lock.version} ({lock.backend}) is not installed and "
                    f"cannot be downloaded. Copy {pack.name} into {pack.parent} "
                    f"and run the install again."
                ) from exc
            return self.install_from_wheel(lock, wheel)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def install_from_wheel(self, lock: PinLock, wheel: Path) -> Path:
        """Install *wheel* as the build of *lock*; return the build directory."""
        _check_licences(lock)
        _check_wheel(lock, wheel)
        final = self.build_dir(lock)
        with file_lock(self.root / _LOCK_DIR / lock.engine, _LOCK_TIMEOUT_S):
            if self._complete_build(lock) is not None:
                return final
            stage = final.parent / f"{_STAGE_PREFIX}{uuid.uuid4().hex}"
            final.parent.mkdir(parents=True, exist_ok=True)
            try:
                _unpack(wheel, stage)
                with zipfile.ZipFile(wheel) as archive:
                    binary = _binary_of(archive)
                self._write_record(stage, lock, binary)
                if final.exists():
                    shutil.rmtree(final)
                stage.rename(final)
            finally:
                shutil.rmtree(stage, ignore_errors=True)
        return final

    @staticmethod
    def _write_record(stage: Path, lock: PinLock, binary: str | None = None) -> None:
        record = {
            "engine": lock.engine,
            "version": lock.version,
            "backend": lock.backend,
            "sha256": lock.sha256,
            "licences": list(lock.licences),
            "installed_at": datetime.now(UTC).isoformat(),
            "binary": binary,
        }
        (stage / RECORD_NAME).write_text(json.dumps(record), encoding="utf-8")


def package_name(lock: PinLock) -> str:
    """The name of the engine's wheel on the package index."""
    return _PACKAGE_PREFIX + re.sub(r"[^a-z0-9]+", "-", lock.engine.lower()).strip("-")


def pip_requirement(lock: PinLock) -> str:
    """The pinned, hashed requirement line the package route reads from standard input."""
    return f"{package_name(lock)}=={lock.version} --hash=sha256:{lock.sha256.lower()}\n"


def pip_install_argv(lock: PinLock, stage: Path, tool: InstallTool = InstallTool.UV) -> list[str]:
    """The command that installs the pinned wheel of *lock* into *stage* from the package index."""
    runner = ["uv", "pip"] if tool == InstallTool.UV else [sys.executable, "-m", "pip"]
    return [
        *runner,
        "install",
        "--target",
        str(stage),
        "--no-deps",
        "--only-binary=:all:",
        "--require-hashes",
        "--index-url",
        f"{_PACKAGE_INDEX}/{lock.backend}/",
        "-r",
        "-",
    ]


def _hub_download(
    *, repo_id: str, revision: str, filename: str, local_dir: str | None = None
) -> str:
    from huggingface_hub import hf_hub_download

    return str(
        hf_hub_download(repo_id=repo_id, revision=revision, filename=filename, local_dir=local_dir)
    )


def fetch_frozen(
    lock: PinLock,
    platform: str,
    *,
    dest: Path,
    downloader: Callable[..., str] = _hub_download,
) -> Path:
    """Download the wheel of *lock* for *platform* at its pinned commit into *dest*, unchecked."""
    from huggingface_hub.errors import (
        EntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    asset = lock.platforms.get(platform)
    if asset is None:
        raise UnsupportedPlatformError(
            f"Engine {lock.engine} {lock.version} has no build for {platform}."
        )
    shortfall = disk_shortfall(dest, asset.repo, asset.size)
    if shortfall is not None:
        raise EngineDiskError(shortfall)
    try:
        return Path(
            retry_transfer(
                lambda: downloader(
                    repo_id=asset.repo,
                    revision=asset.revision,
                    filename=asset.filename,
                    local_dir=str(dest),
                )
            )
        )
    except (EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError) as exc:
        raise FileNotFoundError(
            f"The hub has no {asset.filename} at {asset.revision} in {asset.repo}."
        ) from exc


__all__ = [
    "RECORD_NAME",
    "ChecksumError",
    "EngineConflictError",
    "EngineDiskError",
    "EngineStore",
    "InstallTool",
    "InstalledEngine",
    "LicenceRefusedError",
    "MissingPackError",
    "PinLock",
    "PlatformAsset",
    "UnsafeWheelError",
    "UnsupportedPlatformError",
    "fetch_frozen",
    "licence_permitted",
    "package_name",
    "pip_install_argv",
    "pip_requirement",
]
