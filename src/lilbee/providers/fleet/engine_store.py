"""Engine builds installed side by side on disk, one directory per build, a record written last."""

from __future__ import annotations

import codecs
import errno
import hashlib
import json
import logging
import os
import re
import shutil
import stat
import sys
import uuid
import zipfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from email import message_from_string
from functools import partial
from pathlib import Path, PurePosixPath, PureWindowsPath

from lilbee.catalog.download import (
    RetryCallback,
    disk_shortfall,
    download_url,
    url_filename,
)
from lilbee.catalog.download_progress import ProgressCallback
from lilbee.core.security import FileLockUnavailableError, file_lock
from lilbee.providers.fleet.engine_crates import MPL_CRATE_CHECKSUMS, MPL_PYPI_CHECKSUMS
from lilbee.providers.fleet.engine_record import PACKAGES_URL
from lilbee.providers.fleet.wheel_server import server_entry, server_problems
from lilbee.runtime.cancellation import CancelSignal

log = logging.getLogger(__name__)

RECORD_NAME = "ENGINE-RECORD.json"
PERMISSIVE_LICENCES = frozenset(
    {
        "MIT",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "PSF-2.0",
        "ISC",
        "Zlib",
        "0BSD",
        "MIT-0",
        "Unlicense",
        "CC0-1.0",
        "Unicode-3.0",
        "CNRI-Python",
        "CDLA-Permissive-2.0",
        "Apache-2.0 WITH LLVM-exception",
        "NCSA",
        "BSL-1.0",
        "TCL",
        "bzip2-1.0.6",
        "X11",
        "blessing",
        "Python-2.0",
    }
)


@dataclass(frozen=True)
class NoticeRule:
    """The notice file that a term needs, and the test that its text must pass."""

    file: str
    accepts: Callable[[str], bool]


def _holds(pattern: str) -> Callable[[str], bool]:
    compiled = re.compile(pattern)
    return lambda text: compiled.search(text) is not None


_CRATE_HOST = "static.crates.io"
"""The host of the crate download URL in a pack notice. Nothing fetches it."""
_CRATE_LINE = re.compile(
    rf"https://{re.escape(_CRATE_HOST)}/crates/(?P<crate>[\w-]+)/(?P=crate)"
    r"-(?P<version>\d[\w.+-]*)\.crate[ \t]+(?P<sha256>[0-9a-f]{64})"
)


_PYPI_HOST = "files.pythonhosted.org"
"""The host of the PyPI source archive URL in a pack notice. Nothing fetches it."""
_PYPI_LINE = re.compile(
    rf"https://{re.escape(_PYPI_HOST)}/packages/source/(?P<initial>\w)/(?P<package>[\w.-]+)/"
    r"(?P=package)-(?P<version>\d[\w.+-]*)\.tar\.gz[ \t]+(?P<sha256>[0-9a-f]{64})"
)


def _listed_source(line: str) -> bool:
    """True for a crates.io or PyPI source URL with the sha256 that the list holds for it."""
    if crate := _CRATE_LINE.fullmatch(line):
        return MPL_CRATE_CHECKSUMS.get((crate["crate"], crate["version"])) == crate["sha256"]
    if package := _PYPI_LINE.fullmatch(line):
        return (
            package["initial"] == package["package"][0]
            and MPL_PYPI_CHECKSUMS.get((package["package"], package["version"]))
            == package["sha256"]
        )
    return False


def _mpl_pointer(text: str) -> bool:
    """True when every line is a listed crates.io or PyPI source URL and its sha256."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return bool(lines) and all(_listed_source(line) for line in lines)


# The pin of an engine carries no archive, so the pin check only screens the licence terms. The
# wheel check and the staged-tree check hold every install, and each one needs the notice file
# of a term in this table: a pin that names the term gets no build without it.
NOTICE_RULES: Mapping[str, NoticeRule] = {
    "LicenseRef-NVIDIA-CUDA-EULA": NoticeRule("NOTICE-NVIDIA-CUDA", _holds(r"\S")),
    "LicenseRef-Microsoft-VC-Runtime": NoticeRule(
        "NOTICE-MICROSOFT-VC-RUNTIME",
        _holds(
            re.escape(
                "These files are licensed under the Microsoft terms below, "
                "not under the other terms of this wheel."
            )
        ),
    ),
    "OpenSSL": NoticeRule("NOTICE-OpenSSL", _holds(r"(?s)(?=.*OpenSSL Project)(?=.*Eric Young)")),
    # One line per crate or Python package: the crates.io download URL of the crate or the PyPI
    # URL of the source archive, then its sha256.
    "MPL-2.0": NoticeRule("SOURCE-MPL-2.0", _mpl_pointer),
}

# A GPL, AGPL or LGPL text, by a whole word in its file name or by its title line.
# The title match ignores case and spacing.
# The title needs "version" right after it, so a notice that only names the licence passes.
_COPYLEFT_NAME = re.compile(r"(?i)(?<![a-z])(?:[al]?gplv?|lesser|affero)(?![a-z])")
_COPYLEFT_TITLE = re.compile(r"gnu (?:(?:affero|lesser|library) )?general public license version")
_LICENCE_FILE = re.compile(r"(?i)licen[cs]e|copying|notice|copyright")
# Libraries whose source licence is LGPL or GPL, as a bundled library name.
_COPYLEFT_LIBRARIES = re.compile(
    r"(?i)(?:lib)?(?P<name>quadmath|gmp|gmpxx|mpfr|readline|history)(?:[-.][\w.-]*)?"
)
_LIBRARY_SUFFIXES = (".so", ".dylib", ".dll")
_STATIC_ARCHIVE_SUFFIX = ".a"
LOCK_DIR = ".locks"
_LOCK_TIMEOUT_S = 300.0
_COPY_CHUNK_BYTES = 1024 * 1024
STAGE_PREFIX = ".stage-"
_DOWNLOAD_PREFIX = ".download-"
_TRASH_PREFIX = ".trash-"
_BUNDLED = "bundled"
_NO_WAIT_S = 0.0
_ENGINES_DIRNAME = "engines"
_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
# A version, a hyphen and a backend with no dot: a file with a suffix is no build.
_BUILD_PATTERN = re.compile(r".+-[^.]+")
_PRUNE_SKIPPED = "The engine prune cannot read %s. Run 'lilbee engine list' for the fix."
_LICENCE_JOINER = re.compile(r"\s+(?:OR|AND)\s+")
PACK_DIR = "packs"


class ChecksumError(Exception):
    """The wheel does not match the sha256 in the pin."""


class LicenceRefusedError(Exception):
    """A licence of the engine is not on the permissive list."""


class UnsafeWheelError(Exception):
    """The wheel holds an entry outside its install directory, or no usable server."""


class EngineConflictError(Exception):
    """A build of the same engine, version and backend exists with a different wheel."""


class MissingPackError(Exception):
    """The build is not installed, has no pack on disk, and cannot be downloaded."""


class PackUrlError(Exception):
    """The pin names a download url outside the pack index."""


class UnsupportedPlatformError(Exception):
    """The pin lists no wheel for this platform."""


class EngineDiskError(Exception):
    """The volume of the engines root has too little free space for the wheel."""


class EngineInUseError(Exception):
    """A running launch uses the build, so removing it would pull files from under it."""


class BundledEngineError(Exception):
    """The engine inside the executable has no store entry and cannot be removed."""


class EngineNotInstalledError(Exception):
    """The store holds no build with this engine, version and backend."""


class EngineFolderError(OSError):
    """A folder of one engine in the store holds no build that lilbee can read."""

    def __init__(self, folder: Path, engine: str, fault: OSError | None = None) -> None:
        super().__init__(str(folder))
        self.folder = folder
        self.engine = engine
        self.fault = fault

    def __str__(self) -> str:
        # circular: choice imports engine_store
        from lilbee.catalog.runnable import session_host
        from lilbee.providers.fleet.engines.choice import install_offered

        # A file in the place of a folder is damage with the usual fix, as a denied read is.
        usual = (PermissionError, NotADirectoryError)
        if self.fault is not None and not isinstance(self.fault, usual):
            return (
                f"A read of the engine folder {self.folder} failed: {self.fault.strerror}. "
                f"lilbee leaves the folder as it is. Run the command again."
            )
        offered = install_offered(self.engine, session_host())
        install = f" and run 'lilbee engine install {self.engine}'" if offered else ""
        return (
            f"The engine folder {self.folder} holds no build that lilbee can read. If its "
            f"permissions deny a read, correct them. lilbee cannot remove it until then. "
            f"If they do not, remove it{install}."
        )


class StorePathError(Exception):
    """A name or path resolves outside the engines root, so the store does not touch it."""


@dataclass(frozen=True)
class PlatformAsset:
    """One downloadable wheel of a pin, for one platform."""

    url: str
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
    tool: str
    """The file name of the engine server in the wheel."""
    libraries: tuple[str, ...] = ()
    """The shared libraries, by name without prefix or suffix, that the wheel must hold."""

    def __post_init__(self) -> None:
        for part in (self.engine, self.version, self.backend):
            if not _NAME_PATTERN.fullmatch(part):
                raise ValueError(f"{part!r} cannot name an engine directory")

    def for_cell(self, cell: str) -> PinLock:
        """This pin with the digest of the cell *cell*; a pin with no cell is unchanged."""
        if not cell:
            return self
        return replace(self, sha256=self.platforms[cell].sha256)


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


@dataclass(frozen=True)
class RemoveResult:
    """What a removal deleted, and whether an index may depend on it."""

    path: Path
    canary_warning: bool


def licence_permitted(expression: str, notices: frozenset[str] = frozenset()) -> bool:
    """True when every OR/AND term of *expression* is permissive.

    A term in ``NOTICE_RULES`` passes only when *notices* holds the name of its verified
    notice file.
    """
    terms = [term for term in _LICENCE_JOINER.split(expression.strip()) if term]
    return bool(terms) and all(
        term in PERMISSIVE_LICENCES
        or ((rule := NOTICE_RULES.get(term)) is not None and rule.file in notices)
        for term in terms
    )


def _verified_notices(found: Iterable[tuple[str, str]]) -> frozenset[str]:
    """The names of the notice files, from (name, text) pairs, whose text meets its rule."""
    pairs = list(found)
    return frozenset(
        rule.file
        for rule in NOTICE_RULES.values()
        for name, text in pairs
        if name == rule.file and rule.accepts(text)
    )


def _check_licences(lock: PinLock) -> None:
    if not lock.licences:
        raise LicenceRefusedError(f"Engine {lock.engine} {lock.version} lists no licence.")
    for expression in lock.licences:
        # A pin has no archive, so it cannot hold the notice; the wheel check requires it.
        if not licence_permitted(expression, frozenset(r.file for r in NOTICE_RULES.values())):
            raise LicenceRefusedError(
                f"Engine {lock.engine} {lock.version} uses the licence {expression!r}, "
                f"which is not on the permissive list."
            )


def copy_with_digest(source: Path, target: Path) -> str:
    """Copy *source* to *target* in chunks; return the sha256 of the bytes copied."""
    digest = hashlib.sha256()
    with source.open("rb") as reader, target.open("wb") as writer:
        while chunk := reader.read(_COPY_CHUNK_BYTES):
            digest.update(chunk)
            writer.write(chunk)
    return digest.hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require_pinned_digest(subject: str, digest: str, pinned: str, source: Path | None) -> None:
    """Refuse a file whose *digest* is not *pinned*; *source* is the file the user copied, or None.

    A file the user copied is read again on every run, so the text tells the user to copy it
    again. A download is made again on every run, so the text tells the user to run the
    install again. Every install path states where its file came from through *source*.
    """
    if digest == pinned.lower():
        return
    if source is None:
        raise ChecksumError(
            f"The file for {subject} does not match its pin, so lilbee did not install it. "
            "Run the install again. If this happens again, report it."
        )
    text = (
        f"The file {source} for {subject} does not match its pin, so lilbee did not install "
        f"it. Copy {source.name} again from the lilbee release that matches this version "
        "and run the install again."
    )
    if source.parent.name == PACK_DIR:
        text += f" Or remove it from {source.parent} to download it."
    raise ChecksumError(text)


def _require_pinned(lock: PinLock, digest: str, source: Path | None) -> None:
    require_pinned_digest(
        f"engine {lock.engine} {lock.version} ({lock.backend})", digest, lock.sha256, source
    )


def _check_wheel(lock: PinLock, wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        _refuse_escaping_entries(lock, archive)
        _refuse_linked_binary(lock, archive)
        names = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        text = archive.read(names[0]).decode("utf-8") if names else None
        found = [(name, info) for info in archive.infolist() if (name := _notice_name(info))]
        _refuse_repeated_notices(lock, [info.filename for _, info in found])
        notices = _verified_notices((name, _read_entry(archive, info)) for name, info in found)
        _refuse_copyleft_files(
            lock,
            (
                (info.filename, partial(archive.read, info))
                for info in archive.infolist()
                if not info.is_dir()
            ),
        )
        _check_declared_licence(lock, text, notices)
        _refuse_unusable_server(lock, archive)


def _read_entry(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> str:
    return archive.read(info).decode("utf-8", errors="replace")


def _refuse_repeated_notices(lock: PinLock, entries: list[str]) -> None:
    """Refuse a wheel that holds one notice entry twice: the check and the unpack differ."""
    repeated = sorted(name for name in set(entries) if entries.count(name) > 1)
    if repeated:
        raise LicenceRefusedError(
            f"The wheel for engine {lock.engine} {lock.version} holds two entries named "
            f"{repeated[0]!r}, so lilbee cannot tell which one is the notice."
        )


def _utf16_without_mark(data: bytes) -> str | None:
    """Return the text of a UTF-16 file with no byte order mark, found by its NUL bytes."""
    even, odd = data[0::2], data[1::2]
    if 0 not in even and odd.count(0) * 2 > len(odd):
        return data.decode("utf-16-le", errors="replace")
    if 0 not in odd and even.count(0) * 2 > len(even):
        return data.decode("utf-16-be", errors="replace")
    return None


def _wide_text(data: bytes) -> str | None:
    """Return the text of a UTF-16 or UTF-32 file, found by its mark or its NUL bytes."""
    if data.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        return data.decode("utf-32", errors="replace")
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16", errors="replace")
    return _utf16_without_mark(data)


def _licence_texts(data: bytes) -> list[str]:
    """Return every decoding to test for a title: UTF-8 or Latin-1, then a wide one."""
    try:
        texts = [data.decode("utf-8")]
    except UnicodeDecodeError:
        texts = [data.decode("latin-1")]
    if (wide := _wide_text(data)) is not None:
        texts.append(wide)
    return texts


def _is_copyleft_file(name: str, read: Callable[[], bytes]) -> bool:
    """True when the file *name* is a GPL, AGPL or LGPL licence, or a library under one."""
    *folders, base = PurePosixPath(name).parts
    lower = base.lower()
    if _COPYLEFT_LIBRARIES.fullmatch(base) and (
        lower.endswith(_STATIC_ARCHIVE_SUFFIX) or any(s in lower for s in _LIBRARY_SUFFIXES)
    ):
        return True
    in_scanned_folder = "licenses" in folders or any(_COPYLEFT_NAME.search(f) for f in folders)
    if not (_LICENCE_FILE.search(base) or _COPYLEFT_NAME.search(base) or in_scanned_folder):
        return False
    if _COPYLEFT_NAME.search(base):
        return True
    return any(
        _COPYLEFT_TITLE.search(" ".join(text.split()).casefold()) for text in _licence_texts(read())
    )


def _refuse_copyleft_files(lock: PinLock, files: Iterable[tuple[str, Callable[[], bytes]]]) -> None:
    """Refuse a build that carries a GPL, AGPL or LGPL file, whatever its METADATA declares."""
    for name, read in files:
        if _is_copyleft_file(name, read):
            raise LicenceRefusedError(
                f"The build of engine {lock.engine} {lock.version} carries a GPL, AGPL or "
                f"LGPL file: {name!r}. That licence is not on the permissive list."
            )


def _notice_name(info: zipfile.ZipInfo) -> str | None:
    """The name of a notice file directly in a top-level ``*.dist-info/licenses``."""
    match info.filename.split("/"):
        case [folder, "licenses", name] if folder.endswith(".dist-info"):
            return name if any(name == rule.file for rule in NOTICE_RULES.values()) else None
        case _:
            return None


def _check_declared_licence(
    lock: PinLock, metadata_text: str | None, notices: frozenset[str] = frozenset()
) -> None:
    """Refuse unless the METADATA text declares a permitted ``License-Expression``."""
    metadata = message_from_string(metadata_text) if metadata_text else None
    expression = (metadata.get("License-Expression") if metadata else None) or ""
    if not licence_permitted(expression, notices):
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
    name = _binary_of(archive, lock.tool)
    if name is not None and stat.S_ISLNK(archive.getinfo(name).external_attr >> 16):
        raise UnsafeWheelError(
            f"The wheel for engine {lock.engine} {lock.version} holds the tool {name!r} "
            f"as a symbolic link."
        )


def _refuse_unusable_server(lock: PinLock, archive: zipfile.ZipFile) -> None:
    problems = server_problems(archive, lock.tool, lock.libraries)
    if problems:
        raise UnsafeWheelError(
            f"The wheel for engine {lock.engine} {lock.version} {'; '.join(problems)}."
        )


def _binary_of(archive: zipfile.ZipFile, tool: str) -> str | None:
    """The ``<package>/bin/<tool>`` file of the wheel, as a POSIX path; None when it has none."""
    return server_entry(archive.namelist(), tool)


def _binary_usable(binary: Path, directory: Path) -> bool:
    """True for an executable file in *directory*.

    stat goes first, so resolve meets no link loop: on Python 3.11 and 3.12 it raises
    RuntimeError on one. A loop is damage, as a missing binary is, not a failed read.
    """
    try:
        mode = binary.stat().st_mode
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            return False  # pragma: posix only - a Windows loop is WinError 1921
        raise
    if not binary.resolve().is_relative_to(directory.resolve()):
        return False
    return stat.S_ISREG(mode) and (sys.platform == "win32" or bool(mode & stat.S_IXUSR))


def _unpack(wheel: Path, stage: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(stage)
        for info in archive.infolist():
            mode = info.external_attr >> 16
            if stat.S_ISREG(mode) and mode & stat.S_IXUSR:
                target = stage / info.filename
                target.chmod(target.stat().st_mode | stat.S_IXUSR)


def _any_in_use(
    in_use: Callable[[str, str, str], bool],
    readings: list[tuple[str, str, str]],
    engine: str,
    version: str,
    backend: str,
) -> bool:
    """True when a launch uses any reading of one build folder."""
    return any(in_use(*reading) for reading in readings)


def _raise_walk_error(exc: OSError) -> None:
    raise exc


def _names_engine(name: str) -> bool:
    """True for an engine; the root also holds the router, the files of a server and strays."""
    # circular: the engines import engine_store
    from lilbee.providers.fleet.engines.base import EngineName

    return name in set(EngineName)


def _build_entries(engine: Path) -> list[Path]:
    """The folders of one engine folder, and the files named as a build; none when it is absent."""
    try:
        # The "." entry needs the right to enter the folder as well as the right to list it.
        names = sorted(os.listdir(os.path.join(engine, os.curdir)))
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise EngineFolderError(engine, engine.name, exc) from exc
    builds = [engine / name for name in names if not name.startswith(".")]
    return [b for b in builds if b.is_dir() or _BUILD_PATTERN.fullmatch(b.name)]


def _read_record(directory: Path) -> InstalledEngine | None:
    """The record of *directory* when it is a complete build, else None.

    Raises ``EngineFolderError`` when a read of the build fails: that build is not absent.
    A folder in the place of the record is damage, not a failed read.
    """
    record = directory / RECORD_NAME
    try:
        # Windows reports the read of a folder as a denied read, so stat tells the kind.
        if stat.S_ISDIR(record.stat().st_mode):
            return None
        data = json.loads(record.read_text(encoding="utf-8"))
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
        if found.binary is not None and not _binary_usable(found.binary, directory):
            return None
    except (FileNotFoundError, NotADirectoryError, ValueError, KeyError, TypeError):
        return None
    except OSError as exc:
        raise EngineFolderError(directory, directory.parent.name, exc) from exc
    if (
        directory.name != f"{found.version}-{found.backend}"
        or directory.parent.name != found.engine
    ):
        return None
    return found


class EngineStore:
    """The engines root: ``<root>/<engine>/<version>-<backend>/``, complete only with its record."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def build_dir(self, lock: PinLock) -> Path:
        return self._dir(lock.engine, lock.version, lock.backend)

    def _dir(self, engine: str, version: str, backend: str) -> Path:
        return self.root / engine / f"{version}-{backend}"

    def holds(self, engine: str, version: str, backend: str) -> bool:
        """True when the build folder is a real entry under the engines root, as remove needs."""
        try:
            self._inside_root(self._dir(engine, version, backend))
        except StorePathError:
            return False
        return True

    def build_size(self, engine: str, version: str, backend: str) -> int | None:
        """The bytes on disk of one build folder; None when it is outside the root or unreadable."""
        build = self._dir(engine, version, backend)
        total = 0
        try:
            self._inside_root(build)
            for folder, _dirs, files in os.walk(build, onerror=_raise_walk_error):
                total += sum(os.lstat(os.path.join(folder, name)).st_size for name in files)
        except (OSError, StorePathError) as exc:
            log.warning(
                "The size of engine %s %s (%s) is unknown: %s", engine, version, backend, exc
            )
            return None
        return total

    def installed(self) -> list[InstalledEngine]:
        """Every complete build; a directory without a valid record is not listed."""
        return [found for _, found in self._walk() if isinstance(found, InstalledEngine)]

    def damaged(self) -> list[EngineFolderError]:
        """An error for each engine or build folder that holds no complete build."""
        errors = [
            found or EngineFolderError(folder, folder.parent.name)
            for folder, found in self._walk()
            if not isinstance(found, InstalledEngine)
        ]
        return [error for error in errors if _names_engine(error.engine)]

    def _walk(self) -> Iterator[tuple[Path, InstalledEngine | EngineFolderError | None]]:
        """Each build folder with its record, else None, and each folder whose read failed."""
        for engine in self._engine_entries():
            try:
                builds = _build_entries(engine)
            except EngineFolderError as exc:
                yield engine, exc
                continue
            for build in builds:
                found: InstalledEngine | EngineFolderError | None
                try:
                    found = _read_record(build)
                except EngineFolderError as exc:
                    found = exc
                yield build, found

    def _engine_entries(self) -> list[Path]:
        """The entries of the root that can be an engine folder."""
        if not self.root.is_dir():
            return []
        return [
            entry
            for entry in sorted(self.root.iterdir())
            if entry.name != PACK_DIR and not entry.name.startswith(".")
        ]

    def pack_path(self, lock: PinLock) -> Path:
        """Where a hand-copied wheel of *lock* is looked up."""
        return self.root / PACK_DIR / f"{lock.engine}-{lock.version}-{lock.backend}.whl"

    def complete_build(self, lock: PinLock) -> Path | None:
        """The directory of the complete build of *lock*, None when absent or damaged.

        Raises ``EngineFolderError`` when a read of the engine folder or the build fails.
        """
        _build_entries(self.root / lock.engine)
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
        found = self.complete_build(lock)
        if found is not None:
            return found
        pack = self.pack_path(lock)
        if pack.is_file():
            return self.install_from_wheel(lock, pack)
        with self._scratch() as scratch:
            try:
                wheel = fetch(lock, scratch)
            except (OSError, RuntimeError) as exc:
                raise MissingPackError(
                    f"Engine {lock.engine} {lock.version} ({lock.backend}) is not installed "
                    f"and cannot be downloaded. Copy {pack.name} into {pack.parent} "
                    f"and run the install again."
                ) from exc
            return self._install_private(lock, wheel, file_digest(wheel), None)

    @contextmanager
    def _scratch(self) -> Iterator[Path]:
        """A private download folder under the root, locked while in use, removed on exit."""
        name = f"{_DOWNLOAD_PREFIX}{uuid.uuid4().hex}"
        scratch = self.root / name
        scratch_lock = self.root / LOCK_DIR / name
        try:
            with file_lock(scratch_lock, _LOCK_TIMEOUT_S):
                yield scratch
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
            _unlink_lock_file(scratch_lock)

    def install_from_wheel(self, lock: PinLock, wheel: Path) -> Path:
        """Install a private copy of *wheel* as the build of *lock*; return the build directory.

        A rewrite of *wheel* after the copy cannot reach the install.
        """
        _check_licences(lock)
        with self._scratch() as scratch:
            scratch.mkdir()
            copy = scratch / wheel.name
            return self._install_private(lock, copy, copy_with_digest(wheel, copy), wheel)

    def _refuse_held_replace(self, lock: PinLock) -> None:
        """Refuse to replace a damaged build while a launch uses it, as remove does."""
        if self.build_dir(lock).exists() and launch_uses_engine(
            self.root, lock.engine, lock.version, lock.backend
        ):
            raise EngineInUseError(
                f"Engine {lock.engine} {lock.version} ({lock.backend}) is in use. Stop the "
                f"engine with 'lilbee engine stop' and install it again."
            )

    def _install_private(
        self, lock: PinLock, wheel: Path, digest: str, source: Path | None
    ) -> Path:
        """Install the private file *wheel* once *digest* and the wheel checks match the pin.

        *source* is the file the user copied, or None for a download.
        """
        _require_pinned(lock, digest, source)
        _check_wheel(lock, wheel)
        final = self.build_dir(lock)
        with file_lock(self.root / LOCK_DIR / lock.engine, _LOCK_TIMEOUT_S):
            if self.complete_build(lock) is not None:
                return final
            self._refuse_held_replace(lock)
            stage = final.parent / f"{STAGE_PREFIX}{uuid.uuid4().hex}"
            final.parent.mkdir(parents=True, exist_ok=True)
            try:
                _unpack(wheel, stage)
                with zipfile.ZipFile(wheel) as archive:
                    binary = _binary_of(archive, lock.tool)
                self._swap_in(lock, stage, binary)
            finally:
                shutil.rmtree(stage, ignore_errors=True)
        return final

    def _swap_in(self, lock: PinLock, stage: Path, binary: str | None) -> None:
        """Write the record into *stage*, then rename it onto the final name."""
        final = self.build_dir(lock)
        self._write_record(stage, lock, binary)
        if final.exists():
            shutil.rmtree(final)
        stage.rename(final)

    def _inside_root(self, path: Path) -> Path:
        """*path* when it is a real entry under the engines root, else StorePathError."""
        if path.is_symlink() or not path.resolve().is_relative_to(self.root.resolve()):
            raise StorePathError(f"'{path.name}' is not inside the engines folder {self.root}.")
        return path

    def remove(
        self,
        engine: str,
        version: str,
        backend: str,
        in_use: Callable[[str, str, str], bool],
        *,
        wait_s: float = _LOCK_TIMEOUT_S,
    ) -> RemoveResult:
        """Delete one build; refuse while *in_use* says a launch uses it.

        The install lock of the engine is awaited for *wait_s* seconds.
        """
        if _BUNDLED in (version, backend):
            raise BundledEngineError(
                f"The {engine} engine inside the executable is not an installed build "
                f"and cannot be removed."
            )
        for part in (engine, version, backend):
            if not _NAME_PATTERN.fullmatch(part):
                raise StorePathError(f"{part!r} cannot name an engine build.")
        build = self.root / engine / f"{version}-{backend}"
        with file_lock(self.root / LOCK_DIR / engine, wait_s):
            _build_entries(build.parent)
            if not build.is_dir():
                raise EngineNotInstalledError(
                    f"Engine {engine} {version} ({backend}) is not installed."
                )
            self._inside_root(build)
            _read_record(build)
            if in_use(engine, version, backend):
                raise EngineInUseError(
                    f"Engine {engine} {version} ({backend}) is in use. Stop the engine "
                    f"with 'lilbee engine stop' and remove it again."
                )
            trash = build.parent / f"{_TRASH_PREFIX}{uuid.uuid4().hex}"
            build.rename(trash)
            shutil.rmtree(trash, ignore_errors=True)
        return RemoveResult(path=build, canary_warning=True)

    def prune(
        self, keep: set[tuple[str, str, str]], in_use: Callable[[str, str, str], bool]
    ) -> list[Path]:
        """Remove every build that is not in *keep* and not in use; return what went.

        An empty *keep* is refused: it would remove every build.
        """
        if not keep:
            raise ValueError("Prune needs at least one build to keep.")
        gone: list[Path] = []
        readings: dict[Path, list[tuple[str, str, str]]] = {}
        for engine, version, backend, path in self.build_dirs():
            readings.setdefault(path, []).append((engine, version, backend))
        for path, names in readings.items():
            if any(name in keep for name in names):
                continue
            engine, version, backend = names[0]
            try:
                self.remove(
                    engine,
                    version,
                    backend,
                    in_use=partial(_any_in_use, in_use, names),
                    wait_s=_NO_WAIT_S,
                )
            except (
                BundledEngineError,
                EngineInUseError,
                StorePathError,
                EngineNotInstalledError,
                FileLockUnavailableError,
                OSError,
            ) as exc:
                log.debug("Kept the engine build %s: %s", path, exc)
                continue
            gone.append(path)
        return gone

    def build_dirs(
        self, report: Callable[[str, Path], None] = log.debug
    ) -> list[tuple[str, str, str, Path]]:
        """Every build folder, damaged ones too: (engine, version, backend, path).

        A complete build is listed once, as its record says. A name holds no mark between
        version and backend, so a damaged build is listed once for each hyphen of its name.
        A folder that cannot be read is skipped, and *report* hears of the one of an engine.
        """
        dirs = []
        for build, found in self._walk():
            engine = build.parent.name
            if isinstance(found, EngineFolderError):
                if _names_engine(found.engine):
                    report(_PRUNE_SKIPPED, found.folder)
            elif found is not None:
                dirs.append((engine, found.version, found.backend, build))
            elif build.is_dir():
                dirs.extend(
                    (engine, build.name[:cut], build.name[cut + 1 :], build)
                    for cut in range(1, len(build.name) - 1)
                    if build.name[cut] == "-"
                )
        return dirs

    def sweep(self) -> list[Path]:
        """Remove stage, trash and download folders that no live install holds; never raises.

        The engine lock says who is live: an installer holds it from its stage folder
        to the rename, and a download holds a lock named for its folder. A folder whose
        lock is free belongs to no live install, however recent it is.
        """
        removed: list[Path] = []
        try:
            for engine in sorted(self.root.iterdir()) if self.root.is_dir() else []:
                if engine.name == PACK_DIR or not engine.is_dir():
                    continue
                if engine.name == LOCK_DIR:
                    continue
                if engine.name.startswith(_DOWNLOAD_PREFIX):
                    removed.extend(self._sweep_download(engine))
                elif not engine.name.startswith("."):
                    removed.extend(self._sweep_engine(engine))
        except OSError as exc:
            log.debug("Skipped the engine sweep of %s: %s", self.root, exc)
        return removed

    def _sweep_download(self, scratch: Path) -> list[Path]:
        lock_path = self.root / LOCK_DIR / scratch.name
        try:
            with file_lock(lock_path, _NO_WAIT_S):
                self._inside_root(scratch)
                shutil.rmtree(scratch, ignore_errors=True)
        except (FileLockUnavailableError, StorePathError):
            return []
        _unlink_lock_file(lock_path)
        return [scratch]

    def _sweep_engine(self, engine: Path) -> list[Path]:
        try:
            with file_lock(self.root / LOCK_DIR / engine.name, _NO_WAIT_S):
                leftovers = [
                    child
                    for child in sorted(engine.iterdir())
                    if child.name.startswith((STAGE_PREFIX, _TRASH_PREFIX))
                ]
                removed = [child for child in leftovers if self._remove_leftover(child)]
        except FileLockUnavailableError:
            return []
        return removed

    def _remove_leftover(self, child: Path) -> bool:
        """Delete one stage or trash folder; False when it is a link out of the root."""
        try:
            self._inside_root(child)
        except StorePathError:
            return False
        shutil.rmtree(child, ignore_errors=True)
        return not child.exists()

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


def _unlink_lock_file(lock_path: Path) -> None:
    lock_path.with_name(lock_path.name + ".lock").unlink(missing_ok=True)


def engines_root(models_dir: Path) -> Path:
    """The machine-wide engines folder, beside *models_dir*."""
    return models_dir.parent / _ENGINES_DIRNAME


def launch_uses_engine(root: Path, engine: str, version: str, backend: str) -> bool:
    """True while a launch of this machine runs a binary from this build folder.

    A user whose launches cannot all be named may run any build, so it counts as a user.
    The paths are compared as written: a link inside the build counts as the build.
    """
    # circular: swap_manager -> binary -> first_use -> engine_store
    from lilbee.providers.fleet.swap_manager import recorded_launch_binaries
    from lilbee.runtime.engine_lock import live_users_exist, machine_engine_dir

    slot = machine_engine_dir()
    if not live_users_exist(slot):
        return False
    binaries = recorded_launch_binaries(slot)
    if binaries is None:
        return True
    build = Path(os.path.abspath(root / engine / f"{version}-{backend}"))
    return any(Path(os.path.abspath(binary)).is_relative_to(build) for binary in binaries)


def sweep_engines(models_dir: Path) -> list[Path]:
    """Startup hook: sweep stale install folders; a no-op when nothing is installed."""
    if not models_dir.is_absolute():
        return []
    return EngineStore(engines_root(models_dir)).sweep()


def prune_engines(models_dir: Path) -> list[Path]:
    """Startup hook: remove builds no shipped pin names and no live process uses."""
    if not models_dir.is_absolute():
        return []
    store = EngineStore(engines_root(models_dir))
    try:
        builds = store.build_dirs(log.warning)
    except OSError as exc:
        log.warning("Skipped the engine prune of %s: %s", store.root, exc)
        return []
    if not builds:
        return []
    from lilbee.providers.fleet.engines.choice import release_pins

    keep = release_pins(builds)
    if not keep:
        return []
    try:
        return store.prune(keep, in_use=partial(launch_uses_engine, store.root))
    except OSError as exc:
        log.warning("Skipped the engine prune of %s: %s", store.root, exc)
        return []


def fetch_frozen(
    lock: PinLock,
    platform: str,
    *,
    dest: Path,
    on_progress: ProgressCallback | None = None,
    on_retry: RetryCallback | None = None,
    cancel: CancelSignal | None = None,
) -> Path:
    """Download the wheel of *lock* for the cell *platform* into *dest*, unchecked.

    A pin with no cells ships as a pack only, so it has nothing to download.
    """
    if not lock.platforms:
        raise FileNotFoundError(f"Engine {lock.engine} {lock.version} has no download.")
    asset = lock.platforms.get(platform)
    if asset is None:
        raise UnsupportedPlatformError(
            f"Engine {lock.engine} {lock.version} has no build for {platform}."
        )
    if not PACKAGES_URL:
        raise PackUrlError(
            f"Engine {lock.engine} {lock.version} cannot download: this build of lilbee "
            "lists no pack index."
        )
    if not asset.url.startswith(PACKAGES_URL):
        raise PackUrlError(
            f"Engine {lock.engine} {lock.version} points at {asset.url}, which is not on "
            f"the pack index {PACKAGES_URL}. lilbee downloads packs from the index only."
        )
    shortfall = disk_shortfall(dest, url_filename(asset.url), asset.size)
    if shortfall is not None:
        raise EngineDiskError(shortfall)
    return download_url(
        asset.url, dest, asset.size, on_progress=on_progress, on_retry=on_retry, cancel=cancel
    )


__all__ = [
    "RECORD_NAME",
    "BundledEngineError",
    "ChecksumError",
    "EngineConflictError",
    "EngineDiskError",
    "EngineFolderError",
    "EngineInUseError",
    "EngineNotInstalledError",
    "EngineStore",
    "InstalledEngine",
    "LicenceRefusedError",
    "MissingPackError",
    "PackUrlError",
    "PinLock",
    "PlatformAsset",
    "RemoveResult",
    "StorePathError",
    "UnsafeWheelError",
    "UnsupportedPlatformError",
    "engines_root",
    "fetch_frozen",
    "launch_uses_engine",
    "licence_permitted",
    "prune_engines",
    "require_pinned_digest",
    "sweep_engines",
]
