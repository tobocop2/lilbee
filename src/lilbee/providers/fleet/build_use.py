"""The build folder says whether an engine runs from it.

The install creates a file inside each store build. Every launch of an engine opens that
file read-only in each build it runs and hands the open file to the engine process, so a
launch needs no write to the build. The operating system keeps the file open for as long as the
engine lives, also after the lilbee that started it has exited. A remove or a prune asks the
operating system whether the file is open and refuses when it is. No list of engine
folders, state files or config roots is needed, so one machine-wide store with many
config roots needs none.

On POSIX the file holds a shared ``flock`` that the engine inherits; the probe asks for an
exclusive one without waiting. On Windows the file is open without ``FILE_SHARE_DELETE``,
inherited through the process handle list; the probe opens it with no sharing, and a
sharing violation means in use.

A remove holds the file for itself across the rename and the delete of the build, so a
launch in that time fails at once and names the engine. On POSIX the remove keeps an
exclusive ``flock``; a launch asks for its shared one without waiting. Windows refuses the
rename of a folder while a file inside it is open, so there the remove only proves the
file free.
"""

from __future__ import annotations

import errno
import os
import stat
import subprocess
import sys
from collections.abc import Iterable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

from lilbee.providers.base import ProviderError, ProviderErrorKind

IN_USE_NAME = ".in-use"
_OWNER_ONLY_MODE = 0o600
_BUILD_PATH_PARTS = 3  # engine folder, build folder, a file inside the build
# The errors of a create in a folder that refuses writes.
_UNWRITABLE = (errno.EACCES, errno.EPERM, errno.EROFS)


class BuildInUseError(Exception):
    """An engine holds the in-use file of the build a remove wants."""


def engine_label(engine: str, version: str, backend: str) -> str:
    """The name of one build in every message: ``engine version (backend)``."""
    return f"{engine} {version} ({backend})"


def build_label(build: Path) -> str:
    """The ``engine_label`` of a build folder ``<engine>/<version>-<backend>``.

    Versions are release tags with no hyphen, so the first hyphen ends the version. A
    router build (``router/<version>``) has no backend.
    """
    version, _, backend = build.name.partition("-")
    if not backend:
        return f"{build.parent.name} {version}"
    return engine_label(build.parent.name, version, backend)


def make_in_use_file(stage: Path) -> None:
    """Create the in-use file in the stage folder of an install, before its rename, so a
    launch opens it read-only."""
    os.close(os.open(stage / IN_USE_NAME, os.O_WRONLY | os.O_CREAT, _OWNER_ONLY_MODE))


if sys.platform == "win32":  # pragma: windows only - the Windows CI lanes measure it
    import ctypes
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.CreateFileW.restype = wintypes.HANDLE
    _k32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    _GENERIC_READ = 0x80000000
    _SHARE_READ_WRITE = 0x1 | 0x2
    _OPEN_EXISTING = 3
    _OPEN_ALWAYS = 4
    _FILE_ATTRIBUTE_NORMAL = 0x80
    _INVALID_HANDLE = wintypes.HANDLE(-1).value
    _ERROR_SHARING_VIOLATION = 32

    class _SecurityAttributes(ctypes.Structure):
        _fields_ = [
            ("nLength", wintypes.DWORD),
            ("lpSecurityDescriptor", ctypes.c_void_p),
            ("bInheritHandle", wintypes.BOOL),
        ]

    def _open(path: Path, share: int, attributes: Any) -> int:
        """A handle of *path*, created when absent; a sharing violation is BlockingIOError."""
        handle = _k32.CreateFileW(
            str(path), _GENERIC_READ, share, attributes, _OPEN_ALWAYS, _FILE_ATTRIBUTE_NORMAL, None
        )
        if handle == _INVALID_HANDLE:
            error = ctypes.get_last_error()
            if error == _ERROR_SHARING_VIOLATION:
                raise BlockingIOError(error, "the file is open in another process", str(path))
            raise ctypes.WinError(error)
        return int(handle)

    def _hold(path: Path) -> int:
        attributes = _SecurityAttributes(ctypes.sizeof(_SecurityAttributes), None, True)
        return _open(path, _SHARE_READ_WRITE, ctypes.byref(attributes))

    @contextmanager
    def _claimed(path: Path) -> Iterator[None]:
        _k32.CloseHandle(_open(path, 0, None))
        yield

    def _release(handle: int) -> None:
        _k32.CloseHandle(handle)

    def _spawn_kwargs(handles: list[int]) -> dict[str, Any]:
        info = subprocess.STARTUPINFO(lpAttributeList={"handle_list": handles})
        return {"startupinfo": info}

    def _open_elsewhere(path: Path) -> bool:
        handle = _k32.CreateFileW(
            str(path), _GENERIC_READ, 0, None, _OPEN_EXISTING, _FILE_ATTRIBUTE_NORMAL, None
        )
        if handle == _INVALID_HANDLE:
            return ctypes.get_last_error() == _ERROR_SHARING_VIOLATION
        _k32.CloseHandle(handle)
        return False

else:  # pragma: posix only - Windows takes the branch above
    import fcntl

    def _locked(path: Path, operation: int) -> int | None:
        """A read-only descriptor of *path* under *operation*; None for an older build that
        has no file and refuses to get one (a read-only folder)."""
        try:
            fd = os.open(path, os.O_RDONLY)
        except FileNotFoundError:
            # A build installed before the install made the file; a gone folder raises here.
            try:
                fd = os.open(path, os.O_RDONLY | os.O_CREAT, _OWNER_ONLY_MODE)
            except OSError as exc:
                if exc.errno not in _UNWRITABLE:
                    raise
                return None
        try:
            fcntl.flock(fd, operation | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise
        return fd

    def _hold(path: Path) -> int | None:
        fd = _locked(path, fcntl.LOCK_SH)
        if fd is not None:
            os.set_inheritable(fd, True)
        return fd

    @contextmanager
    def _claimed(path: Path) -> Iterator[None]:
        fd = _locked(path, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fd is not None:
                os.close(fd)

    def _release(handle: int) -> None:
        os.close(handle)

    def _spawn_kwargs(handles: list[int]) -> dict[str, Any]:
        return {"pass_fds": handles}

    def _open_elsewhere(path: Path) -> bool:
        fd = os.open(path, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        finally:
            os.close(fd)
        return False


class BuildHolds:
    """The in-use files of the builds one engine launch runs, open in this process.

    Use ``spawn_kwargs`` in the ``subprocess.Popen`` call that starts the engine, then call
    ``close``: the engine keeps its own copy of each open file.
    """

    def __init__(self, builds: Iterable[Path]) -> None:
        self._handles: list[int] = []
        try:
            for build in dict.fromkeys(builds):
                handle = _hold(build / IN_USE_NAME)
                if handle is not None:
                    self._handles.append(handle)
        except (BlockingIOError, FileNotFoundError) as exc:
            self.close()
            raise ProviderError(
                f"Engine {build_label(build)} is being removed. Install it again "
                f"before you start it.",
                kind=ProviderErrorKind.SERVER,
            ) from exc
        except OSError:
            self.close()
            raise

    @property
    def spawn_kwargs(self) -> dict[str, Any]:
        """The ``Popen`` arguments that hand the open files to the child."""
        return _spawn_kwargs(self._handles) if self._handles else {}

    def close(self) -> None:
        """Close this process's copies; the engine's copies stay open."""
        handles, self._handles = self._handles, []
        for handle in handles:
            _release(handle)


@contextmanager
def removing(build: Path) -> Iterator[None]:
    """Hold the in-use file of *build* for a remove; raise BuildInUseError while an engine
    holds it. A launch in the hold fails at once (see ``BuildHolds``).

    An older build with no file gets one; when its folder refuses that, no launch holds it
    either, and the rename of the remove meets the same refusal.
    """
    with ExitStack() as stack:
        try:
            stack.enter_context(_claimed(build / IN_USE_NAME))
        except BlockingIOError as exc:
            raise BuildInUseError(build) from exc
        yield


def build_in_use(build: Path) -> bool:
    """True while any engine process holds the in-use file of *build* open.

    A slot that is not a real folder (a link, a file) is damage, and a remove only unlinks it.
    """
    try:
        if not stat.S_ISDIR(os.lstat(build).st_mode):
            return False
        return _open_elsewhere(build / IN_USE_NAME)
    except FileNotFoundError:
        return False


def builds_run_by(store_root: Path, binaries: Iterable[str | os.PathLike[str]]) -> list[Path]:
    """The build folders of *store_root* that *binaries* run from, each once.

    A binary outside the store (a path set by the user, a system install) names no build.
    The paths are compared as written: a link inside the build counts as the build.
    """
    root = Path(os.path.abspath(store_root))
    builds = []
    for binary in binaries:
        path = Path(os.path.abspath(binary))
        if path.is_relative_to(root) and len(path.relative_to(root).parts) >= _BUILD_PATH_PARTS:
            engine, build = path.relative_to(root).parts[:2]
            builds.append(root / engine / build)
    return list(dict.fromkeys(builds))
