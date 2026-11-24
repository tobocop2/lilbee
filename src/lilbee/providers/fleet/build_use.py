"""The build folder says whether an engine runs from it.

Every launch of an engine opens a file inside each store build it runs and hands that open
file to the engine process. The operating system keeps the file open for as long as the
engine lives, also after the lilbee that started it has exited. A remove or a prune asks the
operating system whether the file is open and refuses when it is. No list of engine
folders, state files or config roots is needed, so one machine-wide store with many
config roots needs none.

On POSIX the file holds a shared ``flock`` that the engine inherits; the probe asks for an
exclusive one without waiting. On Windows the file is open without ``FILE_SHARE_DELETE``,
inherited through the process handle list; the probe opens it with no sharing, and a
sharing violation means in use.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

IN_USE_NAME = ".in-use"
_OWNER_ONLY_MODE = 0o600
_BUILD_PATH_PARTS = 3  # engine folder, build folder, a file inside the build

if sys.platform == "win32":  # pragma: no cover - proved on windows-latest, not on the CI host
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

    def _hold(path: Path) -> int:
        attributes = _SecurityAttributes(ctypes.sizeof(_SecurityAttributes), None, True)
        handle = _k32.CreateFileW(
            str(path),
            _GENERIC_READ,
            _SHARE_READ_WRITE,
            ctypes.byref(attributes),
            _OPEN_ALWAYS,
            _FILE_ATTRIBUTE_NORMAL,
            None,
        )
        if handle == _INVALID_HANDLE:
            raise ctypes.WinError(ctypes.get_last_error())
        return int(handle)

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

else:
    import fcntl

    def _hold(path: Path) -> int:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, _OWNER_ONLY_MODE)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
            os.set_inheritable(fd, True)
        except OSError:
            os.close(fd)
            raise
        return fd

    def _release(handle: int) -> None:
        os.close(handle)

    def _spawn_kwargs(handles: list[int]) -> dict[str, Any]:
        return {"pass_fds": handles}

    def _open_elsewhere(path: Path) -> bool:
        fd = os.open(path, os.O_RDWR)
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
                self._handles.append(_hold(build / IN_USE_NAME))
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


def build_in_use(build: Path) -> bool:
    """True while any engine process holds the in-use file of *build* open."""
    try:
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
