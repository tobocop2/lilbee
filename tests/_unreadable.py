"""Deny reads on a real path so a test drives the fault the product code sees."""

from __future__ import annotations

import errno
import os
import stat
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

# Windows mode bits do not deny a read, and root bypasses them entirely.
POSIX_DENIES_READS = pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="needs POSIX mode bits and a non-root user to deny a read",
)


def deny_listing(monkeypatch: pytest.MonkeyPatch, folder: Path) -> None:
    """Make listing *folder* fail with the error a denied read raises, on every platform.

    ``os.walk`` lists each folder through ``os.scandir``, so the stub sits at the call
    the product code makes and needs no mode bits.
    """
    real_scandir = os.scandir

    def scandir(path: str | os.PathLike[str] = ".") -> object:
        if Path(path) == folder:
            raise PermissionError(errno.EACCES, "Permission denied", str(folder))
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", scandir)


@contextmanager
def unreadable(path: Path) -> Iterator[Path]:
    """Strip every permission bit from *path* for the block, then restore them."""
    original = stat.S_IMODE(path.stat().st_mode)
    path.chmod(0o000)
    try:
        yield path
    finally:
        path.chmod(original)
