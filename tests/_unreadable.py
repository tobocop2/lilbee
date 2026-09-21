"""Deny reads on a real path so a test drives the fault the product code sees."""

from __future__ import annotations

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


@contextmanager
def unreadable(path: Path) -> Iterator[Path]:
    """Strip every permission bit from *path* for the block, then restore them."""
    original = stat.S_IMODE(path.stat().st_mode)
    path.chmod(0o000)
    try:
        yield path
    finally:
        path.chmod(original)
