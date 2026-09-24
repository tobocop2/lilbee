"""Read POSIX mode bits of a real path."""

from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode bits only")


def file_mode(path: Path) -> int:
    """The permission bits of *path*."""
    return stat.S_IMODE(path.stat().st_mode)
