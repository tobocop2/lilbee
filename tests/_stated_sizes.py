"""Model files whose size a test states, so no test writes or reserves the bytes.

Windows fills a truncated file with real zeros, so a test that truncates a file to
20 GiB hangs there. A test calls ``state_size`` instead: the file stays empty and
every ``Path.stat`` of it reports the stated size. Use the ``stated_sizes`` fixture
from ``conftest.py`` to switch the patch on.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

_sizes: dict[str, int] | None = None


def _key(path: Path) -> str:
    return os.path.normcase(os.path.realpath(path))


class _StatedStat:
    """A stat result that reports a stated size and delegates the rest."""

    def __init__(self, real: os.stat_result, size: int) -> None:
        self._real = real
        self.st_size = size

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def state_size(path: Path, size: int) -> None:
    """Create *path* empty, and make ``Path.stat`` report *size* bytes for it."""
    if _sizes is None:
        raise RuntimeError("Request the stated_sizes fixture before stating a size.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    _sizes[_key(path)] = size


def install(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ``Path.stat`` for the length of one test, with no size stated yet."""
    global _sizes
    sizes: dict[str, int] = {}
    _sizes = sizes
    real_stat = Path.stat

    def stat(self: Path, *, follow_symlinks: bool = True) -> Any:
        result = real_stat(self, follow_symlinks=follow_symlinks)
        stated = sizes.get(_key(self)) if follow_symlinks else None
        return result if stated is None else _StatedStat(result, stated)

    monkeypatch.setattr(Path, "stat", stat)


def uninstall() -> None:
    global _sizes
    _sizes = None
