"""Frozen-build detection shared by package-level code.

``__main__`` keeps its own copy of this predicate on purpose: the
multiprocessing/-m child dispatch path there must import nothing from the
``lilbee`` package (the splash subprocess is deliberately stdlib-only), so it
cannot import this module. Every other caller imports :func:`is_frozen`.
"""

from __future__ import annotations

import sys


def is_frozen() -> bool:
    """True when running from a frozen build.

    PyInstaller/cx_Freeze set ``sys.frozen``; Nuitka never does but injects a
    ``__compiled__`` global into every module it compiles. Both must be checked
    so detection works in the shipped Nuitka onefile binary, not just under
    PyInstaller.
    """
    return bool(sys.__dict__.get("frozen")) or "__compiled__" in globals()
