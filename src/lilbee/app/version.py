"""Installed package version."""

from __future__ import annotations

from importlib.metadata import version as _pkg_version


def get_version() -> str:
    """Return the installed lilbee version."""
    return _pkg_version("lilbee")
