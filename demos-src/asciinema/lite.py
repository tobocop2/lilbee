#!/usr/bin/env python3
"""Shared LITE data root for the startup reels.

first-start, later-start and cold-start are the same launch measured three ways, so they
have to answer from the same knowledge base with the same model or the comparison the
README draws between them is meaningless. One root, built once, reused by all three.
"""
from __future__ import annotations

import pathlib
import shutil

LITE = pathlib.Path.home() / ".cache/lilbee-reel/lite"
SOURCE = pathlib.Path.home() / ".cache/lilbee-reel/whatis"

# The Nuitka onefile payload unpacks here on first run of a given version. Removing the
# directory is what makes a cold start cold; the next launch rebuilds it.
UNPACK_CACHE = pathlib.Path.home() / ".cache/lilbee/0.6.90.420"
BINARY = "/opt/homebrew/bin/lilbee"


def ensure() -> pathlib.Path:
    """Return the LITE root, seeding it from the what_is_lilbee root on first use."""
    if (LITE / "data").exists():
        return LITE
    if not (SOURCE / "data").exists():
        raise SystemExit("build the what_is_lilbee root first; it is the seed for LITE")
    shutil.copytree(SOURCE, LITE)
    return LITE


def go_cold() -> bool:
    """Drop the unpack cache so the next launch shows the one-time unpack bar."""
    if not UNPACK_CACHE.exists():
        return False
    shutil.rmtree(UNPACK_CACHE)
    return True
