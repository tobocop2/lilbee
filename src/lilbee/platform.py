"""OS, environment, and platform helpers for lilbee."""

import os
import sys
from pathlib import Path


def env(key: str, default: str) -> str:
    """Read a LILBEE_<key> environment variable with fallback."""
    return os.environ.get(f"LILBEE_{key}", default)


def env_int(key: str, default: int) -> int:
    """Read a LILBEE_<key> environment variable as int with fallback."""
    raw = os.environ.get(f"LILBEE_{key}")
    if raw is None:
        return default
    return int(raw)


def default_data_dir() -> Path:
    """Return platform-appropriate data directory.

    - macOS:   ~/Library/Application Support/lilbee
    - Windows: %LOCALAPPDATA%/lilbee
    - Linux:   ~/.local/share/lilbee  (XDG_DATA_HOME)
    """
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "lilbee"


def is_ignored_dir(name: str, ignore_dirs: frozenset[str]) -> bool:
    """Return True if a directory name should be skipped during traversal."""
    return name.startswith(".") or name in ignore_dirs or name.endswith(".egg-info")
