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
    try:
        return int(raw)
    except ValueError:
        import logging

        logging.getLogger(__name__).warning(
            "Invalid LILBEE_%s=%r, using default %d", key, raw, default
        )
        return default


def env_float(key: str, default: float | None = None) -> float | None:
    """Read a LILBEE_<key> environment variable as float with fallback."""
    raw = os.environ.get(f"LILBEE_{key}")
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        import logging

        logging.getLogger(__name__).warning(
            "Invalid LILBEE_%s=%r, using default %s", key, raw, default
        )
        return default


def env_int_optional(key: str) -> int | None:
    """Read a LILBEE_<key> environment variable as optional int."""
    raw = os.environ.get(f"LILBEE_{key}")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        import logging

        logging.getLogger(__name__).warning("Invalid LILBEE_%s=%r, ignoring", key, raw)
        return None


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


def find_local_root(start: Path | None = None) -> Path | None:
    """Walk up from start (default: cwd) looking for .lilbee/ directory."""
    current = start or Path.cwd()
    while True:
        candidate = current / ".lilbee"
        if candidate.is_dir():
            return candidate
        parent = current.parent
        if parent == current:
            return None
        current = parent


def canonical_models_dir() -> Path:
    """Return the shared models directory (always in the platform default, never per-project).

    Multiple lilbee instances share this directory so models are downloaded once.
    """
    return default_data_dir() / "models"


def is_ignored_dir(name: str, ignore_dirs: frozenset[str]) -> bool:
    """Return True if a directory name should be skipped during traversal."""
    return name.startswith(".") or name in ignore_dirs or name.endswith(".egg-info")
