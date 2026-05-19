"""OS, environment, and platform helpers for lilbee."""

import os
import sys
from pathlib import Path

#: Directory name for a project-local lilbee knowledge base (sibling of ``.git/``).
LOCAL_ROOT_DIRNAME = ".lilbee"


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
    """Walk up from start (default: cwd) looking for a ``.lilbee/`` directory."""
    start = start or Path.cwd()
    for candidate in (start, *start.parents):
        marker = candidate / LOCAL_ROOT_DIRNAME
        if marker.is_dir():
            return marker
    return None


def canonical_models_dir() -> Path:
    """Return the shared models directory (always in the platform default, never per-project).
    Multiple lilbee instances share this directory so models are downloaded once.
    """
    return default_data_dir() / "models"


def is_ignored_dir(name: str, ignore_dirs: frozenset[str]) -> bool:
    """Return True if a directory name should be skipped during traversal."""
    return name.startswith(".") or name in ignore_dirs or name.endswith(".egg-info")


_CTX_TIER_FLOOR = 8192
_CTX_TIER_TABLE: tuple[tuple[int, int], ...] = (
    # (total_bytes_threshold, target)
    (64 * 1024**3, 24576),
    (32 * 1024**3, 16384),
    (16 * 1024**3, 12288),
)


def chat_ctx_target_for_total_bytes(total_bytes: int) -> int:
    """Pick a chat_n_ctx_target from total host RAM (floor 8192, tiers at 16/32/64 GiB)."""
    if total_bytes <= 0:
        return _CTX_TIER_FLOOR
    for threshold, target in _CTX_TIER_TABLE:
        if total_bytes >= threshold:
            return target
    return _CTX_TIER_FLOOR


def _read_total_memory_bytes() -> int:
    """Total system RAM in bytes, or 0 when introspection is unavailable."""
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:
        # psutil import or platform read failed; the caller falls back to the floor.
        return 0


def scaled_chat_ctx_target_default() -> int:
    """Pick a chat_n_ctx_target from this host's total RAM at config-load time."""
    return chat_ctx_target_for_total_bytes(_read_total_memory_bytes())
