"""OS, environment, and platform helpers for lilbee."""

from pathlib import Path

from platformdirs import user_data_dir

#: Directory name for a project-local lilbee knowledge base (sibling of ``.git/``).
LOCAL_ROOT_DIRNAME = ".lilbee"


def default_data_dir() -> Path:
    """Return the per-user lilbee data directory for the current platform."""
    return Path(user_data_dir("lilbee", appauthor=False))


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
