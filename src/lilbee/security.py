"""Security helpers — path validation, input sanitization."""

from __future__ import annotations

from pathlib import Path


def validate_path_within(path: str | Path, root: Path) -> Path:
    """Resolve *path* and verify it stays within *root*.

    Raises ``ValueError`` if the resolved path escapes the root directory.
    Returns the resolved path on success.
    """
    resolved = Path(path).resolve()
    root_resolved = root.resolve()
    if not resolved.is_relative_to(root_resolved):
        raise ValueError(f"Path escapes allowed directory: {path}")
    return resolved
