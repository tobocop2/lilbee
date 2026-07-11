"""Security helpers: path validation, input sanitization."""

from __future__ import annotations

from pathlib import Path


class PathTraversalError(ValueError):
    """Raised when a caller-supplied path escapes its allowed root.

    Subclasses ``ValueError`` so existing ``except ValueError`` callers keep
    working, while letting handlers catch *only* a traversal (not an unrelated
    downstream ``ValueError`` such as a store dimension mismatch).
    """


def validate_path_within(path: str | Path, root: Path) -> Path:
    """Resolve *path* and verify it stays within *root*.
    Raises :class:`PathTraversalError` if the resolved path escapes the root.
    Returns the resolved path on success.
    """
    resolved = Path(path).resolve()
    root_resolved = root.resolve()
    if not resolved.is_relative_to(root_resolved):
        raise PathTraversalError(f"Path escapes allowed directory: {path}")
    return resolved
