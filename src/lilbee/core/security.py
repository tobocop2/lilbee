"""Security helpers: path validation, input sanitization, secret-file writes."""

from __future__ import annotations

import os
import tempfile
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


def write_private_text(path: Path, text: str) -> None:
    """Write *text* to *path* so it is owner-only for its entire existence.

    Writing with the process umask and narrowing to 0600 afterwards leaves a
    window in which any local user can read the file, which matters because the
    callers here persist a bearer token and provider API keys. ``mkstemp``
    creates the temp file 0600 by construction, and ``os.replace`` carries that
    mode over, so the secret is never observable at wider permissions. The
    replace is also atomic, so a crash mid-write cannot truncate the original.

    Windows has no POSIX mode bits; there these files rely on the inherited
    ``%LOCALAPPDATA%`` DACL for owner-only access.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
