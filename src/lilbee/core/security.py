"""Security helpers: path validation, input sanitization, secret-file writes."""

from __future__ import annotations

import logging
import os
import stat
import sys
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

_OWNER_ONLY_MODE = 0o600


class PathTraversalError(ValueError):
    """Raised when a caller-supplied path escapes its allowed root.

    Subclasses ``ValueError`` so existing ``except ValueError`` callers keep
    working, while letting handlers catch *only* a traversal (not an unrelated
    downstream ``ValueError`` such as a store dimension mismatch).
    """


def validate_path_within(path: str | Path, root: Path) -> Path:
    """Resolve *path* under *root* and verify it stays within it.

    A relative *path* is taken as relative to *root*.
    Raises :class:`PathTraversalError` if the resolved path escapes the root.
    Returns the resolved path on success.
    """
    root_resolved = root.resolve()
    # A relative path resolves against the process CWD, not *root*, so a bare
    # user-supplied name was judged against wherever the process happened to be
    # started. Anchor it to root, which is what the docstring describes; a
    # traversal inside it is still caught by the containment check below.
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else root_resolved / candidate).resolve()
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


def harden_private_file(path: Path) -> None:
    """Narrow *path* to owner-only, tolerating a file we do not own.

    :func:`write_private_text` creates these files owner-only, but one can
    arrive wider: written by a release predating that hardening, restored from
    a backup, copied between machines. Their contents (a bearer token, provider
    API keys) are then read indefinitely without the file ever being rewritten,
    so callers narrow on every load rather than only at creation.

    Never fatal: a file owned by another user must not stop the caller from
    reading it, so a refused chmod warns and returns.

    Windows has no POSIX mode bits; there these files rely on the inherited
    ``%LOCALAPPDATA%`` DACL for owner-only access.
    """
    if sys.platform == "win32":  # pragma: no cover - Windows uses the DACL
        return
    if stat.S_IMODE(path.stat().st_mode) == _OWNER_ONLY_MODE:
        return
    try:
        path.chmod(_OWNER_ONLY_MODE)
    except OSError:
        log.warning("Could not restrict permissions on %s.", path, exc_info=True)
