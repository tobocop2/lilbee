"""Security helpers: path validation, input sanitization, secret-file writes."""

from __future__ import annotations

import logging
import os
import stat
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock
from filelock import Timeout as FileLockTimeout

log = logging.getLogger(__name__)

_OWNER_ONLY_MODE = 0o600


@contextmanager
def file_lock_or_warn(path: Path, timeout_s: float) -> Iterator[None]:
    """Serialize access to *path* across processes via a sibling ``.lock`` file.

    On timeout the caller proceeds unserialized: losing coordination to a stale
    lock file is worse than the rare interleave the lock prevents.
    """
    flock = FileLock(str(path) + ".lock")
    try:
        flock.acquire(timeout=timeout_s)
    except FileLockTimeout:
        log.warning("Timed out waiting for the %s lock; proceeding without it.", path.name)
        yield
        return
    try:
        yield
    finally:
        flock.release()


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
    # Relative paths resolve against the CWD, not *root*, so anchor them here;
    # a traversal inside is still caught by the containment check below.
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else root_resolved / candidate).resolve()
    if not resolved.is_relative_to(root_resolved):
        raise PathTraversalError(f"Path escapes allowed directory: {path}")
    return resolved


def write_private_text(path: Path, text: str) -> None:
    """Write *text* to *path* so it is owner-only for its entire existence.

    Writing under the umask and chmod'ing afterwards leaves a window where any
    local user can read the file, and these callers persist a bearer token and
    API keys. ``mkstemp`` creates at 0600 and ``os.replace`` keeps that mode,
    atomically.

    Windows has no POSIX mode bits; there these rely on the inherited
    ``%LOCALAPPDATA%`` DACL.
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

    A secret file can arrive wider than :func:`write_private_text` leaves it
    (backup, older release) and is then read indefinitely without a rewrite, so
    callers narrow on every load. A refused chmod warns rather than raising: a
    file owned by someone else must not stop the caller from reading it.

    No-op on Windows, which has no POSIX mode bits.
    """
    if sys.platform == "win32":  # pragma: no cover - Windows uses the DACL
        return
    if stat.S_IMODE(path.stat().st_mode) == _OWNER_ONLY_MODE:
        return
    try:
        path.chmod(_OWNER_ONLY_MODE)
    except OSError:
        log.warning("Could not restrict permissions on %s.", path, exc_info=True)
