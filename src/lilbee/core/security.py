"""Security helpers: path validation, input sanitization, secret-file writes."""

from __future__ import annotations

import logging
import os
import secrets
import stat
import sys
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager, suppress
from pathlib import Path

from filelock import FileLock
from filelock import Timeout as FileLockTimeout

log = logging.getLogger(__name__)

OWNER_ONLY_MODE = 0o600
OWNER_ONLY_DIR_MODE = 0o700

_Opener = Callable[[str, int], int]


class FileLockUnavailableError(OSError):
    """Raised when a file lock cannot be taken; the guarded file is untouched."""


@contextmanager
def file_lock(path: Path, timeout_s: float) -> Iterator[None]:
    """Serialize access to *path* across processes via a sibling ``.lock`` file.

    Raises ``FileLockUnavailableError`` when another process holds the lock
    past *timeout_s*, or when the operating system refuses the lock itself.
    The block does not run then, so a writer never writes without the lock.
    A refused chmod on the lock file does not fail the lock: some NFS/FUSE/SMB
    mounts answer ``ENOTSUP``, and the lock is held all the same, so that only
    logs a warning. The lock file is created owner-only, so filelock is given
    no mode and never calls fchmod.
    """
    lock_path = Path(str(path) + ".lock")
    flock = FileLock(lock_path)
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        os.close(os.open(lock_path, os.O_CREAT | os.O_RDWR, OWNER_ONLY_MODE))
        flock.acquire(timeout=timeout_s)
    except FileLockTimeout:
        raise FileLockUnavailableError(
            f"another lilbee process held the lock on {path.name} for {timeout_s:g} seconds"
        ) from None
    except OSError as exc:
        raise FileLockUnavailableError(
            f"the lock on {path.name} cannot be taken ({exc.strerror or exc})"
        ) from exc
    try:
        harden_private_file(lock_path)
        yield
    finally:
        flock.release()


@contextmanager
def file_lock_or_warn(path: Path, timeout_s: float) -> Iterator[None]:
    """Like :func:`file_lock`, but a lock that cannot be taken only logs a warning.

    For a reader, which sees a whole file because writers replace it atomically.
    """
    with ExitStack() as stack:
        try:
            stack.enter_context(file_lock(path, timeout_s))
        except FileLockUnavailableError as exc:
            log.warning("Lock on %s not taken (%s); proceeding without it.", path.name, exc)
        yield


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
    API keys. The temp file is created 0600 and ``os.replace`` keeps that mode.

    Windows has no POSIX mode bits; there these rely on the inherited
    ``%LOCALAPPDATA%`` DACL.
    """
    _write_atomically(path, text, private_opener)


def write_text_atomically(path: Path, text: str) -> None:
    """Write *text* to *path* atomically, with the mode the umask gives a new file."""
    _write_atomically(path, text, None)


def _write_atomically(path: Path, text: str, opener: _Opener | None) -> None:
    """Write a temp file beside *path*, fsync it, then ``os.replace`` it onto *path*.

    The data is fsynced before the rename, so a crash cannot leave the target empty.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with open(tmp, "x", encoding="utf-8", newline="\n", opener=opener) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException as exc:
        with suppress(OSError):
            tmp.unlink(missing_ok=True)
        if isinstance(exc, OSError):
            # The error names the file the user asked to write, not the hidden temp file.
            raise OSError(exc.errno, exc.strerror, str(path)) from None
        raise


def private_opener(path: str, flags: int) -> int:
    """``open(..., opener=)`` hook that creates a missing file owner-only."""
    return os.open(path, flags, OWNER_ONLY_MODE)


def harden_private_file(path: Path) -> None:
    """Narrow *path* to owner-only, tolerating a file we do not own.

    A secret file can arrive wider than :func:`write_private_text` leaves it
    (backup, older release) and is then read indefinitely without a rewrite, so
    callers narrow on every load. A refused chmod warns rather than raising: a
    file owned by someone else must not stop the caller from reading it.

    No-op on Windows, which has no POSIX mode bits.
    """
    _narrow_mode(path, OWNER_ONLY_MODE)


def ensure_private_dir(path: Path) -> None:
    """Create *path* owner-only, or narrow it when it already exists wider.

    Same failure semantics as :func:`harden_private_file`.
    """
    path.mkdir(mode=OWNER_ONLY_DIR_MODE, parents=True, exist_ok=True)
    _narrow_mode(path, OWNER_ONLY_DIR_MODE)


def _narrow_mode(path: Path, mode: int) -> None:
    """Clear the bits of *path* that *mode* lacks; a refused chmod warns.

    Never adds a bit: a file its owner made read-only stays read-only.
    """
    if sys.platform == "win32":  # pragma: no cover - Windows uses the DACL
        return
    current = stat.S_IMODE(path.stat().st_mode)
    narrowed = current & mode
    if narrowed == current:
        return
    try:
        path.chmod(narrowed)
    except OSError:
        log.warning("Could not restrict permissions on %s.", path, exc_info=True)
