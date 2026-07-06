"""OS, environment, and platform helpers for lilbee."""

import os
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

#: Directory name for a project-local lilbee knowledge base (sibling of ``.git/``).
LOCAL_ROOT_DIRNAME = ".lilbee"

_STDERR_LOCK = threading.Lock()


@contextmanager
def stderr_suppressed() -> Iterator[None]:
    """Redirect fd 2 to /dev/null for the duration of the block.

    Silences C-library stderr (native document extractors, GGUF readers) that
    bypasses Python's logging. Holds a process lock so concurrent fd-2 swaps
    can't clobber each other's saved descriptor. Wrap the whole native call, not
    each inner iteration, so the lock doesn't serialize a hot loop.

    On Windows, MSVC-built native extensions use GetStdHandle rather than the
    CRT fd 2, so the fd-dup technique has no effect there. The context manager
    is a no-op on Windows to avoid false suppression expectations.
    """
    if sys.platform == "win32":  # pragma: no cover - Windows-only passthrough
        yield
        return
    with _STDERR_LOCK:
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        os.dup2(devnull, 2)
        try:
            yield
        finally:
            os.dup2(old_stderr, 2)
            os.close(devnull)
            os.close(old_stderr)


def default_data_dir() -> Path:
    """Return platform-appropriate data directory.
    - macOS:   ~/Library/Application Support/lilbee
    - Windows: %LOCALAPPDATA%/lilbee
    - Linux:   ~/.local/share/lilbee  (XDG_DATA_HOME)
    """
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")).expanduser()
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


# Filesystem types whose backing store is a network, where mmap page faults are
# served over the wire and can wedge the model loader in uninterruptible I/O. The
# exact type string a given volume reports (e.g. a RunPod network volume) is
# confirmed on the target host and added here.
_NETWORK_FS_TYPES = frozenset(
    {"nfs", "nfs4", "cifs", "smb3", "smbfs", "9p", "ceph", "glusterfs", "lustre", "beegfs", "afs"}
)
# A /proc/mounts line is "device mountpoint fstype options ...": at least 3 fields.
_PROC_MOUNTS_MIN_FIELDS = 3
_PROC_MOUNTS = Path("/proc/mounts")


def _mount_fstype(path: str, mounts_text: str) -> str:
    """Filesystem type of the longest mount point in *mounts_text* that covers *path*."""
    best_mount = ""
    best_type = ""
    for line in mounts_text.splitlines():
        parts = line.split()
        if len(parts) < _PROC_MOUNTS_MIN_FIELDS:
            continue
        mount_point, fs_type = parts[1], parts[2]
        covers = path == mount_point or path.startswith(mount_point.rstrip("/") + "/")
        if covers and len(mount_point) >= len(best_mount):
            best_mount, best_type = mount_point, fs_type
    return best_type


def is_network_path(path: Path) -> bool:
    """Whether *path* lives on a network filesystem.

    mmap over a network filesystem faults pages over the wire, which can stall a
    large-model load in uninterruptible I/O. Linux-only (reads ``/proc/mounts``);
    returns False on other platforms and on any read failure, so local disk is the
    safe assumption.
    """
    try:
        mounts_text = _PROC_MOUNTS.read_text(encoding="utf-8")
    except OSError:
        return False
    try:
        resolved = str(path.resolve())
    except OSError:
        resolved = str(path)
    fstype = _mount_fstype(resolved, mounts_text)
    return fstype in _NETWORK_FS_TYPES or fstype.startswith("fuse.")
