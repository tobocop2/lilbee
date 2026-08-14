"""OS, environment, and platform helpers for lilbee."""

import os
import shutil
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

#: Directory name for a project-local lilbee knowledge base (sibling of ``.git/``).
LOCAL_ROOT_DIRNAME = ".lilbee"

# Reentrant: a suppressed block can re-enter this (directly or via a native
# helper wrapping its own stderr) and a plain Lock self-deadlocks. Nesting
# restores correctly: the inner exit puts back the outer's devnull.
_STDERR_LOCK = threading.RLock()


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


def default_state_dir() -> Path:
    """Return platform-appropriate directory for live runtime state.
    - macOS:   ~/Library/Application Support/lilbee
    - Windows: %LOCALAPPDATA%/lilbee
    - Linux:   ~/.local/state/lilbee  (XDG_STATE_HOME)

    Deliberately not a cache directory. This holds the machine engine slot: the
    state files recording a running llama-swap's pid and ports, the refcount
    lock dir, and the build lock. Those records are the only handle any
    out-of-process stop has on a running fleet, so a cleaner (or macOS evicting
    ~/Library/Caches under disk pressure) emptying the dir mid-run would orphan
    a fleet holding VRAM and leave the slot looking free to the next process,
    which would then build a second fleet on top of it.
    """
    if sys.platform == "darwin":  # pragma: no cover - platform split
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":  # pragma: no cover - platform split
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")).expanduser()
    else:  # pragma: no cover - platform split
        base = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return base / "lilbee"


def default_cache_dir() -> Path:
    """Return platform-appropriate directory for regenerable caches.

    - macOS:   ~/Library/Caches/lilbee
    - Windows: %LOCALAPPDATA%/lilbee/cache
    - Linux:   ~/.cache/lilbee  (XDG_CACHE_HOME)

    The counterpart to :func:`default_state_dir`. Everything here is derived data
    that costs time, not correctness, to lose, so a cleaner -- or macOS evicting
    ~/Library/Caches under disk pressure -- may empty it freely. Nothing that a
    stop path needs to find a running process belongs here.
    """
    if sys.platform == "darwin":  # pragma: no cover - platform split
        return Path.home() / "Library" / "Caches" / "lilbee"
    if sys.platform == "win32":  # pragma: no cover - platform split
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")).expanduser()
        return base / "lilbee" / "cache"
    return (  # pragma: no cover - platform split
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "lilbee"
    )


def find_local_root(start: Path | None = None) -> Path | None:
    """Walk up from start (default: cwd) looking for a ``.lilbee/`` directory."""
    start = start or Path.cwd()
    for candidate in (start, *start.parents):
        marker = candidate / LOCAL_ROOT_DIRNAME
        if marker.is_dir():
            return marker
    return None


def canonical_data_root(root: Path | str) -> Path:
    """Resolve a data root to one canonical path.

    Session file, port file, and write lock all derive from the data root, so
    two spellings of one directory key two locks. Symlinks, relative paths, a
    leading ``~``, and macOS ``/var`` vs ``/private/var`` each produce a pair.
    A root that does not exist yet resolves to where it will be created.

    Uses ``os.path`` rather than ``Path.expanduser().resolve()``: ``resolve``
    rebuilds via ``type(self)``, which raises for a ``PosixPath`` that exists
    on Windows (``Path()`` picks its flavour from ``os.name``, which tests patch).
    """
    return Path(os.path.realpath(os.path.expanduser(os.fspath(root))))


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
    # The top tier matches AGENT_CHAT_CTX_FLOOR: a 128 GiB host is a server or
    # pod whose GPUs can back an agent-sized window, and the dynamic picker
    # still clamps to trained context and device memory where they cannot.
    (128 * 1024**3, 65536),
    (64 * 1024**3, 24576),
    (32 * 1024**3, 16384),
    (16 * 1024**3, 12288),
)


def chat_ctx_target_for_total_bytes(total_bytes: int) -> int:
    """Pick a chat_n_ctx_target from total host RAM (floor 8192, tiers at 16/32/64/128 GiB)."""
    if total_bytes <= 0:
        return _CTX_TIER_FLOOR
    for threshold, target in _CTX_TIER_TABLE:
        if total_bytes >= threshold:
            return target
    return _CTX_TIER_FLOOR


_CGROUP_ROOT = Path("/sys/fs/cgroup")


def cgroup_memory_limit() -> int | None:
    """Bytes this process's cgroup allows, or ``None`` when unlimited or unreadable.

    cgroup v2 keeps the cap in ``memory.max`` (``max`` for unlimited); v1 uses
    ``memory/memory.limit_in_bytes``, which spells unlimited as a near-int64
    sentinel rather than a word, and so reads as a limit above installed RAM.
    Both are read, matching the CPU quota reader in :mod:`lilbee.runtime.cpu`.

    Every reader of host memory needs this: psutil reports the machine's
    ``/proc/meminfo``, which a memory-capped container sees in full, so a 4 GiB
    container on a 512 GiB machine sizes itself for the machine and is killed by
    the OOM reaper on its first load.
    """
    for path in (_CGROUP_ROOT / "memory.max", _CGROUP_ROOT / "memory" / "memory.limit_in_bytes"):
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if raw == "max":
            return None
        try:
            return int(raw)
        except ValueError:
            return None
    return None


def cgroup_memory_used() -> int | None:
    """Bytes this process's cgroup currently holds, or ``None`` when unreadable."""
    for path in (
        _CGROUP_ROOT / "memory.current",
        _CGROUP_ROOT / "memory" / "memory.usage_in_bytes",
    ):
        try:
            return int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
    return None


def capped_total_memory() -> int:
    """Total RAM this process may use in bytes; raises if the host cannot be read.

    Bounded by the cgroup cap where one applies; a limit above installed RAM is
    no limit at all, which is also how cgroup v1 spells unlimited.
    """
    import psutil

    host_total = int(psutil.virtual_memory().total)
    limit = cgroup_memory_limit()
    return min(host_total, limit) if limit is not None else host_total


def _read_total_memory_bytes() -> int:
    """:func:`capped_total_memory`, or 0 when introspection is unavailable.

    The config default needs an answer at import time and has a floor to fall
    back to, so it swallows the failure. Callers sizing a real placement want the
    exception instead: a budget silently computed from zero refuses every model
    with no reason given.
    """
    try:
        return capped_total_memory()
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


_EXTRA_BIN_DIRS: tuple[str, ...] = ("~/.local/bin", "~/.bun/bin")
_UNIX_BIN_DIRS: tuple[str, ...] = ("/opt/homebrew/bin", "/usr/local/bin")
_WINDOWS_BIN_DIRS: tuple[str, ...] = ("~/AppData/Roaming/npm", "~/AppData/Local/Programs")


def executable_search_path() -> str:
    """PATH plus the directories user-level installers put executables in.

    A server started from a desktop session inherits the login PATH, which
    misses the package-manager and per-user install dirs a shell profile adds.
    """
    platform_dirs = _WINDOWS_BIN_DIRS if sys.platform == "win32" else _UNIX_BIN_DIRS
    entries = [os.environ.get("PATH", "")]
    entries += [str(Path(d).expanduser()) for d in (*_EXTRA_BIN_DIRS, *platform_dirs)]
    return os.pathsep.join(entry for entry in entries if entry)


def find_executable(name: str) -> str | None:
    """Absolute path to the *name* executable, or None when it is not installed."""
    return shutil.which(name, path=executable_search_path())
