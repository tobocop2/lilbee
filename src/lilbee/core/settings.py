"""Persistent settings stored in config.toml alongside the data directory."""

import os
import threading
import tomllib
import weakref
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Any, TypeVar

import tomli_w

from lilbee.config_meta import MODEL_ROLE_FIELDS, WRITABLE_CONFIG_FIELDS
from lilbee.core.config import CONFIG_FILE_NAME, Config, cfg
from lilbee.core.config.model import env_value, toml_value, toml_values
from lilbee.core.security import (
    file_lock,
    file_lock_or_warn,
    harden_private_file,
    write_private_text,
)

_settings_lock = threading.Lock()

T = TypeVar("T")

# A server, a CLI invocation, and an MCP process routinely run against the same
# data root, so the in-process mutex alone lets two of them interleave a
# read-modify-write and silently drop each other's keys.
_CONFIG_LOCK_TIMEOUT_S = 10.0


def _config_path(data_root: Path) -> Path:
    return data_root / CONFIG_FILE_NAME


@contextmanager
def _config_write_lock(data_root: Path) -> Generator[None, None, None]:
    """Serialize a config read-modify-write across threads and processes.

    Raises ``FileLockUnavailableError`` without running the block when the file
    lock cannot be taken. A write without the lock can drop another writer's
    update, and the OS releases the lock when its holder exits, so a stale lock
    file cannot cause the wait.
    """
    path = _config_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _settings_lock, file_lock(path, _CONFIG_LOCK_TIMEOUT_S):
        yield


@contextmanager
def _config_read_lock(data_root: Path) -> Generator[None, None, None]:
    """Keep a writer out of a read of config.toml; a lock that is busy only logs a warning.

    A reader sees a whole file either way, because a writer replaces it atomically.
    """
    path = _config_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _settings_lock, file_lock_or_warn(path, _CONFIG_LOCK_TIMEOUT_S):
        yield


def load(data_root: Path) -> dict[str, Any]:
    """Read all settings from config.toml. Returns {} if file is missing.

    Values keep the types TOML gave them. Stringifying here used to turn a
    ``true`` into ``"True"`` in memory, which the next save then wrote back
    quoted, so the file drifted away from valid types for its own fields.
    """
    path = _config_path(data_root)
    if not path.exists():
        return {}
    harden_private_file(path)
    with path.open("rb") as f:
        return dict(tomllib.load(f))


def save(data_root: Path, settings: dict[str, Any]) -> None:
    """Write *settings* to config.toml.

    ``tomli_w`` is the write half of the stdlib ``tomllib`` used by ``load``.
    The emitter this replaced escaped strings by hand and stringified anything
    that was not a bool or a number, so a list value was persisted as its
    quoted repr and read back as text. A control character it escaped wrongly
    was worse still: the reader discards the whole file on a parse error, so
    one bad value silently wiped every other setting.

    A ``None`` is dropped rather than written. TOML has no null, and the old
    emitter wrote the literal string "None", which then read back as a set
    value instead of an absent one. A path is written as its string: tomli_w
    refuses ``Path`` objects, and the config's path fields hold them after
    validation.
    """
    path = _config_path(data_root)
    present = {
        k: str(v) if isinstance(v, PurePath) else v
        for k, v in sorted(settings.items())
        if v is not None
    }
    # config.toml can hold provider API keys, so it gets the same owner-only
    # treatment as the session token rather than a post-hoc chmod.
    write_private_text(path, tomli_w.dumps(present))


def get(data_root: Path, key: str) -> str | None:
    """Look up a single key from config.toml, as text for callers that want text."""
    value = load(data_root).get(key)
    return None if value is None else str(value)


def set_value(data_root: Path, key: str, value: Any) -> None:
    """Read-modify-write a single key in config.toml."""
    with _config_write_lock(data_root):
        current = load(data_root)
        current[key] = value
        save(data_root, current)


def delete_value(data_root: Path, key: str) -> None:
    """Remove a key from config.toml. No-op if key doesn't exist."""
    with _config_write_lock(data_root):
        current = load(data_root)
        current.pop(key, None)
        save(data_root, current)


def update_values(data_root: Path, updates: dict[str, Any]) -> None:
    """Batch update multiple keys in config.toml (single write)."""
    with _config_write_lock(data_root):
        current = load(data_root)
        current.update(updates)
        save(data_root, current)


def delete_values(data_root: Path, keys: list[str]) -> None:
    """Batch delete multiple keys from config.toml (single write)."""
    with _config_write_lock(data_root):
        current = load(data_root)
        for key in keys:
            current.pop(key, None)
        save(data_root, current)


def mutate_value(data_root: Path, key: str, fn: Callable[[Any], tuple[Any, T]]) -> T:
    """Read-modify-write a single key under the config lock, atomically.

    ``fn`` receives the key's persisted value (or None if absent) read *inside*
    the lock and returns ``(new_value, result)``; the new value is written and
    the result is returned. Unlike a read-then-:func:`set_value`, the whole
    compound update is serialized across threads and processes, so two callers
    updating a dict-valued key cannot lose each other's change.
    """
    with _config_write_lock(data_root):
        current = load(data_root)
        new_value, result = fn(current.get(key))
        current[key] = new_value
        save(data_root, current)
    return result


def _overlay_file(root: Path) -> Path | None:
    """The config.toml an overlay of *root* reads: None without one or under the skip switch."""
    path = _config_path(root)
    if os.environ.get("LILBEE_SKIP_TOML_CONFIG") == "1" or not path.exists():
        return None
    return path


def overlay_stamp(root: Path) -> tuple[int, int, int] | None:
    """What a write of the config.toml an overlay of *root* reads changes; None without one."""
    path = _overlay_file(root)
    if path is None:
        return None
    try:
        stat = path.stat()
    except OSError:
        return None  # deleted since the check above
    return stat.st_mtime_ns, stat.st_size, stat.st_ino


def overlay_persisted_settings(root: Path) -> None:
    """Set on cfg each writable setting ``<root>/config.toml`` holds and no variable sets.

    A value Config refuses is left out with a warning that names the file, and
    the setting keeps the value it had. ``LILBEE_SKIP_TOML_CONFIG=1`` disables it.
    """
    path = _overlay_file(root)
    if path is None:
        return
    harden_private_file(path)
    stored = toml_values(path)
    overlayable = set(WRITABLE_CONFIG_FIELDS) | set(MODEL_ROLE_FIELDS)
    for key, value in stored.items():
        if key in overlayable and env_value(key) is None:
            setattr(cfg, key, value)


@dataclass
class _RegistryWatch:
    """The config.toml state one Config's registry was read at, and how many blocks hold it.

    ``lock`` guards the pin count and the swap of the stamp and the registry. It is never
    held across file IO. ``reading`` is held by the one caller that reads the file; any
    other caller uses the registry as it is.
    """

    stamp: tuple[int, int, int] | None = None
    pins: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    reading: threading.Lock = field(default_factory=threading.Lock)


class _RegistryWatches:
    """One watch per live Config, keyed by identity and dropped with the Config."""

    def __init__(self) -> None:
        self._by_config: dict[int, _RegistryWatch] = {}
        self._lock = threading.Lock()

    def of(self, config: Config) -> _RegistryWatch:
        with self._lock:
            watch = self._by_config.get(id(config))
            if watch is None:
                watch = self._by_config[id(config)] = _RegistryWatch()
                weakref.finalize(config, self._by_config.pop, id(config), None)
            return watch


_registry_watches = _RegistryWatches()


def refresh_registry(config: Config) -> bool:
    """Read the registry from config.toml again if another process changed the file.

    Costs an existence check and one stat() when the file is unchanged. A pinned registry
    is left alone. Returns False when it is pinned. A caller that finds another caller
    reading the file does not wait: it uses the registry as it is. The registry and the
    stamp change together under one small lock, inside the lock a writer takes.
    """
    watch = _registry_watches.of(config)
    if watch.pins:
        return False
    if not watch.reading.acquire(blocking=False):
        return True
    try:
        stamp = overlay_stamp(config.data_root)
        if stamp == watch.stamp:
            return True
        pinned = False

        def swap(roots: Any) -> None:
            nonlocal pinned
            with watch.lock:
                if watch.pins:
                    pinned = True
                    return
                if roots is not None:
                    config.linked_roots = roots
                watch.stamp = stamp

        _read_roots(config, swap)
        return not pinned
    finally:
        watch.reading.release()


@contextmanager
def pin_registry(config: Config) -> Generator[None, None, None]:
    """Hold the registry *config* has now: ``refresh_registry`` does nothing in the block."""
    watch = _registry_watches.of(config)
    with watch.lock:
        watch.pins += 1
    try:
        yield
    finally:
        with watch.lock:
            watch.pins -= 1


@contextmanager
def fresh_registry_pinned(config: Config) -> Generator[None, None, None]:
    """Read the registry again if the file changed, then hold it for the block.

    A block that reads the registry and then resolves keys against it sees one registry.
    """
    refresh_registry(config)
    with pin_registry(config):
        yield


def _read_roots(config: Config, accept: Callable[[Any], None]) -> None:
    """Pass *accept* the registry config.toml holds, None without one, under a writer's lock.

    *accept* runs inside the lock, so a writer cannot change the file between the read
    and the assignment.
    """
    path = _overlay_file(config.data_root)
    if path is None:
        accept(None)
        return
    with _config_read_lock(config.data_root):
        accept(toml_value(path, "linked_roots"))


def overlay_persisted_roots(config: Config) -> None:
    """Set on *config* the source registry its config.toml holds, under the lock a writer takes.

    A file without a registry sets nothing. A file that does not parse, or a
    registry Config refuses, is reported and sets nothing.
    ``LILBEE_SKIP_TOML_CONFIG=1`` disables it.
    """

    def assign(roots: Any) -> None:
        if roots is not None:
            config.linked_roots = roots

    _read_roots(config, assign)
