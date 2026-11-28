"""Persistent settings stored in config.toml alongside the data directory."""

import logging
import os
import threading
import time
import weakref
from collections.abc import Callable, Generator, Iterable, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePath
from typing import Any, TypeVar

import tomlkit
from tomlkit.exceptions import TOMLKitError

from lilbee.config_meta import MODEL_ROLE_FIELDS, WRITABLE_CONFIG_FIELDS
from lilbee.core.config import CONFIG_FILE_NAME, Config, cfg
from lilbee.core.config import resolve as resolve_mod
from lilbee.core.config.defaults import SKIP_TOML_ENV
from lilbee.core.config.enums import SettingSource
from lilbee.core.config.load_warnings import warn_on_load
from lilbee.core.config.model import overlay_source, refusal_sentence, toml_value
from lilbee.core.config.resolve import (
    PROFILE_NAME_KEY,
    PROFILE_TABLE,
    PROFILE_VALUES_KEY,
    ROOT_DERIVED_FIELDS,
    Resolved,
    read_layers,
    resolve_all,
    sanitize_soft_fields,
)
from lilbee.core.security import (
    file_lock,
    file_lock_or_warn,
    harden_private_file,
    write_private_text,
)
from lilbee.core.toml_read import load_toml

log = logging.getLogger(__name__)

_settings_lock = threading.Lock()

T = TypeVar("T")

# A server, a CLI invocation, and an MCP process routinely run against the same
# data root, so the in-process mutex alone lets two of them interleave a
# read-modify-write and silently drop each other's keys.
_CONFIG_LOCK_TIMEOUT_S = 10.0

# An overlay leaves these on cfg when the root gives them no value of its own.
HELD_AT_BUILT_IN = ROOT_DERIVED_FIELDS | set(MODEL_ROLE_FIELDS)


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

    A file that changes while it is read raises ConfigChangedWhileReadError: an editor
    that rewrites the file in place can leave it half written at that moment, and a
    read-modify-write of that half would drop every key it does not hold yet. Any other
    failure to read the file raises ConfigReadFailedError.
    """
    path = _config_path(data_root)
    if not path.exists():
        return {}
    try:
        harden_private_file(path)
        before = _stamp_of(path)
        with path.open("rb") as f:
            values = dict(load_toml(f))
        after = _stamp_of(path)
    except OSError as exc:
        # Name the read: a read-modify-write caller reports this as a failed write otherwise.
        raise ConfigReadFailedError(exc.errno, exc.strerror, exc.filename) from exc
    if after != before:
        raise ConfigChangedWhileReadError(f"{path} changed while it was read; try again")
    return values


class ConfigReadFailedError(OSError):
    """config.toml exists and could not be read."""


class ConfigChangedWhileReadError(ConfigReadFailedError):
    """config.toml changed while it was read, so the reading can hold half of a write."""


class ReadOnlyConfigError(PermissionError):
    """config.toml exists and its owner has made it read-only; a write would replace it."""


def save(data_root: Path, settings: dict[str, Any]) -> None:
    """Write *settings* to config.toml, keeping the comments and the order of the keys it holds.

    Only the keys whose value differs are touched; a key that is not in *settings* is removed.
    A ``None`` is dropped rather than written, since TOML has no null. A path is written as
    its string.
    """
    path = _config_path(data_root)
    if path.exists() and not os.access(path, os.W_OK):
        # The atomic replace swaps the file through a temp file, which a read-only file
        # does not stop: refuse here, the one writer every surface goes through.
        raise ReadOnlyConfigError(f"{path} is read-only; make it writable and try again")
    present = {
        k: str(v) if isinstance(v, PurePath) else v for k, v in settings.items() if v is not None
    }
    # config.toml can hold provider API keys, so it gets the same owner-only
    # treatment as the session token rather than a post-hoc chmod.
    text, newline = _merged_text(path, present)
    write_private_text(path, text, newline=newline)


def _merged_text(path: Path, present: dict[str, Any]) -> tuple[str, str]:
    """The text of *path* with *present* applied, and the line end to write it with.

    A missing or broken file starts new. A file whose every line ends in CRLF keeps
    CRLF; any other file gets LF.
    """
    document = tomlkit.document()
    newline = "\n"
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            held = handle.read()
            if handle.newlines == "\r\n":
                newline = "\r\n"
        with suppress(TOMLKitError, RecursionError):
            document = tomlkit.parse(held)
    _apply(document, present)
    return tomlkit.dumps(document), newline


def _apply(table: Any, wanted: Mapping[str, Any]) -> None:
    """Make *table* hold *wanted*, touching only the keys that differ, so comments stay.

    An empty table is assigned, not merged into: a table that holds only subtables
    has no header of its own, so emptying it would write nothing and lose the key.
    """
    held = table.unwrap()
    for key in [key for key in held if key not in wanted]:
        del table[key]
    for key, value in wanted.items():
        if key not in held:
            table[key] = value
        elif isinstance(value, dict) and value and isinstance(held[key], dict):
            _apply(table[key], value)
        elif not _same(held[key], value):
            table[key] = value


def _same(held: Any, value: Any) -> bool:
    """Equal in type and value, so True over 1 and 1.0 over 1 still write."""
    if isinstance(held, list) and isinstance(value, list):
        return len(held) == len(value) and all(
            _same(a, b) for a, b in zip(held, value, strict=True)
        )
    return type(held) is type(value) and held == value


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
    """Batch delete multiple keys from config.toml (single write, none when no key is present)."""
    with _config_write_lock(data_root):
        current = load(data_root)
        present = [key for key in dict.fromkeys(keys) if key in current]
        if not present:
            return
        for key in present:
            del current[key]
        save(data_root, current)


def write_profile_table(
    data_root: Path, name: str, values: Mapping[str, Any], drop: Iterable[str] = ()
) -> None:
    """Replace the ``[profile]`` table with *name* and *values* and remove the *drop* keys.

    Other top-level keys stay as they are; one write does both.
    """
    table: dict[str, Any] = {PROFILE_NAME_KEY: name}
    if values:
        table[PROFILE_VALUES_KEY] = dict(values)
    with _config_write_lock(data_root):
        current = load(data_root)
        for key in drop:
            current.pop(key, None)
        current[PROFILE_TABLE] = table
        save(data_root, current)


def mutate_value(
    data_root: Path,
    key: str,
    fn: Callable[[Any], tuple[Any, T]],
    after_save: Callable[[T], None] | None = None,
) -> T:
    """Read-modify-write a single key under the config lock, atomically.

    ``fn`` receives the key's persisted value (or None if absent) read *inside*
    the lock and returns ``(new_value, result)``; the new value is written and
    the result is returned. Unlike a read-then-:func:`set_value`, the whole
    compound update is serialized across threads and processes, so two callers
    updating a dict-valued key cannot lose each other's change.

    ``after_save`` receives the result once the file is written, still inside
    the lock, so a live value changes only when the write succeeded.
    """
    with _config_write_lock(data_root):
        current = load(data_root)
        new_value, result = fn(current.get(key))
        current[key] = new_value
        save(data_root, current)
        if after_save is not None:
            after_save(result)
    return result


def _assign_resolved(
    config: Config, resolved: Mapping[str, Resolved], keys: Iterable[str], root: Path
) -> list[str]:
    """Set each of *keys* on *config* to its resolved value, warning on one the field rejects.

    Returns the keys it set.
    """
    sanitized = sanitize_soft_fields(dict(resolved), root)
    assigned: list[str] = []
    for key in keys:
        entry = sanitized[key]
        try:
            setattr(config, key, entry.value)
        except (ValueError, TypeError):
            # A bad variable is the entry point's to refuse, in its one line.
            if entry.source is not SettingSource.ENV:
                warn_on_load(
                    f"Ignoring {entry.source} value in {root}: {refusal_sentence(key, entry.value)}"
                )
            continue
        assigned.append(key)
    return assigned


def sync_from_resolver(config: Config, keys: Iterable[str]) -> None:
    """Set each of *keys* on *config* to the value the resolver gives under its data root."""
    root = config.data_root
    _assign_resolved(config, resolve_all(read_layers(root)), keys, root)


def _overlay_file(root: Path) -> Path | None:
    """The config.toml an overlay of *root* reads: None without one or under the skip switch."""
    path = _config_path(root)
    if os.environ.get(SKIP_TOML_ENV) == "1" or not path.exists():
        return None
    return path


def overlay_stamp(root: Path) -> tuple[int, int, int] | None:
    """What a write of the config.toml an overlay of *root* reads changes; None without one."""
    path = _overlay_file(root)
    return None if path is None else _stamp_of(path)


def _stamp_of(path: Path) -> tuple[int, int, int] | None:
    """What a write of the file at *path* changes; None when it is missing."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size, stat.st_ino


@dataclass(frozen=True)
class OverlayReading:
    """What one overlay read from a root's config.toml: the file's stamp and the values it set."""

    root: Path
    stamp: tuple[int, int, int] | None
    values: Mapping[str, Any]


class _LastOverlay:
    """The reading of the last overlay onto cfg, so a follower started later begins from it."""

    def __init__(self) -> None:
        self._reading: OverlayReading | None = None
        self._lock = threading.Lock()

    def record(self, reading: OverlayReading) -> None:
        with self._lock:
            self._reading = reading

    def of(self, root: Path) -> OverlayReading | None:
        with self._lock:
            reading = self._reading
        return reading if reading is not None and reading.root == root else None


_last_overlay = _LastOverlay()


def overlay_persisted_settings(root: Path) -> None:
    """Set every overlayable key on cfg to its resolved value under ``root``.

    A value in ``<root>/config.toml`` that Config refuses is left out with a warning
    that names the file, and the setting keeps the value it had. A file that does not
    parse is reported and changes nothing. A root-derived path at its built-in keeps the
    value the caller set from ``root``, and a model role at its built-in keeps the
    model cfg holds, so entering a fresh root leaves it usable. A root
    without the file resolves as a root with an empty one, and
    ``LILBEE_SKIP_TOML_CONFIG=1`` disables the overlay. The file is read once, under
    the lock a writer takes, and ``last_overlay`` returns that reading.
    """
    stamp, values = read_persisted(root)
    for key, value in (values or {}).items():
        setattr(cfg, key, value)
    _last_overlay.record(OverlayReading(root, stamp, values or {}))


def last_overlay(root: Path) -> OverlayReading | None:
    """The reading the last overlay took, when it was of *root*; None otherwise."""
    return _last_overlay.of(root)


def persisted_values(root: Path) -> dict[str, Any] | None:
    """The value the overlay of *root* sets on each key, as cfg holds it.

    A key the overlay leaves alone is absent: a value Config refuses, a root-derived
    path at its built-in, and a model role at its built-in. None under the skip switch
    or when the file does not parse, because the overlay then changes nothing.
    """
    if os.environ.get(SKIP_TOML_ENV) == "1":
        return None
    path = _config_path(root)
    data: Mapping[str, Any] = {}
    accepted: Mapping[str, Any] = {}
    if path.exists():
        harden_private_file(path)
        source = overlay_source(path)
        table = resolve_mod.read_config_table(path)
        if table is None:
            source.report_unreadable()
            return None
        data = table
        accepted = source.accept(table)
    layers = read_layers(root, data)
    refused = (layers.user.keys() - accepted.keys()) | layers.refused_user
    resolved = resolve_all(
        replace(layers, user={key: layers.user[key] for key in layers.user.keys() - refused})
    )
    keys = [
        key
        for key in sorted((set(WRITABLE_CONFIG_FIELDS) | MODEL_ROLE_FIELDS) - refused)
        if not (key in HELD_AT_BUILT_IN and resolved[key].source is SettingSource.BUILT_IN)
    ]
    trial = cfg.model_copy()
    return {key: getattr(trial, key) for key in _assign_resolved(trial, resolved, keys, root)}


SETTLE_NS = 1_000_000_000  # a file changed less than this long ago may be mid-rewrite


def read_persisted(root: Path) -> tuple[tuple[int, int, int] | None, dict[str, Any] | None]:
    """The stamp of *root*'s config.toml and ``persisted_values``, from one locked read.

    The read holds the lock a writer takes, so it never sees a write by lilbee in
    progress. Without the file there is nothing to lock, so the read creates no folder
    or lock file. Another program can rewrite the file in place without the lock: when
    the stamp moves during the read, the reading can hold half of that write, so the
    values are None, as for a file that does not parse, and the stamp is None, so the
    next check reads the file again.
    """
    if overlay_stamp(root) is None:
        return _stable_reading(root)
    with _config_read_lock(root):
        return _stable_reading(root)


def _now_ns() -> int:
    """The clock ``read_settled`` measures a file's age against."""
    return time.time_ns()


def read_settled(
    root: Path, last_stamp: tuple[int, int, int] | None
) -> tuple[tuple[int, int, int] | None, dict[str, Any] | None]:
    """``read_persisted``, but an in-place write under ``SETTLE_NS`` old is not taken.

    An editor can truncate the file and write it back a moment later, and the empty
    file in between is stable to ``read_persisted``. An atomic replace (every write by
    lilbee, and the default of some editors) gives the file a new inode and is never
    torn, so it is taken at once. A file with the inode of *last_stamp* and a young
    modification time may be mid-write: the reading has values None and stamp None, so
    the next check reads it again; a reader that runs only on events asks
    ``settle_remaining_s`` when to check again. With no *last_stamp* the read is plain.
    A modification time in the future (a skewed clock) is not young.

    Gaps: a file system whose inode numbers do not change on a replace gets the window
    for every write; an in-place edit that stays torn for more than ``SETTLE_NS`` is taken.
    """
    stamp, values = read_persisted(root)
    if _settle_remaining_ns(stamp, last_stamp) > 0:
        return None, None
    return stamp, values


_SETTLE_MARGIN_S = 0.05


def _settle_remaining_ns(
    stamp: tuple[int, int, int] | None, last_stamp: tuple[int, int, int] | None
) -> int:
    """Nanoseconds until *stamp*, an in-place write of the file *last_stamp* names, is old."""
    if stamp is None or last_stamp is None or stamp[2] != last_stamp[2]:
        return 0
    age = _now_ns() - stamp[0]
    return SETTLE_NS - age if 0 <= age < SETTLE_NS else 0


def settle_remaining_s(root: Path, last_stamp: tuple[int, int, int] | None) -> float:
    """Seconds until ``read_settled`` takes the file of *root* that *last_stamp* names; 0.0 now.

    Above zero, a check that reads the file now skips it. A reader that runs only on
    events calls this after a check that left the file unread (its reading still holds
    *last_stamp*) and checks again after this long.
    """
    remaining = _settle_remaining_ns(overlay_stamp(root), last_stamp)
    return remaining / 1e9 + _SETTLE_MARGIN_S if remaining > 0 else 0.0


def _stable_reading(
    root: Path,
) -> tuple[tuple[int, int, int] | None, dict[str, Any] | None]:
    """``persisted_values`` of *root* and its stamp; both None when the stamp moved during it."""
    stamp = overlay_stamp(root)
    values = persisted_values(root)
    if overlay_stamp(root) != stamp:
        return None, None
    return stamp, values


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
