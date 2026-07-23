"""Persistent settings stored in config.toml alongside the data directory."""

import logging
import os
import threading
import tomllib
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

import tomli_w

from lilbee.config_meta import MODEL_ROLE_FIELDS, WRITABLE_CONFIG_FIELDS
from lilbee.core.config import cfg
from lilbee.core.security import file_lock_or_warn, harden_private_file, write_private_text

_settings_lock = threading.Lock()

T = TypeVar("T")

# A server, a CLI invocation, and an MCP process routinely run against the same
# data root, so the in-process mutex alone lets two of them interleave a
# read-modify-write and silently drop each other's keys.
_CONFIG_LOCK_TIMEOUT_S = 10.0


def _config_path(data_root: Path) -> Path:
    return data_root / "config.toml"


@contextmanager
def _config_write_lock(data_root: Path) -> Generator[None, None, None]:
    """Serialize a config read-modify-write across threads and processes.

    A lock timeout falls through to the write rather than failing the caller:
    losing a settings update to a stale lock file is worse than the interleave
    the lock exists to prevent, which is already rare.
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
    value instead of an absent one.
    """
    path = _config_path(data_root)
    present = {k: v for k, v in sorted(settings.items()) if v is not None}
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


def overlay_persisted_settings(root: Path) -> None:
    """Overlay persisted scalars from ``<root>/config.toml`` onto cfg, skipping bad values.

    An explicit ``LILBEE_<FIELD>`` env var wins over config.toml (the documented
    precedence): cfg already holds the env-loaded value, so a key whose env var is
    set is left untouched rather than overwritten by the persisted file.

    ``LILBEE_SKIP_TOML_CONFIG=1`` disables this overlay entirely, matching the
    pydantic-settings source in ``config/model.py`` so the escape hatch is honored
    on every config-read path (import-time load, CLI callback, MCP server).
    """
    if os.environ.get("LILBEE_SKIP_TOML_CONFIG") == "1":
        return
    log = logging.getLogger(__name__)
    try:
        persisted = load(root)
    except (OSError, ValueError):
        log.warning("Failed to read %s/config.toml; using in-memory defaults", root)
        return
    if not persisted:
        return
    overlayable = set(WRITABLE_CONFIG_FIELDS) | set(MODEL_ROLE_FIELDS)
    env_prefix = cfg.model_config.get("env_prefix", "")
    for key, raw in persisted.items():
        if key not in overlayable:
            continue
        # Non-empty env var wins over config.toml (matches pydantic env_ignore_empty=True).
        if os.environ.get(f"{env_prefix}{key.upper()}", "") != "":
            continue
        # Legacy: set_setting used to persist None as "". Skip rather than
        # warn so a stale config doesn't spam logs on every CLI invocation.
        if raw == "":
            continue
        try:
            setattr(cfg, key, raw)
        except (ValueError, TypeError) as exc:
            log.warning(
                "Ignoring invalid persisted value for %s in %s: %s",
                key,
                root,
                exc,
            )
