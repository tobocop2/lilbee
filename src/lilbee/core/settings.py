"""Persistent settings stored in config.toml alongside the data directory."""

import logging
import os
import threading
import tomllib
from pathlib import Path
from typing import Any

from lilbee.config_meta import MODEL_ROLE_FIELDS, WRITABLE_CONFIG_FIELDS
from lilbee.core.config import cfg
from lilbee.core.security import write_private_text

_settings_lock = threading.Lock()


def _config_path(data_root: Path) -> Path:
    return data_root / "config.toml"


# The escapes TOML gives a short name to. Everything else in the C0 range
# (plus U+007F) has to go out as \uXXXX.
_TOML_NAMED_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\b": "\\b",
    "\f": "\\f",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}


def _escape_toml_string(s: str) -> str:
    """Escape a string for embedding in a TOML double-quoted value.

    TOML forbids every control character from appearing raw in a basic string,
    and the reader responds to a parse failure by discarding the whole file, so
    a single stray ESC or NUL in one setting value would silently wipe every
    other persisted setting on the next start.
    """
    out: list[str] = []
    for char in s:
        named = _TOML_NAMED_ESCAPES.get(char)
        if named is not None:
            out.append(named)
        elif char < "\x20" or char == "\x7f":
            out.append(f"\\u{ord(char):04X}")
        else:
            out.append(char)
    return "".join(out)


def load(data_root: Path) -> dict[str, Any]:
    """Read all settings from config.toml. Returns {} if file is missing.

    Values keep the types TOML gave them. Stringifying here used to turn a
    ``true`` into ``"True"`` in memory, which the next save then wrote back
    quoted, so the file drifted away from valid types for its own fields.
    """
    path = _config_path(data_root)
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return dict(tomllib.load(f))


def _render_toml_value(value: Any) -> str:
    """Render a scalar as TOML: booleans and numbers bare, everything else quoted."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    return f'"{_escape_toml_string(str(value))}"'


def save(data_root: Path, settings: dict[str, Any]) -> None:
    """Write settings dict as simple TOML key-value pairs."""
    path = _config_path(data_root)
    lines = [f"{k} = {_render_toml_value(v)}\n" for k, v in sorted(settings.items())]
    # config.toml can hold provider API keys, so it gets the same owner-only
    # treatment as the session token rather than a post-hoc chmod.
    write_private_text(path, "".join(lines))


def get(data_root: Path, key: str) -> str | None:
    """Look up a single key from config.toml, as text for callers that want text."""
    value = load(data_root).get(key)
    return None if value is None else str(value)


def set_value(data_root: Path, key: str, value: Any) -> None:
    """Read-modify-write a single key in config.toml (thread-safe)."""
    with _settings_lock:
        current = load(data_root)
        current[key] = value
        save(data_root, current)


def delete_value(data_root: Path, key: str) -> None:
    """Remove a key from config.toml. No-op if key doesn't exist."""
    with _settings_lock:
        current = load(data_root)
        current.pop(key, None)
        save(data_root, current)


def update_values(data_root: Path, updates: dict[str, Any]) -> None:
    """Batch update multiple keys in config.toml (single write)."""
    with _settings_lock:
        current = load(data_root)
        current.update(updates)
        save(data_root, current)


def delete_values(data_root: Path, keys: list[str]) -> None:
    """Batch delete multiple keys from config.toml (single write)."""
    with _settings_lock:
        current = load(data_root)
        for key in keys:
            current.pop(key, None)
        save(data_root, current)


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
