"""Persistent settings stored in config.toml alongside the data directory."""

import logging
import os
import sys
import threading
import tomllib
from pathlib import Path

from lilbee.config_meta import MODEL_ROLE_FIELDS, WRITABLE_CONFIG_FIELDS
from lilbee.core.config import cfg

_settings_lock = threading.Lock()


def _config_path(data_root: Path) -> Path:
    return data_root / "config.toml"


def _escape_toml_string(s: str) -> str:
    """Escape a string for embedding in a TOML double-quoted value."""
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        .replace("\b", "\\b")
        .replace("\f", "\\f")
    )


def load(data_root: Path) -> dict[str, str]:
    """Read all settings from config.toml. Returns {} if file is missing."""
    path = _config_path(data_root)
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return {k: str(v) for k, v in tomllib.load(f).items()}


def save(data_root: Path, settings: dict[str, str]) -> None:
    """Write settings dict as simple TOML key-value pairs."""
    path = _config_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f'{k} = "{_escape_toml_string(v)}"\n' for k, v in sorted(settings.items())]
    path.write_text("".join(lines), encoding="utf-8", newline="\n")
    if sys.platform != "win32":
        path.chmod(0o600)  # pragma: no cover - POSIX-only; Windows has no 0600 mode bits


def get(data_root: Path, key: str) -> str | None:
    """Look up a single key from config.toml."""
    return load(data_root).get(key)


def set_value(data_root: Path, key: str, value: str) -> None:
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


def update_values(data_root: Path, updates: dict[str, str]) -> None:
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
