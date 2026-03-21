"""Persistent settings stored in config.toml alongside the data directory."""

import tomllib
from pathlib import Path


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
    path.write_text("".join(lines))


def get(data_root: Path, key: str) -> str | None:
    """Look up a single key from config.toml."""
    return load(data_root).get(key)


def set_value(data_root: Path, key: str, value: str) -> None:
    """Read-modify-write a single key in config.toml."""
    current = load(data_root)
    current[key] = value
    save(data_root, current)


def delete_value(data_root: Path, key: str) -> None:
    """Remove a key from config.toml. No-op if key doesn't exist."""
    current = load(data_root)
    current.pop(key, None)
    save(data_root, current)
