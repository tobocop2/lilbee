"""Persistent settings stored in config.toml alongside the data directory."""

import tomllib
from pathlib import Path


def _config_path() -> Path:
    """Return path to the persistent config file (next to documents/ and data/)."""
    from lilbee.config import _data_root

    return _data_root / "config.toml"


def load() -> dict[str, str]:
    """Read all settings from config.toml. Returns {} if file is missing."""
    path = _config_path()
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return {k: str(v) for k, v in tomllib.load(f).items()}


def save(settings: dict[str, str]) -> None:
    """Write settings dict as simple TOML key-value pairs."""
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f'{k} = "{v}"\n' for k, v in sorted(settings.items())]
    path.write_text("".join(lines))


def get(key: str) -> str | None:
    """Look up a single key from config.toml."""
    return load().get(key)


def set_value(key: str, value: str) -> None:
    """Read-modify-write a single key in config.toml."""
    settings = load()
    settings[key] = value
    save(settings)
