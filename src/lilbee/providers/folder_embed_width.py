"""The output width of an embedding model stored as a folder of config and weight files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_CONFIG = "config.json"
_MODULES = "modules.json"
_TEXT_CONFIG_KEY = "text_config"
_HIDDEN_KEY = "hidden_size"
_POOLING_WIDTH_KEY = "word_embedding_dimension"
_POOLING_MODE_PREFIX = "pooling_mode_"
_DENSE_WIDTH_KEY = "out_features"
_POOLING_SUFFIX = ".Pooling"
_DENSE_SUFFIX = ".Dense"


def _read_json(path: Path) -> object:
    """The JSON value in *path*, or None when the file is absent or unreadable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return None


def read_folder_config(folder: Path) -> dict[str, Any]:
    """The ``config.json`` object of a model folder, or an empty one when it is not an object."""
    return _read_object(folder / _CONFIG)


def _read_object(path: Path) -> dict[str, Any]:
    """The JSON object in *path*, or an empty one when the file is not an object."""
    data = _read_json(path)
    return data if isinstance(data, dict) else {}


def _positive_int(value: object) -> int | None:
    """*value* as an int above zero, else None; a bool is not a width."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


def _hidden_size(folder: Path) -> int | None:
    """``hidden_size`` of the folder's ``config.json``, read from its text section if it has one."""
    config = _read_object(folder / _CONFIG)
    text = config.get(_TEXT_CONFIG_KEY)
    section = text if isinstance(text, dict) else config
    return _positive_int(section.get(_HIDDEN_KEY))


def _module_config(folder: Path, entry: dict[str, Any]) -> dict[str, Any]:
    """The config of the module folder that the ``modules.json`` *entry* names."""
    sub = entry.get("path")
    if not isinstance(sub, str) or not sub:
        return {}
    return _read_object(folder / sub / _CONFIG)


def _module_value(folder: Path, entry: dict[str, Any], key: str) -> int | None:
    """*key* from the config file of the module folder that the ``modules.json`` *entry* names."""
    return _positive_int(_module_config(folder, entry).get(key))


def _pooling_width(folder: Path, entry: dict[str, Any], concatenates: bool) -> int | None:
    """The width of the pooling step: its input width, times the enabled modes if *concatenates*."""
    config = _module_config(folder, entry)
    base = _positive_int(config.get(_POOLING_WIDTH_KEY))
    modes = sum(
        1 for key, flag in config.items() if key.startswith(_POOLING_MODE_PREFIX) and flag is True
    )
    blocks = max(modes, 1) if concatenates else 1
    return base * blocks if base else None


def folder_embedding_width(folder: Path, *, concatenates_pooling: bool = False) -> int | None:
    """The vector width an embedding model folder produces, or None when no file says.

    The folder may list a pipeline in ``modules.json``. The pooling step states
    the width it emits. An engine that serves one pooling mode gives the width
    of one block. An engine that concatenates modes (*concatenates_pooling*)
    gives one block per enabled mode. The last dense step states the width it
    emits, which wins. Without a pipeline the width is the ``hidden_size`` of
    ``config.json``.
    """
    width = _hidden_size(folder)
    modules = _read_json(folder / _MODULES)
    entries = [e for e in modules if isinstance(e, dict)] if isinstance(modules, list) else []
    for entry in entries:
        kind = str(entry.get("type", ""))
        if kind.endswith(_POOLING_SUFFIX):
            width = _pooling_width(folder, entry, concatenates_pooling) or width
        elif kind.endswith(_DENSE_SUFFIX):
            width = _module_value(folder, entry, _DENSE_WIDTH_KEY) or width
    return width
