"""Per-model default generation settings.

Parses and caches generation parameters from key-value parameter text
or GGUF file metadata so that model-specific defaults (temperature, num_ctx,
etc.) are applied automatically when switching models.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, fields
from typing import Any

log = logging.getLogger(__name__)

# Parameter keys we recognise and their target types
_KNOWN_PARAM_TYPES: dict[str, type] = {
    "temperature": float,
    "top_p": float,
    "top_k": int,
    "repeat_penalty": float,
    "num_ctx": int,
    "max_tokens": int,
}

# GGUF metadata keys mapped to ModelDefaults field names
_GGUF_KEY_MAP: dict[str, str] = {
    "general.temperature": "temperature",
    "general.top_p": "top_p",
    "general.top_k": "top_k",
    "general.repeat_penalty": "repeat_penalty",
}


@dataclass(frozen=True)
class ModelDefaults:
    """Frozen snapshot of a model's default generation parameters."""

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repeat_penalty: float | None = None
    num_ctx: int | None = None
    max_tokens: int | None = None


class _DefaultsCache:
    """Encapsulates the per-model defaults cache (no module-level mutable global)."""

    def __init__(self) -> None:
        self._data: dict[str, ModelDefaults] = {}

    def get(self, model_name: str) -> ModelDefaults | None:
        return self._data.get(model_name)

    def set(self, model_name: str, defaults: ModelDefaults) -> None:
        self._data[model_name] = defaults

    def clear(self) -> None:
        self._data.clear()


_defaults_cache = _DefaultsCache()

# Public API — preserves existing call sites.
get_defaults = _defaults_cache.get
set_defaults = _defaults_cache.set
clear_cache = _defaults_cache.clear


def parse_kv_parameters(text: str) -> ModelDefaults:
    """Parse multiline ``key value`` parameter format.
    Example input::

        temperature 0.7
        top_p 0.9
        num_ctx 4096
        stop <|im_end|>

    Unknown keys (like ``stop``) are silently skipped.
    """
    values: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        key, raw_value = parts
        if key not in _KNOWN_PARAM_TYPES:
            continue
        try:
            values[key] = _KNOWN_PARAM_TYPES[key](raw_value)
        except (ValueError, TypeError):
            log.debug("Skipping unparseable param %s=%r", key, raw_value)
    return ModelDefaults(**values)


def read_gguf_defaults(metadata: dict[str, str]) -> ModelDefaults:
    """Extract generation defaults from a GGUF metadata dict.
    Looks for keys like ``general.temperature``, ``context_length`` (via the
    architecture-prefixed key already resolved by the caller into
    ``context_length``).
    """
    values: dict[str, Any] = {}
    for gguf_key, field_name in _GGUF_KEY_MAP.items():
        if gguf_key in metadata:
            try:
                target_type = _field_type(field_name)
                values[field_name] = target_type(metadata[gguf_key])
            except (ValueError, TypeError):
                log.debug("Skipping unparseable GGUF key %s=%r", gguf_key, metadata[gguf_key])
    if "context_length" in metadata:
        with contextlib.suppress(ValueError, TypeError):
            values["num_ctx"] = int(metadata["context_length"])
    return ModelDefaults(**values)


def _field_type(field_name: str) -> type:
    """Return the base type for a ModelDefaults field."""
    for f in fields(ModelDefaults):
        if f.name == field_name:
            return int if "int" in str(f.type) else float
    return float  # pragma: no cover
