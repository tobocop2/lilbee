"""Per-model default generation settings.

Parses and caches generation parameters from Ollama's /api/show response
or GGUF file metadata so that model-specific defaults (temperature, num_ctx,
etc.) are applied automatically when switching models.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, fields
from typing import Any

log = logging.getLogger(__name__)

# Ollama parameter keys we recognise and their target types
_OLLAMA_PARAM_TYPES: dict[str, type] = {
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


_cache: dict[str, ModelDefaults] = {}


def get_defaults(model_name: str) -> ModelDefaults | None:
    """Return cached defaults for *model_name*, or None if not cached."""
    return _cache.get(model_name)


def set_defaults(model_name: str, defaults: ModelDefaults) -> None:
    """Store *defaults* in the in-memory cache keyed by *model_name*."""
    _cache[model_name] = defaults


def clear_cache() -> None:
    """Remove all cached model defaults (for test isolation)."""
    _cache.clear()


def parse_ollama_parameters(text: str) -> ModelDefaults:
    """Parse Ollama's multiline ``key value`` parameter format.

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
        if key not in _OLLAMA_PARAM_TYPES:
            continue
        try:
            values[key] = _OLLAMA_PARAM_TYPES[key](raw_value)
        except (ValueError, TypeError):
            log.debug("Skipping unparseable Ollama param %s=%r", key, raw_value)
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
