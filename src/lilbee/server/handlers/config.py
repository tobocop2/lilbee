"""Config read/update handlers for the HTTP server."""

from __future__ import annotations

import copy
import functools
from typing import Any

from pydantic_core import PydanticUndefined

from lilbee.config_meta import (
    MODEL_ROLE_FIELDS as _MODEL_ROLE_FIELDS,
)
from lilbee.config_meta import (
    PUBLIC_CONFIG_FIELDS as _PUBLIC_CONFIG_FIELDS,
)
from lilbee.config_meta import (
    REINDEX_FIELDS,
    WRITABLE_CONFIG_FIELDS,
)
from lilbee.core import settings
from lilbee.core.config import Config, cfg
from lilbee.providers.sdk_backend import API_KEY_FIELDS
from lilbee.providers.sdk_llm_provider import inject_provider_keys
from lilbee.server.models import ConfigResponse, ConfigUpdateResponse

_MIN_CHUNK_SIZE = 64


def _validate_config_updates(updates: dict[str, Any]) -> None:
    """Reject unknown fields, null values on non-nullable fields, and invalid ranges."""
    for key, value in updates.items():
        if key not in WRITABLE_CONFIG_FIELDS:
            raise ValueError(f"Unknown or read-only config field: {key}")
        if value is None and not WRITABLE_CONFIG_FIELDS[key]:
            raise ValueError(f"Field '{key}' does not accept null")
    chunk_val = updates.get("chunk_size")
    if isinstance(chunk_val, int) and chunk_val < _MIN_CHUNK_SIZE:
        raise ValueError(f"chunk_size must be >= {_MIN_CHUNK_SIZE}")


def _apply_config_updates(updates: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    """Apply updates to the in-memory config, rolling back on error.
    Returns (fields_to_persist, fields_to_delete) for disk write.
    """
    snapshot = {k: getattr(cfg, k) for k in updates}
    to_persist: dict[str, str] = {}
    to_delete: list[str] = []
    try:
        for key, value in updates.items():
            if value is None:
                setattr(cfg, key, None)
                to_delete.append(key)
            else:
                setattr(cfg, key, value)
                to_persist[key] = str(getattr(cfg, key))
    except Exception:
        for k, v in snapshot.items():
            setattr(cfg, k, v)
        raise
    return to_persist, to_delete


async def update_config(updates: dict[str, Any]) -> ConfigUpdateResponse:
    """Partial update of writable config fields.
    Algorithm: validate-then-apply with rollback.

    1. Validate all keys and null-acceptability upfront (no mutations yet).
       This catches typos and bad input before anything changes.
    2. Snapshot current values, then apply each update via setattr (pydantic's
       validate_assignment catches type errors). If any field fails type
       validation, roll back ALL fields from the snapshot so the config
       stays consistent: no half-applied updates.
    3. Persist to disk in batch (one file write for sets, one for deletes)
       rather than per-field, avoiding partial writes on crash.

    Why not just setattr-and-save per field? A multi-field PATCH like
    {"chunk_size": 1024, "chunk_overlap": "bad"} would leave chunk_size
    changed but chunk_overlap unchanged: the caller gets an error but
    the config is silently modified. The snapshot/rollback prevents that.
    """
    _validate_config_updates(updates)
    to_persist, to_delete = _apply_config_updates(updates)
    if to_persist:
        settings.update_values(cfg.data_root, to_persist)
    if to_delete:
        settings.delete_values(cfg.data_root, to_delete)
    if API_KEY_FIELDS & set(updates):
        inject_provider_keys()
    reindex_required = bool(REINDEX_FIELDS & set(updates))
    return ConfigUpdateResponse(updated=list(updates), reindex_required=reindex_required)


async def get_config() -> ConfigResponse:
    """Return all user-facing configuration values."""
    dumped = cfg.model_dump()
    result = {k: v for k, v in dumped.items() if k in _PUBLIC_CONFIG_FIELDS}
    return ConfigResponse(**result)


@functools.cache
def _compute_config_defaults() -> dict[str, Any]:
    """Materialize Config defaults once per process."""
    defaults: dict[str, Any] = {}
    for name, info in Config.model_fields.items():
        is_writable_public = name in WRITABLE_CONFIG_FIELDS and name in _PUBLIC_CONFIG_FIELDS
        if not is_writable_public and name not in _MODEL_ROLE_FIELDS:
            continue
        value = info.get_default(call_default_factory=True)
        if value is PydanticUndefined:  # pragma: no cover
            continue
        defaults[name] = value
    return defaults


async def get_config_defaults() -> ConfigResponse:
    """Return canonical defaults for every public config field.

    Covers writable fields (resettable via PATCH /api/config) and the
    model-role fields (resettable via PUT /api/models/<role>).

    Deepcopies the cached dict so callers that mutate the response
    (list-valued fields like ``crawl_exclude_patterns``) cannot poison
    subsequent calls.
    """
    return ConfigResponse(**copy.deepcopy(_compute_config_defaults()))
