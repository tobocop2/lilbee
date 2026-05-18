"""Config read/update handlers for the HTTP server."""

from __future__ import annotations

import copy
import functools
from typing import Any

from pydantic_core import PydanticUndefined

from lilbee.app.settings import apply_settings_update
from lilbee.config_meta import (
    MODEL_ROLE_FIELDS as _MODEL_ROLE_FIELDS,
)
from lilbee.config_meta import (
    PUBLIC_CONFIG_FIELDS as _PUBLIC_CONFIG_FIELDS,
)
from lilbee.config_meta import (
    WRITABLE_CONFIG_FIELDS,
)
from lilbee.core.config import Config, cfg
from lilbee.server.models import ConfigResponse, ConfigUpdateResponse


async def update_config(updates: dict[str, Any]) -> ConfigUpdateResponse:
    """Partial update of writable config fields.

    Delegates validation, snapshot/rollback, persistence, and cache
    invalidation to ``app.settings.apply_settings_update`` so HTTP, MCP,
    CLI, and the TUI share one write boundary. Model role writes are
    refused at this surface because PUT /api/models/<role> already
    handles them with an install-availability check.
    """
    result = apply_settings_update(updates, allow_model_roles=False)
    return ConfigUpdateResponse(updated=result.updated, reindex_required=result.reindex_required)


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
