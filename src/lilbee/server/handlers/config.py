"""Config read/update handlers for the HTTP server."""

from __future__ import annotations

import copy
import functools
from typing import Any

from pydantic_core import PydanticUndefined

from lilbee.app.settings import (
    SettingInfo,
    apply_settings_update,
    list_settings,
    provider_reset_refused_message,
    requires_services_reset,
)
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
from lilbee.server.models import (
    ConfigFieldSchema,
    ConfigResponse,
    ConfigSchemaResponse,
    ConfigUpdateResponse,
)


async def update_config(updates: dict[str, Any]) -> ConfigUpdateResponse:
    """Partial update of writable config fields.

    Delegates validation, snapshot/rollback, persistence, and cache
    invalidation to ``app.settings.apply_settings_update`` so HTTP, MCP,
    CLI, and the TUI share one write boundary. Model role writes are
    refused at this surface because PUT /api/models/<role> already
    handles them with an install-availability check.

    A provider switch rebuilds the shared Services singleton, unsafe while other
    clients have in-flight requests, so it is refused on the always-concurrent
    HTTP server; do it from the CLI instead.
    """
    if requires_services_reset(updates):
        raise ValueError(provider_reset_refused_message("Switching"))
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
        # The writable conjunct is redundant today (every public field is
        # writable or a model role) but keeps a future public-but-not-writable
        # field out of this payload.
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


def _field_schema(info: SettingInfo) -> ConfigFieldSchema:
    """Render one setting's metadata for the wire.

    ``writable`` is the PATCH /api/config contract, so it is false for the
    model role slots, which PUT /api/models/<role> owns.
    """
    return ConfigFieldSchema(
        key=info.key,
        type=info.type,
        nullable=info.nullable,
        writable=info.key in WRITABLE_CONFIG_FIELDS,
        reindex_required=info.reindex_required,
        group=info.group,
        help=info.help_text,
        choices=list(info.choices) if info.choices else None,
    )


async def get_config_schema() -> ConfigSchemaResponse:
    """Return per-field metadata for every public configuration field.

    The field list comes from the same settings boundary that MCP
    ``settings_list`` reads, so a new setting appears here with no route
    change and no restated value set.
    """
    return ConfigSchemaResponse(fields=[_field_schema(info) for info in list_settings()])
