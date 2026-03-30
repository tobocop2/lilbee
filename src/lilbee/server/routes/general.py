"""General routes — health, status, config."""

from __future__ import annotations

from typing import Any

from litestar import get, patch
from pydantic import ValidationError

from lilbee.server.auth import read_only
from lilbee.server.models import ConfigResponse, ConfigUpdateResponse, HealthResponse


@get("/api/health")
@read_only
async def health_route() -> HealthResponse:
    """Service health check returning server version and uptime status."""
    from lilbee.server import handlers

    raw = await handlers.health()
    return HealthResponse(**raw)


@get("/api/status")
@read_only
async def status_route() -> dict[str, Any]:
    """Current configuration, indexed document sources, and chunk counts."""
    from lilbee.server import handlers

    return await handlers.status()


@get("/api/config")
@read_only
async def config_route() -> ConfigResponse:
    """Return all user-facing configuration values."""
    from lilbee.server import handlers

    return await handlers.get_config()


@patch("/api/config")
async def config_update_route(data: dict[str, Any]) -> ConfigUpdateResponse:
    """Partial update of writable configuration fields."""
    from lilbee.server import handlers

    try:
        return await handlers.update_config(data)
    except (ValueError, ValidationError) as exc:
        from litestar.exceptions import ValidationException

        raise ValidationException(str(exc)) from exc
