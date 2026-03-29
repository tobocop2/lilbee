"""General routes — health, status, config."""

from __future__ import annotations

from typing import Any

from litestar import get

from lilbee.server.auth import read_only
from lilbee.server.models import HealthResponse


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
async def config_route() -> dict[str, Any]:
    """Return all user-facing configuration values."""
    from lilbee.server import handlers

    return await handlers.get_config()
