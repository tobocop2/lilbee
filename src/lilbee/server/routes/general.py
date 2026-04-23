"""General routes — health, status, config."""

from __future__ import annotations

from typing import Any

from litestar import Response, get, patch
from pydantic import ValidationError

from lilbee.server import handlers
from lilbee.server.auth import read_only
from lilbee.server.models import (
    ConfigResponse,
    ConfigUpdateResponse,
    HealthResponse,
    SourceContentResponse,
    StatusResponse,
)


@get("/api/health")
@read_only
async def health_route() -> HealthResponse:
    """Service health check returning server version and uptime status."""
    return await handlers.health()


@get("/api/status")
@read_only
async def status_route() -> StatusResponse:
    """Current configuration, indexed document sources, and chunk counts."""
    return await handlers.status()


@get("/api/config")
@read_only
async def config_route() -> ConfigResponse:
    """Return all user-facing configuration values."""
    return await handlers.get_config()


@get("/api/config/defaults")
@read_only
async def config_defaults_route() -> ConfigResponse:
    """Return canonical defaults for every writable, public configuration field."""
    return await handlers.get_config_defaults()


@patch("/api/config")
async def config_update_route(data: dict[str, Any]) -> ConfigUpdateResponse:
    """Partial update of writable configuration fields."""
    try:
        return await handlers.update_config(data)
    except (ValueError, ValidationError) as exc:
        from litestar.exceptions import ValidationException

        raise ValidationException(str(exc)) from exc


@get("/api/source")
@read_only
async def source_content_route(
    source: str, raw: bool = False
) -> SourceContentResponse | Response[bytes]:
    """Return stored source file content.

    ``raw=0`` (default) returns a JSON body with markdown text + content type.
    ``raw=1`` streams the raw bytes with the guessed Content-Type header so
    clients can render PDFs, images, or other binary formats directly.
    """
    from litestar.exceptions import NotFoundException

    try:
        result = await handlers.get_source_content(source, raw=raw)
    except FileNotFoundError as exc:
        raise NotFoundException(f"source not found: {source}") from exc
    except ValueError as exc:
        from litestar.exceptions import ValidationException

        raise ValidationException(str(exc)) from exc

    if raw:
        body, content_type = result  # type: ignore[misc]
        return Response(content=body, media_type=content_type, status_code=200)
    return result  # type: ignore[return-value]
