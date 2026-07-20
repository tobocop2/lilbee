"""General routes: health, status, config, source, warm.

Every route here needs the session token, ``/api/health`` included. Health
looks like the one endpoint that could stay open, but it reports the chat
engine's last error string, which carries model paths and loader failures, so
an unauthenticated liveness probe would hand out the most useful reconnaissance
on the box. A local probe runs as the user and can read the token out of
server.json the same way every other local client does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from litestar import Response, get, patch
from litestar.exceptions import NotFoundException, ValidationException
from litestar.response import Stream
from pydantic import ValidationError

from lilbee.server import handlers
from lilbee.server.models import (
    ConfigResponse,
    ConfigUpdateResponse,
    HealthResponse,
    SourceContentResponse,
    StatusResponse,
)


@get("/api/health")
async def health_route() -> HealthResponse:
    """Service health check returning server version and uptime status."""
    return await handlers.health()


@get("/api/warm/stream")
async def warm_stream_route() -> Stream:
    """Stream chat-model cold-load progress as SSE for a launcher's warm indicator."""
    return Stream(handlers.warm_stream(), media_type="text/event-stream")


@get("/api/status")
async def status_route() -> StatusResponse:
    """Current configuration, indexed document sources, and chunk counts."""
    return await handlers.status()


@get("/api/config")
async def config_route() -> ConfigResponse:
    """Return all user-facing configuration values."""
    return await handlers.get_config()


@get("/api/config/defaults")
async def config_defaults_route() -> ConfigResponse:
    """Return canonical defaults for every writable, public configuration field."""
    return await handlers.get_config_defaults()


@patch("/api/config")
async def config_update_route(data: dict[str, Any]) -> ConfigUpdateResponse:
    """Partial update of writable configuration fields."""
    try:
        return await handlers.update_config(data)
    except (ValueError, ValidationError) as exc:
        raise ValidationException(str(exc)) from exc


@get("/api/source")
async def source_content_route(
    source: str, raw: bool = False
) -> SourceContentResponse | Response[bytes]:
    """Return stored source file as JSON (``raw=0``) or raw bytes (``raw=1``)."""
    try:
        result = await handlers.get_source_content(source, raw=raw)
    except FileNotFoundError as exc:
        raise NotFoundException(f"source not found: {source}") from exc
    except ValueError as exc:
        raise ValidationException(str(exc)) from exc

    # ``raw=True`` returns ``(bytes, content_type)``; narrow via ``isinstance``
    # so mypy sees the tuple branch without leaning on ``type: ignore``.
    if isinstance(result, tuple):
        body, content_type = result
        # nosniff blocks browser MIME-sniffing fallbacks; attachment forces a
        # download for any type the handler degraded to octet-stream so
        # attacker-named files don't render inline anywhere.
        headers = {"X-Content-Type-Options": "nosniff"}
        if content_type == "application/octet-stream":
            headers["Content-Disposition"] = f'attachment; filename="{Path(source).name}"'
        return Response(content=body, media_type=content_type, status_code=200, headers=headers)
    return result
