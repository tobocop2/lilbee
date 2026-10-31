"""Server preset routes: list, add, edit and remove. Every rule lives in the preset service."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TypeVar

from litestar import delete, get, post, put
from litestar.connection import ASGIConnection
from litestar.exceptions import HTTPException
from litestar.handlers.base import BaseRouteHandler

import lilbee.app.presets as service
from lilbee.core.config import cfg
from lilbee.server.models import (
    PresetAddBody,
    PresetEditBody,
    PresetListResponse,
    PresetRemovedResponse,
    PresetResponse,
)

_T = TypeVar("_T")
_HTTP_NOT_FOUND = 404
_HTTP_CONFLICT = 409
_HTTP_UNPROCESSABLE = 422


async def _call(action: Callable[[], _T]) -> _T:
    """Run a service call off the loop; a refusal becomes the status the error kind names."""
    try:
        return await asyncio.to_thread(action)
    except service.PresetNotFoundError as exc:
        raise HTTPException(status_code=_HTTP_NOT_FOUND, detail=service.failure_text(exc)) from exc
    except service.PresetExistsError as exc:
        raise HTTPException(status_code=_HTTP_CONFLICT, detail=service.failure_text(exc)) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(
            status_code=_HTTP_UNPROCESSABLE, detail=service.failure_text(exc)
        ) from exc


def _require_engines_enabled(_connection: ASGIConnection, _handler: BaseRouteHandler) -> None:
    """Refuse a write while allow_http_engines is off. A guard runs before the body is parsed."""
    if not cfg.allow_http_engines:
        raise HTTPException(status_code=_HTTP_CONFLICT, detail=service.engines_refused_message())


@get("/api/presets")
async def presets_list_route() -> PresetListResponse:
    """The stored server presets. A key shows as has_key and never as a value."""
    presets = await _call(service.list_presets)
    return PresetListResponse(presets=[PresetResponse.from_view(p) for p in presets])


@post("/api/presets", status_code=201, guards=[_require_engines_enabled])
async def presets_add_route(data: PresetAddBody) -> PresetResponse:
    """Add a server preset. Refused unless allow_http_engines is enabled."""
    preset = await _call(
        lambda: service.add_preset(
            data.name,
            data.url,
            roles=data.roles,
            api_key=data.api_key,
            rerank_wire=data.rerank_wire,
        )
    )
    return PresetResponse.from_view(preset)


@put("/api/presets/{name:str}", status_code=200, guards=[_require_engines_enabled])
async def presets_edit_route(name: str, data: PresetEditBody) -> PresetResponse:
    """Change a server preset. Refused unless allow_http_engines is enabled."""
    preset = await _call(
        lambda: service.edit_preset(
            name,
            url=data.url,
            roles=data.roles,
            api_key=data.api_key,
            rerank_wire=data.rerank_wire,
            clear_key=data.clear_key,
        )
    )
    return PresetResponse.from_view(preset)


@delete("/api/presets/{name:str}", status_code=200, guards=[_require_engines_enabled])
async def presets_remove_route(name: str) -> PresetRemovedResponse:
    """Remove a server preset. Refused unless allow_http_engines is enabled."""
    await _call(lambda: service.remove_preset(name))
    return PresetRemovedResponse(removed=name)
