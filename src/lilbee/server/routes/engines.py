"""Engine choice routes: read the engine of each role and write the choice."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TypeVar

from litestar import get, post, put
from litestar.connection import ASGIConnection
from litestar.exceptions import HTTPException
from litestar.handlers.base import BaseRouteHandler
from litestar.params import FromPath

import lilbee.app.engine_service as service
from lilbee.core.config import cfg
from lilbee.server.models import (
    EngineChosenResponse,
    EngineInstallBody,
    EngineInstalledResponse,
    EnginePreferBody,
    EngineRoleResponse,
    EngineRolesResponse,
    EngineUseBody,
)

_T = TypeVar("_T")
_HTTP_BAD_REQUEST = 400
_HTTP_CONFLICT = 409
_HTTP_UNPROCESSABLE = 422
_HTTP_UNAVAILABLE = 503


async def _call(action: Callable[[], _T]) -> _T:
    """Run a service call off the loop; a refusal is a 422 and a failed save a 503."""
    try:
        return await asyncio.to_thread(action)
    except service.EngineRefusedError as exc:
        raise HTTPException(status_code=_HTTP_UNPROCESSABLE, detail=str(exc)) from exc
    except service.EngineNotSavedError as exc:
        raise HTTPException(status_code=_HTTP_UNAVAILABLE, detail=str(exc)) from exc


def _require_engines_enabled(_connection: ASGIConnection, _handler: BaseRouteHandler) -> None:
    """Refuse a write while allow_http_engines is off. A guard runs before the body is parsed."""
    if not cfg.allow_http_engines:
        raise HTTPException(status_code=_HTTP_CONFLICT, detail=service.http_refused_message())


@get("/api/engines/roles")
async def engine_roles_route() -> EngineRolesResponse:
    """The engine of every role, with the notice for a role that fell back or has none."""
    views = await _call(service.roles)
    return EngineRolesResponse(
        roles=[
            EngineRoleResponse(
                role=v.role, model=v.model, format=v.format, engine=v.engine, notice=v.notice
            )
            for v in views
        ]
    )


@put("/api/engines/roles/{role:str}", status_code=200, guards=[_require_engines_enabled])
async def engine_use_route(role: FromPath[str], data: EngineUseBody) -> EngineChosenResponse:
    """Choose the engine for the model of one role. Refused unless allow_http_engines is enabled."""
    try:
        parsed = service.parse_role(role)
    except service.EngineRefusedError as exc:
        raise HTTPException(status_code=_HTTP_BAD_REQUEST, detail=str(exc)) from exc
    chosen = await _call(lambda: service.use_for_role(parsed, data.model, data.engine))
    return EngineChosenResponse(engine=chosen)


@post("/api/engines/install", status_code=200, guards=[_require_engines_enabled])
async def engine_install_route(data: EngineInstallBody) -> EngineInstalledResponse:
    """Install an engine build from its pack. Refused unless allow_http_engines is enabled."""
    built = await _call(lambda: service.install(data.engine))
    return EngineInstalledResponse(
        engine=built.engine, version=built.version, backend=built.backend
    )


@put("/api/engines/preferred", status_code=200, guards=[_require_engines_enabled])
async def engine_prefer_route(data: EnginePreferBody) -> EngineChosenResponse:
    """Set the global engine preference. Refused unless allow_http_engines is enabled."""
    chosen = await _call(lambda: service.prefer(data.engine))
    return EngineChosenResponse(engine=chosen)
