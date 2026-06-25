"""Placement routes: inspect and preview GPU placement.

Mutation (set/clear) is intentionally not served here. Applying a placement
rebuilds the shared fleet, which is unsafe on the always-concurrent HTTP
daemon, so PUT/DELETE are refused and placement is changed from the CLI or TUI.
"""

from __future__ import annotations

import json

from litestar import delete, get, post, put
from litestar.exceptions import HTTPException

from lilbee.app.settings import provider_reset_refused_message
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet.placement_spec import PlacementError
from lilbee.server import handlers
from lilbee.server.auth import read_only
from lilbee.server.models import GpuInfoResponse, PlacementResponse, PlacementSpecBody

_HTTP_UNPROCESSABLE = 422
_HTTP_CONFLICT = 409
_HTTP_UNAVAILABLE = 503
_INPUT_ERRORS = (PlacementError, ValueError, OSError)


def _spec_json(body: PlacementSpecBody) -> str | None:
    """Serialize the optional spec dict to JSON, or return None when absent."""
    return json.dumps(body.spec) if body.spec is not None else None


def _refused() -> HTTPException:
    return HTTPException(
        status_code=_HTTP_CONFLICT, detail=provider_reset_refused_message("Changing placement on")
    )


@get("/api/placement")
@read_only
async def placement_route() -> PlacementResponse:
    """Current effective placement."""
    try:
        return await handlers.placement()
    except ProviderError as exc:
        raise HTTPException(status_code=_HTTP_UNAVAILABLE, detail=str(exc)) from exc


@post("/api/placement/preview", status_code=200)
async def placement_preview_route(data: PlacementSpecBody) -> PlacementResponse:
    """Preview a candidate spec (or auto when no spec). Requires auth: runs subprocess probes."""
    try:
        return await handlers.placement_preview(_spec_json(data))
    except ProviderError as exc:
        raise HTTPException(status_code=_HTTP_UNAVAILABLE, detail=str(exc)) from exc
    except _INPUT_ERRORS as exc:
        raise HTTPException(status_code=_HTTP_UNPROCESSABLE, detail=str(exc)) from exc


@put("/api/placement")
async def placement_set_route() -> PlacementResponse:
    """Refused on the HTTP daemon: applying placement rebuilds the shared fleet."""
    raise _refused()


@delete("/api/placement")
async def placement_clear_route() -> None:
    """Refused on the HTTP daemon: clearing placement rebuilds the shared fleet."""
    raise _refused()


@get("/api/gpus")
@read_only
async def gpus_route() -> list[GpuInfoResponse]:
    """Detected GPUs with free/total VRAM."""
    try:
        return await handlers.gpus()
    except ProviderError as exc:
        raise HTTPException(status_code=_HTTP_UNAVAILABLE, detail=str(exc)) from exc
