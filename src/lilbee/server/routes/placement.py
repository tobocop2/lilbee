"""Placement routes: inspect, preview, set, and clear GPU placement."""

from __future__ import annotations

import json

from litestar import delete, get, post, put
from litestar.exceptions import HTTPException

from lilbee.providers.fleet.placement_spec import PlacementError
from lilbee.server import handlers
from lilbee.server.auth import read_only
from lilbee.server.models import GpuInfoResponse, PlacementResponse, PlacementSpecBody


def _spec_json(body: PlacementSpecBody) -> str | None:
    """Serialize the optional spec dict to JSON, or return None when absent."""
    return json.dumps(body.spec) if body.spec is not None else None


@get("/api/placement")
@read_only
async def placement_route() -> PlacementResponse:
    """Current effective placement."""
    return await handlers.placement()


@post("/api/placement/preview", status_code=200)
@read_only
async def placement_preview_route(data: PlacementSpecBody) -> PlacementResponse:
    """Preview a candidate spec (or auto when no spec)."""
    try:
        return await handlers.placement_preview(_spec_json(data))
    except PlacementError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@put("/api/placement")
async def placement_set_route(data: PlacementSpecBody) -> PlacementResponse:
    """Validate, persist, and apply a manual placement spec."""
    if data.spec is None:
        raise HTTPException(status_code=422, detail="spec is required")
    try:
        return await handlers.placement_set(json.dumps(data.spec))
    except PlacementError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@delete("/api/placement", status_code=200)
async def placement_clear_route() -> PlacementResponse:
    """Clear the manual placement, returning to auto."""
    return await handlers.placement_clear()


@get("/api/gpus")
@read_only
async def gpus_route() -> list[GpuInfoResponse]:
    """Detected GPUs with free/total VRAM."""
    return await handlers.gpus()
