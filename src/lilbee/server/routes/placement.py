"""Placement routes: inspect, preview, and (when enabled) apply GPU placement.

Every route requires auth: even the reads run resolve_placement_plan, which
spawns subprocess device probes. Every route needs the token. Applying or
clearing placement restarts the shared fleet's moved roles, which is unsafe
across concurrent HTTP clients, so PUT/DELETE are refused by default. They are gated on the
``allow_http_placement`` flag (LILBEE_ALLOW_HTTP_PLACEMENT), which an operator
turns on for a single-client / owned deployment to get the same apply/clear
capability the CLI and TUI have.
"""

from __future__ import annotations

import asyncio
import json
import logging

from litestar import delete, get, post, put
from litestar.exceptions import HTTPException
from litestar.response import Stream

from lilbee.core.config import cfg
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet.placement_spec import PlacementError
from lilbee.server import handlers
from lilbee.server.models import GpusResponse, PlacementResponse, PlacementSpecBody

log = logging.getLogger(__name__)

_HTTP_UNPROCESSABLE = 422
_HTTP_CONFLICT = 409
_HTTP_UNAVAILABLE = 503
# OSError is deliberately absent: on this path it means the device probe could
# not run (no nvidia-smi on PATH, no permission on the device node, a failed
# spawn), which is a host fault and not something the caller's spec can fix.
_INPUT_ERRORS = (PlacementError, ValueError)
_PROBE_FAILED_DETAIL = (
    "Could not probe the GPUs. Check that the vendor tool (nvidia-smi or "
    "rocm-smi) is installed and this process may read the device."
)
_MISSING_SPEC_DETAIL = "spec is required to apply placement; send {} to DELETE for auto."


def _spec_json(body: PlacementSpecBody) -> str | None:
    """Serialize the optional spec dict to JSON, or return None when absent."""
    return json.dumps(body.spec) if body.spec is not None else None


def _refused() -> HTTPException:
    from lilbee.app.placement import placement_refused_message

    return HTTPException(status_code=_HTTP_CONFLICT, detail=placement_refused_message())


@get("/api/placement")
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
    except OSError as exc:
        log.exception("GPU probe failed")
        raise HTTPException(status_code=_HTTP_UNAVAILABLE, detail=_PROBE_FAILED_DETAIL) from exc
    except _INPUT_ERRORS as exc:
        raise HTTPException(status_code=_HTTP_UNPROCESSABLE, detail=str(exc)) from exc


@put("/api/placement")
async def placement_set_route(data: PlacementSpecBody) -> PlacementResponse:
    """Apply a manual spec. Refused unless allow_http_placement is enabled."""
    if not cfg.allow_http_placement:
        raise _refused()
    spec_json = _spec_json(data)
    if spec_json is None:
        raise HTTPException(status_code=_HTTP_UNPROCESSABLE, detail=_MISSING_SPEC_DETAIL)
    try:
        return await handlers.placement_set(spec_json)
    except ProviderError as exc:
        raise HTTPException(status_code=_HTTP_UNAVAILABLE, detail=str(exc)) from exc
    except OSError as exc:
        log.exception("GPU probe failed")
        raise HTTPException(status_code=_HTTP_UNAVAILABLE, detail=_PROBE_FAILED_DETAIL) from exc
    except _INPUT_ERRORS as exc:
        raise HTTPException(status_code=_HTTP_UNPROCESSABLE, detail=str(exc)) from exc


@delete("/api/placement", status_code=200)
async def placement_clear_route() -> PlacementResponse:
    """Clear placement (back to auto). Refused unless allow_http_placement is enabled."""
    if not cfg.allow_http_placement:
        raise _refused()
    try:
        return await handlers.placement_clear()
    except ProviderError as exc:
        raise HTTPException(status_code=_HTTP_UNAVAILABLE, detail=str(exc)) from exc


@get("/api/gpus")
async def gpus_route() -> GpusResponse:
    """Detected GPUs with free/total VRAM, plus the host-level Intel util notice."""
    try:
        return await handlers.gpus()
    except ProviderError as exc:
        raise HTTPException(status_code=_HTTP_UNAVAILABLE, detail=str(exc)) from exc


@get("/api/gpus/stream")
async def gpu_stats_stream_route() -> Stream:
    """Live per-GPU utilization + free memory as SSE for the placement view."""
    from lilbee.app.placement import get_placement

    try:
        # get_placement runs resolve_placement_plan, which spawns subprocess
        # device probes that can wedge for the probe timeout on a broken driver;
        # offload so the probe never blocks the event loop.
        view = await asyncio.to_thread(get_placement)
    except ProviderError as exc:
        raise HTTPException(status_code=_HTTP_UNAVAILABLE, detail=str(exc)) from exc
    return Stream(handlers.gpu_stats_stream(view.gpus), media_type="text/event-stream")
