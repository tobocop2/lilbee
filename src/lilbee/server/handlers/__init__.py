"""Framework-agnostic route handlers for the lilbee HTTP server.

Every public function is a plain async callable; no framework imports.
Return types are dicts (JSON responses), lists, or async generators of SSE strings.

Handlers are grouped by concern (sse, rag, models, ingest, config, documents,
crawl) under sibling submodules. The names re-exported below are the public
API consumed by ``server/routes/*.py``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from lilbee.app.services import get_services
from lilbee.app.status import gather_status
from lilbee.app.version import get_version
from lilbee.providers.roles import WorkerRole
from lilbee.providers.warm_progress import WarmPhase, WarmProgress
from lilbee.runtime.progress import SseEvent
from lilbee.server.handlers.config import (
    get_config,
    get_config_defaults,
    update_config,
)
from lilbee.server.handlers.crawl import crawl_stream
from lilbee.server.handlers.documents import (
    delete_documents,
    get_source_content,
    list_documents,
)
from lilbee.server.handlers.ingest import (
    MAX_ADD_FILES,
    add_files_stream,
    import_stream,
    sync_stream,
    validate_add_paths,
)
from lilbee.server.handlers.models import (
    TASK_ENDPOINT_PATH,
    ModelCatalogSection,
    ModelsResponse,
    enforce_pull_arch_compat,
    format_task_mismatch,
    list_external_models,
    list_models,
    models_catalog,
    models_delete,
    models_installed,
    models_pull,
    models_show,
    set_chat_model,
    set_embedding_model,
    set_reranker_model,
    set_vision_model,
)
from lilbee.server.handlers.rag import (
    ask,
    ask_stream,
    chat,
    chat_stream,
    search,
)
from lilbee.server.handlers.sse import (
    SseStream,
    classify_load_error,
    sse_done,
    sse_error,
    sse_event,
)
from lilbee.server.models import HealthResponse, StatusResponse

if TYPE_CHECKING:
    from lilbee.app.placement import PlacementView
    from lilbee.server.models import GpuInfoResponse, PlacementResponse

# How often the warm stream re-snapshots provider state; sub-second so the read
# bar advances smoothly without busy-spinning.
_WARM_POLL_INTERVAL_S = 0.25
# Upper bound on the warm stream; the launcher hands off when this elapses, so a
# model still loading past it just warms on the client's first call. Generous to
# cover a cold tensor-split giant off a slow filesystem.
_WARM_STREAM_TIMEOUT_S = 1800.0


async def health() -> HealthResponse:
    """Return service health, version, and whether the chat engine is warm."""
    provider = get_services().provider
    return HealthResponse(
        status="ok",
        version=get_version(),
        chat_ready=provider.role_ready(WorkerRole.CHAT),
        chat_ctx=provider.served_chat_ctx(),
    )


async def warm_stream() -> AsyncGenerator[str, None]:
    """Stream chat-model cold-load progress as SSE until the engine is ready.

    A launcher subscribes to render granular warm feedback. Each
    :data:`SseEvent.WARM` event carries a :class:`WarmProgress` snapshot; a
    terminal :data:`SseEvent.DONE` closes the stream once the chat role is ready
    or has failed, or when the budget elapses (the caller proceeds either way, so
    a still-loading model just warms on its first call). When nothing is loading
    because the engine is already warm, a single ready snapshot is emitted.
    """
    provider = get_services().provider
    deadline = time.monotonic() + _WARM_STREAM_TIMEOUT_S
    while time.monotonic() < deadline:
        snapshot = provider.warm_progress()
        if snapshot is None:
            if provider.role_ready(WorkerRole.CHAT):
                yield sse_event(SseEvent.WARM, WarmProgress(phase=WarmPhase.READY).model_dump())
                break
            yield sse_event(SseEvent.WARM, WarmProgress(phase=WarmPhase.STARTING).model_dump())
        else:
            yield sse_event(SseEvent.WARM, snapshot.model_dump())
            if snapshot.phase in (WarmPhase.READY, WarmPhase.ERROR):
                break
        await asyncio.sleep(_WARM_POLL_INTERVAL_S)
    yield sse_done({})


_GPU_STATS_INTERVAL_S = 1.0


async def gpu_stats_stream(
    interval_s: float = _GPU_STATS_INTERVAL_S,
    max_ticks: int | None = None,
) -> AsyncGenerator[str, None]:
    """Stream live per-GPU utilization + free memory as SSE for the placement view.

    Structural devices are probed once (a subprocess); each tick only runs the
    light ``nvidia-smi`` query, so the loop is cheap. The client (placement view)
    keeps the stream open while visible; ``max_ticks`` bounds it for tests.
    """
    from lilbee.app.placement import get_placement
    from lilbee.providers.fleet.gpu_stats import probe_gpu_stats
    from lilbee.server.models import GpuStatEvent

    devices = get_placement().gpus
    tick = 0
    while max_ticks is None or tick < max_ticks:
        stats = probe_gpu_stats(devices)
        payload = {
            "gpus": [
                GpuStatEvent(
                    index=s.index,
                    utilization_pct=s.utilization_pct,
                    free_bytes=s.free_bytes,
                    total_bytes=s.total_bytes,
                ).model_dump()
                for s in stats.values()
            ]
        }
        yield sse_event(SseEvent.GPU_STATS, payload)
        tick += 1
        if max_ticks is None or tick < max_ticks:
            await asyncio.sleep(interval_s)


async def status() -> StatusResponse:
    """Return config, sources, and chunk counts."""
    raw = gather_status()
    return StatusResponse(**raw.model_dump(exclude_none=True))


def _placement_response(view: PlacementView) -> PlacementResponse:
    """Map a PlacementView to a PlacementResponse."""
    from lilbee.server.models import GpuInfoResponse, PlacementResponse, RolePlacementResponse

    return PlacementResponse(
        gpus=[GpuInfoResponse(**vars(g)) for g in view.gpus],
        roles=[
            RolePlacementResponse(
                role=r.role,
                model=r.model,
                devices=list(r.devices),
                tensor_split=list(r.tensor_split) if r.tensor_split else None,
                replicas=r.replicas,
            )
            for r in view.roles
        ],
        unplaceable=[r.value for r in view.unplaceable],
        manual=view.manual,
        spec_json=view.spec_json,
    )


async def placement() -> PlacementResponse:
    """Current effective placement."""
    from lilbee.app.placement import get_placement

    return _placement_response(get_placement())


async def placement_preview(spec_json: str | None) -> PlacementResponse:
    """Preview a candidate spec (or auto when spec_json is None). No persistence."""
    from lilbee.app.placement import preview_placement
    from lilbee.providers.fleet.placement_spec import PlacementSpec

    spec = PlacementSpec.from_json(spec_json) if spec_json else None
    return _placement_response(preview_placement(spec))


async def placement_set(spec_json: str) -> PlacementResponse:
    """Apply a manual placement spec; persists and rebuilds the fleet."""
    from lilbee.app.placement import set_placement
    from lilbee.providers.fleet.placement_spec import PlacementSpec

    return _placement_response(set_placement(PlacementSpec.from_json(spec_json)))


async def placement_clear() -> PlacementResponse:
    """Clear manual placement; returns to the auto planner and rebuilds the fleet."""
    from lilbee.app.placement import set_placement

    return _placement_response(set_placement(None))


async def gpus() -> list[GpuInfoResponse]:
    """Detected GPUs with free/total VRAM."""
    from lilbee.app.placement import get_placement
    from lilbee.server.models import GpuInfoResponse as _GpuInfoResponse

    return [_GpuInfoResponse(**vars(g)) for g in get_placement().gpus]


__all__ = [
    "MAX_ADD_FILES",
    "TASK_ENDPOINT_PATH",
    "ModelCatalogSection",
    "ModelsResponse",
    "SseStream",
    "add_files_stream",
    "ask",
    "ask_stream",
    "chat",
    "chat_stream",
    "classify_load_error",
    "crawl_stream",
    "delete_documents",
    "enforce_pull_arch_compat",
    "format_task_mismatch",
    "get_config",
    "get_config_defaults",
    "get_source_content",
    "gpu_stats_stream",
    "gpus",
    "health",
    "import_stream",
    "list_documents",
    "list_external_models",
    "list_models",
    "models_catalog",
    "models_delete",
    "models_installed",
    "models_pull",
    "models_show",
    "placement",
    "placement_clear",
    "placement_preview",
    "placement_set",
    "search",
    "set_chat_model",
    "set_embedding_model",
    "set_reranker_model",
    "set_vision_model",
    "sse_done",
    "sse_error",
    "sse_event",
    "status",
    "sync_stream",
    "update_config",
    "validate_add_paths",
    "warm_stream",
]
