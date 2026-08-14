"""Framework-agnostic route handlers for the lilbee HTTP server.

Every public function is a plain async callable; no framework imports.
Return types are dicts (JSON responses), lists, or async generators of SSE strings.

Handlers are grouped by concern (sse, rag, models, ingest, config, documents,
crawl) under sibling submodules. The names re-exported below are the public
API consumed by ``server/routes/*.py``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from collections.abc import AsyncGenerator, Callable, Sequence
from typing import TYPE_CHECKING, Literal

from lilbee.app.services import get_services
from lilbee.app.status import gather_status
from lilbee.app.version import get_version
from lilbee.core.config import cfg
from lilbee.providers.roles import WorkerRole
from lilbee.providers.warm_progress import WarmPhase, WarmProgress
from lilbee.runtime.progress import SseEvent
from lilbee.server.handlers.agent_config import agent_config, agent_config_index
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
    add_files_stream,
    add_uploads_stream,
    import_stream,
    sync_stream,
    validate_add_paths,
    validate_upload_names,
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
from lilbee.server.handlers.wiki import (
    wiki_build_stream,
    wiki_generate_stream,
    wiki_synthesize_stream,
)
from lilbee.server.models import (
    GpusResponse,
    HealthResponse,
    PlacementResponse,
    ShutdownResponse,
    StatusResponse,
)

if TYPE_CHECKING:
    from lilbee.app.placement import GpuInfo, PlacementView
    from lilbee.providers.base import LLMProvider

log = logging.getLogger(__name__)

# How often the warm stream re-snapshots provider state; sub-second so the read
# bar advances smoothly without busy-spinning.
_WARM_POLL_INTERVAL_S = 0.25
# Upper bound on the warm stream; the launcher hands off when this elapses, so a
# model still loading past it just warms on the client's first call. Generous to
# cover a cold tensor-split giant off a slow filesystem.
_WARM_STREAM_TIMEOUT_S = 1800.0


def _chat_status(
    provider: LLMProvider,
) -> tuple[Literal["ready", "loading", "not_started", "error"], str | None]:
    """Classify the chat engine's readiness for /api/health, with the error reason.

    ``ready`` once the role serves; ``error`` when warm-up failed (paired with the
    warm tracker's reason); ``loading`` while a warm is in flight; ``not_started``
    when nothing is warming and the role isn't up (no chat model planned, or chat
    is swapped out for its co-tenant; the next chat request loads it).
    """
    if provider.role_ready(WorkerRole.CHAT):
        return "ready", None
    snapshot = provider.warm_progress()
    if snapshot is None:
        return "not_started", None
    if snapshot.phase is WarmPhase.ERROR:
        return "error", snapshot.error
    return "loading", None


async def health() -> HealthResponse:
    """Return service health, version, and whether the chat engine is warm."""
    provider = get_services().provider
    chat_status, chat_error = _chat_status(provider)
    return HealthResponse(
        status="ok",
        version=get_version(),
        chat_ready=provider.role_ready(WorkerRole.CHAT),
        chat_status=chat_status,
        chat_error=chat_error,
        chat_ctx=provider.served_chat_ctx(),
        chat_slots=provider.served_chat_slots(),
    )


async def shutdown() -> ShutdownResponse:
    """Accept an API-requested stop; the route's background task sends SIGTERM.

    Litestar runs that task only after the response has been handed to the
    transport, so the signal cannot beat the 202 out and no wall-clock delay
    has to be guessed. Routing through SIGTERM keeps the fleet teardown and
    shutdown logging identical however the stop arrives.
    """
    log.info("Shutdown requested via the API")
    return ShutdownResponse(status="shutting_down")


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
    devices: Sequence[GpuInfo],
    interval_s: float = _GPU_STATS_INTERVAL_S,
    max_ticks: int | None = None,
) -> AsyncGenerator[str, None]:
    """Stream live per-GPU utilization + free memory as SSE for the placement view.

    Devices are resolved by the caller before the stream starts so a ProviderError
    surfaces as a 503 at route time, not mid-stream. The client keeps the stream
    open while visible; ``max_ticks`` bounds it for tests. A heartbeat is emitted
    every ``cfg.sse_heartbeat_interval`` seconds of idle so clients don't time out.

    The per-vendor probe runs on a worker thread, not here. It is not light: every
    backend shells out to an SMI tool with a five-second timeout, and the Intel
    paths sleep and scan /proc on top of that. Driven inline it held the event
    loop for the whole subprocess on every tick, once per connected client, which
    stalls chat, search and embedding requests along with it.
    """
    from lilbee.cli.tui import messages as msg
    from lilbee.providers.fleet.gpu_stats import intel_util_hint, probe_gpu_stats_shared

    last_heartbeat = time.monotonic()
    tick = 0
    while max_ticks is None or tick < max_ticks:
        stats = await asyncio.to_thread(probe_gpu_stats_shared, devices)
        payload: dict[str, object] = {"gpus": [dataclasses.asdict(s) for s in stats.values()]}
        hint = intel_util_hint(devices, stats)
        if hint:
            payload["notice"] = msg.intel_util_hint_text(hint)
        yield sse_event(SseEvent.GPU_STATS, payload)
        tick += 1
        if max_ticks is None or tick < max_ticks:
            await asyncio.sleep(interval_s)
            now = time.monotonic()
            heartbeat_interval = cfg.sse_heartbeat_interval
            if heartbeat_interval > 0 and now - last_heartbeat >= heartbeat_interval:
                last_heartbeat = now
                yield sse_event(SseEvent.HEARTBEAT, {"ts": time.time()})


async def status() -> StatusResponse:
    """Return config, sources, and chunk counts."""
    raw = gather_status()
    return StatusResponse(**raw.model_dump(exclude_none=True))


async def placement() -> PlacementResponse:
    """Current effective placement."""
    from lilbee.app.placement import get_placement

    return await _placement_response_off_loop(get_placement)


async def placement_preview(spec_json: str | None) -> PlacementResponse:
    """Preview a candidate spec (or auto when no spec). No persistence."""
    from lilbee.app.placement import preview_placement
    from lilbee.providers.fleet.placement_spec import PlacementSpec

    spec = PlacementSpec.from_json(spec_json) if spec_json else None
    return await _placement_response_off_loop(lambda: preview_placement(spec))


async def placement_set(spec_json: str) -> PlacementResponse:
    """Apply a manual placement spec; persists and rebuilds the fleet."""
    from lilbee.app.placement import set_placement
    from lilbee.providers.fleet.placement_spec import PlacementSpec

    spec = PlacementSpec.from_json(spec_json)
    return await _placement_response_off_loop(lambda: set_placement(spec))


async def placement_clear() -> PlacementResponse:
    """Clear manual placement; returns to the auto planner and rebuilds the fleet."""
    from lilbee.app.placement import set_placement

    return await _placement_response_off_loop(lambda: set_placement(None))


async def _placement_response_off_loop(action: Callable[[], PlacementView]) -> PlacementResponse:
    """Run a placement action and serialize it off the event loop.

    Placement actions and the Intel util notice both shell out to GPU probes,
    so neither may run on the loop.
    """
    return await asyncio.to_thread(lambda: _placement_response(action()))


def _placement_response(view: PlacementView) -> PlacementResponse:
    """Serialize a placement view with the host-level Intel util notice attached."""
    resp = PlacementResponse.from_view(view)
    resp.notice = _intel_notice_text(view.gpus)
    return resp


def _intel_notice_text(devices: Sequence[GpuInfo]) -> str | None:
    """Formatted Intel util fix for the JSON surfaces, or None when util reads fine."""
    from lilbee.cli.tui import messages as msg
    from lilbee.providers.fleet.gpu_stats import probe_intel_util_hint

    hint = probe_intel_util_hint(devices)
    return msg.intel_util_hint_text(hint) if hint else None


async def gpus() -> GpusResponse:
    """Detected GPUs with free/total VRAM, plus the host-level Intel util notice."""
    from lilbee.app.placement import get_placement

    def _body() -> GpusResponse:
        view = get_placement()
        return GpusResponse(
            gpus=PlacementResponse.from_view(view).gpus,
            notice=_intel_notice_text(view.gpus),
        )

    return await asyncio.to_thread(_body)


__all__ = [
    "TASK_ENDPOINT_PATH",
    "ModelCatalogSection",
    "ModelsResponse",
    "SseStream",
    "add_files_stream",
    "add_uploads_stream",
    "agent_config",
    "agent_config_index",
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
    "validate_upload_names",
    "warm_stream",
    "wiki_build_stream",
    "wiki_generate_stream",
    "wiki_synthesize_stream",
]
