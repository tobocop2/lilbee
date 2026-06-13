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

from lilbee.app.services import get_services
from lilbee.app.status import gather_status
from lilbee.app.version import get_version
from lilbee.providers.roles import WorkerRole
from lilbee.providers.warm_progress import WarmPhase, WarmProgress
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

# SSE event name carrying a WarmProgress snapshot on the warm stream.
_WARM_EVENT = "warm"
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
        chat_warm=provider.warm_progress(),
    )


async def warm_stream() -> AsyncGenerator[str, None]:
    """Stream chat-model cold-load progress as SSE until the engine is ready.

    A launcher subscribes to render granular warm feedback. Each ``warm`` event
    carries a :class:`WarmProgress` snapshot; the stream ends with ``[DONE]`` once
    the chat role is ready or has failed, or when the budget elapses (the caller
    proceeds either way, so a still-loading model just warms on its first call).
    When nothing is loading because the engine is already warm, a single ``ready``
    snapshot is emitted and the stream closes.
    """
    provider = get_services().provider
    deadline = time.monotonic() + _WARM_STREAM_TIMEOUT_S
    while time.monotonic() < deadline:
        snapshot = provider.warm_progress()
        if snapshot is None:
            if provider.role_ready(WorkerRole.CHAT):
                yield sse_event(_WARM_EVENT, WarmProgress(phase=WarmPhase.READY).model_dump())
                break
            yield sse_event(_WARM_EVENT, WarmProgress(phase=WarmPhase.STARTING).model_dump())
        else:
            yield sse_event(_WARM_EVENT, snapshot.model_dump())
            if snapshot.phase in (WarmPhase.READY, WarmPhase.ERROR):
                break
        await asyncio.sleep(_WARM_POLL_INTERVAL_S)
    yield sse_done({})


async def status() -> StatusResponse:
    """Return config, sources, and chunk counts."""
    raw = gather_status()
    return StatusResponse(**raw.model_dump(exclude_none=True))


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
    "format_task_mismatch",
    "get_config",
    "get_config_defaults",
    "get_source_content",
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
