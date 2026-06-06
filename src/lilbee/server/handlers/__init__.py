"""Framework-agnostic route handlers for the lilbee HTTP server.

Every public function is a plain async callable; no framework imports.
Return types are dicts (JSON responses), lists, or async generators of SSE strings.

Handlers are grouped by concern (sse, rag, models, ingest, config, documents,
crawl) under sibling submodules. The names re-exported below are the public
API consumed by ``server/routes/*.py``.
"""

from __future__ import annotations

from lilbee.app.status import gather_status
from lilbee.app.version import get_version
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


async def health() -> HealthResponse:
    """Return service health and version."""
    return HealthResponse(status="ok", version=get_version())


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
]
