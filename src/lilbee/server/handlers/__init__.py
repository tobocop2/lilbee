"""Framework-agnostic route handlers for the lilbee HTTP server.

Every public function is a plain async callable; no framework imports.
Return types are dicts (JSON responses), lists, or async generators of SSE strings.

This subpackage groups handlers by concern (sse, rag, models, ingest, config,
documents, crawl). The names re-exported below are the public API consumed by
``server/routes/*.py`` and the tests; importing from
``lilbee.server.handlers`` continues to work unchanged.
"""

from __future__ import annotations

# ``time`` is patched by the external-models cache tests; expose it at the
# package top-level so the patch path ``lilbee.server.handlers.time`` still
# resolves to the same module object the submodule uses.
import time

from lilbee.cli.helpers import gather_status, get_version

# Re-export the ``settings`` and ``time`` modules under the package namespace so
# legacy ``mock.patch("lilbee.server.handlers.settings.set_value")`` and
# ``mock.patch("lilbee.server.handlers.time")`` patches keep working.
from lilbee.core import settings

# ``get_services`` is patched by tests at the legacy
# ``lilbee.server.handlers.get_services`` path for the few tests that target
# the package-level binding instead of a submodule.
from lilbee.core.services import get_services
from lilbee.modelhub.model_manager import get_model_manager
from lilbee.server.handlers import models as _models_module
from lilbee.server.handlers.config import (
    _apply_config_updates,
    _compute_config_defaults,
    _validate_config_updates,
    get_config,
    get_config_defaults,
    update_config,
)
from lilbee.server.handlers.crawl import crawl_stream
from lilbee.server.handlers.documents import (
    _RAW_INLINE_RENDER_DENY,
    _is_safe_for_inline_render,
    delete_documents,
    get_source_content,
    list_documents,
)
from lilbee.server.handlers.ingest import (
    _INGEST_LOCK_REGISTRY,
    _INGEST_LOCKS,
    MAX_ADD_FILES,
    _acquire_add_locks,
    _canonical_source_name,
    _get_registry_lock,
    _parse_ocr_params,
    _release_add_locks,
    _reset_ingest_locks,
    _run_add,
    _run_sync_with_sentinel,
    _try_acquire_source,
    add_files_stream,
    sync_stream,
    validate_add_paths,
)
from lilbee.server.handlers.models import (
    _TASK_TO_FIELD,
    TASK_ENDPOINT_PATH,
    ModelCatalogEntry,
    ModelCatalogSection,
    ModelsResponse,
    _build_task_to_field,
    _catalog_section,
    _external_cache,
    _ExternalModelsCache,
    _parse_source,
    _require_model_available,
    _require_model_for_task,
    _resolve_via_catalog,
    _resolve_via_parse,
    _set_model,
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
    _run_llm_stream,
    _stream_rag_response,
    ask,
    ask_stream,
    chat,
    chat_stream,
    search,
)
from lilbee.server.handlers.sse import (
    SseStream,
    _resolve_generation_options,
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
    "_INGEST_LOCKS",
    "_INGEST_LOCK_REGISTRY",
    "_RAW_INLINE_RENDER_DENY",
    "_TASK_TO_FIELD",
    "ModelCatalogEntry",
    "ModelCatalogSection",
    "ModelsResponse",
    "SseStream",
    "_ExternalModelsCache",
    "_acquire_add_locks",
    "_apply_config_updates",
    "_build_task_to_field",
    "_canonical_source_name",
    "_catalog_section",
    "_compute_config_defaults",
    "_external_cache",
    "_get_registry_lock",
    "_is_safe_for_inline_render",
    "_models_module",
    "_parse_ocr_params",
    "_parse_source",
    "_release_add_locks",
    "_require_model_available",
    "_require_model_for_task",
    "_reset_ingest_locks",
    "_resolve_generation_options",
    "_resolve_via_catalog",
    "_resolve_via_parse",
    "_run_add",
    "_run_llm_stream",
    "_run_sync_with_sentinel",
    "_set_model",
    "_stream_rag_response",
    "_try_acquire_source",
    "_validate_config_updates",
    "add_files_stream",
    "ask",
    "ask_stream",
    "chat",
    "chat_stream",
    "crawl_stream",
    "delete_documents",
    "format_task_mismatch",
    "gather_status",
    "get_config",
    "get_config_defaults",
    "get_model_manager",
    "get_services",
    "get_source_content",
    "get_version",
    "health",
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
    "settings",
    "sse_done",
    "sse_error",
    "sse_event",
    "status",
    "sync_stream",
    "time",
    "update_config",
    "validate_add_paths",
]
