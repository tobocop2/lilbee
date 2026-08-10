"""Litestar application factory: imports routes from modules, creates app with lifespan."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import anyio.to_thread
from litestar import Litestar, MediaType, Request, Response
from litestar.config.cors import CORSConfig
from litestar.exceptions import HTTPException
from litestar.exceptions.responses import create_exception_response
from litestar.middleware.base import DefineMiddleware
from litestar.openapi import OpenAPIConfig

from lilbee.app.services import get_services, peek_services
from lilbee.app.version import get_version
from lilbee.core.config import cfg
from lilbee.providers.sdk_llm_provider import inject_provider_keys
from lilbee.server.auth import AuthMiddleware, session_manager
from lilbee.server.chat_completions_api.routes import completions_router
from lilbee.server.mcp_mount import build_mcp_mount
from lilbee.server.routes.crawl import crawl_route
from lilbee.server.routes.documents import (
    add_route,
    add_upload_route,
    documents_list_route,
    documents_remove_route,
    export_route,
    import_route,
    sync_route,
)
from lilbee.server.routes.general import (
    config_defaults_route,
    config_route,
    config_update_route,
    health_route,
    shutdown_route,
    source_content_route,
    status_route,
    warm_stream_route,
)
from lilbee.server.routes.memory import (
    memories_list_route,
    memories_remember_route,
    memories_remove_route,
    memories_update_route,
)
from lilbee.server.routes.models import (
    models_catalog_route,
    models_delete_route,
    models_external_route,
    models_installed_route,
    models_list_route,
    models_pull_route,
    models_set_chat_route,
    models_set_embedding_route,
    models_set_reranker_route,
    models_set_vision_route,
    models_show_route,
)
from lilbee.server.routes.placement import (
    gpu_stats_stream_route,
    gpus_route,
    placement_clear_route,
    placement_preview_route,
    placement_route,
    placement_set_route,
)
from lilbee.server.routes.search import (
    ask_route,
    ask_stream_route,
    chat_route,
    chat_stream_route,
    search_route,
)
from lilbee.server.routes.sessions import (
    session_add_message_route,
    session_claim_route,
    session_create_route,
    session_delete_route,
    session_get_route,
    session_rename_route,
    session_set_summary_route,
    sessions_list_route,
)
from lilbee.server.routes.setup import (
    setup_crawler_route,
    setup_crawler_status_route,
)
from lilbee.server.wiki import (
    wiki_build_route,
    wiki_citations_reverse_route,
    wiki_draft_accept_route,
    wiki_draft_diff_route,
    wiki_draft_reject_route,
    wiki_drafts_route,
    wiki_generate_route,
    wiki_index_route,
    wiki_lint_route,
    wiki_list_route,
    wiki_prune_route,
    wiki_read_route,
    wiki_status_route,
    wiki_synthesize_route,
    wiki_update_route,
    wiki_wipe_route,
)

if TYPE_CHECKING:
    from lilbee.retrieval.embedder import Embedder

log = logging.getLogger(__name__)


# Below this soft open-file limit a large agent fleet meets the limit as
# connection failures before it saturates the machine.
_FD_SOFT_LIMIT_NUDGE = 4096


def _warn_if_few_file_descriptors() -> None:
    """Log an advisory when the open-file limit is low enough to cap the agent fleet."""
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows has no resource module
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft == resource.RLIM_INFINITY or soft >= _FD_SOFT_LIMIT_NUDGE:
        return
    log.info(
        "Open-file limit is %d. Each connected agent holds a socket, so a large "
        "fleet will hit this before it saturates the machine; raise it with "
        "'ulimit -n %d' before starting the server (this shell allows up to %s).",
        soft,
        _FD_SOFT_LIMIT_NUDGE,
        "unlimited" if hard == resource.RLIM_INFINITY else hard,
    )


def _raise_thread_pool_ceiling() -> None:
    """Set anyio's shared thread-pool size to ``mcp_tool_threads``.

    Resizes anyio's own default limiter, not a private one, so every offload in
    the process is lifted (Litestar and MCP sync handlers included), not only ours.
    Must run on the server event loop, where the default limiter lives.
    """
    limiter = anyio.to_thread.current_default_thread_limiter()
    if limiter.total_tokens != cfg.mcp_tool_threads:
        limiter.total_tokens = cfg.mcp_tool_threads


class _ServerLoop:
    """Holds the running server's event loop so a config change off the loop can reach it.

    A settings write from a worker thread (the MCP handler) needs the loop to
    resize its thread pool there; nothing is held when no server runs (CLI/TUI).
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None

    def set(self, loop: asyncio.AbstractEventLoop | None) -> None:
        self._loop = loop

    def running(self) -> asyncio.AbstractEventLoop | None:
        loop = self._loop
        return loop if loop is not None and not loop.is_closed() else None


_server_loop = _ServerLoop()


def reapply_thread_pool_ceiling() -> None:
    """Resize the running server's thread pool after ``mcp_tool_threads`` changes.

    Marshalled onto the server loop, so it is safe from the HTTP handler (on the
    loop) and the MCP handler (a worker thread) alike. With no server running the
    new value simply takes effect at the next start.
    """
    loop = _server_loop.running()
    if loop is not None:
        loop.call_soon_threadsafe(_raise_thread_pool_ceiling)


def _log_embedding_model_state(embedder: Embedder) -> None:
    """Report whether embeddings will work, without letting that check stop startup.

    A model that reports itself unavailable and one whose check raises both leave
    the server usable for everything that does not embed, so neither is fatal.
    """
    try:
        validated = embedder.validate_model()
    except Exception:
        log.warning("Failed to validate embedding model", exc_info=True)
        return
    if validated:
        log.info("Embedding model validated")
    else:
        log.warning(
            "Embedding model %s is unavailable; search and chat will run without embeddings",
            cfg.embedding_model,
        )


@asynccontextmanager
async def _lifespan(app: Litestar) -> AsyncIterator[None]:
    """Pre-load LLM provider and embedding model on server startup."""
    _raise_thread_pool_ceiling()
    _server_loop.set(asyncio.get_running_loop())
    _warn_if_few_file_descriptors()
    session_manager.load_or_generate()

    inject_provider_keys()

    try:
        services = get_services()  # pre-load all services (provider, embedder, etc.)
        log.info("LLM provider pre-loaded")
    except Exception:
        log.warning("Failed to pre-load LLM provider", exc_info=True)
    else:
        _log_embedding_model_state(services.embedder)
    try:
        yield
    finally:
        _server_loop.set(None)
        session_manager.cleanup()
        # Terminate the provider's worker/fleet subprocesses so they don't
        # outlive the server (e.g. on parent-death shutdown in managed mode).
        svc = peek_services()
        if svc is not None:
            svc.provider.shutdown()


def _json_error_response(request: Request, exc: Exception) -> Response:
    """Answer errors as JSON even on a route whose success media type is SSE.

    Litestar builds an error response with the route handler's media type, so a
    streaming route would label a JSON error body text/event-stream.
    """
    response = create_exception_response(request, exc)
    response.media_type = MediaType.JSON
    return response


def create_app() -> Litestar:
    """Create the Litestar application instance."""
    cors = CORSConfig(
        allow_origins=cfg.cors_origins,
        allow_origin_regex=cfg.cors_origin_regex,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        allow_headers=["Content-Type", "Authorization"],
    )
    mcp_route, mcp_session_lifespan = build_mcp_mount()
    return Litestar(
        lifespan=[_lifespan, mcp_session_lifespan],
        middleware=[DefineMiddleware(AuthMiddleware)],
        exception_handlers={HTTPException: _json_error_response},
        route_handlers=[
            mcp_route,
            health_route,
            warm_stream_route,
            status_route,
            shutdown_route,
            config_route,
            config_defaults_route,
            config_update_route,
            source_content_route,
            search_route,
            ask_route,
            ask_stream_route,
            chat_route,
            chat_stream_route,
            completions_router,
            sync_route,
            add_route,
            add_upload_route,
            models_list_route,
            models_external_route,
            models_set_chat_route,
            models_set_embedding_route,
            models_set_vision_route,
            models_set_reranker_route,
            models_catalog_route,
            models_installed_route,
            models_pull_route,
            models_show_route,
            models_delete_route,
            documents_list_route,
            documents_remove_route,
            memories_list_route,
            memories_remember_route,
            memories_update_route,
            memories_remove_route,
            sessions_list_route,
            session_get_route,
            session_create_route,
            session_add_message_route,
            session_claim_route,
            session_set_summary_route,
            session_rename_route,
            session_delete_route,
            export_route,
            import_route,
            placement_route,
            placement_preview_route,
            placement_set_route,
            placement_clear_route,
            gpus_route,
            gpu_stats_stream_route,
            crawl_route,
            setup_crawler_route,
            setup_crawler_status_route,
            wiki_list_route,
            wiki_read_route,
            wiki_drafts_route,
            wiki_draft_diff_route,
            wiki_draft_accept_route,
            wiki_draft_reject_route,
            wiki_citations_reverse_route,
            wiki_lint_route,
            wiki_prune_route,
            wiki_index_route,
            wiki_generate_route,
            wiki_wipe_route,
            wiki_build_route,
            wiki_update_route,
            wiki_status_route,
            wiki_synthesize_route,
        ],
        request_max_body_size=10 * 1024 * 1024,
        cors_config=cors,
        openapi_config=OpenAPIConfig(
            title="lilbee",
            description="Local knowledge base REST API",
            version=get_version(),
            path="/schema",
        ),
    )
