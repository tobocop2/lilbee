"""Litestar application factory: imports routes from modules, creates app with lifespan."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from litestar import Litestar
from litestar.config.cors import CORSConfig
from litestar.middleware.base import DefineMiddleware
from litestar.openapi import OpenAPIConfig

from lilbee.app.services import get_services
from lilbee.app.version import get_version
from lilbee.core.config import cfg
from lilbee.providers.sdk_llm_provider import inject_provider_keys
from lilbee.server.auth import AuthMiddleware, session_manager
from lilbee.server.routes.crawl import crawl_route
from lilbee.server.routes.documents import (
    add_route,
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
    source_content_route,
    status_route,
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
from lilbee.server.routes.search import (
    ask_route,
    ask_stream_route,
    chat_route,
    chat_stream_route,
    search_route,
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
    wiki_lint_route,
    wiki_list_route,
    wiki_prune_route,
    wiki_read_route,
    wiki_status_route,
    wiki_synthesize_route,
    wiki_update_route,
)

log = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: Litestar) -> AsyncIterator[None]:
    """Pre-load LLM provider and embedding model on server startup."""
    session_manager.load_or_generate()

    inject_provider_keys()

    try:
        get_services()  # pre-load all services (provider, embedder, etc.)
        log.info("LLM provider pre-loaded")
    except Exception:
        log.warning("Failed to pre-load LLM provider", exc_info=True)
    try:
        get_services().embedder.validate_model()
        log.info("Embedding model validated")
    except Exception:
        log.warning("Failed to validate embedding model", exc_info=True)
    try:
        yield
    finally:
        session_manager.cleanup()


def create_app() -> Litestar:
    """Create the Litestar application instance."""
    cors = CORSConfig(
        allow_origins=cfg.cors_origins,
        allow_origin_regex=cfg.cors_origin_regex,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        allow_headers=["Content-Type", "Authorization"],
    )
    return Litestar(
        lifespan=[_lifespan],
        middleware=[DefineMiddleware(AuthMiddleware)],
        route_handlers=[
            health_route,
            status_route,
            config_route,
            config_defaults_route,
            config_update_route,
            source_content_route,
            search_route,
            ask_route,
            ask_stream_route,
            chat_route,
            chat_stream_route,
            sync_route,
            add_route,
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
            export_route,
            import_route,
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
