"""Litestar application factory — imports routes from modules, creates app with lifespan."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from litestar import Litestar
from litestar.config.cors import CORSConfig
from litestar.middleware.base import DefineMiddleware
from litestar.openapi import OpenAPIConfig

from lilbee.cli.helpers import get_version
from lilbee.config import cfg
from lilbee.server.auth import (
    AuthMiddleware,
    cleanup_session_token,
    generate_session_token,
)
from lilbee.server.routes.crawl import crawl_route
from lilbee.server.routes.documents import (
    add_route,
    documents_list_route,
    documents_remove_route,
    sync_route,
)
from lilbee.server.routes.general import config_route, health_route, status_route
from lilbee.server.routes.models import (
    models_catalog_route,
    models_delete_route,
    models_installed_route,
    models_list_route,
    models_pull_route,
    models_set_chat_route,
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

log = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: Litestar) -> AsyncIterator[None]:
    """Pre-load LLM provider and embedding model on server startup."""
    generate_session_token()
    from lilbee.services import get_services

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
        cleanup_session_token()


def create_app() -> Litestar:
    """Create the Litestar application instance."""
    cors = CORSConfig(
        allow_origins=cfg.cors_origins,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Content-Type", "Authorization"],
    )
    return Litestar(
        lifespan=[_lifespan],
        middleware=[DefineMiddleware(AuthMiddleware)],
        route_handlers=[
            health_route,
            status_route,
            config_route,
            search_route,
            ask_route,
            ask_stream_route,
            chat_route,
            chat_stream_route,
            sync_route,
            add_route,
            models_list_route,
            models_set_chat_route,
            models_set_vision_route,
            models_catalog_route,
            models_installed_route,
            models_pull_route,
            models_show_route,
            models_delete_route,
            documents_list_route,
            documents_remove_route,
            crawl_route,
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
