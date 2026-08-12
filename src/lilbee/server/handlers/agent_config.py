"""Agent-client config handlers for the HTTP server."""

from __future__ import annotations

import asyncio

from lilbee.app.agent_configs.detect import detect_agent_clients
from lilbee.app.agent_configs.document import (
    AgentClient,
    AgentConfigDocument,
    build_agent_config,
    client_serves_models,
)
from lilbee.app.models import installed_chat_model_refs
from lilbee.app.services import get_services
from lilbee.core.config import cfg
from lilbee.providers.model_ref import with_configured_remote_chat
from lilbee.server.auth import session_manager
from lilbee.server.models import AgentConfigIndexResponse, AgentConfigResponse


async def agent_config_index() -> AgentConfigIndexResponse:
    """Every supported client, with whether its CLI is installed on this machine."""
    # Probing walks PATH and stats candidate directories; keep it off the loop.
    detections = await asyncio.to_thread(detect_agent_clients)
    return AgentConfigIndexResponse.from_detections(detections)


async def agent_config(client: AgentClient, base_url: str) -> AgentConfigResponse:
    """One client's config, carrying *base_url* and this server's current token."""
    # The registry walk and the served-window probe both block.
    document = await asyncio.to_thread(_build_document, client, base_url)
    return AgentConfigResponse.from_document(document)


def _model_inputs(client: AgentClient) -> tuple[list[str] | None, int | None]:
    """The chat refs and served window to advertise, or ``None`` for an MCP-only client."""
    if not client_serves_models(client):
        return None, None
    refs = with_configured_remote_chat(installed_chat_model_refs(), cfg.chat_model)
    return refs, get_services().provider.served_chat_ctx()


def _build_document(client: AgentClient, base_url: str) -> AgentConfigDocument:
    """Assemble *client*'s document from live server state."""
    model_refs, chat_ctx = _model_inputs(client)
    return build_agent_config(
        client,
        base_url=base_url,
        # A server with auth disabled accepts an empty bearer, which is what a
        # client configured from it should send.
        api_key=session_manager.token or "",
        model_refs=model_refs,
        default_ref=str(cfg.chat_model) if model_refs is not None else None,
        chat_ctx=chat_ctx,
    )
