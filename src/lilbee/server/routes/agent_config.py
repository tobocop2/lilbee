"""Agent-client config routes: which clients are installed, and their live config."""

from __future__ import annotations

from litestar import Request, get
from litestar.exceptions import NotFoundException
from litestar.params import FromPath

from lilbee.app.agent_configs.document import parse_agent_client
from lilbee.server import handlers
from lilbee.server.models import AgentConfigIndexResponse, AgentConfigResponse


def _base_url(request: Request) -> str:
    """The origin this request arrived on, so the config points back at this server."""
    return str(request.base_url).rstrip("/")


@get("/api/agent-config")
async def agent_config_index_route() -> AgentConfigIndexResponse:
    """Supported AI clients, with whether each one's CLI is installed on this machine."""
    return await handlers.agent_config_index()


@get("/api/agent-config/{client:str}")
async def agent_config_route(client: FromPath[str], request: Request) -> AgentConfigResponse:
    """One client's paste-ready config, carrying this server's URL and current token."""
    try:
        parsed = parse_agent_client(client)
    except ValueError as exc:
        raise NotFoundException(str(exc)) from exc
    return await handlers.agent_config(parsed, _base_url(request))
