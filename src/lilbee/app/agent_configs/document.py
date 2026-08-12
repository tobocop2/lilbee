"""The per-client config document `lilbee agent-config` prints and /api/agent-config serves."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import yaml

from lilbee.app.agent_configs.claude import claude_http_config, claude_stdio_config
from lilbee.app.agent_configs.hermes import hermes_config
from lilbee.app.agent_configs.opencode import opencode_config


class AgentClient(StrEnum):
    """An AI client lilbee builds a config document for."""

    CLAUDE = "claude"
    HERMES = "hermes"
    OPENCODE = "opencode"


class ConfigFormat(StrEnum):
    """The serialization the client's own config file is written in."""

    JSON = "json"
    YAML = "yaml"


class AgentSurface(StrEnum):
    """A lilbee surface a client's config wires the client up to."""

    MODEL_PROVIDER = "model_provider"
    MCP = "mcp"


@dataclass(frozen=True)
class AgentConfigDocument:
    """One client's paste-ready lilbee config, carrying a live base URL and token.

    ``config`` holds the block for a JSON client and ``content`` the rendered
    text for a YAML one; exactly one of the two is set. ``stdio_config`` is an
    alternative block for a client that can also run lilbee as a subprocess.
    """

    client: AgentClient
    format: ConfigFormat
    surfaces: tuple[AgentSurface, ...]
    config: dict[str, Any] | None = None
    content: str | None = None
    stdio_config: dict[str, Any] | None = None


@dataclass(frozen=True)
class _ClientInputs:
    """The live server state every client document is rendered from."""

    base_url: str
    api_key: str
    model_refs: list[str]
    default_ref: str | None
    chat_ctx: int | None


CLIENT_SURFACES: dict[AgentClient, tuple[AgentSurface, ...]] = {
    # Claude Code takes lilbee's MCP tools only. It brings its own model, so
    # there is no provider block to write.
    AgentClient.CLAUDE: (AgentSurface.MCP,),
    AgentClient.HERMES: (AgentSurface.MODEL_PROVIDER, AgentSurface.MCP),
    AgentClient.OPENCODE: (AgentSurface.MODEL_PROVIDER, AgentSurface.MCP),
}


def client_serves_models(client: AgentClient) -> bool:
    """True when *client*'s document registers lilbee as its model provider."""
    return AgentSurface.MODEL_PROVIDER in CLIENT_SURFACES[client]


def parse_agent_client(value: str) -> AgentClient:
    """Return the client named *value*, or raise ValueError naming the valid ones."""
    try:
        return AgentClient(value)
    except ValueError as exc:
        valid = ", ".join(client.value for client in AgentClient)
        raise ValueError(f"lilbee has no config for '{value}'. Ask for one of: {valid}.") from exc


def _claude_document(inputs: _ClientInputs) -> AgentConfigDocument:
    """Claude Code's MCP registration, over http with the stdio form alongside."""
    return AgentConfigDocument(
        client=AgentClient.CLAUDE,
        format=ConfigFormat.JSON,
        surfaces=CLIENT_SURFACES[AgentClient.CLAUDE],
        config=claude_http_config(base_url=inputs.base_url, api_key=inputs.api_key),
        stdio_config=claude_stdio_config(),
    )


def _hermes_document(inputs: _ClientInputs) -> AgentConfigDocument:
    """hermes's config.yaml fragment, rendered to the text the user pastes."""
    fragment = hermes_config(
        base_url=inputs.base_url,
        api_key=inputs.api_key,
        model_refs=inputs.model_refs,
        default_ref=inputs.default_ref,
        chat_ctx=inputs.chat_ctx,
    )
    return AgentConfigDocument(
        client=AgentClient.HERMES,
        format=ConfigFormat.YAML,
        surfaces=CLIENT_SURFACES[AgentClient.HERMES],
        content=yaml.safe_dump(fragment, sort_keys=False),
    )


def _opencode_document(inputs: _ClientInputs) -> AgentConfigDocument:
    """opencode.json's provider plus MCP block."""
    return AgentConfigDocument(
        client=AgentClient.OPENCODE,
        format=ConfigFormat.JSON,
        surfaces=CLIENT_SURFACES[AgentClient.OPENCODE],
        config=opencode_config(
            base_url=inputs.base_url,
            api_key=inputs.api_key,
            model_refs=inputs.model_refs,
            default_ref=inputs.default_ref,
            chat_ctx=inputs.chat_ctx,
        ),
    )


_BUILDERS: dict[AgentClient, Callable[[_ClientInputs], AgentConfigDocument]] = {
    AgentClient.CLAUDE: _claude_document,
    AgentClient.HERMES: _hermes_document,
    AgentClient.OPENCODE: _opencode_document,
}


def build_agent_config(
    client: AgentClient,
    *,
    base_url: str,
    api_key: str,
    model_refs: list[str] | None = None,
    default_ref: str | None = None,
    chat_ctx: int | None = None,
) -> AgentConfigDocument:
    """Build *client*'s config document against a live server URL and token.

    The model arguments describe what lilbee serves and are ignored by a client
    that only takes the MCP tools (see :func:`client_serves_models`).
    """
    inputs = _ClientInputs(
        base_url=base_url,
        api_key=api_key,
        model_refs=model_refs or [],
        default_ref=default_ref,
        chat_ctx=chat_ctx,
    )
    return _BUILDERS[client](inputs)
