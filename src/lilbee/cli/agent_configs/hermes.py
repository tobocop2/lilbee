"""hermes config.yaml fragment builder: lilbee as a provider plus MCP search tool."""

from __future__ import annotations

from typing import Any

from lilbee.cli.agent_configs.merge import LILBEE_PROVIDER_KEY

_V1_SUFFIX = "/v1"
_MCP_SUFFIX = "/mcp"
_MCP_TRANSPORT = "streamable-http"
_MCP_TIMEOUT_S = 120
# hermes's custom provider defaults max_tokens to the full window, leaving ~no
# room for input. Pin a sane output cap so the system prompt + history fit.
_MAX_OUTPUT_TOKENS = 8192


def hermes_config(
    *,
    base_url: str,
    api_key: str,
    model_refs: list[str],
    default_ref: str | None = None,
    chat_ctx: int | None = None,
    include_mcp: bool = True,
) -> dict[str, Any]:
    """Return the hermes config fragment registering lilbee as a provider (and MCP).

    ``api_key`` is embedded verbatim: a literal token for the paste path, or the
    ``${LILBEE_TOKEN}`` reference for the launcher (hermes expands it from the env
    at load, so the on-disk file never holds the literal token)."""
    pin = default_ref or (model_refs[0] if model_refs else None)
    provider: dict[str, Any] = {
        "api": f"{base_url}{_V1_SUFFIX}",
        "api_key": api_key,
        "max_tokens": _MAX_OUTPUT_TOKENS,
    }
    if pin is not None:
        provider["default_model"] = pin
    if chat_ctx is not None:
        provider["context_length"] = chat_ctx
    config: dict[str, Any] = {"providers": {LILBEE_PROVIDER_KEY: provider}}
    if include_mcp:
        config["mcp_servers"] = {
            LILBEE_PROVIDER_KEY: {
                "url": f"{base_url}{_MCP_SUFFIX}",
                "transport": _MCP_TRANSPORT,
                "headers": {"Authorization": f"Bearer {api_key}"},
                "timeout": _MCP_TIMEOUT_S,
            }
        }
    if pin is not None:
        # Dict form binds the active model to the lilbee provider explicitly, so the
        # pin is unambiguous even when another provider could serve the same ref.
        config["model"] = {"default": pin, "provider": LILBEE_PROVIDER_KEY}
    return config
