"""hermes config.yaml fragment builder: lilbee as a provider plus MCP search tool."""

from __future__ import annotations

from typing import Any

from lilbee.cli.agent_configs.merge import LILBEE_PROVIDER_KEY

_V1_SUFFIX = "/v1"
_MCP_SUFFIX = "/mcp"
# hermes's custom provider defaults max_tokens to the full window, leaving ~no
# room for input. Pin a sane output cap so the system prompt + history fit.
_MAX_OUTPUT_TOKENS = 8192
# hermes truncates a project context file (AGENTS.md and the like) to this many
# chars by default, printing a visible TRUNCATED warning. Raised to the served
# window below so large context files load intact.
_HERMES_DEFAULT_CONTEXT_FILE_MAX_CHARS = 20000


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
    if chat_ctx is not None:
        # Size hermes's context-file budget to the served window so a large project
        # context file loads instead of being cut to the 20k default. chat_ctx is
        # tokens; used as a char budget it stays within the model window.
        config["context_file_max_chars"] = max(_HERMES_DEFAULT_CONTEXT_FILE_MAX_CHARS, chat_ctx)
    if include_mcp:
        # An `url` (no `transport` key) is hermes's HTTP MCP shape; `headers`
        # carries the bearer with ${VAR} env resolution. A `transport` string
        # makes hermes reject the entry (it must be a mapping) -> 0 connected.
        config["mcp_servers"] = {
            LILBEE_PROVIDER_KEY: {
                "url": f"{base_url}{_MCP_SUFFIX}",
                "headers": {"Authorization": f"Bearer {api_key}"},
            }
        }
    if pin is not None:
        # Dict form binds the active model to the lilbee provider explicitly, so the
        # pin is unambiguous even when another provider could serve the same ref.
        # `model.max_tokens` is hermes's documented winning output cap: without it
        # hermes requests the full window as output, so input+output overflows and
        # it misreads the error as "prompt too long" and compresses to nothing.
        config["model"] = {
            "default": pin,
            "provider": LILBEE_PROVIDER_KEY,
            "max_tokens": _MAX_OUTPUT_TOKENS,
        }
    return config
