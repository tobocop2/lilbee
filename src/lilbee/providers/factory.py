"""Factory for creating LLM provider singletons."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lilbee.providers.base import LLMProvider

_provider: LLMProvider | None = None


def get_provider() -> LLMProvider:
    """Return the configured LLM provider singleton."""
    global _provider
    if _provider is not None:
        return _provider

    from lilbee.config import cfg

    provider_name = cfg.llm_provider

    if provider_name == "auto":
        from lilbee.providers.routing_provider import RoutingProvider

        _provider = RoutingProvider()
    elif provider_name == "llama-cpp":
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        _provider = LlamaCppProvider()
    elif provider_name in ("litellm", "ollama"):
        from lilbee.providers.litellm_provider import LiteLLMProvider

        if not LiteLLMProvider.available():
            from lilbee.providers.base import ProviderError

            raise ProviderError(
                "litellm is not installed. Install with: pip install 'lilbee[litellm]'"
            )
        _provider = LiteLLMProvider(
            base_url=cfg.litellm_base_url, api_key=cfg.llm_api_key
        )
    else:
        from lilbee.providers.base import ProviderError

        raise ProviderError(f"Unknown LLM provider: {provider_name!r}")

    return _provider


def reset_provider() -> None:
    """Clear the provider singleton. For testing only."""
    global _provider
    _provider = None
