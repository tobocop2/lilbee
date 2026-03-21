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

    if provider_name == "llama-cpp":
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        _provider = LlamaCppProvider()
    elif provider_name == "ollama":
        from lilbee.providers.litellm_provider import LiteLLMProvider

        _provider = LiteLLMProvider(base_url=cfg.llm_base_url)
    elif provider_name == "litellm":
        from lilbee.providers.litellm_provider import LiteLLMProvider

        _provider = LiteLLMProvider(base_url=cfg.llm_base_url, api_key=cfg.llm_api_key)
    else:
        from lilbee.providers.base import ProviderError

        raise ProviderError(f"Unknown LLM provider: {provider_name!r}")

    return _provider


def reset_provider() -> None:
    """Clear the provider singleton. For testing only."""
    global _provider
    _provider = None
