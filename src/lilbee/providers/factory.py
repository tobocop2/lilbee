"""Factory for creating LLM provider instances."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lilbee.providers.base import ProviderError

if TYPE_CHECKING:
    from lilbee.core.config import Config
    from lilbee.providers.base import LLMProvider


def create_provider(config: Config) -> LLMProvider:
    """Create a new LLM provider instance from the given config."""
    provider_name = config.llm_provider

    if provider_name == "auto":
        # circular: providers/__init__ -> factory -> routing_provider -> config
        from lilbee.providers.routing_provider import RoutingProvider

        return RoutingProvider()

    if provider_name == "llama-cpp":
        # circular: providers/__init__ -> factory -> llama_cpp_provider -> config
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        return LlamaCppProvider()

    if provider_name == "remote":
        # THIS is the swap line: the single import that changes when
        # migrating to a different SDK. Replace LitellmSdkBackend here
        # with the new adapter and the rest of lilbee is untouched.
        # circular: providers/__init__ -> factory -> sdk_llm_provider -> config
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        backend = LitellmSdkBackend()
        if not backend.available():
            raise ProviderError(
                "SDK backend adapter is not installed. Install with: pip install 'lilbee[litellm]'"
            )
        return SdkLLMProvider(
            backend,
            base_url=config.remote_base_url,
            api_key=config.llm_api_key,
        )  # pragma: no cover

    raise ProviderError(f"Unknown LLM provider: {provider_name!r}")
