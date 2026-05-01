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
        # heavy: routing_provider eagerly imports litellm_sdk (>50ms; pulls litellm fanout)
        from lilbee.providers.routing_provider import RoutingProvider

        return RoutingProvider()

    if provider_name == "llama-cpp":
        # heavy: llama_cpp loads native Metal/CUDA dylibs at module top
        from lilbee.providers.llama_cpp import LlamaCppProvider

        return LlamaCppProvider()

    if provider_name == "remote":
        # THIS is the swap line: the single import that changes when
        # migrating to a different SDK. Replace LitellmSdkBackend here
        # with the new adapter and the rest of lilbee is untouched.
        # heavy: litellm_sdk loads litellm provider fanout (>50ms)
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
