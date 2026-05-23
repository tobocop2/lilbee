"""Factory for creating LLM provider instances."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

from lilbee.core.config.enums import LlmProvider
from lilbee.providers.base import ProviderError

if TYPE_CHECKING:
    from lilbee.core.config import Config
    from lilbee.providers.base import LLMProvider


def create_provider(config: Config) -> LLMProvider:
    """Create a new LLM provider instance from the given config."""
    match config.llm_provider:
        case LlmProvider.AUTO:
            # heavy: routing_provider eagerly imports litellm_sdk (>50ms; litellm fanout)
            from lilbee.providers.routing_provider import RoutingProvider

            return RoutingProvider()

        case LlmProvider.LLAMA_CPP:
            # heavy: llama_cpp loads native Metal/CUDA dylibs at module top
            from lilbee.providers.llama_cpp import LlamaCppProvider

            return LlamaCppProvider()

        case LlmProvider.REMOTE:
            # THIS is the swap line: the single import that changes when
            # migrating to a different SDK. Replace LitellmSdkBackend here
            # with the new adapter and the rest of lilbee is untouched.
            # heavy: litellm_sdk loads litellm provider fanout (>50ms)
            from lilbee.providers.litellm_sdk import LitellmSdkBackend
            from lilbee.providers.sdk_llm_provider import SdkLLMProvider

            backend = LitellmSdkBackend()
            if not backend.available():
                raise ProviderError(
                    "SDK backend adapter is not installed. Install with: "
                    "pip install 'lilbee[litellm]'"
                )
            return SdkLLMProvider(
                backend,
                base_url=config.remote_base_url,
                api_key=config.llm_api_key,
            )  # pragma: no cover

    assert_never(config.llm_provider)  # pragma: no cover
