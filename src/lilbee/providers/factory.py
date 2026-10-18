"""Factory for creating LLM provider instances."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

from lilbee.core.config.enums import LlmProvider
from lilbee.providers.base import ProviderError

if TYPE_CHECKING:
    from lilbee.core.config import Config
    from lilbee.providers.base import LLMProvider


def create_provider(config: Config, *, hold_warm: bool = False) -> LLMProvider:
    """Create a new LLM provider instance from the given config.

    *hold_warm* marks the provider as serving an interactive session, so a local
    fleet bound to that session keeps its weights resident instead of
    idle-unloading for as long as the session owns the process (a fleet shared
    with other users, or kept warm, outlives it and keeps its idle window).
    Irrelevant to a remote provider, which has no local engine to hold.
    """
    match config.llm_provider:
        case LlmProvider.AUTO:
            # heavy: routing_provider eagerly imports liter_llm_sdk (native module load)
            from lilbee.providers.routing_provider import RoutingProvider

            return RoutingProvider(hold_warm=hold_warm)

        case LlmProvider.REMOTE:
            # heavy: liter_llm_sdk pulls in the native liter-llm module
            from lilbee.providers.liter_llm_sdk import LITER_LLM_MISSING_MSG, LiterLlmSdkBackend
            from lilbee.providers.sdk_llm_provider import SdkLLMProvider

            backend = LiterLlmSdkBackend()
            if not backend.available():
                raise ProviderError(LITER_LLM_MISSING_MSG)
            return SdkLLMProvider(backend)  # pragma: no cover

    assert_never(config.llm_provider)  # pragma: no cover
