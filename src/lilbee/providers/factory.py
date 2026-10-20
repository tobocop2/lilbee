"""Factory for creating LLM provider instances."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

from lilbee.core.config.enums import LlmProvider
from lilbee.providers.backends.liter_llm import (
    PROVIDER_SDK_MISSING_MESSAGE,
    SDK_MODULE,
    create_sdk_backend,
    provider_sdk_available,
)
from lilbee.providers.base import ProviderError

if TYPE_CHECKING:
    from lilbee.core.config import Config
    from lilbee.providers.base import LLMProvider

__all__ = [
    "PROVIDER_SDK_MISSING_MESSAGE",
    "SDK_MODULE",
    "create_provider",
    "create_sdk_backend",
    "provider_sdk_available",
]


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
            # circular: routing_provider -> factory via create_sdk_backend
            from lilbee.providers.routing_provider import RoutingProvider

            return RoutingProvider(hold_warm=hold_warm)

        case LlmProvider.REMOTE:
            # heavy: the SDK provider imports numpy, and setup.py imports this module at CLI startup
            from lilbee.providers.sdk_llm_provider import SdkLLMProvider

            backend = create_sdk_backend()
            if not backend.available():
                raise ProviderError(PROVIDER_SDK_MISSING_MESSAGE)
            return SdkLLMProvider(backend)

    assert_never(config.llm_provider)  # pragma: no cover
