"""Factory for creating LLM provider instances."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, assert_never

from lilbee.core.config.enums import LlmProvider
from lilbee.providers import availability
from lilbee.providers.base import ProviderError

if TYPE_CHECKING:
    from lilbee.core.config import Config
    from lilbee.providers.base import LLMProvider
    from lilbee.providers.fleet.engines.base import EngineAdapter
    from lilbee.providers.sdk_backend import LlmSdkBackend
    from lilbee.providers.sdk_llm_provider import SdkLLMProvider

__all__ = ["SdkLibrary", "create_provider", "create_sdk_provider", "sdk_library"]


@dataclass(frozen=True)
class SdkLibrary:
    """The client library behind the SDK backend."""

    dist: str
    """Distribution name, as package metadata lists it."""
    module: str
    """Top-level import name."""
    create_backend: Callable[[], LlmSdkBackend]
    """Builds a new backend over the library."""


def sdk_library() -> SdkLibrary:
    """Return the client library's names and backend builder, importing nothing heavy."""
    # heavy: setup.py imports this module at CLI startup; the backend package
    # defers its backend module and the library to create_backend
    from lilbee.providers.backends.liter_llm import SDK_DIST, SDK_MODULE, create_sdk_backend

    return SdkLibrary(dist=SDK_DIST, module=SDK_MODULE, create_backend=create_sdk_backend)


def create_sdk_provider() -> SdkLLMProvider:
    """Return a new provider over the SDK backend, for remote and API models."""
    from lilbee.providers.sdk_llm_provider import SdkLLMProvider

    return SdkLLMProvider(sdk_library().create_backend())


def create_provider(
    config: Config, *, engine: EngineAdapter, hold_warm: bool = False
) -> LLMProvider:
    """Create a new LLM provider instance from the given config.

    *engine* is the inference engine a local fleet runs on.

    *hold_warm* marks the provider as serving an interactive session, so a local
    fleet bound to that session keeps its weights resident instead of
    idle-unloading for as long as the session owns the process (a fleet shared
    with other users, or kept warm, outlives it and keeps its idle window).
    Irrelevant to a remote provider, which has no local engine to hold.
    """
    match config.llm_provider:
        case LlmProvider.AUTO:
            # circular: routing_provider -> factory via create_sdk_provider
            from lilbee.providers.routing_provider import RoutingProvider

            return RoutingProvider(engine=engine, hold_warm=hold_warm)

        case LlmProvider.REMOTE:
            if not availability.provider_sdk_available():
                raise ProviderError(availability.PROVIDER_SDK_MISSING_MESSAGE)
            return create_sdk_provider()

    assert_never(config.llm_provider)  # pragma: no cover
