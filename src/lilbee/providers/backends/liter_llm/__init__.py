"""liter-llm backend for the provider layer.

The rest of lilbee reaches this package through ``create_sdk_backend`` and
``provider_sdk_available``. The client library loads on the first request, never at import.
"""

from __future__ import annotations

from lilbee.providers.backends.liter_llm.availability import (
    PROVIDER_SDK_MISSING_MESSAGE,
    SDK_MODULE,
    provider_sdk_available,
)
from lilbee.providers.backends.liter_llm.backend import LiterLlmBackend
from lilbee.providers.sdk_backend import LlmSdkBackend

__all__ = [
    "PROVIDER_SDK_MISSING_MESSAGE",
    "SDK_MODULE",
    "create_sdk_backend",
    "provider_sdk_available",
]


def create_sdk_backend() -> LlmSdkBackend:
    """Return a new backend over the provider client library."""
    return LiterLlmBackend()
