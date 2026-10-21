"""liter-llm backend for the provider layer.

The provider factory is the only caller.
The client library loads on the first request, never at import.
"""

from __future__ import annotations

from lilbee.providers.backends.liter_llm.backend import LiterLlmBackend
from lilbee.providers.sdk_backend import LlmSdkBackend

__all__ = ["create_sdk_backend"]


def create_sdk_backend() -> LlmSdkBackend:
    """Return a new backend over the provider client library."""
    return LiterLlmBackend()
