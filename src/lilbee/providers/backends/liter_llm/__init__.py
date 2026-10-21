"""liter-llm backend for the provider layer.

The provider factory is the only module outside this package that imports it.
Importing this package loads nothing heavy: the backend module loads inside
``create_sdk_backend``, and the client library loads on the first request.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lilbee.providers.sdk_backend import LlmSdkBackend

__all__ = ["SDK_DIST", "SDK_MODULE", "create_sdk_backend"]

# Distribution and import names of the client library.
SDK_DIST = "liter-llm"
SDK_MODULE = "liter_llm"


def create_sdk_backend() -> LlmSdkBackend:
    """Return a new backend over the provider client library."""
    from lilbee.providers.backends.liter_llm.backend import LiterLlmBackend

    return LiterLlmBackend()
