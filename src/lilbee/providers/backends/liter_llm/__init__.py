"""liter-llm backend for the provider layer.

The provider factory is the only caller of ``create_sdk_backend``. Importing this
package loads nothing heavy: the backend module loads inside that function, and the
client library loads on the first request.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lilbee.providers.sdk_backend import LlmSdkBackend

__all__ = ["create_sdk_backend"]


def create_sdk_backend() -> LlmSdkBackend:
    """Return a new backend over the provider client library."""
    from lilbee.providers.backends.liter_llm.backend import LiterLlmBackend

    return LiterLlmBackend()
