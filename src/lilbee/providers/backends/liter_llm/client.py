"""Import and construct the liter-llm client."""

from __future__ import annotations

import functools
from typing import Any

from lilbee.providers.backends.liter_llm.availability import PROVIDER_SDK_MISSING_MESSAGE
from lilbee.providers.backends.liter_llm.errors import _PROVIDER_NAME
from lilbee.providers.base import ProviderError
from lilbee.providers.local_servers import local_server_for_key

# Seconds a single provider request may run. liter-llm has no unbounded default,
# so a long non-streaming generation needs an explicit ceiling.
_REQUEST_TIMEOUT_S = 600
# Retries after a failed request. A local server answers or is down, so a retry only
# repeats a long load; a hosted provider gets two (three attempts).
_LOCAL_SERVER_RETRIES = 0
_HOSTED_RETRIES = 2


def _require_sdk() -> Any:
    """Import ``liter_llm`` or raise a user-facing ProviderError with install steps."""
    try:
        import liter_llm
    except ImportError as exc:
        raise ProviderError(PROVIDER_SDK_MISSING_MESSAGE, provider=_PROVIDER_NAME) from exc
    return liter_llm


def _max_retries(model: str) -> int:
    """Return the retry count for a ``provider/model`` string: none for a local server."""
    provider = model.partition("/")[0]
    return _LOCAL_SERVER_RETRIES if local_server_for_key(provider) is not None else _HOSTED_RETRIES


@functools.lru_cache(maxsize=32)
def _client(model: str, base_url: str | None, api_key: str | None) -> Any:
    """Return the liter-llm client for one model, endpoint and key (cached)."""
    liter_llm = _require_sdk()
    options = liter_llm.ClientOptions(
        api_key=api_key or "",
        base_url=base_url,
        timeout_secs=_REQUEST_TIMEOUT_S,
        max_retries=_max_retries(model),
        model_hint=model,
    )
    return liter_llm.create_client_with_options(options)
