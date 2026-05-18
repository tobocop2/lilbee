"""Thin wrapper around the provider's tool-capability probe."""

from __future__ import annotations

import logging

from lilbee.app.services import get_services
from lilbee.providers.base import ProviderError

log = logging.getLogger(__name__)


def model_supports_tools(model_ref: str) -> bool:
    """Return True iff the active provider reports tool support for *model_ref*.

    A ``ProviderError`` from the probe (model file unavailable, backend down)
    is treated as "no tools" so the route layer returns a 400 instead of 500.
    Other exceptions are intentionally NOT caught here so genuine bugs
    surface as 500s instead of being silently downgraded.
    """
    try:
        return bool(get_services().provider.supports_tools(model_ref))
    except ProviderError:
        log.debug("supports_tools probe raised ProviderError for %s", model_ref, exc_info=True)
        return False
