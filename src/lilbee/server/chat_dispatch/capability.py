"""Thin wrapper around the provider's tool-capability probe.

Lives at the dispatch boundary so the provider stays the single source
of truth for whether a model can carry tool calls. ``False`` is the
conservative default when the probe raises.
"""

from __future__ import annotations

import logging

from lilbee.app.services import get_services

log = logging.getLogger(__name__)


def model_supports_tools(model_ref: str) -> bool:
    """Return True iff the active provider reports tool support for *model_ref*."""
    try:
        return bool(get_services().provider.supports_tools(model_ref))
    except Exception:
        log.debug("supports_tools probe raised for %s", model_ref, exc_info=True)
        return False
