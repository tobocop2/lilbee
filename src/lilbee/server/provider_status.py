"""The HTTP status for provider failures, and the retry hint for the ones that clear."""

from __future__ import annotations

from lilbee.providers.base import ProviderErrorKind

CLIENT_FAULT_STATUSES: dict[ProviderErrorKind, int] = {
    ProviderErrorKind.BAD_REQUEST: 400,
}

UNAVAILABLE_RETRY_AFTER_SECONDS = 5

# Failures that end by themselves: a server that is not up yet, a busy engine, a port
# that frees. Every other 503 kind needs a change from the user (a missing library, an
# unsupported CPU, a broken model file), so a retry hint would mislead.
_CLEARING_KINDS = frozenset(
    {
        ProviderErrorKind.CONNECTION,
        ProviderErrorKind.CAPACITY,
        ProviderErrorKind.PORT_CONFLICT,
        ProviderErrorKind.STARTING,
    }
)


def retry_after_headers(kind: ProviderErrorKind | None = None) -> dict[str, str]:
    """The Retry-After header for a 503 that clears by itself, else none.

    With no *kind*, the failure is one that clears whatever its source, such as a
    config lock another process holds.
    """
    if kind is not None and kind not in _CLEARING_KINDS:
        return {}
    return {"Retry-After": str(UNAVAILABLE_RETRY_AFTER_SECONDS)}
