"""The HTTP status for the provider failures that are the caller's fault."""

from __future__ import annotations

from lilbee.providers.base import ProviderErrorKind

CLIENT_FAULT_STATUSES: dict[ProviderErrorKind, int] = {
    ProviderErrorKind.BAD_REQUEST: 400,
}
