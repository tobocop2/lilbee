"""Hosted provider API-key status: one authenticated call per key, cached."""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import httpx
from cachetools import TTLCache, cached

from lilbee.catalog.types import KeyStatus
from lilbee.providers.sdk_backend import PROVIDER_API_KEY_ENV, get_provider_api_key

log = logging.getLogger(__name__)

KEY_CHECK_TTL_S = 300
KEY_CHECK_TIMEOUT_S = 5.0
_KEYS_CACHED_PER_PROVIDER = 4
_AUTH_REJECTED_STATUS = frozenset({httpx.codes.UNAUTHORIZED, httpx.codes.FORBIDDEN})
_ANTHROPIC_API_VERSION = "2023-06-01"
_GEMINI_INVALID_KEY_REASON = "API_KEY_INVALID"


def _bearer(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _anthropic_headers(api_key: str) -> dict[str, str]:
    return {"x-api-key": api_key, "anthropic-version": _ANTHROPIC_API_VERSION}


def _gemini_headers(api_key: str) -> dict[str, str]:
    return {"x-goog-api-key": api_key}


def _auth_rejected(resp: httpx.Response) -> bool:
    return resp.status_code in _AUTH_REJECTED_STATUS


def _gemini_rejected(resp: httpx.Response) -> bool:
    """Gemini answers a bad key with 400 and the structured reason ``API_KEY_INVALID``."""
    if _auth_rejected(resp):
        return True
    if resp.status_code != httpx.codes.BAD_REQUEST:
        return False
    try:
        details = resp.json()["error"]["details"]
    except (ValueError, KeyError, TypeError):
        return False
    # Untyped JSON error body: details must be a list, and only dict entries carry a reason.
    if not isinstance(details, list):
        return False
    return any(
        isinstance(item, dict) and item.get("reason") == _GEMINI_INVALID_KEY_REASON
        for item in details
    )


@dataclass(frozen=True)
class _KeyProbe:
    """An authenticated GET that fails only when the key is bad."""

    url: str
    headers: Callable[[str], dict[str, str]]
    rejected: Callable[[httpx.Response], bool] = _auth_rejected


# Not litellm: get_models raises untyped errors; check_valid_key and get_valid_models hide them.
# OpenRouter's model list is public, so its probe reads the key's own record instead.
_PROBES: dict[str, _KeyProbe] = {
    "openrouter": _KeyProbe("https://openrouter.ai/api/v1/key", _bearer),
    "gemini": _KeyProbe(
        "https://generativelanguage.googleapis.com/v1beta/models", _gemini_headers, _gemini_rejected
    ),
    "anthropic": _KeyProbe("https://api.anthropic.com/v1/models", _anthropic_headers),
    "openai": _KeyProbe("https://api.openai.com/v1/models", _bearer),
    "mistral": _KeyProbe("https://api.mistral.ai/v1/models", _bearer),
    "deepseek": _KeyProbe("https://api.deepseek.com/models", _bearer),
}

_probe_condition = threading.Condition()


def _http_get(url: str, *, headers: dict[str, str]) -> httpx.Response:
    """GET one probe URL (module seam; tests stub here)."""
    return httpx.get(url, headers=headers, timeout=KEY_CHECK_TIMEOUT_S)


def _sendable(api_key: str) -> bool:
    """True when the key can travel in an HTTP header: printable ASCII only."""
    return api_key.isascii() and api_key.isprintable()


def _probe_status(provider: str, api_key: str) -> KeyStatus:
    """Send one probe and read the verdict from its status."""
    probe = _PROBES[provider]
    resp = _http_get(probe.url, headers=probe.headers(api_key))
    if probe.rejected(resp):
        return KeyStatus.INVALID_KEY
    if resp.is_error:
        log.warning("Could not verify the %s API key: HTTP %d", provider, resp.status_code)
    return KeyStatus.READY


@cached(
    TTLCache(maxsize=len(_PROBES) * _KEYS_CACHED_PER_PROVIDER, ttl=KEY_CHECK_TTL_S),
    lock=_probe_condition,
    condition=_probe_condition,
)
def _checked_key_status(provider: str, api_key: str) -> KeyStatus:
    """INVALID_KEY for a key that cannot be sent or is refused; an unanswered check stays READY."""
    if not _sendable(api_key):
        return KeyStatus.INVALID_KEY
    try:
        return _probe_status(provider, api_key)
    except Exception as exc:  # a failed check must never fail the catalog or chat routing
        log.warning("Could not verify the %s API key: %s", provider, type(exc).__name__)
        return KeyStatus.READY


def _key_in_use(provider: str) -> str | None:
    """The key the SDK sends: its env var when set, else the lilbee config field."""
    return os.environ.get(PROVIDER_API_KEY_ENV[provider]) or get_provider_api_key(provider)


def provider_key_set(provider: str) -> bool:
    """True when *provider* has an API key in its env var or the lilbee config."""
    return _key_in_use(provider) is not None


def provider_key_status(provider: str) -> KeyStatus:
    """Whether *provider*'s API key is missing, rejected, or usable."""
    api_key = _key_in_use(provider)
    if api_key is None:
        return KeyStatus.MISSING_KEY
    return _checked_key_status(provider, api_key)


def provider_key_statuses(providers: list[str]) -> dict[str, KeyStatus]:
    """Key status per provider, the uncached checks running concurrently."""
    if not providers:
        return {}
    with ThreadPoolExecutor(max_workers=len(providers)) as pool:
        return dict(zip(providers, pool.map(provider_key_status, providers), strict=True))
