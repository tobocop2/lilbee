"""Resolve each local server's configured base URL, with spec-default fallback."""

from __future__ import annotations

from collections.abc import Callable

from lilbee.core.config.model import cfg
from lilbee.providers.local_servers.lm_studio import LM_STUDIO
from lilbee.providers.local_servers.ollama import OLLAMA
from lilbee.providers.local_servers.registry import LOCAL_SERVERS, local_server_for_key
from lilbee.providers.local_servers.spec import LocalServerSpec

_URL_ACCESSOR_BY_KEY: dict[str, Callable[[], str]] = {
    OLLAMA.key: lambda: cfg.ollama_base_url,
    LM_STUDIO.key: lambda: cfg.lm_studio_base_url,
}


def _configured_url(spec: LocalServerSpec) -> str:
    return _URL_ACCESSOR_BY_KEY[spec.key]().strip() or spec.default_base_url


def base_url_for(server_key: str) -> str:
    """Configured URL for a server key, falling back to its spec default."""
    spec = local_server_for_key(server_key)
    if spec is None:
        raise KeyError(f"Unknown local server: {server_key!r}")
    return _configured_url(spec)


def configured_local_servers() -> list[tuple[LocalServerSpec, str]]:
    """Return (spec, resolved-url) for every known local server."""
    return [(spec, _configured_url(spec)) for spec in LOCAL_SERVERS]
