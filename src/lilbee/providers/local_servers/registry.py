"""Registry of known local servers and lookup helpers over it."""

from __future__ import annotations

from lilbee.providers.local_servers.lm_studio import LM_STUDIO
from lilbee.providers.local_servers.ollama import OLLAMA
from lilbee.providers.local_servers.spec import LocalServerSpec

LOCAL_SERVERS: tuple[LocalServerSpec, ...] = (OLLAMA, LM_STUDIO)

LOCAL_SERVER_KEYS: frozenset[str] = frozenset(spec.key for spec in LOCAL_SERVERS)


def openai_models_url(base_url: str) -> str:
    """Build the ``/v1/models`` URL, tolerating a base that already ends in ``/v1``."""
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/v1"):
        trimmed = trimmed[: -len("/v1")]
    return f"{trimmed}/v1/models"


def detect_local_server(base_url: str) -> LocalServerSpec | None:
    """Return the local server whose URL patterns match *base_url*, if any."""
    url_lower = base_url.lower()
    for spec in LOCAL_SERVERS:
        if any(pattern in url_lower for pattern in spec.url_patterns):
            return spec
    return None


def local_server_for_key(key: str) -> LocalServerSpec | None:
    """Return the spec with routing *key*, or ``None``."""
    for spec in LOCAL_SERVERS:
        if spec.key == key:
            return spec
    return None


def canonical_local_ref(name: str, source_key: str) -> str:
    """Prefix a bare *name* with its local server's wire prefix.

    *source_key* is a ModelSource value (``"ollama"``). No-op for non-local
    sources and for names already carrying the prefix, so callers can dedup
    the installed and catalog views.
    """
    spec = local_server_for_key(source_key)
    if spec is None or name.startswith(spec.wire_prefix):
        return name
    return spec.qualify(name)


def local_server_for_label(label: str) -> LocalServerSpec | None:
    """Return the spec matching *label* by routing key or display name.

    Discovery stamps the display value (``"LM Studio"``); a parsed ref carries
    the routing key (``"lm_studio"``). Both resolve to the same spec.
    """
    lowered = label.lower()
    for spec in LOCAL_SERVERS:
        if spec.key == lowered or spec.display_name.lower() == lowered:
            return spec
    return None
