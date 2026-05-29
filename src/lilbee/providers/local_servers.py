"""Local OpenAI-compatible model servers (Ollama, LM Studio).

One abstraction over the local servers lilbee can drive through the
litellm SDK. Each server is described by a :class:`LocalServerSpec`
carrying its routing key, litellm wire prefix, default base URL, the URL
substrings that identify it, and which catalog operations it supports.

Kept dependency-free on purpose so the routing layer (``model_ref``,
``sdk_backend``) can import it without pulling in httpx or the modelhub.
The HTTP behaviour (model discovery, pull) lives in the modelhub layer,
dispatched off the spec returned here.
"""

from __future__ import annotations

from dataclasses import dataclass

from lilbee.providers.backend_names import BackendName


@dataclass(frozen=True)
class LocalServerSpec:
    """Identity and capabilities of one local model server."""

    key: str
    """Routing key and ref-prefix stem (``ollama``, ``lm_studio``)."""

    display_name: BackendName
    """Human-facing backend name shown in the UI."""

    wire_prefix: str
    """litellm ``provider/`` prefix the SDK routes on (``ollama/``)."""

    default_base_url: str
    """Base URL the server listens on out of the box."""

    url_patterns: tuple[str, ...]
    """Lowercase substrings that identify this server in a base URL."""

    appends_latest_tag: bool
    """Whether a bare model name gets a ``:latest`` tag (Ollama convention)."""

    supports_pull: bool
    """Whether the server exposes an HTTP model-pull endpoint."""

    supports_show: bool
    """Whether the server exposes an HTTP model-metadata endpoint."""

    def qualify(self, name: str) -> str:
        """Render *name* as a canonical ``<wire_prefix><name>`` ref."""
        return f"{self.wire_prefix}{name}"

    def normalize_name(self, name: str) -> str:
        """Apply the server's bare-name convention (Ollama's ``:latest``)."""
        if self.appends_latest_tag and ":" not in name:
            return f"{name}:latest"
        return name


OLLAMA = LocalServerSpec(
    key="ollama",
    display_name=BackendName.OLLAMA,
    wire_prefix="ollama/",
    default_base_url="http://localhost:11434",
    url_patterns=("localhost:11434", "127.0.0.1:11434", "ollama"),
    appends_latest_tag=True,
    supports_pull=True,
    supports_show=True,
)

LM_STUDIO = LocalServerSpec(
    # litellm's lm_studio provider posts to ``{api_base}/chat/completions``, so
    # the base URL must carry ``/v1`` (matching what LM Studio's server panel
    # shows). It also injects a placeholder key, so no API key is required.
    key="lm_studio",
    display_name=BackendName.LM_STUDIO,
    wire_prefix="lm_studio/",
    default_base_url="http://localhost:1234/v1",
    url_patterns=("localhost:1234", "127.0.0.1:1234"),
    appends_latest_tag=False,
    supports_pull=False,
    supports_show=False,
)

LOCAL_SERVERS: tuple[LocalServerSpec, ...] = (OLLAMA, LM_STUDIO)

LOCAL_SERVER_KEYS: frozenset[str] = frozenset(spec.key for spec in LOCAL_SERVERS)


def openai_models_url(base_url: str) -> str:
    """Build the ``/v1/models`` listing URL, tolerating a trailing ``/v1``.

    OpenAI-compatible servers are configured with the base that chat uses
    (``.../v1`` for LM Studio) or without it (Ollama, generic). Both resolve
    to a single ``/v1/models`` so the listing URL never doubles the segment.
    """
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


def local_server_for_label(label: str) -> LocalServerSpec | None:
    """Return the spec matching *label* by routing key or display name.

    Callers pass either form: discovery stamps the ``BackendName`` display
    value (``"LM Studio"``) onto a model, while a parsed ref carries the
    routing key (``"lm_studio"``). Both resolve to the same spec.
    """
    lowered = label.lower()
    for spec in LOCAL_SERVERS:
        if spec.key == lowered or spec.display_name.lower() == lowered:
            return spec
    return None
