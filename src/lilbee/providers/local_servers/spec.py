"""The LocalServerSpec type: identity and capabilities of one local server."""

from __future__ import annotations

from dataclasses import dataclass

from lilbee.providers.backend_names import BackendName


@dataclass(frozen=True)
class LocalServerSpec:
    """Identity and capabilities of one local model server.

    ``key`` doubles as the model-source string: ``ModelSource(spec.key)`` is
    the source these models report as (asserted by an invariant test). The
    type lives in ``lilbee.catalog`` and is not imported here to keep this
    module light for the routing layer.
    """

    key: str  # routing key, ref-prefix stem, and ModelSource value ("ollama")
    display_name: BackendName  # human-facing backend name
    wire_prefix: str  # litellm provider/ prefix ("ollama/")
    default_base_url: str  # URL the server listens on out of the box
    url_patterns: tuple[str, ...]  # lowercase substrings that identify it in a URL
    appends_latest_tag: bool  # bare name gets a ":latest" tag (Ollama)
    supports_pull: bool  # exposes an HTTP model-pull endpoint
    supports_show: bool  # exposes an HTTP model-metadata endpoint

    def qualify(self, name: str) -> str:
        """Render *name* as a canonical ``<wire_prefix><name>`` ref."""
        return f"{self.wire_prefix}{name}"

    def normalize_name(self, name: str) -> str:
        """Apply the server's bare-name convention (Ollama's ``:latest``)."""
        if self.appends_latest_tag and ":" not in name:
            return f"{name}:latest"
        return name
