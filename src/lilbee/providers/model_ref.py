"""Model reference parsing and option translation.

Single source of truth for classifying model strings and translating
generation options per provider type. This module must NOT import from
lilbee.config or lilbee.models to avoid circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lilbee.providers.base import filter_options
from lilbee.providers.local_servers import (
    LOCAL_SERVER_KEYS,
    local_server_for_key,
    local_server_for_label,
)

_API_PROVIDERS = frozenset(
    {
        "openrouter",
        "gemini",
        "anthropic",
        "openai",
        "mistral",
        "deepseek",
    }
)

# All provider prefixes that route a ref away from the local registry:
# API providers plus the local OpenAI-compatible servers (ollama, lm_studio).
PROVIDER_PREFIXES: frozenset[str] = frozenset(_API_PROVIDERS | LOCAL_SERVER_KEYS)


@dataclass(frozen=True)
class ProviderModelRef:
    """Parsed model reference with provider routing information."""

    raw: str
    provider: str  # "local" or any value in PROVIDER_PREFIXES
    name: str  # provider-specific name with tag normalization applied

    @property
    def is_api(self) -> bool:
        return self.provider in _API_PROVIDERS

    @property
    def is_local(self) -> bool:
        return self.provider == "local"

    @property
    def is_remote(self) -> bool:
        """True if this model must route through a remote SDK (API or Ollama).

        Remote means "not a locally-loaded GGUF". Both Ollama (HTTP
        localhost server) and hosted API providers share the same
        dispatch path; they go through whichever SDK backend is wired
        up.
        """
        return self.provider != "local"

    def for_openai_prefix(self) -> str:
        """Name formatted with canonical ``provider/model`` prefix.

        The prefix convention is the same one used by OpenAI-compatible
        SDKs: ``openai/gpt-4o``, ``ollama/llama3.2:1b``, etc. Every
        dispatching SDK accepts this shape.
        """
        spec = local_server_for_key(self.provider)
        if spec is not None:
            return spec.qualify(self.name)
        if self.is_api:
            return f"{self.provider}/{self.name}"
        return self.name

    def for_display(self) -> str:
        """Human-readable name for UI."""
        return self.raw

    @property
    def needs_api_base(self) -> bool:
        """True if the SDK needs an explicit api_base (Ollama/local)."""
        return not self.is_api


def format_remote_ref(name: str, provider: str) -> str:
    """Render a remote model as a canonical ``provider/name`` ref.

    *provider* may be a routing key (``"ollama"``) or a backend display
    name (``"LM Studio"``); local-server labels are normalised to the
    routing key so the prefix survives. API providers fall through to
    their lowercase key unchanged.
    """
    spec = local_server_for_label(provider)
    key = spec.key if spec is not None else provider.lower()
    return ProviderModelRef(raw=name, provider=key, name=name).for_openai_prefix()


def parse_model_ref(raw: str) -> ProviderModelRef:
    """Classify a model string by its prefix and return the routing ref.

    Native HuggingFace refs are ``<org>/<repo>/<file>.gguf``. Remote
    providers use prefixes from :data:`PROVIDER_PREFIXES` (the local
    servers ``ollama/`` and ``lm_studio/`` plus every API provider).
    """
    if "/" not in raw:
        known = ", ".join(f"{p}/" for p in sorted(PROVIDER_PREFIXES))
        raise ValueError(
            f"Model ref {raw!r} must be a HuggingFace ref "
            f"('<org>/<repo>/<filename>.gguf') or carry a known provider prefix ({known})."
        )
    prefix, rest = raw.split("/", 1)
    if prefix in _API_PROVIDERS:
        return ProviderModelRef(raw=raw, provider=prefix, name=rest)
    spec = local_server_for_key(prefix)
    if spec is not None:
        return ProviderModelRef(raw=raw, provider=spec.key, name=spec.normalize_name(rest))
    return ProviderModelRef(raw=raw, provider="local", name=raw)


def translate_options(options: dict[str, Any], ref: ProviderModelRef) -> dict[str, Any]:
    """Translate generation options for the target provider."""
    filtered = filter_options(options)
    if ref.is_api:
        # API providers use max_tokens, not num_predict
        if "num_predict" in filtered:
            filtered["max_tokens"] = filtered.pop("num_predict")
        # num_ctx is a model-load param, not per-call
        filtered.pop("num_ctx", None)
        # top_k kept for local llama.cpp, stripped for API providers: litellm
        # forwards it (into extra_body for OpenAI-compatible) without erroring,
        # but hosted APIs ignore it, so dropping it keeps the wire request clean.
        filtered.pop("top_k", None)
    return filtered
