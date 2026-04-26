"""Model reference parsing and option translation.

Single source of truth for classifying model strings and translating
generation options per provider type. This module must NOT import from
lilbee.config or lilbee.models to avoid circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lilbee.providers.base import filter_options

_API_PROVIDERS = {"openai", "anthropic", "gemini"}

# All provider prefixes that route a ref away from the local registry.
# Includes API providers and ollama (which keeps its own name:tag shape).
PROVIDER_PREFIXES: frozenset[str] = frozenset(_API_PROVIDERS | {"ollama"})

OLLAMA_PREFIX = "ollama/"


@dataclass(frozen=True)
class ProviderModelRef:
    """Parsed model reference with provider routing information."""

    raw: str
    provider: str  # "local", "ollama", "openai", "anthropic", "gemini"
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
        if self.provider == "ollama":
            return f"{OLLAMA_PREFIX}{self.name}"
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


def parse_model_ref(raw: str) -> ProviderModelRef:
    """Parse a model string into a ProviderModelRef.

    Classifies model strings by prefix:
    - ``openai/gpt-4o`` -> API provider
    - ``anthropic/claude-sonnet-4-20250514`` -> API provider
    - ``ollama/llama3.2:1b`` -> Ollama provider (keeps its own ``name:tag``)
    - ``<org>/<repo>/<file>.gguf`` -> local HuggingFace native model
    - ``<org>/<repo>`` -> local, repo-only ref (filename resolved later)
    - Any other ``name:tag`` shape is rejected as a legacy ref.
    """
    if "/" in raw:
        prefix, rest = raw.split("/", 1)
        if prefix in _API_PROVIDERS:
            return ProviderModelRef(raw=raw, provider=prefix, name=rest)
        if prefix == "ollama":
            name = rest if ":" in rest else f"{rest}:latest"
            return ProviderModelRef(raw=raw, provider="ollama", name=name)
        return ProviderModelRef(raw=raw, provider="local", name=raw)
    if ":" in raw:
        raise ValueError(
            f"Legacy model ref {raw!r} is no longer supported. "
            "Use the HuggingFace shape '<org>/<repo>/<filename>.gguf'. "
            "See release notes for the upgrade path."
        )
    raise ValueError(
        f"Model ref {raw!r} is not recognized. Native models use "
        "'<org>/<repo>/<filename>.gguf'; remote models use a provider "
        "prefix like 'ollama/' or 'openai/'."
    )


def translate_options(options: dict[str, Any], ref: ProviderModelRef) -> dict[str, Any]:
    """Translate generation options for the target provider."""
    filtered = filter_options(options)
    if ref.is_api:
        # API providers use max_tokens, not num_predict
        if "num_predict" in filtered:
            filtered["max_tokens"] = filtered.pop("num_predict")
        # num_ctx is a model-load param, not per-call
        filtered.pop("num_ctx", None)
        # top_k not supported by most API providers
        filtered.pop("top_k", None)
    return filtered
