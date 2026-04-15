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
    def needs_litellm(self) -> bool:
        """True if this model must route through litellm (API or Ollama)."""
        return self.provider != "local"

    def for_litellm(self) -> str:
        """Name formatted for litellm dispatch."""
        if self.provider == "ollama":
            return f"ollama/{self.name}"
        if self.is_api:
            return f"{self.provider}/{self.name}"
        return self.name

    def for_display(self) -> str:
        """Human-readable name for UI."""
        return self.raw

    @property
    def needs_api_base(self) -> bool:
        """True if litellm needs an explicit api_base (Ollama/local)."""
        return not self.is_api


def parse_model_ref(raw: str) -> ProviderModelRef:
    """Parse a model string into a ProviderModelRef.

    Classifies model strings by prefix:
    - ``openai/gpt-4o`` -> API provider, no tag normalization
    - ``anthropic/claude-sonnet-4-20250514`` -> API provider
    - ``ollama/qwen3:8b`` -> Ollama provider
    - ``org/model-name`` -> local (HuggingFace-style), tag normalization applied
    - ``qwen3:8b`` -> local, already has tag
    - ``qwen3`` -> local, ``:latest`` appended
    """
    if "/" in raw:
        prefix, rest = raw.split("/", 1)
        if prefix in _API_PROVIDERS:
            return ProviderModelRef(raw=raw, provider=prefix, name=rest)
        if prefix == "ollama":
            name = rest if ":" in rest else f"{rest}:latest"
            return ProviderModelRef(raw=raw, provider="ollama", name=name)
        # Unknown prefix (HuggingFace org/model). Treat as local.
        name = raw if ":" in raw else f"{raw}:latest"
        return ProviderModelRef(raw=raw, provider="local", name=name)
    # No prefix = local model. Add :latest if no tag.
    name = raw if ":" in raw else f"{raw}:latest"
    return ProviderModelRef(raw=raw, provider="local", name=name)


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
