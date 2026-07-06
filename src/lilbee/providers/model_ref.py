"""Model reference parsing and option translation.

Single source of truth for classifying model strings and translating
generation options per provider type. This module must NOT import from
lilbee.config or lilbee.models to avoid circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lilbee.catalog.refs import NATIVE_GGUF_REF_MIN_SLASHES
from lilbee.providers.base import filter_options, normalize_generation_options
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

# Provider value for refs served from the local registry (native GGUF).
LOCAL_PROVIDER = "local"


def is_native_gguf_ref(raw: str) -> bool:
    """True when *raw* has the native HuggingFace GGUF shape ``<org>/<repo>/<file>.gguf``.

    The suffix check is case-sensitive on purpose: repo extraction
    (:func:`lilbee.catalog.refs.hf_repo_from_ref`) only recognises the
    lowercase ``.gguf`` suffix, and classification must agree with it.
    """
    return raw.endswith(".gguf") and raw.count("/") >= NATIVE_GGUF_REF_MIN_SLASHES


def routes_to_native_gguf(raw: str) -> bool:
    """True when *raw* is a native GGUF shape not claimed by a local-server prefix.

    Local-server prefixes (``ollama/``, ``lm_studio/``) are exempt from the
    shape rule: those servers report model ids that can themselves look like
    GGUF paths, so the prefix wins over the shape.
    """
    first_segment = raw.split("/", 1)[0]
    return first_segment not in LOCAL_SERVER_KEYS and is_native_gguf_ref(raw)


@dataclass(frozen=True)
class ProviderModelRef:
    """Parsed model reference with provider routing information."""

    raw: str
    provider: str  # LOCAL_PROVIDER or any value in PROVIDER_PREFIXES
    name: str  # provider-specific name with tag normalization applied

    @property
    def is_api(self) -> bool:
        return self.provider in _API_PROVIDERS

    @property
    def is_local(self) -> bool:
        return self.provider == LOCAL_PROVIDER

    @property
    def is_remote(self) -> bool:
        """True if this model routes through a remote SDK (any non-``local`` provider)."""
        return self.provider != LOCAL_PROVIDER

    def for_openai_prefix(self) -> str:
        """Name with its canonical ``provider/model`` prefix (``ollama/llama3.2:1b``)."""
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
    """Classify a model string and return the routing ref, native shape first.

    Native HuggingFace refs are ``<org>/<repo>/<file>.gguf``; that shape
    routes locally even when the org collides with an API provider prefix
    (``openai/``, ``mistral/``, ``deepseek/`` are real HF orgs). Local-server
    prefixes (``ollama/``, ``lm_studio/``) are exempt from the shape rule:
    those servers report model ids that can themselves look like GGUF paths
    (LM Studio 0.2.x uses full relative GGUF paths), so the prefix wins there.
    Remote providers use prefixes from :data:`PROVIDER_PREFIXES`.
    """
    if routes_to_native_gguf(raw):
        return ProviderModelRef(raw=raw, provider=LOCAL_PROVIDER, name=raw)
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
    return ProviderModelRef(raw=raw, provider=LOCAL_PROVIDER, name=raw)


def default_first(refs: list[str], default_ref: str) -> list[str]:
    """Order so *default_ref* leads, leaving the rest in their existing order."""
    if default_ref not in refs:
        return list(refs)
    return [default_ref, *(ref for ref in refs if ref != default_ref)]


def with_configured_remote_chat(refs: list[str], configured: str) -> list[str]:
    """Return *refs* with *configured* prepended when it is a remote ref not already listed.

    A remote-configured chat model (``ollama/...``, ``openai/...``) is served
    through known-model resolution without appearing in the native registry;
    prepending it keeps a model listing truthful and puts the model lilbee
    actually serves first. *configured* must parse; ``cfg.chat_model`` is
    validated and canonicalized at the write boundary.
    """
    if configured in refs or not parse_model_ref(configured).is_remote:
        return list(refs)
    return [configured, *refs]


def translate_options(options: dict[str, Any], ref: ProviderModelRef) -> dict[str, Any]:
    """Translate generation options for the target provider.

    A local ref forced through the SDK keeps the raw filtered options; an API ref
    gets the shared per-call mapping (``num_predict`` -> ``max_tokens``, drop
    ``num_ctx``) plus a ``top_k`` drop: litellm would forward ``top_k`` (into
    ``extra_body`` for OpenAI-compatible) without erroring, but hosted APIs ignore
    it, so dropping it keeps the wire request clean.
    """
    if not ref.is_api:
        return filter_options(options)
    api_options = normalize_generation_options(options)
    api_options.pop("top_k", None)
    return api_options
