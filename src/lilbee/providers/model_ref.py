"""Model reference parsing and option translation.

Single source of truth for classifying model strings and translating
generation options per provider type. This module must NOT import from
lilbee.config or lilbee.models to avoid circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

from lilbee.catalog.refs import is_gguf_filename, is_native_gguf_shape, model_format
from lilbee.catalog.types import ModelFormat, ModelSource
from lilbee.providers.base import (
    ProviderError,
    ProviderErrorKind,
    filter_options,
    normalize_generation_options,
)
from lilbee.providers.local_servers import local_server_for_key, local_server_for_label
from lilbee.providers.server_registry import (
    API_PROVIDERS,
    BUILTIN_PREFIXES,
    LOCAL_PROVIDER,
    route_for,
)

OLLAMA_NO_THINKING = "none"


def is_native_gguf_ref(raw: str) -> bool:
    """True when *raw* is a GGUF with the ``<org>/<repo>/<file>.gguf`` shape or an absolute path.

    An absolute Windows path carries no forward slash, so ``C:\\models\\x.gguf``
    classifies like ``/models/x.gguf``. ``PureWindowsPath`` reads Windows roots
    on a POSIX host and answers the same on every supported Python, and it
    keeps a drive-relative ``C:models\\x.gguf`` out: the fleet cannot load it.

    The suffix check is case-sensitive on purpose: repo extraction
    (:func:`lilbee.catalog.refs.hf_repo_from_ref`) only recognises the
    lowercase ``.gguf`` suffix, and classification must agree with it.
    """
    return is_native_gguf_shape(raw) or (
        is_gguf_filename(raw) and PureWindowsPath(raw).is_absolute()
    )


def is_loose_model_file(raw: str) -> bool:
    """True when *raw* is an absolute path to an existing file, loadable without a manifest.

    A directory answers False. It exists, but the engine cannot load it, and
    every caller here treats a True as "this ref serves as it stands".
    """
    candidate = Path(raw)
    return candidate.is_absolute() and candidate.is_file()


def routes_to_native_gguf(raw: str) -> bool:
    """True when *raw* is a native GGUF shape not claimed by a local-server prefix.

    Local-server prefixes (``ollama/``, ``lm_studio/``) are exempt from the
    shape rule: those servers report model ids that can themselves look like
    GGUF paths, so the prefix wins over the shape.
    """
    return not _has_local_server_prefix(raw) and is_native_gguf_ref(raw)


def routes_to_native_model(raw: str, *, read_presets: bool = False) -> bool:
    """True when *raw* is a native model in any format.

    A GGUF shape follows :func:`routes_to_native_gguf`. A ``#<format>`` ref
    routes natively unless its prefix names a provider: ``openai/x#mlx`` and
    ``ollama/x#mlx`` keep their provider. A malformed format qualifier is not a
    native ref. With *read_presets*, a stored server preset name is a provider.
    """
    if routes_to_native_gguf(raw):
        return True
    if model_format(raw) == ModelFormat.GGUF:
        return False
    prefix = raw.split("/", 1)[0]
    return prefix not in BUILTIN_PREFIXES and not (read_presets and route_for(prefix))


def _has_local_server_prefix(raw: str) -> bool:
    return local_server_for_key(raw.split("/", 1)[0]) is not None


@dataclass(frozen=True)
class ProviderModelRef:
    """Parsed model reference with provider routing information."""

    raw: str
    provider: str  # LOCAL_PROVIDER, a built-in prefix or the name of a server preset
    name: str  # provider-specific name with tag normalization applied

    @property
    def is_api(self) -> bool:
        return self.provider in API_PROVIDERS

    @property
    def is_local(self) -> bool:
        return self.provider == LOCAL_PROVIDER

    @property
    def is_preset(self) -> bool:
        """True when the provider is a stored server preset rather than a built-in prefix."""
        return self.is_remote and not self.is_api and local_server_for_key(self.provider) is None

    @property
    def is_remote(self) -> bool:
        """True if this model routes through a remote SDK (any non-``local`` provider)."""
        return self.provider != LOCAL_PROVIDER

    def for_openai_prefix(self) -> str:
        """Name with its canonical ``provider/model`` prefix (``ollama/llama3.2:1b``)."""
        spec = local_server_for_key(self.provider)
        if spec is not None:
            return spec.qualify(self.name)
        if self.is_remote:
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
    their lowercase key unchanged. A label that is no built-in routing key
    (``"Remote"``) adds no prefix: the name stays as the server reported it.
    """
    spec = local_server_for_label(provider)
    key = spec.key if spec is not None else provider.lower()
    if key not in BUILTIN_PREFIXES:
        return name
    return ProviderModelRef(raw=name, provider=key, name=name).for_openai_prefix()


def canonical_model_ref(raw: str) -> str:
    """The canonical text of a model ref, decided from the text alone.

    A ref that names a stored server preset and a ref that names none both keep
    the text as written, so this reads no stored preset. The config validators call it.
    """
    return _classify(raw, read_presets=False).for_openai_prefix()


def parse_model_ref(raw: str) -> ProviderModelRef:
    """Classify a model string and return the routing ref, native shape first.

    Native HuggingFace refs are ``<org>/<repo>/<file>.gguf``; that shape
    routes locally even when the org collides with an API provider prefix
    (``openai/``, ``mistral/``, ``deepseek/`` are real HF orgs). Local-server
    prefixes (``ollama/``, ``lm_studio/``) are exempt from the shape rule:
    those servers report model ids that can themselves look like GGUF paths
    (LM Studio 0.2.x uses full relative GGUF paths), so the prefix wins there.
    Remote providers use a built-in prefix or the name of a stored server preset.
    """
    return _classify(raw, read_presets=True)


def _classify(raw: str, *, read_presets: bool) -> ProviderModelRef:
    if routes_to_native_model(raw, read_presets=read_presets):
        return ProviderModelRef(raw=raw, provider=LOCAL_PROVIDER, name=raw)
    if "/" not in raw:
        known = ", ".join(f"{p}/" for p in sorted(BUILTIN_PREFIXES))
        raise ValueError(
            f"Model ref {raw!r} must be a HuggingFace ref "
            f"('<org>/<repo>/<filename>.gguf') or carry a known provider prefix ({known}) "
            "or the name of a server preset."
        )
    prefix, rest = raw.split("/", 1)
    if prefix in API_PROVIDERS:
        return ProviderModelRef(raw=raw, provider=prefix, name=rest)
    spec = local_server_for_key(prefix)
    if spec is not None:
        return ProviderModelRef(raw=raw, provider=spec.key, name=spec.normalize_name(rest))
    route = route_for(prefix) if read_presets else None
    if route is not None:
        return ProviderModelRef(raw=raw, provider=route.name, name=rest)
    return ProviderModelRef(raw=raw, provider=LOCAL_PROVIDER, name=raw)


def default_first(refs: list[str], default_ref: str) -> list[str]:
    """Order so *default_ref* leads, leaving the rest in their existing order."""
    if default_ref not in refs:
        return list(refs)
    return [default_ref, *(ref for ref in refs if ref != default_ref)]


# Samplers the Anthropic API has no field for.
_ANTHROPIC_UNSUPPORTED = ("seed", "frequency_penalty", "presence_penalty")


def _refuse_unsupported(
    options: dict[str, Any], ref: ProviderModelRef, unsupported: tuple[str, ...]
) -> None:
    """Raise a bad-request error when *options* sets a name in *unsupported* to any value."""
    refused = [name for name in unsupported if options.get(name) is not None]
    if refused:
        raise ProviderError(
            f"The provider rejected the request for {ref.raw}. "
            f"It does not support {', '.join(refused)}. Remove the setting and try again.",
            provider=ModelSource.REMOTE,
            kind=ProviderErrorKind.BAD_REQUEST,
        )


def translate_options(options: dict[str, Any], ref: ProviderModelRef) -> dict[str, Any]:
    """Translate generation options for the target provider.

    A local ref forced through the SDK keeps the raw filtered options; an API ref
    gets the shared per-call mapping (``num_predict`` -> ``max_tokens``, drop
    ``num_ctx``) plus a ``top_k`` and ``repeat_penalty`` drop: hosted APIs do not
    take either, so dropping them keeps the wire request clean. An Anthropic ref
    refuses a ``seed`` or penalty set to any value, zero included, with a bad-request
    error; when ``response_format`` is set it drops them and sends the request.
    """
    if not ref.is_api:
        filtered = filter_options(options)
        think = filtered.pop("think", None)
        if think is False and ref.provider == ModelSource.OLLAMA:
            # Ollama maps reasoning_effort onto its think field; any value
            # outside low/medium/high turns thinking off.
            filtered["reasoning_effort"] = OLLAMA_NO_THINKING
        return filtered
    api_options = normalize_generation_options(options)
    if ref.provider == "anthropic":
        if "response_format" not in api_options:
            _refuse_unsupported(api_options, ref, _ANTHROPIC_UNSUPPORTED)
        for name in _ANTHROPIC_UNSUPPORTED:
            api_options.pop(name, None)
    api_options.pop("top_k", None)
    api_options.pop("repeat_penalty", None)
    api_options.pop("think", None)
    return api_options
