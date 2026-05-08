"""Validate persisted chat/embedding refs against current installation state.

Persisted refs in ``~/.lilbee/config.toml`` (or any other config source)
become stale when the user removes a GGUF, swaps providers, or moves
between machines. The TUI / server / CLI all read these refs at startup
and should not get a "model not found" error from the very first prompt.

The helpers here are pure and side-effect-free: callers decide what to
do with the result (swap in-memory ``cfg`` field, surface a banner, log
a warning, etc.). The persisted file is never rewritten, so the user's
declared intent is preserved across reinstalls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from lilbee.core.config import cfg
from lilbee.modelhub.model_manager.discovery import discover_api_models
from lilbee.modelhub.model_manager.types import ValidationResult
from lilbee.modelhub.registry import ModelRegistry
from lilbee.providers.model_ref import format_remote_ref, parse_model_ref
from lilbee.providers.sdk_backend import PROVIDER_API_KEY_FIELD, get_provider_api_key

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CanonicalRef:
    """Result of canonicalizing a persisted ref.

    ``effective`` is what callers should use this session. ``original``
    is what the user persisted; if it differs from ``effective`` the
    caller should surface the swap.
    """

    original: str
    effective: str
    status: ValidationResult


def _is_local_installed(ref: str) -> bool:
    """True iff ``ref`` resolves to an installed GGUF in the local registry."""
    try:
        registry = ModelRegistry(cfg.models_dir)
        installed = {m.ref for m in registry.list_installed()} | {
            m.hf_repo for m in registry.list_installed()
        }
        return ref in installed
    except Exception:  # pragma: no cover - defensive for fresh installs
        log.debug("Local registry probe failed for %r", ref, exc_info=True)
        return False


def _is_api_ref_with_key(ref: str) -> ValidationResult:
    """Classify a non-local ref. Returns OK / NO_KEY / UNKNOWN.

    OK when the parsed provider matches a configured API key. NO_KEY
    when the provider is recognized but the user hasn't set the key.
    UNKNOWN for malformed strings or unfamiliar providers.
    """
    try:
        parsed = parse_model_ref(ref)
    except Exception:
        return ValidationResult.UNKNOWN
    provider = (parsed.provider or "").lower()
    if provider not in PROVIDER_API_KEY_FIELD:
        return ValidationResult.UNKNOWN
    return ValidationResult.OK if get_provider_api_key(provider) else ValidationResult.NO_KEY


def validate_persisted_model(ref: str) -> ValidationResult:
    """Classify a persisted chat/embedding ref against current state.

    Pure function. Reads cfg and the local registry; never mutates.
    """
    if not ref:
        return ValidationResult.UNKNOWN
    if _is_local_installed(ref):
        return ValidationResult.OK
    return _is_api_ref_with_key(ref)


def _first_available_api_chat_ref() -> str | None:
    """Return the first cloud chat ref backed by a configured API key, or ``None``."""
    try:
        groups = discover_api_models()
    except Exception:
        log.debug("discover_api_models failed during canonicalization", exc_info=True)
        return None
    for _provider, models in groups.items():
        if models:
            return format_remote_ref(models[0])
    return None


def _first_installed_local_ref() -> str | None:
    """Return the first installed local ref, or ``None`` if none."""
    try:
        registry = ModelRegistry(cfg.models_dir)
        installed = list(registry.list_installed())
    except Exception:
        log.debug("Local registry probe failed during canonicalization", exc_info=True)
        return None
    return installed[0].ref if installed else None


def _canonicalize(original: str, *, allow_api: bool) -> CanonicalRef:
    """Resolve a persisted ref to its effective session value.

    ``allow_api`` controls the fallback chain: chat allows an API
    fallback first; embedding is local-only because most providers
    have no embedding equivalent.
    """
    status = validate_persisted_model(original)
    if status == ValidationResult.OK:
        return CanonicalRef(original=original, effective=original, status=status)
    candidates: list[str | None] = []
    if allow_api:
        candidates.append(_first_available_api_chat_ref())
    candidates.append(_first_installed_local_ref())
    effective = next((c for c in candidates if c), original)
    return CanonicalRef(original=original, effective=effective, status=status)


def canonicalize_chat_model() -> CanonicalRef:
    """Effective chat ref for this session, falling back API -> local -> original."""
    return _canonicalize(cfg.chat_model, allow_api=True)


def canonicalize_embedding_model() -> CanonicalRef:
    """Effective embedding ref for this session, falling back local -> original."""
    return _canonicalize(cfg.embedding_model, allow_api=False)
