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

from lilbee.catalog.types import ModelTask
from lilbee.core.config import cfg
from lilbee.modelhub.model_manager.discovery import (
    classify_remote_models,
    discover_api_models,
    reclassify_by_name,
)
from lilbee.modelhub.model_manager.types import ValidationResult
from lilbee.modelhub.registry import ModelRegistry
from lilbee.providers.local_servers.registry import LOCAL_SERVER_KEYS
from lilbee.providers.model_ref import ProviderModelRef, format_remote_ref, parse_model_ref
from lilbee.providers.sdk_backend import PROVIDER_API_KEY_FIELD, get_provider_api_key

log = logging.getLogger(__name__)

# Surface-neutral reasons explaining why a persisted ref is unusable, so
# every surface (TUI toast, server log, CLI) can show the same wording.
REASON_LITELLM_MISSING = "the litellm extra isn't installed; run pip install 'lilbee[litellm]'"
REASON_SERVER_UNREACHABLE = "the model server at {base_url} isn't reachable"
REASON_NO_API_KEY = "no API key is configured for {provider}"
REASON_NOT_INSTALLED = "it isn't installed"
REASON_UNAVAILABLE = "it isn't available"

# Local-server reachability probe budget. Short so a down server does not
# stall startup; the probe only runs for ollama/lm_studio refs.
_PROBE_TIMEOUT_S = 1.0


@dataclass(frozen=True)
class CanonicalRef:
    """Result of canonicalizing a persisted ref.

    ``effective`` is what callers should use this session. ``original``
    is what the user persisted; if it differs from ``effective`` the
    caller should surface the swap. ``reason`` is a human-readable
    explanation of why ``original`` was unusable, set whenever
    ``status`` is not ``OK``.
    """

    original: str
    effective: str
    status: ValidationResult
    reason: str | None = None


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


def _local_server_reachable(base_url: str) -> bool:
    """True if the local model server lists at least one model within the probe budget."""
    try:
        return bool(classify_remote_models(base_url, timeout=_PROBE_TIMEOUT_S))
    except Exception:
        log.debug("Local model server probe failed for %r", base_url, exc_info=True)
        return False


def _classify_local_server_ref() -> tuple[ValidationResult, str | None]:
    """Classify an ollama/lm_studio ref: needs the litellm extra and a live server."""
    from lilbee.providers.litellm_sdk import litellm_available

    if not litellm_available():
        return ValidationResult.UNKNOWN, REASON_LITELLM_MISSING
    if not _local_server_reachable(cfg.remote_base_url):
        return ValidationResult.UNKNOWN, REASON_SERVER_UNREACHABLE.format(
            base_url=cfg.remote_base_url
        )
    return ValidationResult.OK, None


def _classify_uninstalled_ref(parsed: ProviderModelRef) -> tuple[ValidationResult, str | None]:
    """Classify a parsed ref that is not installed locally, by provider kind."""
    provider = (parsed.provider or "").lower()
    if provider in LOCAL_SERVER_KEYS:
        return _classify_local_server_ref()
    if provider in PROVIDER_API_KEY_FIELD:
        if get_provider_api_key(provider):
            return ValidationResult.OK, None
        return ValidationResult.NO_KEY, REASON_NO_API_KEY.format(provider=provider)
    if not parsed.is_remote:
        # A native GGUF ref that no longer resolves to a file on disk.
        return ValidationResult.NOT_INSTALLED, REASON_NOT_INSTALLED
    return ValidationResult.UNKNOWN, REASON_UNAVAILABLE


def _classify_ref(ref: str) -> tuple[ValidationResult, str | None]:
    """Classify a persisted ref, returning its status and a human-readable reason.

    Reads cfg, the local registry, and (for ollama/lm_studio refs) probes
    the configured model server. Never mutates persisted state.
    """
    if not ref:
        return ValidationResult.UNKNOWN, REASON_UNAVAILABLE
    if _is_local_installed(ref):
        return ValidationResult.OK, None
    try:
        parsed = parse_model_ref(ref)
    except Exception:
        return ValidationResult.UNKNOWN, REASON_UNAVAILABLE
    return _classify_uninstalled_ref(parsed)


def validate_persisted_model(ref: str) -> ValidationResult:
    """Classify a persisted chat/embedding ref against current state."""
    status, _reason = _classify_ref(ref)
    return status


def _first_available_api_chat_ref() -> str | None:
    """Return the first cloud chat ref backed by a configured API key, or ``None``."""
    try:
        groups = discover_api_models()
    except Exception:
        log.debug("discover_api_models failed during canonicalization", exc_info=True)
        return None
    for _provider, models in groups.items():
        if models:
            first = models[0]
            return format_remote_ref(first.name, first.provider)
    return None


def _first_installed_local_ref(want: ModelTask) -> str | None:
    """Return the first installed local ref whose task matches *want*.

    The task is name-reclassified before comparison so the pick agrees
    with the role validator that runs on the subsequent swap; falling
    back to a wrong-task model (e.g. a chat model for the embedding
    slot) would be rejected downstream and is never a valid candidate.
    """
    try:
        registry = ModelRegistry(cfg.models_dir)
        installed = list(registry.list_installed())
    except Exception:
        log.debug("Local registry probe failed during canonicalization", exc_info=True)
        return None
    for manifest in installed:
        if reclassify_by_name(manifest.ref, manifest.task) == want:
            return manifest.ref
    return None


def _canonicalize(original: str, *, allow_api: bool, want_task: ModelTask) -> CanonicalRef:
    """Resolve a persisted ref to its effective session value.

    ``allow_api`` controls the fallback chain: chat allows an API
    fallback first; embedding is local-only because most providers
    have no embedding equivalent. The local fallback is restricted to
    installed models whose task is ``want_task``.
    """
    status, reason = _classify_ref(original)
    if status == ValidationResult.OK:
        return CanonicalRef(original=original, effective=original, status=status)
    candidates: list[str | None] = []
    if allow_api:
        candidates.append(_first_available_api_chat_ref())
    candidates.append(_first_installed_local_ref(want_task))
    effective = next((c for c in candidates if c), original)
    return CanonicalRef(original=original, effective=effective, status=status, reason=reason)


def canonicalize_chat_model() -> CanonicalRef:
    """Effective chat ref for this session, falling back API -> local -> original."""
    return _canonicalize(cfg.chat_model, allow_api=True, want_task=ModelTask.CHAT)


def canonicalize_embedding_model() -> CanonicalRef:
    """Effective embedding ref for this session, falling back local -> original."""
    return _canonicalize(cfg.embedding_model, allow_api=False, want_task=ModelTask.EMBEDDING)
