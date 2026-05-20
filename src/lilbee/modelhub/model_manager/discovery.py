"""Remote model discovery and task classification."""

import logging
import os
import time
from threading import Lock

import httpx

from lilbee.app.services import get_services
from lilbee.catalog.types import ModelTask
from lilbee.core.config.model import cfg
from lilbee.modelhub.model_manager.types import RemoteModel
from lilbee.providers.model_ref import format_remote_ref
from lilbee.providers.sdk_backend import (
    PROVIDER_KEYS,
    detect_backend_name,
)

log = logging.getLogger(__name__)

_EMBEDDING_FAMILIES = frozenset({"bert", "nomic-bert", "e5", "bge"})
_VISION_NAME_PATTERNS = frozenset({"llava", "vision", "moondream", "ocr", "minicpm-v"})
# Reranker detection runs BEFORE embedding detection so ``bge-reranker-*``
# (family "bge" but clearly a reranker) does not get misclassified as
# EMBEDDING. ``cross-encoder`` covers the SBERT-style naming convention
# for cross-encoder rerankers.
_RERANKER_NAME_PATTERNS = frozenset({"reranker", "rerank", "cross-encoder"})

_CLASSIFY_DEFAULT_TIMEOUT_S = 5.0


def _classify_remote_task(name: str, family: str) -> ModelTask:
    """Classify a remote model as chat, embedding, vision, or rerank.

    Reranker detection runs first so ``bge-reranker-base`` (family
    ``bge``) does not get dragged into the embedding bucket by the
    family check. After reranker: embedding by family tag, vision by
    name pattern, else chat.
    """
    name_lower = name.lower()
    if any(rp in name_lower for rp in _RERANKER_NAME_PATTERNS):
        return ModelTask.RERANK
    family_lower = family.lower()
    if any(ef in family_lower for ef in _EMBEDDING_FAMILIES):
        return ModelTask.EMBEDDING
    if any(vp in name_lower for vp in _VISION_NAME_PATTERNS):
        return ModelTask.VISION
    return ModelTask.CHAT


def reclassify_by_name(ref: str, declared_task: str) -> str:
    """Override declared_task to RERANK / VISION when ref names a known role.

    Defends against pre-fix manifests that stored ``task="chat"`` for
    models whose ref obviously identifies them as rerankers (e.g.
    ``bge-reranker-*``) or vision loaders. The model bar uses this so a
    historical mis-tag does not surface a reranker in the chat picker.
    """
    name_lower = ref.lower()
    if any(rp in name_lower for rp in _RERANKER_NAME_PATTERNS):
        return ModelTask.RERANK
    if any(vp in name_lower for vp in _VISION_NAME_PATTERNS):
        return ModelTask.VISION
    return declared_task


def classify_remote_models(
    base_url: str = "http://localhost:11434",
    *,
    timeout: float = _CLASSIFY_DEFAULT_TIMEOUT_S,
) -> list[RemoteModel]:
    """Discover and classify all models from the SDK backend by task.

    Uses /api/tags family metadata for embedding detection and name
    patterns for reranker and vision detection. Returns an empty list
    on any error (including timeout) so callers in read-only code paths
    can stay responsive when the backend is down.
    """
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=timeout)
        resp.raise_for_status()
        raw_models = resp.json().get("models", [])
    except Exception:
        log.debug("Failed to classify remote models", exc_info=True)
        return []

    provider = detect_backend_name(base_url)
    result: list[RemoteModel] = []
    for model in raw_models:
        name = model.get("name", "")
        details = model.get("details", {})
        family = details.get("family", "")
        param_size = details.get("parameter_size", "")
        task = _classify_remote_task(name, family)
        result.append(
            RemoteModel(
                name=name, task=task, family=family, parameter_size=param_size, provider=provider
            )
        )
    return result


def _has_provider_key(cfg_field: str, env_var: str) -> bool:
    """Return True if a usable API key exists via env var or lilbee config."""
    if os.environ.get(env_var):
        return True
    return bool(getattr(cfg, cfg_field, ""))


def discover_api_models() -> dict[str, list[RemoteModel]]:
    """Return frontier chat models grouped by provider.

    Returns whatever the active provider's backend exposes for each
    configured API key, no curation. Short-circuits before touching
    the SDK when no keys are present.
    """
    active = [
        (prov, cfg_f, env, label)
        for prov, cfg_f, env, label in PROVIDER_KEYS
        if _has_provider_key(cfg_f, env)
    ]
    if not active:
        return {}

    provider = get_services().provider

    result: dict[str, list[RemoteModel]] = {}
    for prov, _cfg_field, _env_var, display_name in active:
        chat_models = [
            RemoteModel(
                name=model_name,
                task=ModelTask.CHAT,
                family="",
                parameter_size="",
                provider=display_name,
            )
            for model_name in provider.list_chat_models(prov)
        ]
        if chat_models:
            result[display_name] = chat_models
    return result


def detect_remote_embedding_models(base_url: str = "http://localhost:11434") -> list[str]:
    """Return names of models classified as embedding from the SDK backend."""
    return [m.name for m in classify_remote_models(base_url) if m.task == ModelTask.EMBEDDING]


def gather_known_model_refs() -> set[str]:
    """Compose canonical refs from native registry + Ollama tags + frontier APIs.

    Reuses the existing primitives: ``registry.list_installed`` for native
    GGUF installs, ``classify_remote_models`` for Ollama / OpenAI-compatible
    local backends, and ``discover_api_models`` for frontier providers.
    Each primitive already swallows its own failures, so a backend being
    down contributes an empty subset rather than raising.
    """
    refs = {m.ref for m in get_services().registry.list_installed()}
    for rm in classify_remote_models(cfg.remote_base_url):
        refs.add(format_remote_ref(rm.name, rm.provider))
    for models in discover_api_models().values():
        for rm in models:
            refs.add(format_remote_ref(rm.name, rm.provider))
    return refs


class KnownModelCache:
    """TTL-cached union of every model ref reachable by lilbee right now.

    The chat-completions route consults this to validate the incoming
    ``model`` field before forwarding. Caching keeps the per-request HTTP
    probe of Ollama's ``/api/tags`` off the hot path; the picker UI uses
    the same underlying primitives but renders infrequently enough to
    skip caching.

    Mutable state lives in a regular class (not a frozen dataclass)
    because the cached set and expiry timestamp are rewritten on each
    refresh. The instance itself is held on ``Services``.
    """

    DEFAULT_TTL_S = 30.0

    def __init__(self, ttl_s: float = DEFAULT_TTL_S) -> None:
        self._ttl_s = ttl_s
        self._refs: set[str] = set()
        self._expires_at: float = 0.0
        self._lock = Lock()

    def refs(self) -> set[str]:
        """Return the cached canonical-ref set, refreshing past the TTL."""
        with self._lock:
            now = time.monotonic()
            if now >= self._expires_at:
                self._refs = gather_known_model_refs()
                self._expires_at = now + self._ttl_s
            return self._refs

    def resolve(self, model: str) -> str | None:
        """Resolve *model* to its canonical ref, or return None if unknown.

        Direct membership wins. If the request used Ollama's bare
        ``name:tag`` convention (no slash, has a colon) and lilbee has
        an ``ollama/<name>`` entry in the discovered set, that canonical
        form is returned so downstream parsing handles it like any other
        prefixed ref. The probe is anchored in real discovery rather
        than a parse-time shape guess, so a typo with the right shape
        doesn't silently route anywhere.
        """
        refs = self.refs()
        if model in refs:
            return model
        if "/" not in model and ":" in model:
            prefixed = f"ollama/{model}"
            if prefixed in refs:
                return prefixed
        return None

    def invalidate(self) -> None:
        """Force the next ``refs()`` call to re-probe.

        Wired by code paths that mutate the native registry (model pull,
        model remove) so the cache reflects the new state without the
        full TTL lag. Remote-side changes (``ollama pull``) we cannot
        observe; the TTL bounds the lag in that direction.
        """
        with self._lock:
            self._expires_at = 0.0
