"""Remote model discovery and task classification."""

import logging
import os
import time
from collections.abc import Callable
from threading import Lock

import httpx

from lilbee.app.services import get_services
from lilbee.catalog.query import (
    EMBEDDING_NAME_PATTERNS,
    RERANKER_NAME_PATTERNS,
    VISION_NAME_PATTERNS,
)
from lilbee.catalog.types import ModelTask
from lilbee.core.config.model import cfg
from lilbee.modelhub.model_manager.types import RemoteModel
from lilbee.providers.backend_names import BackendName
from lilbee.providers.local_servers import (
    LM_STUDIO,
    OLLAMA,
    LocalServerSpec,
    openai_models_url,
)
from lilbee.providers.local_servers.config_urls import configured_local_servers
from lilbee.providers.model_ref import format_remote_ref
from lilbee.providers.sdk_backend import PROVIDER_KEYS

log = logging.getLogger(__name__)

_EMBEDDING_FAMILIES = frozenset({"bert", "nomic-bert", "e5", "bge"})

_CLASSIFY_DEFAULT_TIMEOUT_S = 5.0


def _classify_remote_task(name: str, family: str) -> ModelTask:
    """Classify a remote model as rerank, embedding, vision, or chat (in that order).

    Embedding matches by family tag or name pattern; the name path covers
    servers like LM Studio that report no family.
    """
    name_lower = name.lower()
    if any(rp in name_lower for rp in RERANKER_NAME_PATTERNS):
        return ModelTask.RERANK
    family_lower = family.lower()
    if any(ef in family_lower for ef in _EMBEDDING_FAMILIES) or any(
        ep in name_lower for ep in EMBEDDING_NAME_PATTERNS
    ):
        return ModelTask.EMBEDDING
    if any(vp in name_lower for vp in VISION_NAME_PATTERNS):
        return ModelTask.VISION
    return ModelTask.CHAT


def classify_remote_models(
    base_url: str,
    spec: LocalServerSpec,
    *,
    timeout: float = _CLASSIFY_DEFAULT_TIMEOUT_S,
) -> list[RemoteModel]:
    """Discover and classify all models from one local server by task.

    The strategy and provider label come from *spec* (Ollama ``/api/tags`` vs
    LM Studio ``/v1/models``), so a server reached at a non-default host is
    classified correctly. Returns ``[]`` on any error so read-only callers stay
    responsive when the backend is down.
    """
    discover = _DISCOVERY_BY_KEY[spec.key]
    return discover(base_url, spec.display_name, timeout)


def classify_all_remote_models(
    *,
    timeout: float = _CLASSIFY_DEFAULT_TIMEOUT_S,
) -> list[RemoteModel]:
    """Classify models across every configured local server, source-labeled."""
    result: list[RemoteModel] = []
    for spec, base_url in configured_local_servers():
        result.extend(classify_remote_models(base_url, spec, timeout=timeout))
    return result


def _discover_via_ollama_tags(
    base_url: str, provider: BackendName, timeout: float
) -> list[RemoteModel]:
    """Classify models from Ollama's ``/api/tags`` using family metadata."""
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=timeout)
        resp.raise_for_status()
        raw_models = resp.json().get("models", [])
    except Exception:
        log.debug("Failed to classify remote models", exc_info=True)
        return []

    result: list[RemoteModel] = []
    for model in raw_models:
        name = model.get("name", "")
        details = model.get("details", {})
        family = details.get("family", "")
        param_size = details.get("parameter_size", "")
        task = _classify_remote_task(name, family)
        result.append(
            RemoteModel(
                name=name,
                task=task,
                family=family,
                parameter_size=param_size,
                provider=provider,
            )
        )
    return result


def _discover_via_openai_models(
    base_url: str, provider: BackendName, timeout: float
) -> list[RemoteModel]:
    """Classify models from an OpenAI-compatible ``/v1/models`` endpoint.

    These servers report only ids (no family), so task detection runs off the
    name patterns, which LM Studio ids usually carry. Every id is surfaced: LM
    Studio presents LM Link remote/cloud models here as if local, so the list
    is intentionally not filtered to locally-downloaded models.
    """
    try:
        resp = httpx.get(openai_models_url(base_url), timeout=timeout)
        resp.raise_for_status()
        raw_models = resp.json().get("data", [])
    except Exception:
        log.debug("Failed to classify remote models", exc_info=True)
        return []

    result: list[RemoteModel] = []
    for model in raw_models:
        name = model.get("id", "")
        if not name:
            continue
        task = _classify_remote_task(name, "")
        result.append(
            RemoteModel(
                name=name,
                task=task,
                family="",
                parameter_size="",
                provider=provider,
            )
        )
    return result


# Listing strategy per local-server routing key. Module-level so it stays a
# single source of truth as servers are added to the registry.
_DISCOVERY_BY_KEY: dict[str, Callable[[str, BackendName, float], list[RemoteModel]]] = {
    OLLAMA.key: _discover_via_ollama_tags,
    LM_STUDIO.key: _discover_via_openai_models,
}


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


def detect_remote_embedding_models() -> list[str]:
    """Return embedding-model names across every configured local server."""
    return [m.name for m in classify_all_remote_models() if m.task == ModelTask.EMBEDDING]


def _installed_native_refs() -> set[str]:
    """Canonical refs from the native registry; empty set if the walk fails."""
    try:
        return {m.ref for m in get_services().registry.list_installed()}
    except Exception:
        log.warning("Native registry walk failed; contributing no installed refs", exc_info=True)
        return set()


def gather_known_model_refs() -> set[str]:
    """Canonical refs from the native registry, every configured local server, and APIs.

    Each primitive swallows its own failures, so a backend being down contributes an
    empty subset rather than raising.
    """
    refs = _installed_native_refs()
    for rm in classify_all_remote_models():
        refs.add(format_remote_ref(rm.name, rm.provider))
    for models in discover_api_models().values():
        for rm in models:
            refs.add(format_remote_ref(rm.name, rm.provider))
    return refs


class KnownModelCache:
    """TTL-cached union of native + remote + frontier model refs."""

    DEFAULT_TTL_S = 30.0

    def __init__(self, ttl_s: float = DEFAULT_TTL_S) -> None:
        self._ttl_s = ttl_s
        self._refs: frozenset[str] = frozenset()
        self._expires_at: float = 0.0
        self._generation: int = 0
        self._lock = Lock()

    def refs(self) -> frozenset[str]:
        """Cached canonical-ref set, refreshing past the TTL (fan-out runs off the lock)."""
        with self._lock:
            if time.monotonic() < self._expires_at:
                return self._refs
            captured_generation = self._generation
        fresh = frozenset(gather_known_model_refs())
        with self._lock:
            self._refs = fresh
            if self._generation == captured_generation:
                self._expires_at = time.monotonic() + self._ttl_s
            return self._refs

    def resolve(self, model: str) -> str | None:
        """Resolve *model* to its canonical ref, or None if unknown."""
        refs = self.refs()
        if model in refs:
            return model
        if "/" not in model and ":" in model:
            prefixed = f"ollama/{model}"
            if prefixed in refs:
                return prefixed
        return None

    def invalidate(self) -> None:
        """Force the next ``refs()`` call to re-probe."""
        with self._lock:
            self._expires_at = 0.0
            self._generation += 1
