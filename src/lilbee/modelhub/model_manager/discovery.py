"""Remote model discovery and task classification."""

import logging
import os
from collections.abc import Callable

import httpx

from lilbee.app.services import get_services
from lilbee.catalog.types import ModelTask
from lilbee.core.config.model import cfg
from lilbee.modelhub.model_manager.types import RemoteModel
from lilbee.providers.backend_names import BackendName
from lilbee.providers.local_servers import (
    LM_STUDIO,
    OLLAMA,
    detect_local_server,
    openai_models_url,
)
from lilbee.providers.sdk_backend import PROVIDER_KEYS, detect_backend_name

log = logging.getLogger(__name__)

_EMBEDDING_FAMILIES = frozenset({"bert", "nomic-bert", "e5", "bge"})
# Embedding detection by name, for OpenAI-compatible servers (LM Studio) that
# expose ids but no family metadata. ``embed`` covers the common naming, the
# rest match family-named embedders (``bge-base``, ``multilingual-e5-large``,
# ``gte-large``). The trailing hyphen on the family tags avoids matching a chat
# model that merely contains the letters (``bge`` keeps ``bge-reranker`` out of
# here too, though reranker detection already runs first).
_EMBEDDING_NAME_PATTERNS = frozenset({"embed", "bge-", "e5-", "gte-"})
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
    ``bge``) does not get dragged into the embedding bucket. After
    reranker: embedding by family tag OR by name pattern (the name path
    covers servers like LM Studio that report no family), vision by name
    pattern, else chat.
    """
    name_lower = name.lower()
    if any(rp in name_lower for rp in _RERANKER_NAME_PATTERNS):
        return ModelTask.RERANK
    family_lower = family.lower()
    if any(ef in family_lower for ef in _EMBEDDING_FAMILIES) or any(
        ep in name_lower for ep in _EMBEDDING_NAME_PATTERNS
    ):
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
    """Discover and classify all models from the local server by task.

    Dispatches the listing strategy off the server behind *base_url*:
    Ollama's ``/api/tags`` (rich family metadata) or LM Studio's
    OpenAI-compatible ``/v1/models``. Unknown URLs default to the Ollama
    strategy. The provider label always comes from ``detect_backend_name``,
    so a hosted API URL keeps its own name. Returns an empty list on any
    error (including timeout) so read-only callers stay responsive when the
    backend is down.
    """
    spec = detect_local_server(base_url)
    provider = detect_backend_name(base_url)
    key = spec.key if spec is not None else OLLAMA.key
    discover = _DISCOVERY_BY_KEY.get(key, _discover_via_ollama_tags)
    return discover(base_url, provider, timeout)


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

    LM Studio exposes only model ids, no family metadata, so embedding /
    reranker / vision detection falls back to the name patterns (which
    LM Studio ids usually carry, e.g. ``nomic-embed``, ``bge-reranker``).
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


def detect_remote_embedding_models(base_url: str = "http://localhost:11434") -> list[str]:
    """Return names of models classified as embedding from the SDK backend."""
    return [m.name for m in classify_remote_models(base_url) if m.task == ModelTask.EMBEDDING]
