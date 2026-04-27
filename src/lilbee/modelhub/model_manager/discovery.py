"""Remote model discovery and task classification."""

import logging
import os

import httpx

from lilbee.core.config.model import cfg
from lilbee.modelhub.model_manager.types import RemoteModel
from lilbee.modelhub.models import ModelTask
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


def _classify_remote_task(name: str, family: str) -> str:
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

    Checks which provider API keys are available (env vars or config),
    then asks the active provider's backend for each provider's chat
    catalog. Going through ``SdkLLMProvider.list_chat_models`` ensures
    ``cfg.json_mode`` suppression is applied before the SDK import,
    so ``lilbee --json status`` does not leak a debug banner.

    Short-circuits before touching the SDK when no keys are present.
    """
    active = [
        (prov, cfg_f, env, label)
        for prov, cfg_f, env, label in PROVIDER_KEYS
        if _has_provider_key(cfg_f, env)
    ]
    if not active:
        return {}

    from lilbee.core.services import get_services

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
