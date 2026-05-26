"""Public API for reading model architecture metadata from GGUF files."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from lilbee.core.config import cfg

log = logging.getLogger(__name__)


@dataclass
class ModelArchInfo:
    """Architecture metadata for installed models."""

    chat_arch: str = "unknown"
    embed_arch: str = "unknown"
    vision_projector: str = "unknown"
    active_handler: str = "not loaded"


# Cache: (chat_model_ref, embed_model_ref, vision_model_ref) -> ModelArchInfo.
# Reading GGUF headers is hundreds of ms cold (file open + parse + first
# llama_cpp import); the result is stable as long as the configured refs
# stay the same. Status screen visits, MCP status calls, and any other
# read-side caller share this cache. ``invalidate_cache`` lets settings
# updates clear it explicitly when a model ref changes.
_arch_cache: dict[tuple[str, str, str], ModelArchInfo] = {}


def _cache_key() -> tuple[str, str, str]:
    return (cfg.chat_model or "", cfg.embedding_model or "", cfg.vision_model or "")


def invalidate_cache() -> None:
    """Drop the architecture cache. Call when a model ref changes."""
    _arch_cache.clear()


def get_model_architecture() -> ModelArchInfo:
    """Return architecture metadata for the currently configured models.

    Memoized on (chat_model, embed_model, vision_model). First call
    reads GGUF headers for each; subsequent calls under the same refs
    return the cached result instantly. Falls back gracefully if
    llama-cpp-python is not installed or models are not available.
    """
    key = _cache_key()
    cached = _arch_cache.get(key)
    if cached is not None:
        return cached
    info = ModelArchInfo()
    try:
        import lilbee.providers.llama_cpp  # noqa: F401

        info = _read_chat_arch(info)
        info = _read_embed_arch(info)
        info = _read_vision_arch(info)
    except ImportError:
        pass  # llama_cpp is optional; arch info degrades gracefully
    _arch_cache[key] = info
    return info


def _read_chat_arch(info: ModelArchInfo) -> ModelArchInfo:
    """Read chat model architecture from GGUF metadata."""
    try:
        from lilbee.providers.gguf_meta import read_gguf_metadata
        from lilbee.providers.engine_params import resolve_model_path

        path = resolve_model_path(cfg.chat_model)
        meta = read_gguf_metadata(path)
        if meta:
            info.chat_arch = meta.get("architecture", "unknown")
            info.active_handler = "llama-cpp"
    except Exception:
        log.debug("Failed to read chat model architecture", exc_info=True)
    return info


def _read_embed_arch(info: ModelArchInfo) -> ModelArchInfo:
    """Read embedding model architecture from GGUF metadata."""
    try:
        from lilbee.providers.gguf_meta import read_gguf_metadata
        from lilbee.providers.engine_params import resolve_model_path

        path = resolve_model_path(cfg.embedding_model)
        meta = read_gguf_metadata(path)
        if meta:
            info.embed_arch = meta.get("architecture", "unknown")
    except Exception:
        log.debug("Failed to read embedding model architecture", exc_info=True)
    return info


def _read_vision_arch(info: ModelArchInfo) -> ModelArchInfo:
    """Read vision projector type from GGUF metadata for ``cfg.vision_model``.

    Reads the vision model name from the global ``cfg`` singleton (same
    pattern as :func:`_read_chat_arch` / :func:`_read_embed_arch`) rather
    than taking it as a parameter. The chat model is never inspected for
    vision capability here: role separation is explicit. Returns the
    input unchanged when no vision model is configured.
    """
    if not cfg.vision_model:
        return info
    try:
        from lilbee.providers.gguf_meta import (
            find_mmproj_for_model,
            read_mmproj_projector_type,
        )
        from lilbee.providers.engine_params import resolve_model_path

        path = resolve_model_path(cfg.vision_model)
        mmproj = find_mmproj_for_model(path)
        proj_type = read_mmproj_projector_type(mmproj)
        info.vision_projector = proj_type or "unknown"
    except Exception:
        log.debug("Failed to read vision projector type", exc_info=True)
    return info
