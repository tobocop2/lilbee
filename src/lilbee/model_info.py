"""Public API for reading model architecture metadata from GGUF files."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from lilbee.config import cfg

log = logging.getLogger(__name__)


@dataclass
class ModelArchInfo:
    """Architecture metadata for installed models."""

    chat_arch: str = "unknown"
    embed_arch: str = "unknown"
    vision_projector: str = "unknown"
    active_handler: str = "not loaded"


def get_model_architecture() -> ModelArchInfo:
    """Return architecture metadata for the currently configured models.
    Reads GGUF headers for chat, embedding, and (optionally) vision models.
    Falls back gracefully if llama-cpp-python is not installed or models
    are not available.
    """
    info = ModelArchInfo()
    try:
        import lilbee.providers.llama_cpp_provider  # noqa: F401

        info = _read_chat_arch(info)
        info = _read_embed_arch(info)
        info = _read_vision_arch(info)
    except ImportError:
        pass
    return info


def _read_chat_arch(info: ModelArchInfo) -> ModelArchInfo:
    """Read chat model architecture from GGUF metadata."""
    try:
        from lilbee.providers.llama_cpp_provider import read_gguf_metadata, resolve_model_path

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
        from lilbee.providers.llama_cpp_provider import read_gguf_metadata, resolve_model_path

        path = resolve_model_path(cfg.embedding_model)
        meta = read_gguf_metadata(path)
        if meta:
            info.embed_arch = meta.get("architecture", "unknown")
    except Exception:
        log.debug("Failed to read embedding model architecture", exc_info=True)
    return info


def _read_vision_arch(info: ModelArchInfo) -> ModelArchInfo:
    """Read vision projector type from GGUF metadata."""
    if not cfg.vision_model:
        return info
    try:
        from lilbee.providers.llama_cpp_provider import (
            find_mmproj_for_model,
            read_mmproj_projector_type,
            resolve_model_path,
        )

        path = resolve_model_path(cfg.vision_model)
        mmproj = find_mmproj_for_model(path)
        proj_type = read_mmproj_projector_type(mmproj)
        info.vision_projector = proj_type or "unknown"
    except Exception:
        log.debug("Failed to read vision projector type", exc_info=True)
    return info
