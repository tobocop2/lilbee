"""Public API for reading model architecture metadata for the configured models."""

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


def get_model_architecture() -> ModelArchInfo:
    """Return architecture metadata for the currently configured models.

    Read on every call: the header reader caches by file, so a repeat read
    costs microseconds, and the chat handler follows the engine choice.
    Each reader degrades gracefully when its model is unset or unavailable.
    """
    return _read_vision_arch(_read_embed_arch(_read_chat_arch(ModelArchInfo())))


def _read_chat_arch(info: ModelArchInfo) -> ModelArchInfo:
    """Read the chat architecture from the GGUF header and the handler from the chat engine."""
    try:
        from lilbee.app.engine_choice import resolve_engines
        from lilbee.providers.engine_params import resolve_model_path
        from lilbee.providers.fleet.engines.llama import LLAMA
        from lilbee.providers.roles import WorkerRole

        path = resolve_model_path(cfg.chat_model)
        engine = resolve_engines(cfg).for_role(WorkerRole.CHAT).adapter
        info.active_handler = engine.request_profile(WorkerRole.CHAT).label
        meta = LLAMA.model_meta(path)
        if meta:
            info.chat_arch = meta.get("architecture", "unknown")
    except Exception:
        log.debug("Failed to read chat model architecture", exc_info=True)
    return info


def _read_embed_arch(info: ModelArchInfo) -> ModelArchInfo:
    """Read embedding model architecture from GGUF metadata."""
    try:
        from lilbee.providers.engine_params import resolve_model_path
        from lilbee.providers.gguf_meta import read_gguf_metadata

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
        from lilbee.providers.engine_params import resolve_model_path
        from lilbee.providers.gguf_meta import (
            find_mmproj_for_model,
            read_mmproj_projector_type,
        )

        path = resolve_model_path(cfg.vision_model)
        mmproj = find_mmproj_for_model(path)
        proj_type = read_mmproj_projector_type(mmproj)
        info.vision_projector = proj_type or "unknown"
    except Exception:
        log.debug("Failed to read vision projector type", exc_info=True)
    return info
