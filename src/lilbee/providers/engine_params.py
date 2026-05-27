"""Engine-neutral parameter helpers: model-path resolution, context and GPU-layer
sizing, and chat-option translation.

These derive launch/generation parameters from cfg + a model's GGUF metadata
without loading the model, so both the local llama-server engine and any other
provider compute them identically. No native binding.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from lilbee.app.services import get_services
from lilbee.core.config import DEFAULT_NUM_CTX, cfg
from lilbee.core.config.enums import KV_CACHE_TYPE_BYTES
from lilbee.providers.base import ProviderError, ProviderErrorKind, filter_options
from lilbee.providers.gguf_meta import read_gguf_metadata, train_ctx_from_meta
from lilbee.providers.model_cache import (
    compute_dynamic_ctx,
    get_available_memory,
    kv_bytes_per_token,
)

log = logging.getLogger(__name__)

EMBED_FALLBACK_CTX = 2048
"""Context used for embed/rerank when a GGUF reports junk (e.g. context_length=0)."""

_VISION_FALLBACK_N_CTX = 4096
"""Context for a vision load when the GGUF reports no usable context_length."""

_N_GPU_LAYERS_AUTO = -1
"""llama.cpp's "offload every layer" sentinel for n_gpu_layers."""


def chat_options_to_kwargs(options: dict[str, Any] | None) -> dict[str, Any]:
    """Translate user-facing chat options into generation kwargs.

    The output keys (``temperature``/``top_p``/``top_k``/``seed``/``max_tokens``/
    ``repeat_penalty``) are accepted by llama-server's OpenAI body, so every
    provider translates options identically. ``num_predict`` becomes
    ``max_tokens`` and ``num_ctx`` is dropped (a model-load param, not per-call).
    ``filter_options`` also validates against ``LLMOptions``.
    """
    if not options:
        return {}
    filtered = filter_options(options)
    if "num_predict" in filtered:
        filtered["max_tokens"] = filtered.pop("num_predict")
    filtered.pop("num_ctx", None)
    return filtered


def resolve_model_path(model: str) -> Path:
    """Resolve a model name to a .gguf file path.

    Resolution order: (1) registry (canonical source for installed models),
    (2) an absolute path to an existing file.
    """
    registry = get_services().registry
    try:
        return registry.resolve(model)
    except (KeyError, ValueError):
        pass

    candidate = Path(model)
    if candidate.is_absolute():
        if candidate.exists():
            return candidate
        raise ProviderError(
            f"Model file not found: {model}",
            provider="llama-server",
            kind=ProviderErrorKind.NOT_FOUND,
        )

    raise ProviderError(
        f"Model {model!r} is not installed. Run 'lilbee model pull {model}' to download it.",
        provider="llama-server",
        kind=ProviderErrorKind.NOT_FOUND,
    )


def _kv_elem_bytes_for_cfg() -> int:
    """Bytes per KV element implied by the configured cache type."""
    return KV_CACHE_TYPE_BYTES[cfg.kv_cache_type]


def resolve_chat_ctx(model_path: Path, meta: dict[str, str] | None) -> int:
    """Pick n_ctx aiming for ``cfg.chat_n_ctx_target``, clamped to model + host.

    When ``cfg.num_ctx_max`` is ``None`` the model's training_ctx is the only
    ceiling, so a long-context model can grow past the target if the host has
    the RAM to back it. Setting ``num_ctx_max`` explicitly caps below
    training_ctx for per-host policy reasons.
    """
    training_ctx = train_ctx_from_meta(meta, fallback=DEFAULT_NUM_CTX, model_path=model_path)
    ceiling = cfg.num_ctx_max if cfg.num_ctx_max is not None else training_ctx

    try:
        model_bytes = model_path.stat().st_size
        available = get_available_memory(cfg.gpu_memory_fraction)
        kv_per_tok = kv_bytes_per_token(meta, _kv_elem_bytes_for_cfg())
        return compute_dynamic_ctx(
            model_bytes=model_bytes,
            available_bytes=available,
            training_ctx=training_ctx,
            kv_bytes_per_tok=kv_per_tok,
            ceiling=ceiling,
            target=cfg.chat_n_ctx_target,
        )
    except (OSError, ValueError):
        log.debug("dynamic ctx sizing failed for %s, using static cap", model_path, exc_info=True)
        return min(training_ctx, cfg.chat_n_ctx_target)


def resolve_n_gpu_layers(*, embedding: bool) -> int:
    """Resolve ``cfg.n_gpu_layers`` (None=all) to llama.cpp's offload integer."""
    if embedding or cfg.n_gpu_layers is None:
        return _N_GPU_LAYERS_AUTO
    return cfg.n_gpu_layers


def resolve_vision_ctx(model_path: Path) -> int:
    """Pick n_ctx for a vision load from the model's training context.

    Reads ``<arch>.context_length`` and uses it directly. The chat-tuned
    ``cfg.num_ctx`` is not propagated: a vision pass packs image-token
    embeddings plus the prompt (often hundreds to a few thousand tokens per
    page), and clamping to a small chat ctx truncates OCR output. An explicit
    value keeps the OOM-retry path working (it cannot bisect from 0).
    """
    try:
        meta = read_gguf_metadata(model_path)
    except Exception:
        log.debug("read_gguf_metadata failed for vision %s", model_path, exc_info=True)
        meta = None
    return train_ctx_from_meta(meta, fallback=_VISION_FALLBACK_N_CTX, model_path=model_path)
