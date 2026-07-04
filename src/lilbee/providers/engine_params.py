"""Engine-neutral parameter helpers: model-path resolution, context and GPU-layer
sizing, and chat-option translation.

These derive launch/generation parameters from cfg + a model's GGUF metadata
without loading the model, so both the local llama-server engine and any other
provider compute them identically. No native binding.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lilbee.app.services import get_services

if TYPE_CHECKING:
    from lilbee.modelhub.registry import ModelRegistry
from lilbee.core.config import DEFAULT_NUM_CTX, cfg
from lilbee.core.config.enums import KV_CACHE_TYPE_BYTES
from lilbee.providers.base import (
    ProviderError,
    ProviderErrorKind,
    normalize_generation_options,
)
from lilbee.providers.gguf_meta import read_gguf_metadata, train_ctx_from_meta
from lilbee.providers.model_cache import (
    compute_dynamic_ctx,
    get_available_memory,
    kv_bytes_per_token,
)

log = logging.getLogger(__name__)

EMBED_FALLBACK_CTX = 2048
"""Context used for embed/rerank when a GGUF reports junk (e.g. context_length=0)."""

# Sized above chunk_size so BOS re-added on re-tokenization doesn't overflow a full-chunk input.
_EMBED_CTX_MARGIN = 8


def resolve_embed_ctx(meta: dict[str, str] | None, model_path: Path) -> int:
    """Embed/rerank context: worst-case chunk tokenization, capped by trained context.

    ``chunk_size`` is token-denominated but the chunker enforces a CHARACTER
    budget (``chunk_size * CHARS_PER_TOKEN``). A BPE token is at least one
    character, so that char budget is also the PROVABLE token ceiling for any
    chunk the chunker can emit: size the context to it and embed-time
    truncation becomes impossible, not merely rare. (Observed live before the
    fix: numeric-table chunks at ~1.5 chars/token reached 1982 tokens against a
    2x-chunk_size cap and lost their tails -- silently unsearchable text.)"""
    from lilbee.data.chunk import CHARS_PER_TOKEN

    train_ctx = train_ctx_from_meta(meta, fallback=EMBED_FALLBACK_CTX, model_path=model_path)
    return min(train_ctx, cfg.chunk_size * CHARS_PER_TOKEN + _EMBED_CTX_MARGIN)


_LLM_RERANK_HEADROOM = 512
"""Tokens reserved above chunk_size for an LLM reranker's query, prompt, and 1-token answer."""


def resolve_llm_rerank_ctx(meta: dict[str, str] | None, model_path: Path) -> int:
    """LLM-reranker context: a query+candidate pair, capped by the model's trained context."""
    train_ctx = train_ctx_from_meta(meta, fallback=EMBED_FALLBACK_CTX, model_path=model_path)
    return min(train_ctx, cfg.chunk_size + _LLM_RERANK_HEADROOM)


_VISION_FALLBACK_N_CTX = 4096
"""Context for a vision load when the GGUF reports no usable context_length."""

_VISION_PAGE_CTX_CAP = 32768
"""Per-page ceiling on a vision OCR server's context: covers a single high-res page's
image tokens plus prompt, while keeping a long-context VLM placeable beside a chat giant."""

_N_GPU_LAYERS_AUTO = -1
"""llama.cpp's "offload every layer" sentinel for n_gpu_layers."""


def chat_options_to_kwargs(options: dict[str, Any] | None) -> dict[str, Any]:
    """Translate user-facing chat options into generation kwargs.

    The output keys (``temperature``/``top_p``/``top_k``/``seed``/``max_tokens``/
    ``repeat_penalty``) are accepted by llama-server's OpenAI body. ``top_k`` is
    kept (local llama.cpp honors it), unlike the SDK/API translator which drops it.
    """
    return normalize_generation_options(options)


def resolve_model_path(model: str, registry: ModelRegistry | None = None) -> Path:
    """Resolve a model name to a .gguf file path.

    Resolution order: (1) registry (canonical source for installed models),
    (2) an absolute path to an existing file. Pass *registry* to resolve without
    reaching for ``get_services()`` (callers running inside its construction).
    """
    if registry is None:
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


def chat_ctx_ceiling(meta: dict[str, str] | None, model_path: Path) -> int:
    """Hard upper bound on a chat per-slot n_ctx: trained context, capped by ``cfg.num_ctx_max``."""
    training_ctx = train_ctx_from_meta(meta, fallback=DEFAULT_NUM_CTX, model_path=model_path)
    if cfg.num_ctx_max is not None:
        return min(training_ctx, cfg.num_ctx_max)
    return training_ctx


def resolve_chat_ctx(
    model_path: Path, meta: dict[str, str] | None, *, available_bytes: int | None = None
) -> int:
    """Pick a single-GPU n_ctx aiming for ``cfg.chat_n_ctx_target``, clamped to model + host.

    When ``cfg.num_ctx_max`` is ``None`` the model's training_ctx is the only
    ceiling, so a long-context model can grow past the target if the host has
    the RAM to back it. A multi-GPU tensor-split chat is sized separately by the
    fleet against its per-device headroom (see :func:`lilbee.providers.fleet.ctx.fit_split_ctx`).
    ``available_bytes`` overrides the live memory read: the fleet planner passes
    its clean-box snapshot so a reload sizes ctx like the boot did, not against
    VRAM its own loaded fleet is holding.
    """
    training_ctx = train_ctx_from_meta(meta, fallback=DEFAULT_NUM_CTX, model_path=model_path)
    ceiling = cfg.num_ctx_max if cfg.num_ctx_max is not None else training_ctx

    try:
        model_bytes = model_path.stat().st_size
        kv_per_tok = kv_bytes_per_token(meta, _kv_elem_bytes_for_cfg())
        if available_bytes is None:
            available_bytes = get_available_memory(cfg.gpu_memory_fraction)
        return compute_dynamic_ctx(
            model_bytes=model_bytes,
            available_bytes=available_bytes,
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
    """Pick n_ctx for a vision OCR load: the model's training context, capped per page.

    Uses the model's ``<arch>.context_length`` (not the chat-tuned ``cfg.num_ctx``: a
    vision pass packs image-token embeddings plus the prompt, and a small chat ctx
    truncates OCR output) but caps it at ``_VISION_PAGE_CTX_CAP``. OCR processes one page
    per request, so a single page never exceeds the cap, yet a long-context VLM's full
    context would otherwise estimate too large to place alongside a chat giant.
    """
    try:
        meta = read_gguf_metadata(model_path)
    except Exception:
        log.debug("read_gguf_metadata failed for vision %s", model_path, exc_info=True)
        meta = None
    train_ctx = train_ctx_from_meta(meta, fallback=_VISION_FALLBACK_N_CTX, model_path=model_path)
    return min(train_ctx, _VISION_PAGE_CTX_CAP)
