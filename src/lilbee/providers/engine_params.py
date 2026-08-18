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

if TYPE_CHECKING:
    from lilbee.modelhub.registry import ModelRegistry
from lilbee.core.config import DEFAULT_NUM_CTX, cfg
from lilbee.core.config.enums import KV_CACHE_TYPE_BYTES, KvCacheType
from lilbee.providers.base import (
    CONTEXT_WINDOW_MARGIN_TOKENS,
    GENERATION_RESERVE_TOKENS,
    ProviderError,
    ProviderErrorKind,
    estimate_budget_tokens,
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
    character, so that char budget is also the provable token ceiling for any
    chunk the chunker can emit: size the context to it and embed-time
    truncation becomes impossible. Token-dense text (numeric tables, dense
    identifiers) otherwise reaches ~2x chunk_size tokens against a 1x cap and
    silently loses its tail at embed time."""
    from lilbee.data.extract.chunk import CHARS_PER_TOKEN

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
"""llama.cpp's "fit as many layers as the device holds" value for n_gpu_layers.

The engine measures free VRAM at load and picks the count itself, spilling the
rest to system memory. That is a better answer than any number lilbee can
compute ahead of time, because it is taken on the real device after every other
tenant, so the planner passes this rather than a layer count of its own.
"""
# llama.cpp's "offload nothing"; the user's CPU-only opt-out rather than a budget.
_N_GPU_LAYERS_NONE = 0


def chat_options_to_kwargs(options: dict[str, Any] | None) -> dict[str, Any]:
    """Translate user-facing chat options into generation kwargs.

    The output keys (``temperature``/``top_p``/``top_k``/``seed``/``max_tokens``/
    ``repeat_penalty``) are accepted by llama-server's OpenAI body. ``top_k`` is
    kept (local llama.cpp honors it), unlike the SDK/API translator which drops it.
    ``think`` becomes ``chat_template_kwargs.enable_thinking``, which thinking
    templates honor and others ignore.
    """
    kwargs = normalize_generation_options(options)
    think = kwargs.pop("think", None)
    if think is not None:
        kwargs["chat_template_kwargs"] = {"enable_thinking": think}
    return kwargs


def resolve_model_path(model: str, registry: ModelRegistry | None = None) -> Path:
    """Resolve a model name to a .gguf file path.

    Resolution order: (1) registry (canonical source for installed models),
    (2) an absolute path to an existing file. Pass *registry* to resolve without
    reaching for ``get_services()`` (callers running inside its construction).
    """
    if not model:
        raise ProviderError(
            "No model is configured for this role. Pick one from the catalog "
            "or run 'lilbee model pull <model>'.",
            provider="llama-server",
            kind=ProviderErrorKind.NOT_FOUND,
        )
    if registry is None:
        # call-time import: keeps the app-layer container off this module's import graph
        from lilbee.app.services import get_services

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


def chat_kv_elem_bytes() -> tuple[float, float]:
    """Per-element (K, V) byte costs of the KV cache a chat launch allocates.

    Reads the same flags the launch passes
    (:func:`lilbee.providers.fleet.planning.chat_cache_type_flags`): K carries
    ``cfg.kv_cache_type``, while V carries it only when flash attention is
    certain to be on, because llama.cpp refuses a quantized V cache without it
    and the launch then leaves V at f16. Budgeting from the launch flags keeps
    the granted window in step with the cache the engine actually allocates.
    """
    # call-time import: planning imports this module at load
    from lilbee.providers.fleet.planning import chat_cache_type_flags

    def elem_bytes(flag: str | None) -> float:
        return KV_CACHE_TYPE_BYTES[KvCacheType(flag) if flag else KvCacheType.F16]

    k_flag, v_flag = chat_cache_type_flags()
    return elem_bytes(k_flag), elem_bytes(v_flag)


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

    A gguf-parser fit answers first
    (:func:`lilbee.providers.fleet.planning.fit_chat_ctx`), because it prices the
    cache each layer of this architecture holds. Header math takes over when the
    estimator cannot answer; it charges every layer as dense attention over the
    whole window, which under-grants linear-attention, sliding-window and MLA
    models. Either way the window stops at the smallest of the trained context,
    ``cfg.num_ctx_max`` and the target.

    A multi-GPU tensor-split chat is sized separately by the fleet against its
    per-device headroom (see :func:`lilbee.providers.fleet.ctx.fit_split_ctx`).
    ``available_bytes`` overrides the live host-memory read, and every caller
    that is sizing a real launch passes it: the fleet and the surfaces that
    mirror it hand over
    :func:`lilbee.providers.fleet.planning.plan_sizing_budget`, which reports the
    memory of the GPU that will run the model and holds a clean-box snapshot so
    a reload sizes ctx like the boot did.
    """
    # call-time import: planning imports this module at load
    from lilbee.providers.fleet.planning import fit_chat_ctx

    training_ctx = train_ctx_from_meta(meta, fallback=DEFAULT_NUM_CTX, model_path=model_path)
    ceiling = cfg.num_ctx_max if cfg.num_ctx_max is not None else training_ctx
    if available_bytes is None:
        available_bytes = get_available_memory(cfg.gpu_memory_fraction)
    upper = min(training_ctx, ceiling, cfg.chat_n_ctx_target)

    try:
        return fit_chat_ctx(model_path, meta, available_bytes=available_bytes, ctx_ceiling=upper)
    except (ProviderError, OSError, ValueError):
        log.debug("gguf-parser ctx fit failed for %s, using header math", model_path, exc_info=True)

    try:
        model_bytes = model_path.stat().st_size
        kv_per_tok = kv_bytes_per_token(meta, *chat_kv_elem_bytes())
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


# Tokens the minimum grounded prompt allows for the question plus the context
# template's framing, beyond the system prompt and one retrieved source.
_GROUNDED_QUESTION_TOKENS = 128


def min_usable_chat_ctx() -> int:
    """Smallest chat window that serves one grounded answer: the system prompt,
    one retrieved source, the question, and the generation reserve plus margin."""
    return (
        estimate_budget_tokens(cfg.rag_system_prompt)
        + cfg.chunk_size
        + _GROUNDED_QUESTION_TOKENS
        + GENERATION_RESERVE_TOKENS
        + CONTEXT_WINDOW_MARGIN_TOKENS
    )


def resolve_n_gpu_layers(*, embedding: bool) -> int:
    """Resolve ``cfg.n_gpu_layers`` (None=all) to llama.cpp's offload integer.

    Zero is honoured for every role. It is not a layer budget but the way a user
    says "run this on the CPU", and the search roles used to take the
    full-offload sentinel before the setting was read, so embed, rerank and
    vision kept loading onto the GPU that had just been excluded.

    Any other value is a chat-shaped budget and says nothing useful about a small
    embedding model, which still offloads fully.
    """
    if cfg.n_gpu_layers == _N_GPU_LAYERS_NONE:
        return _N_GPU_LAYERS_NONE
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
