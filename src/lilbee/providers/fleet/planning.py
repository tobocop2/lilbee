"""Launch planning for the fleet: device probe, VRAM estimate, placement, argv."""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lilbee.core.config.enums import KvCacheType
from lilbee.providers import model_cache
from lilbee.providers.fleet.adapters import (
    LLM_RERANK_CONCURRENCY,
    ROLE_SPECS,
    RoleServerSpec,
    build_server_argv,
    embed_spec,
    rerank_spec,
    resolve_rerank_mode,
)
from lilbee.providers.fleet.binary import llama_server_runtime_env, resolve_llama_server
from lilbee.providers.fleet.devices import FleetDevice, probe_devices, visible_env
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.placement import (
    InstancePlan,
    ModelPlacementInput,
    PeakEstimator,
    Placement,
    SplitCtxFitter,
    placement_from_spec,
    plan_placement,
)
from lilbee.providers.fleet.placement_spec import PlacementSpec
from lilbee.providers.fleet.replicas import resolve_replica_count
from lilbee.providers.fleet.vram import USABLE_VRAM_FRACTION, estimate_instance_footprint
from lilbee.providers.roles import RerankMode, WorkerRole

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# Fleet-only concurrency: continuous-batching slots (--parallel) per server.
_CHAT_SLOTS = 4
# A tensor-split chat fills its cards with one full-context sequence; concurrent
# slots are for a chat that fits a single GPU, not a multi-card giant.
_SPLIT_CHAT_SLOTS = 1
# Floor context the PLACEMENT estimate reserves KV for, so a large model is never
# single-carded into a KV corner too small for real use (a 17GB model on a 24GB
# card leaves ~no KV room -> n_ctx collapses to a few hundred tokens). Sizing the
# placement reserve against this floor forces a tensor-split when one card cannot
# hold weights + a usable context; the served ctx is then grown to the chosen
# cards' real headroom by resolve_chat_ctx (single) / fit_split_ctx (split).
_MIN_USABLE_CHAT_CTX = 8192
_AUX_SLOTS = 1
# A tensor-split needs at least this many GPUs; below it the chat context objective
# (a gguf read) is pointless because the model can only single-card or stay unplaced.
_MIN_SPLIT_GPUS = 2
_EMBED_ROLES = (WorkerRole.EMBED, WorkerRole.RERANK)
# Roles whose loaders offload every layer regardless of cfg.n_gpu_layers; only
# chat honors cfg.n_gpu_layers.
_ALL_LAYER_ROLES = (WorkerRole.EMBED, WorkerRole.RERANK, WorkerRole.VISION)
_FLASH_ON = "on"
_FLASH_OFF = "off"
_DEFAULT_THREADS = 4
# Roles to which flash attention applies; embed/rerank run without it.
_FLASH_ROLES = (WorkerRole.CHAT, WorkerRole.VISION)

# Server roles -> model-ref accessor. chat/embed are always configured;
# reranker_model/vision_model may be "" (unconfigured) -> skipped, so that role
# has no server. Vision additionally needs an mmproj projector.
_SERVER_ROLE_PARAMS: dict[WorkerRole, Callable[[Any], str]] = {
    WorkerRole.CHAT: lambda c: str(c.chat_model),
    WorkerRole.EMBED: lambda c: str(c.embedding_model),
    WorkerRole.RERANK: lambda c: str(c.reranker_model),
    WorkerRole.VISION: lambda c: str(c.vision_model),
}


# Cap vision's own KV footprint at this fraction of usable VRAM when sizing its
# batching slots, leaving room for the weights and any co-located role.
_VISION_VRAM_FRACTION = 0.5

# Cap chat's footprint at this fraction of usable VRAM when sizing its batching
# slots, reserving room for the co-located embed/rerank servers and the decode
# compute buffers (which the flat overhead term only partly covers).
_CHAT_VRAM_FRACTION = 0.8

# Cap an LLM reranker's footprint at this fraction of usable VRAM when sizing its
# slots; its per-slot ctx is tiny, so a normal GPU fits the full fan-out and a
# small one steps down toward 1.
_LLM_RERANK_VRAM_FRACTION = 0.5

# RAM kept free for the OS when placing against system memory (no discrete GPU):
# a quarter of total RAM, capped at 4 GiB. A fixed 4 GiB floor leaves a small
# host (7-8 GB) with no budget at all, refusing to serve even tiny models.
_SYSTEM_MEMORY_FLOOR_CAP_BYTES = 4 * 1024**3
_SYSTEM_MEMORY_FLOOR_DIVISOR = 4

# llama.cpp split-GGUF shard naming ("%s-%05d-of-%05d.gguf"); the cold-load
# timeout must scale with the SUM of the shards, not the first file alone.
_SPLIT_GGUF_NAME = re.compile(r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$")


def _weights_bytes(model_path: Path) -> int:
    """Total weights size on disk; a split GGUF sums every sibling shard."""
    match = _SPLIT_GGUF_NAME.fullmatch(model_path.name)
    if match is None:
        return model_path.stat().st_size
    return sum(
        sibling.stat().st_size
        for sibling in model_path.parent.iterdir()
        if _is_sibling_shard(sibling.name, match)
    )


def _is_sibling_shard(name: str, match: re.Match[str]) -> bool:
    """Whether *name* is a shard of the same split GGUF as *match*."""
    shard = _SPLIT_GGUF_NAME.fullmatch(name)
    return (
        shard is not None
        and shard["prefix"] == match["prefix"]
        and shard["total"] == match["total"]
    )


def _slots_for(
    role: WorkerRole,
    model_path: Path,
    ctx: int,
    *,
    mmproj_path: Path | None = None,
    unified_budget: int | None = None,
    chat_reservation: int = 0,
    rerank_mode: RerankMode | None = None,
) -> int:
    """Continuous-batching slots (--parallel) for a role's server.

    Chat batches concurrent turns; vision batches concurrent OCR pages since a
    one-page decode underutilizes the GPU; an LLM reranker batches its per-candidate
    chat requests; embed and cross-encoder rerank are single-slot (their batching is
    request-side). The memory-aware roles drop toward 1 on a small or shared host
    instead of overcommitting. ``unified_budget`` caps sizing against free system RAM
    with no discrete GPU; ``chat_reservation`` is the search-role footprint held back
    from chat.
    """
    if role is WorkerRole.CHAT:
        return _resolve_chat_slots(
            model_path,
            ctx,
            mmproj_path=mmproj_path,
            unified_budget=unified_budget,
            chat_reservation=chat_reservation,
        )
    if role is WorkerRole.VISION:
        return _resolve_vision_slots(
            model_path, ctx, mmproj_path=mmproj_path, unified_budget=unified_budget
        )
    if role is WorkerRole.RERANK and rerank_mode is RerankMode.LLM:
        return _resolve_llm_rerank_slots(model_path, ctx, unified_budget=unified_budget)
    return _AUX_SLOTS


def _resolve_chat_slots(
    model_path: Path,
    ctx: int,
    *,
    mmproj_path: Path | None = None,
    unified_budget: int | None = None,
    chat_reservation: int = 0,
) -> int:
    """Largest chat slot count (<= ``_CHAT_SLOTS``) whose footprint fits the budget
    after reserving the search roles; steps to 1 when none fit."""
    budget = _slot_budget(_CHAT_VRAM_FRACTION, unified_budget) - chat_reservation
    return _fit_slots(
        _CHAT_SLOTS,
        WorkerRole.CHAT,
        model_path,
        ctx,
        mmproj_path=mmproj_path,
        unified=unified_budget is not None,
        budget=budget,
    )


def _resolve_vision_slots(
    model_path: Path,
    ctx: int,
    *,
    mmproj_path: Path | None = None,
    unified_budget: int | None = None,
) -> int:
    """Largest OCR batching slot count (<= ``cfg.vision_ocr_concurrency``) that fits
    the memory budget; 1 when the ceiling is 1 or nothing larger fits."""
    from lilbee.core.config import cfg

    ceiling = max(1, cfg.vision_ocr_concurrency)
    if ceiling == 1:
        return 1
    return _fit_slots(
        ceiling,
        WorkerRole.VISION,
        model_path,
        ctx,
        mmproj_path=mmproj_path,
        unified=unified_budget is not None,
        budget=_slot_budget(_VISION_VRAM_FRACTION, unified_budget),
    )


def _resolve_llm_rerank_slots(
    model_path: Path,
    ctx: int,
    *,
    unified_budget: int | None = None,
) -> int:
    """Largest LLM-reranker slot count (<= ``LLM_RERANK_CONCURRENCY``) that fits the
    memory budget; 1 when nothing larger fits. Matches the client's request fan-out."""
    return _fit_slots(
        LLM_RERANK_CONCURRENCY,
        WorkerRole.RERANK,
        model_path,
        ctx,
        mmproj_path=None,
        unified=unified_budget is not None,
        budget=_slot_budget(_LLM_RERANK_VRAM_FRACTION, unified_budget),
    )


def _slot_budget(vram_fraction: float, unified_budget: int | None) -> int:
    """Memory budget for slot sizing: *vram_fraction* of usable VRAM, capped by
    ``unified_budget`` (free system RAM) when there is no discrete GPU so the count
    steps down to fit free memory instead of overcommitting."""
    from lilbee.core.config import cfg

    budget = int(model_cache.get_available_memory(cfg.gpu_memory_fraction) * vram_fraction)
    if unified_budget is not None:
        budget = min(budget, unified_budget)
    return budget


def _fit_slots(
    ceiling: int,
    role: WorkerRole,
    model_path: Path,
    ctx: int,
    *,
    mmproj_path: Path | None,
    unified: bool,
    budget: int,
) -> int:
    """Largest slot count in ``1..ceiling`` whose instance footprint fits *budget*;
    1 when none larger fit."""
    for slots in range(ceiling, 1, -1):
        est = estimate_instance_footprint(
            model_path,
            ctx=ctx,
            slots=slots,
            gpu_layers=_role_gpu_layers(role),
            flash_attn=_role_flash(role),
            kv_cache_type=_role_kv_cache_type(role),
            mmproj_path=mmproj_path,
        )
        if est.footprint(unified=unified) <= budget:
            return slots
    return 1


def _role_ctx(role: WorkerRole, model_path: Path, meta: dict[str, str] | None) -> int:
    """Per-slot context for a role, derived as the in-process loader does.

    Embed/rerank use the embedding model's training context; vision uses the
    vision loader's training-context picker; chat honors ``cfg.num_ctx`` then
    falls back to the single-GPU dynamic chat-ctx picker. A tensor-split chat is
    sized against its per-device headroom instead (see :func:`fit_split_ctx`).
    """
    from lilbee.core.config import cfg
    from lilbee.providers.engine_params import (
        resolve_chat_ctx,
        resolve_embed_ctx,
        resolve_llm_rerank_ctx,
        resolve_vision_ctx,
    )

    if role is WorkerRole.EMBED:
        return resolve_embed_ctx(meta, model_path)
    if role is WorkerRole.RERANK:
        if _rerank_mode_for(meta) is RerankMode.LLM:
            return resolve_llm_rerank_ctx(meta, model_path)
        return resolve_embed_ctx(meta, model_path)
    if role is WorkerRole.VISION:
        return resolve_vision_ctx(model_path)
    if cfg.num_ctx is not None:
        return cfg.num_ctx
    return resolve_chat_ctx(model_path, meta)


def _rerank_mode_for(meta: dict[str, str] | None) -> RerankMode:
    """Resolve the RERANK serving mode from cfg + the reranker GGUF arch."""
    from lilbee.core.config import cfg

    arch = meta.get("architecture") if meta else None
    return resolve_rerank_mode(cfg.reranker_type, arch)


def _role_rerank_mode(role: WorkerRole, meta: dict[str, str] | None) -> RerankMode | None:
    """The RERANK serving mode for *role*, or ``None`` for every other role."""
    return _rerank_mode_for(meta) if role is WorkerRole.RERANK else None


def _server_spec(
    role: WorkerRole, rerank_mode: RerankMode | None, meta: dict[str, str] | None
) -> RoleServerSpec:
    """The llama-server spec for a launch: rerank mode, decoder-aware embed pooling,
    or the role default. EMBED forces ``--pooling last`` for decoder-only archs."""
    if rerank_mode is not None:
        return rerank_spec(rerank_mode)
    if role is WorkerRole.EMBED:
        return embed_spec(meta)
    return ROLE_SPECS[role]


def _pooled_batch_size(role: WorkerRole, rerank_mode: RerankMode | None, ctx: int) -> int | None:
    """The ``--batch-size``/``--ubatch-size`` the launch raises for pooled
    embed/cross-encoder rerank (the full context), or ``None`` for other roles."""
    if role in _EMBED_ROLES and rerank_mode is not RerankMode.LLM:
        return ctx
    return None


def _role_gpu_layers(role: WorkerRole) -> int:
    """GPU-layer offload: chat honors ``cfg.n_gpu_layers``, others offload all layers."""
    from lilbee.providers.engine_params import resolve_n_gpu_layers

    return resolve_n_gpu_layers(embedding=role in _ALL_LAYER_ROLES)


def _flash_enabled() -> bool:
    """Flash attention is on unless ``cfg.flash_attention`` is explicitly ``False``."""
    from lilbee.core.config import cfg

    return cfg.flash_attention is not False


def _flash_attn_flag() -> str:
    """``--flash-attn`` argv value for chat and vision."""
    return _FLASH_ON if _flash_enabled() else _FLASH_OFF


def _role_flash(role: WorkerRole) -> bool:
    """Flash attention applies to chat and vision; embed/rerank run without it."""
    return role in _FLASH_ROLES and _flash_enabled()


def _role_kv_cache_type(role: WorkerRole) -> KvCacheType:
    """Chat honors ``cfg.kv_cache_type``; embed/rerank/vision run f16 KV."""
    from lilbee.core.config import cfg

    return cfg.kv_cache_type if role is WorkerRole.CHAT else KvCacheType.F16


def _replica_count(role: WorkerRole, device_count: int) -> int:
    """Requested data-parallel instances for *role* via the shared resolver."""
    return resolve_replica_count(role, device_count)


def _cache_type_flag() -> str | None:
    """KV cache type for chat, or ``None`` to leave llama-server's f16 default."""
    from lilbee.core.config import cfg
    from lilbee.core.config.enums import KvCacheType

    if cfg.kv_cache_type is KvCacheType.F16:
        return None
    return cfg.kv_cache_type.value


def _vision_mmproj(model_ref: str) -> Path | None:
    """Resolve a vision model's mmproj sidecar, or ``None`` if absent."""
    from lilbee.providers.base import ProviderError
    from lilbee.providers.engine_params import resolve_model_path
    from lilbee.providers.gguf_meta import find_mmproj_for_model

    try:
        return find_mmproj_for_model(resolve_model_path(model_ref))
    except (ProviderError, OSError, ValueError, KeyError):
        return None


def _estimate_role(
    role: WorkerRole,
    model_ref: str,
    *,
    slots: int | None = None,
    unified_budget: int | None = None,
    chat_reservation: int = 0,
    device_count: int = 0,
) -> ModelPlacementInput:
    """Estimate one role-model's footprint via gguf-parser (+ mmproj for vision).

    ``slots`` defaults to the role's resolved batching slots (chat and vision are
    memory-aware); ``chat_reservation`` shrinks chat to leave room for the search
    roles; ``device_count`` resolves an auto (0) replica knob to one per GPU.
    Charges the unified footprint with no discrete GPU, else the VRAM one.
    """
    from lilbee.providers.engine_params import resolve_model_path
    from lilbee.providers.gguf_meta import read_gguf_metadata

    path = resolve_model_path(model_ref)
    mmproj = _vision_mmproj(model_ref) if role is WorkerRole.VISION else None
    meta = read_gguf_metadata(path)
    # Size the single-instance footprint against the placement reserve (a usable
    # KV floor for chat), so a model that fits weights-only on one card but not
    # weights + a usable context falls through to a tensor-split instead of being
    # single-carded into a tiny n_ctx. Non-chat roles keep their launch ctx.
    ctx = _placement_estimate_ctx(role, path, meta)
    rerank_mode = _role_rerank_mode(role, meta)
    if slots is None:
        slots = _slots_for(
            role,
            path,
            ctx,
            mmproj_path=mmproj,
            unified_budget=unified_budget,
            chat_reservation=chat_reservation,
            rerank_mode=rerank_mode,
        )
    est = estimate_instance_footprint(
        path,
        ctx=ctx,
        slots=slots,
        gpu_layers=_role_gpu_layers(role),
        flash_attn=_role_flash(role),
        kv_cache_type=_role_kv_cache_type(role),
        mmproj_path=mmproj,
        batch_size=_pooled_batch_size(role, rerank_mode, ctx),
    )
    fp = est.footprint(unified=unified_budget is not None)
    if role is WorkerRole.CHAT and unified_budget is None:
        fp = _chat_serve_budget_footprint(fp)
    return ModelPlacementInput(
        role=role, est_vram_bytes=fp, replicas=_replica_count(role, device_count)
    )


def _chat_serve_budget_footprint(footprint: int) -> int:
    """Charge a chat instance against the serve budget, not the placement headroom.

    The planner fits instances within ``USABLE_VRAM_FRACTION`` of a card, but a
    single-card chat then sizes its KV cache against the smaller
    ``cfg.gpu_memory_fraction`` budget (``resolve_chat_ctx``). A model that fills a
    card at 0.9 leaves no room for KV at 0.75 and collapses to a few hundred tokens,
    so scale its placement footprint by the budget ratio: it then needs a
    tensor-split (pooling VRAM across cards) whenever single-carding it would starve
    its context. Small models are unaffected -- they fit the serve budget with KV
    room to spare.
    """
    from lilbee.core.config import cfg

    return int(footprint * (USABLE_VRAM_FRACTION / cfg.gpu_memory_fraction))


def _placement_estimate_ctx(role: WorkerRole, model_path: Path, meta: dict[str, str] | None) -> int:
    """Per-slot context the placement estimate sizes a role against.

    For chat this reserves KV for a usable floor (``_MIN_USABLE_CHAT_CTX``, or the
    user's ``cfg.num_ctx`` pin), capped by the model's trained ceiling -- not the
    single-GPU dynamic ctx (which shrinks to fit one card and then confirms a
    single-card placement) nor the full trained ceiling (which over-reserves). A
    model that cannot hold weights + this floor on one card is tensor-split.
    """
    from lilbee.core.config import cfg
    from lilbee.providers.engine_params import chat_ctx_ceiling

    if role is WorkerRole.CHAT:
        if cfg.num_ctx is not None:
            return cfg.num_ctx
        return min(
            chat_ctx_ceiling(meta, model_path), max(cfg.chat_n_ctx_target, _MIN_USABLE_CHAT_CTX)
        )
    return _role_ctx(role, model_path, meta)


def _placement_estimate_slots(role: WorkerRole, meta: dict[str, str] | None) -> int:
    """The slot count a role launches with, so the estimate's total ctx matches the launch.

    A tensor-split chat serves a single full-context sequence, so the placement total
    is the per-sequence ceiling, not ``ceiling x _CHAT_SLOTS`` (KV no launch allocates).
    """
    from lilbee.core.config import cfg

    if role is WorkerRole.CHAT:
        return _SPLIT_CHAT_SLOTS
    if role is WorkerRole.VISION:
        return max(1, cfg.vision_ocr_concurrency)
    if role is WorkerRole.RERANK and _rerank_mode_for(meta) is RerankMode.LLM:
        return LLM_RERANK_CONCURRENCY
    return _AUX_SLOTS


def _peak_estimator(model_refs: dict[WorkerRole, str]) -> PeakEstimator:
    """Per-device VRAM-vector estimator for the planner, bound to the configured models.

    Estimates each role at its launch ceiling (ctx x slots) with the candidate
    tensor-split ratio, so the planner reserves enough cards for the busiest one.
    """
    from lilbee.providers.engine_params import resolve_model_path
    from lilbee.providers.gguf_meta import read_gguf_metadata

    def estimate_peak(role: WorkerRole, ratio: tuple[int, ...]) -> tuple[int, ...]:
        path = resolve_model_path(model_refs[role])
        meta = read_gguf_metadata(path)
        mmproj = _vision_mmproj(model_refs[role]) if role is WorkerRole.VISION else None
        slots = _placement_estimate_slots(role, meta)
        ctx = _placement_estimate_ctx(role, path, meta)
        est = estimate_instance_footprint(
            path,
            ctx=ctx * slots,
            slots=slots,
            gpu_layers=_role_gpu_layers(role),
            flash_attn=_role_flash(role),
            kv_cache_type=_role_kv_cache_type(role),
            mmproj_path=mmproj,
            tensor_split=ratio,
            batch_size=_pooled_batch_size(role, _role_rerank_mode(role, meta), ctx),
        )
        return est.per_device_vram

    return estimate_peak


def _chat_split_ctx_objective(
    model_refs: dict[WorkerRole, str],
) -> tuple[SplitCtxFitter | None, int]:
    """The chat split's context fitter and target, or ``(None, 0)`` with no chat model.

    The fitter sizes a candidate shard's served context exactly as the launch does
    (:func:`fit_split_ctx`), so the planner widens chat onto idle cards only when a
    tighter shard would starve KV below the target. See docs/architecture.md.
    """
    if WorkerRole.CHAT not in model_refs:
        return None, 0
    from lilbee.providers.engine_params import resolve_model_path
    from lilbee.providers.gguf_meta import read_gguf_metadata

    path = resolve_model_path(model_refs[WorkerRole.CHAT])
    meta = read_gguf_metadata(path)
    target = _placement_estimate_ctx(WorkerRole.CHAT, path, meta)

    def fit(ratio: tuple[int, ...], per_device_free_bytes: Sequence[int]) -> int:
        # circular: fleet.ctx -> engine_params -> app.services
        from lilbee.providers.fleet.ctx import fit_split_ctx

        return fit_split_ctx(
            path,
            meta=meta,
            slots=_SPLIT_CHAT_SLOTS,
            ratio=ratio,
            per_device_free_bytes=per_device_free_bytes,
            gpu_layers=_role_gpu_layers(WorkerRole.CHAT),
            flash_attn=_role_flash(WorkerRole.CHAT),
            kv_cache_type=_role_kv_cache_type(WorkerRole.CHAT),
        )

    return fit, target


def _search_reservation(inputs: dict[WorkerRole, ModelPlacementInput]) -> int:
    """Total footprint of the placed search roles (all replicas), held back ahead
    of chat."""
    return sum(
        inputs[role].est_vram_bytes * inputs[role].replicas
        for role in _EMBED_ROLES
        if role in inputs
    )


def _server_model_inputs(
    roles: tuple[WorkerRole, ...] | None = None,
    *,
    unified_budget: int | None = None,
    device_count: int = 0,
) -> tuple[list[ModelPlacementInput], dict[WorkerRole, str], int]:
    """Build placement inputs for the configured server roles.

    The search and vision roles are estimated first; chat is then sized against the
    budget minus the search footprint (the ``reservation``) so a large chat cannot
    starve embed/rerank on a shared-memory host. ``device_count`` resolves an auto
    replica knob to one per GPU. When *roles* is given, only those are considered.
    Skips an unconfigured optional role, a vision model with no resolvable mmproj
    projector, and a role whose model is not installed on disk.
    """
    from lilbee.core.config import cfg
    from lilbee.providers.base import ProviderError

    inputs: dict[WorkerRole, ModelPlacementInput] = {}
    model_refs: dict[WorkerRole, str] = {}

    def consider(role: WorkerRole, *, chat_reservation: int = 0) -> None:
        if roles is not None and role not in roles:
            return
        ref = _SERVER_ROLE_PARAMS[role](cfg)
        if not ref:
            return  # unconfigured optional role -> no server
        if role is WorkerRole.VISION and _vision_mmproj(ref) is None:
            return  # no projector -> vision can't run on a server
        try:
            estimate = _estimate_role(
                role,
                ref,
                unified_budget=unified_budget,
                chat_reservation=chat_reservation,
                device_count=device_count,
            )
        except (ProviderError, OSError):
            # The configured model is not installed/resolvable. Skip this role
            # rather than failing the whole fleet build: search-only indexing
            # must not require an installed chat model, and a genuinely-needed
            # role surfaces a clear per-role error on first use instead of a
            # build-time traceback.
            log.warning("Skipping %s server: model %r is not installed.", role.value, ref)
            return
        inputs[role] = estimate
        model_refs[role] = ref

    # Estimate every non-chat role first so the search footprint is known, then size
    # chat against the remainder. The reservation only applies on a shared-memory
    # host; discrete GPUs pin each role to its own VRAM and pack independently.
    for role in _SERVER_ROLE_PARAMS:
        if role is not WorkerRole.CHAT:
            consider(role)
    reservation = _search_reservation(inputs) if unified_budget is not None else 0
    consider(WorkerRole.CHAT, chat_reservation=reservation)

    ordered = [inputs[role] for role in _SERVER_ROLE_PARAMS if role in inputs]
    return ordered, model_refs, reservation


def _launch_for(
    plan: InstancePlan,
    model_ref: str,
    binary: Path,
    by_index: dict[int, FleetDevice],
    *,
    unified_budget: int | None = None,
    chat_reservation: int = 0,
) -> InstanceLaunch:
    """Build the launch spec (argv + device-pinning env) for one planned instance."""
    from lilbee.providers.engine_params import (
        _EMBED_CTX_MARGIN,
        resolve_model_path,
    )  # circular: fleet.planning -> engine_params -> app.services
    from lilbee.providers.gguf_meta import read_gguf_metadata

    model_path = resolve_model_path(model_ref)
    weights_bytes = _weights_bytes(model_path)
    meta = read_gguf_metadata(model_path)
    from lilbee.core.config import cfg

    chosen = tuple(by_index[i] for i in plan.devices)
    is_chat = plan.role is WorkerRole.CHAT
    is_vision = plan.role is WorkerRole.VISION
    mmproj = _vision_mmproj(model_ref) if is_vision else None
    # A tensor-split chat serves one full-context sequence sized against the busiest
    # card's headroom. A cfg.num_ctx pin overrides the fit (handled by _role_ctx).
    multi_card_chat = is_chat and len(chosen) > 1
    split_chat = multi_card_chat and cfg.num_ctx is None
    if split_chat:
        # circular: fleet.ctx -> engine_params -> app.services
        from lilbee.providers.fleet.ctx import fit_split_ctx

        ctx = fit_split_ctx(
            model_path,
            meta=meta,
            slots=_SPLIT_CHAT_SLOTS,
            ratio=plan.tensor_split,
            per_device_free_bytes=[d.free_bytes for d in chosen],
            gpu_layers=_role_gpu_layers(WorkerRole.CHAT),
            flash_attn=_role_flash(WorkerRole.CHAT),
            kv_cache_type=_role_kv_cache_type(WorkerRole.CHAT),
        )
    else:
        ctx = _role_ctx(plan.role, model_path, meta)
    rerank_mode = _role_rerank_mode(plan.role, meta)
    is_llm_rerank = rerank_mode is RerankMode.LLM
    # A multi-card chat runs one slot; other roles size --parallel against the budget
    # the same way the estimator did so the launch matches the placement reservation.
    slots = (
        _SPLIT_CHAT_SLOTS
        if multi_card_chat
        else _slots_for(
            plan.role,
            model_path,
            ctx,
            mmproj_path=mmproj,
            unified_budget=unified_budget,
            chat_reservation=chat_reservation,
            rerank_mode=rerank_mode,
        )
    )
    spec = _server_spec(plan.role, rerank_mode, meta)
    # Cross-encoder embed/rerank pools the whole input in one batch; an LLM reranker
    # is generative and uses the default batching plus flash attention.
    cross_encoder_pooled = plan.role in _EMBED_ROLES and not is_llm_rerank
    argv = build_server_argv(
        binary=binary,
        spec=spec,
        model_path=model_path,
        devices=plan.devices,
        n_gpu_layers=_role_gpu_layers(plan.role),
        slots=slots,
        ctx_per_slot=ctx,
        tensor_split=plan.tensor_split,
        mmproj=mmproj,
        flash_attn=_flash_attn_flag() if (is_chat or is_vision or is_llm_rerank) else None,
        cache_type=_cache_type_flag() if is_chat else None,
        batch_size=_pooled_batch_size(plan.role, rerank_mode, ctx),
        threads=(os.cpu_count() or _DEFAULT_THREADS) if is_vision else None,
    )
    return InstanceLaunch(
        role=plan.role,
        argv=argv,
        env_overrides={**visible_env(chosen), **llama_server_runtime_env()},
        model=model_ref,
        # token_cap drives cross-encoder/embed input truncation; the LLM rerank path
        # doesn't truncate (it relies on the per-slot ctx headroom), so leave it None.
        token_cap=max(1, ctx - _EMBED_CTX_MARGIN) if cross_encoder_pooled else None,
        # Weights size scales the cold-load ready timeout (larger model = longer).
        weights_bytes=weights_bytes,
        # Slots is the chat concurrency the gate admits; ctx is what a client fits to.
        slots=slots,
        ctx=ctx,
        replica=plan.replica,
        rerank_mode=rerank_mode,
        device_vram=dict(zip(plan.devices, plan.per_device_vram, strict=False)),
    )


def resolve_devices(binary: Path) -> list[FleetDevice]:
    """Enumerate devices in the binary's index space, or the Vulkan VRAM probe.

    The binary's ``--list-devices`` is authoritative; when it enumerates nothing,
    fall back to the Vulkan VRAM probe, which reports the same index space.
    """
    from lilbee.providers.fleet.cuda_runtime import assert_cuda_devices_usable
    from lilbee.providers.fleet.gpu_select import enumerate_gpu_vram

    devices = probe_devices(binary)
    # A CUDA build that links a runtime it cannot init a GPU with must fail loud,
    # not silently fall back to CPU (the Vulkan VRAM probe below would mask it).
    assert_cuda_devices_usable(binary, devices)
    if not devices and model_cache.has_nvidia_gpu():
        log.warning(
            "This host has an NVIDIA GPU but the engine's device probe "
            "(%s --list-devices) reported none; placement is falling back to "
            "shared-memory mode with unpinned GPUs. Check the GPU driver, "
            "CUDA_VISIBLE_DEVICES, and that the llama-server build has CUDA support.",
            binary,
        )
    if not devices:
        devices = [
            FleetDevice("Vulkan", idx, "", vram, vram) for idx, vram in (enumerate_gpu_vram() or [])
        ]
    return devices


_DEVICE_PROBE_TTL_S = 2.0


class _ReadDeviceCache:
    """Short-TTL device-probe cache for the read/view path.

    Inspecting placement (GET placement/gpus, preview, ``placement show``)
    resolves devices on every call, which spawns a ``llama-server --list-devices``
    subprocess; a brief TTL collapses a burst of reads onto one probe. The launch
    path is never served from here -- it reaps stale servers first and sizes KV
    cache against live free VRAM, so it always probes fresh.
    """

    def __init__(self, ttl_s: float) -> None:
        self._ttl_s = ttl_s
        self._lock = threading.Lock()
        self._at: float | None = None
        self._devices: list[FleetDevice] | None = None

    def get(self, binary: Path) -> list[FleetDevice]:
        with self._lock:
            fresh = self._at is not None and time.monotonic() - self._at < self._ttl_s
            if self._devices is None or not fresh:
                self._devices = resolve_devices(binary)
                self._at = time.monotonic()
            return self._devices

    def clear(self) -> None:
        with self._lock:
            self._at = None
            self._devices = None


_read_device_cache = _ReadDeviceCache(_DEVICE_PROBE_TTL_S)


def clear_read_device_cache() -> None:
    """Drop the read-path device probe cache (e.g. after the fleet is reconfigured)."""
    _read_device_cache.clear()


def _unified_memory_budget(devices: list[FleetDevice]) -> int | None:
    """Shared-RAM placement budget (free RAM minus the OS floor) when there is no
    discrete GPU, else ``None``. Discrete GPUs load into dedicated VRAM, so system
    RAM is not the constraint there."""
    if devices:
        return None
    floor = min(
        _SYSTEM_MEMORY_FLOOR_CAP_BYTES,
        model_cache.total_system_memory() // _SYSTEM_MEMORY_FLOOR_DIVISOR,
    )
    return max(0, model_cache.free_system_memory() - floor)


def _resolve_placement(
    placement: PlacementSpec | None,
    inputs: list[ModelPlacementInput],
    model_refs: dict[WorkerRole, str],
    devices: list[FleetDevice],
    *,
    unified_budget: int | None,
) -> Placement:
    """Resolve a Placement from the manual spec when set, else the auto planner."""
    estimate_peak = _peak_estimator(model_refs)
    # Size placement against each card's TOTAL capacity, not its instantaneous
    # free VRAM. plan_launches always plans the complete fleet, so the plan
    # defines the full intended residency; charging it against live free_bytes
    # double-counts models the fleet has already loaded (a warm get_placement or
    # reload would then falsely report the plan as unplaceable). bb-a8f.
    capacity = {d.index: d.total_bytes for d in devices}
    if placement is not None:
        return placement_from_spec(
            placement,
            tuple(model_refs),
            capacity,
            estimate_peak=estimate_peak,
        )
    # The chat split's card count is decided against live free VRAM (what the launch
    # sizes its context against) so placement and launch agree; charging still uses
    # total capacity above, preserving the bb-a8f no-double-count invariant. A split
    # needs >=2 GPUs, so skip the chat model's gguf read entirely below that.
    chat_ctx_fit, chat_ctx_target = (
        _chat_split_ctx_objective(model_refs) if len(capacity) >= _MIN_SPLIT_GPUS else (None, 0)
    )
    return plan_placement(
        inputs,
        [(idx, total) for idx, total in capacity.items()],
        estimate_peak=estimate_peak,
        unified_budget=unified_budget,
        chat_ctx_fit=chat_ctx_fit,
        chat_ctx_target=chat_ctx_target,
        free_headroom={d.index: d.free_bytes for d in devices},
    )


@dataclass(frozen=True)
class ResolvedPlacement:
    """Devices + resolved instance plans + model refs for the placement view."""

    devices: tuple[FleetDevice, ...]
    instances: tuple[InstancePlan, ...]
    unplaceable_roles: tuple[WorkerRole, ...]
    model_refs: dict[WorkerRole, str]


def resolve_placement_plan(placement: PlacementSpec | None) -> ResolvedPlacement:
    """Probe devices and resolve the auto-or-manual placement, without launching."""
    from lilbee.providers.fleet.cuda_runtime import apply_cuda_runtime_env
    from lilbee.providers.fleet.gpu_env import apply_fleet_gpu_env

    apply_fleet_gpu_env()
    binary = resolve_llama_server()
    apply_cuda_runtime_env()
    devices = _read_device_cache.get(binary)
    unified_budget = _unified_memory_budget(devices)
    inputs, model_refs, _ = _server_model_inputs(None, unified_budget=unified_budget)
    resolved = _resolve_placement(
        placement, inputs, model_refs, devices, unified_budget=unified_budget
    )
    return ResolvedPlacement(
        devices=tuple(devices),
        instances=resolved.instances,
        unplaceable_roles=resolved.unplaceable_roles,
        model_refs=model_refs,
    )


def plan_launches(
    roles: tuple[WorkerRole, ...] | None,
    binary: Path,
    by_index: dict[int, FleetDevice],
    devices: list[FleetDevice],
) -> list[InstanceLaunch]:
    """Plan placement for *roles* (``None`` = all configured) and build their launches."""
    from lilbee.core.config import cfg

    unified_budget = _unified_memory_budget(devices)
    inputs, model_refs, reservation = _server_model_inputs(
        roles, unified_budget=unified_budget, device_count=len(devices)
    )
    spec = PlacementSpec.from_json(cfg.placement) if cfg.placement else None
    placement = _resolve_placement(spec, inputs, model_refs, devices, unified_budget=unified_budget)
    for role in placement.unplaceable_roles:
        log.warning(
            "%s model %s does not fit available memory and will not be served; "
            "free up memory or use a smaller model.",
            role.value,
            model_refs[role],
        )
    return [
        _launch_for(
            plan,
            model_refs[plan.role],
            binary,
            by_index,
            unified_budget=unified_budget,
            chat_reservation=reservation,
        )
        for plan in placement.instances
    ]


def plan_all_launches() -> list[InstanceLaunch]:
    """Apply GPU env, probe devices, and plan launches for every configured role.

    Disables crash-prone Vulkan layers / dual-vendor ICDs and applies any
    ``cfg.gpu_devices`` pin before the probe and plan (both inherit the env).
    """
    from lilbee.providers.fleet.cuda_runtime import apply_cuda_runtime_env
    from lilbee.providers.fleet.gpu_env import apply_fleet_gpu_env

    apply_fleet_gpu_env()
    binary = resolve_llama_server()
    # Put the CUDA-runtime wheels on the process path so the device probe sees the
    # same runtime the servers will, before resolve_devices enumerates GPUs.
    apply_cuda_runtime_env()
    devices = resolve_devices(binary)
    by_index = {d.index: d for d in devices}
    return plan_launches(None, binary, by_index, devices)
