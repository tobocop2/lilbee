"""Launch planning for the fleet: device probe, VRAM estimate, placement, argv."""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from lilbee.core.config.enums import KvCacheType
from lilbee.core.system import is_network_path
from lilbee.providers import model_cache
from lilbee.providers.base import ProviderError
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
from lilbee.providers.fleet.devices import (
    VULKAN_BACKEND,
    FleetDevice,
    host_lacks_nvlink,
    probe_devices,
    visible_env,
)
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
from lilbee.providers.fleet.placement_spec import PlacementError, PlacementSpec
from lilbee.providers.fleet.replicas import resolve_replica_count
from lilbee.providers.fleet.vram import estimate_instance_footprint, usable_vram_fraction
from lilbee.providers.model_cache import free_system_memory, total_system_memory
from lilbee.providers.model_ref import parse_model_ref
from lilbee.providers.roles import ROLE_REGISTRY, RerankMode, WorkerRole

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

# Fleet-only concurrency: continuous-batching slots (--parallel) per server.
_CHAT_SLOTS = 4
# Slots the PLACEMENT estimate reserves KV for on a tensor-split chat: one full
# window, the minimum any split must hold. The launch may serve more than this
# (see _resolve_split_chat_slots) when the cards measurably have room for several
# full windows, so this is a planning floor, not the served slot count.
_SPLIT_CHAT_SLOTS = 1
# Floor context the PLACEMENT estimate reserves KV for, so a large model is never
# single-carded into a KV corner too small for real use (a 17GB model on a 24GB
# card leaves ~no KV room -> n_ctx collapses to a few hundred tokens). Sizing the
# placement reserve against this floor forces a tensor-split when one card cannot
# hold weights + a usable context; the served ctx is then grown by resolve_chat_ctx
# (single) / fit_split_ctx (split) toward the cards' real headroom, with each
# sequence capped at the working-context target. A split may then serve several
# such sequences, so the served total can exceed the single-window reserve; the
# per-device headroom test in fit_split_ctx is what bounds it.
_MIN_USABLE_CHAT_CTX = 8192
# Embed and cross-encoder rerank serve one request at a time. Raising it was
# tried and measured worse: on 8xA40 with an 8B Q8 embedder and one ~100-token
# passage per request, --parallel 1 gave 133 docs/sec at 81% SM while
# --parallel 8 gave 100 at 63%. At one slot the card is already busy, so there is
# no stall for slot-batching to reclaim and the extra slots only add
# continuous-batching and KV-fragmentation overhead. Batch on the request side
# (embed_batch_sequences) instead.
_AUX_SLOTS = 1
# A tensor-split needs at least this many GPUs; below it the chat context objective
# (a gguf read) is pointless because the model can only single-card or stay unplaced.
_MIN_SPLIT_GPUS = 2
# Pooled single-slot search roles (embed/cross-encoder rerank) whose whole input
# batches in one pass; derived from the role registry.
_EMBED_ROLES = tuple(role for role, info in ROLE_REGISTRY.items() if info.pooled)
# Roles whose loaders offload every layer regardless of cfg.n_gpu_layers; only
# chat honors cfg.n_gpu_layers.
_ALL_LAYER_ROLES = tuple(role for role, info in ROLE_REGISTRY.items() if info.offload_all_layers)
_FLASH_ON = "on"
_FLASH_OFF = "off"
_FLASH_AUTO = "auto"
# llama-server's documented way to say "offload nothing": --device none.
_NO_DEVICE = "none"
# Backends pinned by the name the engine printed rather than through an env var,
# because their variables index a different space than --list-devices reports.
_NAME_PINNED_BACKENDS = frozenset({VULKAN_BACKEND, "SYCL"})
# Backends whose flash-attention coverage in llama.cpp is complete enough to ask
# for it outright. Vulkan and SYCL are behind CUDA's and have been incomplete on
# Intel's mesa driver, so those are left to the engine's own auto, which enables
# flash attention only where the backend really supports it.
_TRUSTED_FLASH_BACKENDS = frozenset({"CUDA", "ROCm", "HIP", "MTL", "Metal"})
# Roles to which flash attention applies; embed/rerank run without it.
_FLASH_ROLES = tuple(role for role, info in ROLE_REGISTRY.items() if info.flash_attn)


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
_SYSTEM_MEMORY_FLOOR_DIVISOR = 4
# A GPU driver still initializing at boot answers with no devices. Ask again
# before letting that decide the daemon's whole run; two extra probes cost a
# couple of seconds only on a host that has a card the engine could not see.
_PROBE_RETRIES = 2
_PROBE_RETRY_DELAY_S = 1.0

# A network filesystem makes mmap dangerous (page faults served over the wire can
# wedge the loader in uninterruptible I/O), so the chat server loads its weights
# into a malloc'd host copy (--no-mmap) whenever that copy fits in this fraction
# of total system RAM. Local disk keeps mmap: its lazy paging gives a faster first
# token on a cold cache -- the common desktop first launch -- and --no-mmap's
# buffered full read only wins on an already-hot cache (#474: 33s vs 43s for a
# 112GB model on 3 GPUs) while pessimizing cold start. Keyed on TOTAL memory
# (stable), not free (fluctuates), so replans do not flap the launch argv. The
# exact ceiling is tuned on a network-volume host.
_NO_MMAP_NETWORK_RAM_FRACTION = 0.85

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
    device: FleetDevice | None = None,
) -> int:
    """Continuous-batching slots (--parallel) for a role's server.

    Chat batches concurrent turns; vision batches concurrent OCR pages since a
    one-page decode underutilizes the GPU; an LLM reranker batches its per-candidate
    chat requests; embed and cross-encoder rerank are single-slot (their batching is
    request-side). The memory-aware roles drop toward 1 on a small or shared host
    instead of overcommitting. ``unified_budget`` caps sizing against free system RAM
    with no discrete GPU; ``chat_reservation`` is the search-role footprint held back
    from chat; ``device`` is the card the role was placed on, whose memory the
    budget comes from once placement has chosen one.
    """
    if role is WorkerRole.CHAT:
        return _resolve_chat_slots(
            model_path,
            ctx,
            mmproj_path=mmproj_path,
            unified_budget=unified_budget,
            chat_reservation=chat_reservation,
            device=device,
        )
    if role is WorkerRole.VISION:
        return _resolve_vision_slots(
            model_path, ctx, mmproj_path=mmproj_path, unified_budget=unified_budget, device=device
        )
    if role is WorkerRole.RERANK and rerank_mode is RerankMode.LLM:
        return _resolve_llm_rerank_slots(
            model_path, ctx, unified_budget=unified_budget, device=device
        )
    return _AUX_SLOTS


def _resolve_split_chat_slots(fit_fn: Callable[[int], int]) -> tuple[int, int]:
    """Largest split-chat slot count whose sequences each keep the full window.

    ``fit_fn(n)`` is the per-slot context that fits when serving ``n`` sequences
    (``fit_split_ctx``, capped at the working target and verified against real
    per-card headroom). More slots divide the KV, so a split whose cards hold
    several full windows can serve that many agents concurrently instead of one.
    Returns ``(slots, per_slot_ctx)``, falling to one slot when only one full
    window fits (or the fit degenerated to the floor), which preserves the
    max-context single-sequence behaviour on a tight card.

    Found by bisection rather than a scan because every ``fit_fn`` call is a
    complete binary search whose probes each shell out to gguf-parser, and the
    whole thing runs while this process holds the cross-process build lock that
    every other lilbee start waits on without a deadline. A descending scan paid
    for all of ``_CHAT_SLOTS - 1`` searches in exactly the tight-card case where
    none of them fit. Bisection is sound here because the fit is non-increasing
    in the slot count: more sequences divide the same headroom, so once a count
    fails no larger one can succeed.
    """
    full = fit_fn(1)
    if full <= model_cache._DYNAMIC_CTX_FLOOR:
        return 1, full
    low, high = 1, _CHAT_SLOTS
    while low < high:
        mid = (low + high + 1) // 2
        if fit_fn(mid) >= full:
            low = mid
        else:
            high = mid - 1
    return low, full


def _resolve_chat_slots(
    model_path: Path,
    ctx: int,
    *,
    mmproj_path: Path | None = None,
    unified_budget: int | None = None,
    chat_reservation: int = 0,
    device: FleetDevice | None = None,
) -> int:
    """Largest chat slot count (<= ``_CHAT_SLOTS``) whose footprint fits the budget
    after reserving the search roles; steps to 1 when none fit."""
    budget = _slot_budget(_CHAT_VRAM_FRACTION, unified_budget, device) - chat_reservation
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
    device: FleetDevice | None = None,
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
        budget=_slot_budget(_VISION_VRAM_FRACTION, unified_budget, device),
    )


def _resolve_llm_rerank_slots(
    model_path: Path,
    ctx: int,
    *,
    unified_budget: int | None = None,
    device: FleetDevice | None = None,
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
        budget=_slot_budget(_LLM_RERANK_VRAM_FRACTION, unified_budget, device),
        rerank_mode=RerankMode.LLM,
    )


def _slot_budget(
    vram_fraction: float, unified_budget: int | None, device: FleetDevice | None = None
) -> int:
    """Memory budget for slot sizing: *vram_fraction* of the usable memory on *device*
    (the fleet's smallest when placement has not chosen one yet), capped by
    ``unified_budget`` (free system RAM) when there is no discrete GPU so the count
    steps down to fit free memory instead of overcommitting."""
    budget = int(plan_sizing_budget(device) * vram_fraction)
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
    rerank_mode: RerankMode | None = None,
) -> int:
    """Largest slot count in ``1..ceiling`` whose instance footprint fits *budget*;
    1 when none larger fit."""
    from lilbee.providers.base import ProviderError

    for slots in range(ceiling, 1, -1):
        try:
            est = estimate_instance_footprint(
                model_path,
                ctx=ctx,
                slots=slots,
                gpu_layers=_role_gpu_layers(role),
                flash_attn=_role_flash(role, rerank_mode),
                kv_cache_type=_role_kv_cache_type(role),
                kv_cache_type_v=_role_kv_cache_type_v(role),
                mmproj_path=mmproj_path,
                expert_offload=_role_expert_offload(model_path),
            )
        except (ProviderError, OSError):
            # An unsizable model runs a single slot; the load decides the rest.
            return 1
        if est.footprint(unified=unified) <= budget:
            return slots
    return 1


def _role_ctx(
    role: WorkerRole,
    model_path: Path,
    meta: dict[str, str] | None,
    device: FleetDevice | None = None,
) -> int:
    """Per-slot context for a role, derived as the in-process loader does.

    Embed/rerank use the embedding model's training context; vision uses the
    vision loader's training-context picker; chat honors ``cfg.num_ctx`` then
    falls back to the single-GPU dynamic chat-ctx picker, sized against *device*
    once placement has chosen one. A tensor-split chat is sized against its
    per-device headroom instead (see :func:`fit_split_ctx`).
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
        return _pinned_chat_ctx(model_path, meta)
    return resolve_chat_ctx(model_path, meta, available_bytes=plan_sizing_budget(device))


def _pinned_chat_ctx(model_path: Path, meta: dict[str, str] | None) -> int:
    """``cfg.num_ctx``, clamped to what the model was trained for.

    Every unpinned resolver already clamps, and both docstrings here claimed the
    pin did too. It did not, so a pin past the trained window was passed straight
    to the engine, which clamps it silently and serves a different number than
    every budget was sized for.

    Only against a window that is actually known. A GGUF whose header cannot be
    read falls back to a default that is a guess, and contradicting an explicit
    pin with a guess would break the hosts where the header is the thing that is
    broken.
    """
    from lilbee.core.config import cfg

    pinned = cfg.num_ctx
    assert pinned is not None  # noqa: S101 - callers check; this documents the contract
    ceiling = _known_chat_ceiling(model_path, meta)
    if ceiling is None or pinned <= ceiling:
        return pinned
    log.warning(
        "num_ctx is set to %d but %s was trained for %d, so %d is what will be served. "
        "Lower num_ctx to stop planning against a window this model does not have.",
        pinned,
        model_path.name,
        ceiling,
        ceiling,
    )
    return ceiling


def _known_chat_ceiling(model_path: Path, meta: dict[str, str] | None) -> int | None:
    """The largest chat window this model is known to support, or ``None``.

    ``None`` when the GGUF header gave no usable context length and the user set
    no ``cfg.num_ctx_max``: there is then no measured ceiling, only a default.
    """
    from lilbee.core.config import cfg
    from lilbee.providers.gguf_meta import train_ctx_from_meta

    sentinel = -1
    trained = train_ctx_from_meta(meta, fallback=sentinel, model_path=model_path)
    known = [value for value in (trained, cfg.num_ctx_max) if value is not None and value > 0]
    return min(known) if known else None


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


def _fleet_backend() -> str | None:
    """The engine backend this host plans onto, or ``None`` when unknown.

    Prefers the plan snapshot so a whole planning pass answers consistently, and
    falls back to the short-TTL read cache rather than a fresh probe.
    """
    probe = _plan_probe_store.get()
    if probe is not None:
        return probe.devices[0].backend if probe.devices else None
    try:
        devices = _read_device_cache.get(resolve_llama_server())
    except (ProviderError, OSError):
        return None
    return devices[0].backend if devices else None


def _flash_attention_is_trusted() -> bool:
    """Whether to ask for flash attention outright rather than let the engine decide.

    Unknown backends answer yes, which keeps every host that works today on the
    argv it has now; only the backends known to lag get the engine's own auto.
    """
    backend = _fleet_backend()
    return backend is None or backend in _TRUSTED_FLASH_BACKENDS


def flash_attn_flag() -> str:
    """``--flash-attn`` argv value for chat and vision."""
    if not _flash_enabled():
        return _FLASH_OFF
    return _FLASH_ON if _flash_attention_is_trusted() else _FLASH_AUTO


def _role_launches_with_flash(role: WorkerRole, rerank_mode: RerankMode | None = None) -> bool:
    """Whether the launch asks the engine for flash attention on *role*.

    The one place that answers this. The registry marks RERANK as a non-flash
    role because a cross-encoder pools in one batch, but an LLM reranker is
    generative and launches exactly like chat, so the mode decides there.
    """
    if role is WorkerRole.RERANK:
        return rerank_mode is RerankMode.LLM
    return role in _FLASH_ROLES


def _role_flash(role: WorkerRole, rerank_mode: RerankMode | None = None) -> bool:
    """Whether the estimate may assume flash attention for *role*.

    The launch's own answer, narrowed to a definite ``on``. Under ``auto`` the
    engine decides at load time, and assuming it would size the KV cache below
    what the launch may need.
    """
    return _role_launches_with_flash(role, rerank_mode) and flash_attn_flag() == _FLASH_ON


def _role_kv_cache_type(role: WorkerRole) -> KvCacheType:
    """Chat honors ``cfg.kv_cache_type``; embed/rerank/vision run f16 KV."""
    from lilbee.core.config import cfg

    return cfg.kv_cache_type if role is WorkerRole.CHAT else KvCacheType.F16


def _replica_count(role: WorkerRole, device_count: int) -> int:
    """Requested data-parallel instances for *role* via the shared resolver."""
    return resolve_replica_count(role, device_count)


def _role_kv_cache_type_v(role: WorkerRole) -> KvCacheType:
    """The V cache type for *role*: the configured one only when flash attention is on.

    llama.cpp refuses a quantized V cache without flash attention ("V cache
    quantization requires flash_attn") and the server never starts, while a
    quantized K cache needs nothing. So V follows the setting only where flash
    attention is certain, and is f16 under ``auto`` or ``off``. That costs memory
    rather than a launch, and the estimate moves with it.
    """
    from lilbee.core.config.enums import KvCacheType

    configured = _role_kv_cache_type(role)
    return configured if flash_attn_flag() == _FLASH_ON else KvCacheType.F16


def chat_cache_type_flags() -> tuple[str | None, str | None]:
    """``(--cache-type-k, --cache-type-v)`` for chat; ``None`` leaves the f16 default."""
    from lilbee.core.config.enums import KvCacheType

    def flag(kind: KvCacheType) -> str | None:
        return None if kind is KvCacheType.F16 else kind.value

    return flag(_role_kv_cache_type(WorkerRole.CHAT)), flag(_role_kv_cache_type_v(WorkerRole.CHAT))


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
        flash_attn=_role_flash(role, rerank_mode),
        kv_cache_type=_role_kv_cache_type(role),
        kv_cache_type_v=_role_kv_cache_type_v(role),
        mmproj_path=mmproj,
        batch_size=_pooled_batch_size(role, rerank_mode, ctx),
        expert_offload=_role_expert_offload(path),
    )
    fp = est.footprint(unified=unified_budget is not None)
    if role is WorkerRole.CHAT and unified_budget is None:
        fp = _chat_serve_budget_footprint(fp)
    return ModelPlacementInput(
        role=role,
        est_vram_bytes=fp,
        replicas=_replica_count(role, device_count),
        est_ram_bytes=est.ram_bytes,
    )


def _chat_serve_budget_footprint(footprint: int) -> int:
    """Charge a chat instance against the serve budget, not the placement headroom.

    The planner fits instances within ``cfg.usable_vram_fraction`` of a card, but a
    single-card chat then sizes its KV cache against the smaller
    ``cfg.gpu_memory_fraction`` budget (``resolve_chat_ctx``). A model that fills a
    card at 0.9 leaves no room for KV at 0.75 and collapses to a few hundred tokens,
    so scale its placement footprint by the budget ratio: it then needs a
    tensor-split (pooling VRAM across cards) whenever single-carding it would starve
    its context. Small models are unaffected -- they fit the serve budget with KV
    room to spare.
    """
    from lilbee.core.config import cfg

    # Never below 1.0. The ratio only compensates while the serve budget is the
    # smaller of the two; a gpu_memory_fraction raised past the usable fraction
    # inverts it, and the same line that exists to charge chat more starts
    # charging it less than the model takes.
    return int(footprint * max(1.0, usable_vram_fraction() / cfg.gpu_memory_fraction))


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
            return _pinned_chat_ctx(model_path, meta)
        return apply_ctx_downshift(
            role,
            min(
                chat_ctx_ceiling(meta, model_path),
                max(cfg.chat_n_ctx_target, _MIN_USABLE_CHAT_CTX),
            ),
        )
    return apply_ctx_downshift(role, _role_ctx(role, model_path, meta))


def _placement_estimate_slots(role: WorkerRole, meta: dict[str, str] | None) -> int:
    """The slot count the placement estimate reserves KV for.

    A tensor-split chat reserves one full-context sequence here: a conservative
    floor for the card-count decision. The launch then fills the placed cards'
    real headroom with as many full-context slots as fit (``_resolve_split_chat_slots``),
    never exceeding what those cards hold, so a larger launch count can't OOM.
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
        rerank_mode = _role_rerank_mode(role, meta)
        est = estimate_instance_footprint(
            path,
            ctx=ctx,
            slots=slots,
            gpu_layers=_role_gpu_layers(role),
            flash_attn=_role_flash(role, rerank_mode),
            kv_cache_type=_role_kv_cache_type(role),
            kv_cache_type_v=_role_kv_cache_type_v(role),
            mmproj_path=mmproj,
            tensor_split=ratio,
            batch_size=_pooled_batch_size(role, rerank_mode, ctx),
            expert_offload=_role_expert_offload(path),
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
            kv_cache_type_v=_role_kv_cache_type_v(WorkerRole.CHAT),
            ctx_ceiling=target,
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


def _role_weights_bytes(role: WorkerRole, ref: str) -> int:
    """The model's weight bytes on disk (plus the mmproj for vision): a
    ground-truth lower bound on residency. 0 when the file cannot be resolved."""
    from lilbee.providers.base import ProviderError
    from lilbee.providers.engine_params import resolve_model_path

    try:
        size = _weights_bytes(resolve_model_path(ref))
        if role is WorkerRole.VISION:
            mmproj = _vision_mmproj(ref)
            if mmproj is not None:
                size += int(mmproj.stat().st_size)
    except (ProviderError, OSError):
        return 0
    return size


def _is_moe(meta: dict[str, str] | None) -> bool:
    """Whether the GGUF declares routed experts, so its experts can be offloaded."""
    count = (meta or {}).get("expert_count")
    try:
        return int(count) > 0 if count is not None else False
    except ValueError:
        return False


def expert_offload_all(meta: dict[str, str] | None) -> bool:
    """Whether to keep every layer's experts in system memory; MoE models only."""
    from lilbee.core.config import cfg

    return bool(cfg.cpu_moe) and _is_moe(meta)


def expert_offload_layers(meta: dict[str, str] | None) -> int | None:
    """How many layers' experts to keep in system memory, or None for no split.

    A non-positive ``n_cpu_moe`` offloads nothing (it would emit a no-op
    ``--n-cpu-moe 0``), so it reads as unset.
    """
    from lilbee.core.config import cfg

    if cfg.n_cpu_moe is None or cfg.n_cpu_moe < 1 or not _is_moe(meta):
        return None
    return cfg.n_cpu_moe


def _role_expert_offload(model_path: Path) -> tuple[str, ...]:
    """Expert patterns the launch will offload, for sizing the same way it runs.

    Reads the GGUF (cached) rather than taking metadata as an argument so every
    estimate site charges the same tensors the launch moves off the GPU.
    """
    from lilbee.providers.fleet.adapters import expert_offload_patterns
    from lilbee.providers.gguf_meta import read_gguf_metadata

    meta = read_gguf_metadata(model_path)
    return expert_offload_patterns(
        cpu_moe=expert_offload_all(meta), n_cpu_moe=expert_offload_layers(meta)
    )


def _expert_offload_configured() -> bool:
    """Whether the user asked for expert offload that would actually take effect.

    A non-positive ``n_cpu_moe`` offloads nothing, so it does not count.
    """
    from lilbee.core.config import cfg

    return bool(cfg.cpu_moe) or (cfg.n_cpu_moe is not None and cfg.n_cpu_moe >= 1)


def _weights_exceed_everything(size: int, *, total_vram: int, total_ram: int) -> bool:
    """True when a model's weights fit neither the GPUs nor system memory.

    File size is ground truth, not an estimate, so this bound cannot repeat the
    false-refusal class: no estimator error makes a 40 GiB file fit a 1 GiB box.
    Past both pools there is nowhere for a layer to go and no launch can win, so
    saying so beats a load that thrashes and then dies.
    """
    ceiling = total_vram + total_ram
    return ceiling > 0 and size > ceiling


def _weights_exceed_hardware(size: int, total_vram: int, *, is_moe: bool) -> bool:
    """True when this model cannot be served on this machine at all.

    Exceeding VRAM alone is not that. The engine chooses how many layers fit and
    keeps the rest in system memory, so a model larger than every card is a
    partial offload and lilbee's job is to launch it and say what will happen.
    Refusing there meant the fit never ran and the role was skipped, which left
    the user hand-tuning n_gpu_layers to get back what the engine does by itself.

    What still refuses is a model past VRAM and system memory together, where no
    arrangement of layers exists. A user-set n_gpu_layers or expert offload keeps
    standing the bound down entirely, since the user has said where the weights
    should go.
    """
    from lilbee.core.config import cfg

    if cfg.n_gpu_layers is not None:
        return False
    if is_moe and _expert_offload_configured():
        return False
    return _weights_exceed_everything(
        size, total_vram=total_vram, total_ram=model_cache.total_system_memory()
    )


def _vision_without_mmproj(role: WorkerRole, ref: str) -> bool:
    """True (with a warning) for a configured vision model whose mmproj is missing.

    The skip would silently disable OCR; the warning names the cause and the fix.
    """
    if role is not WorkerRole.VISION or _vision_mmproj(ref) is not None:
        return False
    log.warning(
        "Vision model %s has no mmproj (CLIP projector); OCR is disabled. "
        "Re-run 'lilbee model pull %s' to fetch the projector.",
        ref,
        ref,
    )
    return True


def _estimate_or_fallback(
    role: WorkerRole,
    ref: str,
    *,
    unified_budget: int | None,
    chat_reservation: int,
    device_count: int,
    total_vram: int,
    skipped_not_installed: dict[WorkerRole, str],
    host_committed: int = 0,
) -> ModelPlacementInput | None:
    """Size *role* for placement, degrading rather than refusing.

    A missing model is skipped and recorded; a sizing failure on an installed
    model falls back to its weight bytes; weights alone exceeding the physical
    VRAM refuse with a plain message (ground truth, not an estimate).
    """
    from lilbee.providers.base import ProviderError, ProviderErrorKind

    try:
        estimate = _estimate_role(
            role,
            ref,
            unified_budget=unified_budget,
            chat_reservation=chat_reservation,
            device_count=device_count,
        )
    except (ProviderError, OSError) as exc:
        if isinstance(exc, ProviderError) and exc.kind is ProviderErrorKind.NOT_FOUND:
            log.warning("Skipping %s server: model %r is not installed.", role.value, ref)
            skipped_not_installed[role] = ref
            return None
        return _sizing_failure_fallback(
            role,
            ref,
            exc,
            device_count=device_count,
            total_vram=total_vram,
            host_committed=host_committed,
        )
    return _admit_estimate(
        _floor_implausible_estimate(estimate, role, ref),
        role,
        ref,
        total_vram=total_vram,
        ram_bytes=estimate.est_ram_bytes,
        host_committed=host_committed,
    )


def _admit_estimate(
    estimate: ModelPlacementInput,
    role: WorkerRole,
    ref: str,
    *,
    total_vram: int,
    ram_bytes: int,
    host_committed: int = 0,
) -> ModelPlacementInput | None:
    """*estimate*, or ``None`` when this model cannot load on this machine.

    Two hardware bounds, one per kind of memory: the weights must fit the GPUs
    unless something offloads, and whatever offloading puts in system memory must
    fit the system.
    """
    weights = _role_weights_bytes(role, ref)
    if _weights_exceed_hardware(weights, total_vram, is_moe=_ref_is_moe(ref)):
        _warn_weights_exceed(role, ref, weights, total_vram)
        return None
    if total_vram > 0 and weights > total_vram:
        _warn_weights_spill(role, ref, weights, total_vram)
    if _host_memory_refuses(role, ref, ram_bytes, host_committed):
        return None
    return estimate


def _analytic_footprint_floor(
    weights: int, *, meta: dict[str, str] | None, ctx: int, slots: int
) -> int:
    """The least this instance can occupy: weights, its KV cache, and overhead.

    Used when the estimator cannot answer. Charging weight bytes alone was a
    knowing under-charge: the engine allocates a KV cache sized by context and
    slot count, plus compute buffers, and omitting all of it lets placement fit a
    model that cannot fit. The comment said the load would decide, and it did, by
    running out of memory.

    A floor rather than an estimate. It is derived from the header the same way
    the in-process sizing path derives it, and it is deliberately the smallest
    defensible number, because refusing a model that would have fit is its own
    failure. Without a readable header the per-token fallback still applies:
    zero is the one answer that is certainly wrong.
    """
    from lilbee.providers import model_cache
    from lilbee.providers.engine_params import _kv_elem_bytes_for_cfg

    kv_bytes = model_cache.kv_bytes_per_token(meta, _kv_elem_bytes_for_cfg()) * ctx * max(slots, 1)
    overhead = int(weights * model_cache._BUFFER_OVERHEAD_FRACTION)
    return weights + kv_bytes + overhead


def _estimate_is_implausible(*, estimated: int, floor: int) -> bool:
    """Whether *estimated* describes a load that cannot exist.

    Below the analytic floor, which is the model's own weight bytes plus the
    cache and buffers it was asked to hold, there is no arrangement of memory
    that serves it. A floor of zero means nothing could be computed to compare
    against, and a guess is not grounds to discard the only measurement there is.
    """
    return floor > 0 and 0 < estimated < floor


def _floor_implausible_estimate(
    estimate: ModelPlacementInput, role: WorkerRole, ref: str
) -> ModelPlacementInput:
    """*estimate*, or the analytic floor when the estimator returned less than one.

    The fallback floor otherwise fires only when the estimator cannot answer, so
    an answer that is well formed and impossible went straight through, and
    placement committed a card against a number the load then overran.
    """
    floor = _fallback_floor_for(role, ref, _role_weights_bytes(role, ref))
    if not _estimate_is_implausible(estimated=estimate.est_vram_bytes, floor=floor):
        return estimate
    log.warning(
        "The estimator sized the %s model %s at %.1f GiB, below the %.1f GiB its "
        "weights and cache alone need. Charging the floor instead; the estimate "
        "cannot be describing this load.",
        role.value,
        ref,
        estimate.est_vram_bytes / 1024**3,
        floor / 1024**3,
    )
    return replace(estimate, est_vram_bytes=floor)


def _sizing_failure_fallback(
    role: WorkerRole,
    ref: str,
    exc: Exception,
    *,
    device_count: int,
    total_vram: int,
    host_committed: int = 0,
) -> ModelPlacementInput | None:
    """Analytic-floor placement input for an installed model the estimator cannot
    size; ``None`` skips the role (the file is unresolvable, its weights alone
    exceed the hardware, or offloading it would exceed system memory).

    The host bound applies here too. Charging the whole floor to VRAM and
    skipping it let an unsizable model past a check every sized model faces."""
    weights = _role_weights_bytes(role, ref)
    if weights == 0:
        log.warning("Skipping %s server: could not size model %r (%s).", role.value, ref, exc)
        return None
    if _weights_exceed_hardware(weights, total_vram, is_moe=_ref_is_moe(ref)):
        _warn_weights_exceed(role, ref, weights, total_vram)
        return None
    floor = _fallback_floor_for(role, ref, weights)
    log.warning(
        "Could not size the %s model %s (%s). Charging %.1f GiB, its weights plus the "
        "cache and buffers it will allocate, which is a floor rather than an estimate: "
        "the load may still need more.",
        role.value,
        ref,
        exc,
        floor / 1024**3,
    )
    if _host_memory_refuses(role, ref, floor, host_committed):
        return None
    return ModelPlacementInput(
        role=role, est_vram_bytes=floor, replicas=_replica_count(role, device_count)
    )


def _fallback_floor_for(role: WorkerRole, ref: str, weights: int) -> int:
    """:func:`_analytic_footprint_floor` for *role*, reading what metadata it can."""
    from lilbee.providers.base import ProviderError
    from lilbee.providers.engine_params import resolve_model_path
    from lilbee.providers.gguf_meta import read_gguf_metadata

    try:
        path = resolve_model_path(ref)
        meta = read_gguf_metadata(path)
    except (ProviderError, OSError, ValueError):
        meta = None
        path = None
    ctx = _placement_estimate_ctx(role, path, meta) if path is not None else _MIN_USABLE_CHAT_CTX
    return _analytic_footprint_floor(
        weights, meta=meta, ctx=ctx, slots=_placement_estimate_slots(role, meta)
    )


def _ref_is_moe(ref: str) -> bool:
    """Whether *ref*'s GGUF declares routed experts; False when it cannot be read."""
    from lilbee.providers.base import ProviderError
    from lilbee.providers.engine_params import resolve_model_path
    from lilbee.providers.gguf_meta import read_gguf_metadata

    try:
        return _is_moe(read_gguf_metadata(resolve_model_path(ref)))
    except (ProviderError, OSError):
        return False


def _cpu_offload_in_play() -> bool:
    """Whether this configuration puts any of a model's weights in system memory.

    Expert offload moves the experts, a partial ``n_gpu_layers`` moves whole
    layers, and zero moves the model. Without one of these the engine keeps
    everything on the card and the estimator's host figure describes memory
    nobody will allocate.
    """
    from lilbee.core.config import cfg

    return _expert_offload_configured() or cfg.n_gpu_layers is not None


def _host_bytes_must_be_resident(role: WorkerRole, ref: str) -> bool:
    """Whether *role*'s host bytes have to fit RAM rather than page in and out.

    The estimator's host figure counts mmap pages, and llama.cpp maps CPU-side
    weights over that mapping instead of allocating them, so with mmap they are
    evictable page cache: a model far larger than RAM streams from disk and
    serves, which is a practiced setup for a large mixture-of-experts. Only
    ``--no-mmap`` turns them into a buffered read that must be resident, and the
    single path that asks for it is a chat model on a network filesystem.

    Anything this cannot determine counts as mappable, because a false refusal
    here has no override and costs the user a model that would have run.
    """
    if role is not WorkerRole.CHAT:
        return False
    try:
        from lilbee.providers.engine_params import resolve_model_path

        path = resolve_model_path(ref)
    except (ProviderError, OSError, ValueError):
        return False
    if not is_network_path(path):
        return False
    return _chat_no_mmap(_role_weights_bytes(role, ref), on_network_fs=True)


def _host_committed(admitted: Mapping[WorkerRole, ModelPlacementInput]) -> int:
    """System-memory bytes the roles already admitted to this plan will hold."""
    return sum(inp.est_ram_bytes for inp in admitted.values())


def _host_memory_refuses(role: WorkerRole, ref: str, ram_bytes: int, committed: int) -> bool:
    """Whether *role*'s system-memory half is too big for this machine to load.

    Charged only when something actually offloads, and only when the bytes must
    be resident: refusing a mapped model that would have streamed from disk is a
    false refusal with no override, which is worse than a slow load.

    Measured against the whole plan, not this role alone. Every role was
    previously compared to the entire machine on its own, so two roles that each
    fit and together do not were both admitted.
    """
    if not _cpu_offload_in_play() or ram_bytes <= 0:
        return False
    wanted = committed + ram_bytes
    total = total_system_memory()
    if total and wanted > total and _host_bytes_must_be_resident(role, ref):
        log.warning(
            "The %s model %s cannot load: this plan puts %.1f GiB in system memory, which "
            "cannot be paged out here, and the machine has %.1f GiB in total. Use a smaller "
            "model, or offload less.",
            role.value,
            ref,
            wanted / 1024**3,
            total / 1024**3,
        )
        return True
    free = free_system_memory()
    if free and wanted > free:
        log.warning(
            "Offloading the %s model %s brings this plan to %.1f GiB in system memory and "
            "only %.1f GiB is free. It will still load; close other programs if it swaps "
            "or runs slowly.",
            role.value,
            ref,
            wanted / 1024**3,
            free / 1024**3,
        )
    return False


def _warn_weights_exceed(role: WorkerRole, ref: str, weights: int, total_vram: int) -> None:
    log.warning(
        "The %s model %s cannot load: its weights are %.1f GiB and this machine has "
        "%.1f GiB of GPU memory and %.1f GiB of system memory, so there is nowhere "
        "for its layers to go. Use a smaller model or a smaller quantization.",
        role.value,
        ref,
        weights / 1024**3,
        total_vram / 1024**3,
        model_cache.total_system_memory() / 1024**3,
    )


def _warn_weights_spill(role: WorkerRole, ref: str, weights: int, total_vram: int) -> None:
    """Say that a model larger than the GPUs will run partly in system memory."""
    log.warning(
        "The %s model %s is %.1f GiB and this machine has %.1f GiB of GPU memory, so "
        "the engine will keep the layers that fit on the GPU and the rest in system "
        "memory. It will run, and it will be slower than a model that fits.",
        role.value,
        ref,
        weights / 1024**3,
        total_vram / 1024**3,
    )


def placeable_total_vram() -> int:
    """Physical VRAM across all cards, for the weights-exceed placeability bound.

    Physical total is box-state-independent (a running incumbent doesn't skew
    it), so it is safe to read without a clean box. Reuses the plan probe when
    one is captured; otherwise probes best-effort and returns ``0`` on failure,
    which disables only the weights-exceed filter (its own ``total > 0`` guard).
    """
    probe = _plan_probe_store.get()
    if probe is not None:
        return sum(d.total_bytes for d in probe.devices)
    from lilbee.providers.base import ProviderError
    from lilbee.providers.fleet.gpu_env import apply_fleet_gpu_env

    try:
        apply_fleet_gpu_env()
        return sum(d.total_bytes for d in resolve_devices(resolve_llama_server()))
    except (ProviderError, OSError):
        return 0


def role_model_placeable(role: WorkerRole, ref: str, total_vram: int) -> bool:
    """Whether a fresh plan would actually serve *role* on *ref*.

    Mirrors the planner's own drop conditions (SDK-routed role, vision without a
    projector, model not installed, weights exceeding physical VRAM) using the
    same primitives, so the acquisition ladder binds and replaces against what
    an engine can serve rather than the raw config. Without this a
    configured-but-unplaceable role keeps bind from ever matching a running
    engine and restarts the shared engine on every process start.
    """
    if parse_model_ref(ref).is_remote or _vision_without_mmproj(role, ref):
        return False
    weights = _role_weights_bytes(role, ref)  # 0 when not installed / unresolvable
    if weights == 0:
        return False
    return not _weights_exceed_hardware(weights, total_vram, is_moe=_ref_is_moe(ref))


def _server_model_inputs(
    roles: tuple[WorkerRole, ...] | None = None,
    *,
    unified_budget: int | None = None,
    device_count: int = 0,
    total_vram: int = 0,
) -> tuple[list[ModelPlacementInput], dict[WorkerRole, str], int, dict[WorkerRole, str]]:
    """Build placement inputs for the configured server roles.

    The search and vision roles are estimated first; chat is then sized against the
    budget minus the search footprint (the ``reservation``) so a large chat cannot
    starve embed/rerank on a shared-memory host. ``device_count`` resolves an auto
    replica knob to one per GPU. When *roles* is given, only those are considered.
    Skips an unconfigured optional role, a vision model with no resolvable mmproj
    projector, a role whose model is not installed on disk (returned as
    ``skipped_not_installed`` so a surface can say so), and a model whose weight
    bytes alone exceed ``total_vram`` (physically unloadable under all-GPU layers).
    A model the estimator cannot size is enrolled at its file size instead of
    skipped, so the load, not the estimator, decides.
    """
    from lilbee.core.config import cfg

    inputs: dict[WorkerRole, ModelPlacementInput] = {}
    model_refs: dict[WorkerRole, str] = {}
    skipped_not_installed: dict[WorkerRole, str] = {}

    def consider(role: WorkerRole, *, chat_reservation: int = 0) -> None:
        if roles is not None and role not in roles:
            return
        # chat/embed are always configured; reranker_model/vision_model may be ""
        # (unconfigured) -> skipped, so that role has no server.
        ref = str(getattr(cfg, ROLE_REGISTRY[role].config_field))
        if not ref:
            return  # unconfigured optional role -> no server
        if parse_model_ref(ref).is_remote:
            return  # SDK-routed role: no local server to plan, not a missing install
        if _vision_without_mmproj(role, ref):
            return  # no projector -> vision can't run on a server
        estimate = _estimate_or_fallback(
            role,
            ref,
            unified_budget=unified_budget,
            chat_reservation=chat_reservation,
            device_count=device_count,
            total_vram=total_vram,
            skipped_not_installed=skipped_not_installed,
            host_committed=_host_committed(inputs),
        )
        if estimate is None:
            return
        inputs[role] = estimate
        model_refs[role] = ref

    # Estimate every non-chat role first so the search footprint is known, then size
    # chat against the remainder. The reservation only applies on a shared-memory
    # host; discrete GPUs pin each role to its own VRAM and pack independently.
    for role in ROLE_REGISTRY:
        if role is not WorkerRole.CHAT:
            consider(role)
    reservation = _search_reservation(inputs) if unified_budget is not None else 0
    consider(WorkerRole.CHAT, chat_reservation=reservation)

    ordered = [inputs[role] for role in ROLE_REGISTRY if role in inputs]
    return ordered, model_refs, reservation, skipped_not_installed


def _non_chat_reservation(
    instances: Sequence[InstancePlan],
    inputs: Sequence[ModelPlacementInput],
    co_tenants: frozenset[WorkerRole] = frozenset(),
) -> dict[int, int]:
    """Per-device VRAM the non-chat role servers occupy, keyed by device index.

    A tensor-split chat shard must size its KV against the headroom left after the
    embed/rerank/vision servers on the same card, not the card's raw free VRAM, or
    it over-commits and OOMs at launch. Chat is excluded because it sizes its own
    weights. Chat's own swap-group siblings are excluded too: they are evicted while
    chat is resident, so their VRAM is chat's to use. That only holds when chat is
    itself a co-tenant; a co-tenant group that does not include chat runs behind its
    own swap process and can be resident beside chat, so it is charged normally.
    Non-chat roles are single-device, so each charges its full footprint (once per
    replica) to its card.
    """
    chat_siblings = co_tenants if WorkerRole.CHAT in co_tenants else frozenset()
    charge_by_role = {inp.role: inp.est_vram_bytes for inp in inputs}
    reserved: dict[int, int] = {}
    for inst in instances:
        if inst.role is WorkerRole.CHAT or inst.role in chat_siblings:
            continue
        charge = charge_by_role[inst.role]
        for device in inst.devices:
            reserved[device] = reserved.get(device, 0) + charge
    return reserved


def _charge_by_device(
    chosen: tuple[FleetDevice, ...], ratio: tuple[int, ...], total: int
) -> dict[str, int]:
    """What each of *chosen* was charged, keyed by the name the engine prints.

    A single-card instance carries the whole charge. A split carries it in the
    proportions it launches with, which is what the planner decided and therefore
    what the engine's own report should be compared against.
    """
    from lilbee.providers.fleet.readback import device_label

    if total <= 0 or not chosen:
        return {}
    if len(chosen) == 1:
        return {device_label(chosen[0]): total}
    weights = ratio if len(ratio) == len(chosen) else (1,) * len(chosen)
    denominator = sum(weights) or len(chosen)
    return {
        device_label(device): total * weight // denominator
        for device, weight in zip(chosen, weights, strict=True)
    }


def _launch_for(
    plan: InstancePlan,
    model_ref: str,
    binary: Path,
    by_index: dict[int, FleetDevice],
    *,
    unified_budget: int | None = None,
    chat_reservation: int = 0,
    reserved_by_device: dict[int, int] | None = None,
    est_vram_bytes: int = 0,
    model_path: Path | None = None,
) -> InstanceLaunch:
    """Build the launch spec (argv + device-pinning env) for one planned instance."""
    from lilbee.providers.engine_params import (
        _EMBED_CTX_MARGIN,
        resolve_model_path,
    )  # circular: fleet.planning -> engine_params -> app.services
    from lilbee.providers.gguf_meta import read_gguf_metadata

    # The self-check holds a downloaded file rather than a configured reference,
    # so it hands the path over instead of asking for one to be resolved.
    model_path = model_path or resolve_model_path(model_ref)
    weights_bytes = _weights_bytes(model_path)
    meta = read_gguf_metadata(model_path)
    from lilbee.core.config import cfg

    chosen = tuple(by_index[i] for i in plan.devices)
    # ctx and slots are sized against the card this role landed on, not the fleet's
    # smallest, which is all the pre-placement estimate had to go on. A role spread
    # over several cards has no one budget; the split chat, the only such role today,
    # sizes against per-device headroom below.
    placed_device = chosen[0] if len(chosen) == 1 else None
    is_chat = plan.role is WorkerRole.CHAT
    is_vision = plan.role is WorkerRole.VISION
    mmproj = _vision_mmproj(model_ref) if is_vision else None
    chat_on_network_fs = is_chat and is_network_path(model_path)
    if chat_on_network_fs and not _chat_no_mmap(weights_bytes, on_network_fs=True):
        log.warning(
            "Chat model %s is served from a network filesystem and is too large to load "
            "into host RAM; mmap over the network can stall the load in uninterruptible "
            "I/O. Stage it on local disk for a reliable load.",
            model_ref,
        )
    # A tensor-split chat serves one full-context sequence sized against the busiest
    # card's headroom. A cfg.num_ctx pin overrides the fit (handled by _role_ctx).
    multi_card_chat = is_chat and len(chosen) > 1
    split_chat = multi_card_chat and cfg.num_ctx is None
    if multi_card_chat and host_lacks_nvlink():
        log.warning(
            "Chat model %s is tensor-split across GPUs %s on a host without NVLink; "
            "generation is PCIe all-reduce bound and can be very slow. A model that fits "
            "on fewer cards will generate faster.",
            model_ref,
            list(plan.devices),
        )
    split_slots = _SPLIT_CHAT_SLOTS
    if split_chat:
        # circular: fleet.ctx -> engine_params -> app.services
        from lilbee.providers.fleet.ctx import fit_split_ctx

        reserved = reserved_by_device or {}
        # Headroom left after the embed/rerank servers on each shared card, not the
        # card's raw free VRAM, so the chat KV doesn't over-commit.
        per_device_free = [max(0, d.free_bytes - reserved.get(d.index, 0)) for d in chosen]

        def _split_fit(slots: int) -> int:
            return fit_split_ctx(
                model_path,
                meta=meta,
                slots=slots,
                ratio=plan.tensor_split,
                per_device_free_bytes=per_device_free,
                gpu_layers=_role_gpu_layers(WorkerRole.CHAT),
                flash_attn=_role_flash(WorkerRole.CHAT),
                kv_cache_type=_role_kv_cache_type(WorkerRole.CHAT),
                kv_cache_type_v=_role_kv_cache_type_v(WorkerRole.CHAT),
                ctx_ceiling=_placement_estimate_ctx(WorkerRole.CHAT, model_path, meta),
            )

        split_slots, ctx = _resolve_split_chat_slots(_split_fit)
    else:
        # Downshifted here and not only in the estimate: the role resolvers are
        # pure functions of model and config, so without this the retry after a
        # load OOM re-emits a byte-identical argv and dies the same way. The
        # split branch above already inherits it through its ctx_ceiling.
        ctx = apply_ctx_downshift(plan.role, _role_ctx(plan.role, model_path, meta, placed_device))
    rerank_mode = _role_rerank_mode(plan.role, meta)
    is_llm_rerank = rerank_mode is RerankMode.LLM
    # A multi-card chat runs as many full-context slots as its cards' KV headroom
    # holds (split_slots, one when a num_ctx pin skips the fit); other roles size
    # --parallel against the budget the same way the estimator did.
    slots = (
        split_slots
        if multi_card_chat
        else _slots_for(
            plan.role,
            model_path,
            ctx,
            mmproj_path=mmproj,
            unified_budget=unified_budget,
            chat_reservation=chat_reservation,
            rerank_mode=rerank_mode,
            device=placed_device,
        )
    )
    spec = _server_spec(plan.role, rerank_mode, meta)
    # Cross-encoder embed/rerank pools the whole input in one batch; an LLM reranker
    # is generative and uses the default batching plus flash attention.
    cross_encoder_pooled = plan.role in _EMBED_ROLES and not is_llm_rerank
    cache_type_k, cache_type_v = chat_cache_type_flags() if is_chat else (None, None)
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
        flash_attn=flash_attn_flag() if _role_launches_with_flash(plan.role, rerank_mode) else None,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        batch_size=_pooled_batch_size(plan.role, rerank_mode, ctx),
        no_mmap=is_chat and _chat_no_mmap(weights_bytes, on_network_fs=chat_on_network_fs),
        cpu_moe=expert_offload_all(meta),
        n_cpu_moe=expert_offload_layers(meta),
        device_names=_device_names(chosen) or _cpu_pin_when_every_device_was_refused(),
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
        # What placement charged this instance, for the post-launch check against
        # the engine's own report of what it really allocated.
        est_vram_bytes=est_vram_bytes,
        est_vram_by_device=_charge_by_device(chosen, plan.tensor_split, est_vram_bytes),
        est_unreported_bytes=_unreported_bytes(plan.role, mmproj),
    )


def build_single_role_launch(role: WorkerRole, model_path: Path) -> InstanceLaunch:
    """The launch the fleet would build for *role* serving *model_path*, alone.

    One construction path. The self-check used to assemble its own beside this
    one and the two disagreed on slot count, on the context that follows from it,
    on device pinning and on the tensor split, so a green check proved nothing
    about the launch serving actually performs, and a red one could be a
    configuration serving would never have chosen.

    Placement is the planner's, on the devices the plan snapshot holds, so the
    check runs on the card the role would really land on.
    """
    from lilbee.providers.fleet.cuda_runtime import apply_cuda_runtime_env
    from lilbee.providers.fleet.gpu_env import apply_fleet_gpu_env

    apply_fleet_gpu_env()
    binary = resolve_llama_server()
    apply_cuda_runtime_env(binary)
    devices = _plan_devices(binary)
    by_index = {d.index: d for d in devices}
    # The whole machine, since nothing else is resident during a self-check.
    placed = (min(by_index),) if by_index else ()
    plan = InstancePlan(role=role, devices=placed)
    return _launch_for(
        plan,
        str(model_path),
        binary,
        by_index,
        unified_budget=_unified_memory_budget(devices),
        model_path=model_path,
    )


def resolve_devices(binary: Path) -> list[FleetDevice]:
    """Enumerate devices in the binary's index space, or the Vulkan VRAM probe."""
    return _resolve_devices_and_refusal(binary)[0]


# The visibility variable each vendor's runtime reads, named in the warning so
# the reader checks the one that applies to the card they actually have.
_VENDOR_VISIBILITY_HINT = {
    "NVIDIA": "CUDA_VISIBLE_DEVICES",
    "AMD": "ROCR_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES",
    "Intel": "ONEAPI_DEVICE_SELECTOR",
}


def _warn_gpu_present_but_unenumerated(binary: Path) -> None:
    """Say so when the host has a GPU the engine did not list.

    Previously asked only whether an NVIDIA card was present, so an AMD or Intel
    host whose engine enumerated nothing produced the identical symptom, a fleet
    quietly planned for CPU, and said nothing. The vendor lookup is the same one
    the Vulkan ICD rules use, and it works on Windows as well as Linux.
    """
    from lilbee.providers.fleet.gpu_hardware import installed_gpu_vendor_ids
    from lilbee.providers.fleet.gpu_select import PCIVendorID

    present = installed_gpu_vendor_ids()
    names = sorted(
        v.name.title() if v.name == "INTEL" else v.name for v in PCIVendorID if v in present
    )
    if not names:
        return
    hints = sorted(
        {_VENDOR_VISIBILITY_HINT[name] for name in names if name in _VENDOR_VISIBILITY_HINT}
    )
    log.warning(
        "This host has a %s GPU but the engine's device probe (%s --list-devices) "
        "reported none; placement is falling back to shared-memory mode with unpinned "
        "GPUs. Check the GPU driver, %s, and that this llama-server build supports "
        "that GPU.",
        " and ".join(names),
        binary,
        " / ".join(hints) if hints else "the vendor's visibility variable",
    )


def _resolve_devices_and_refusal(binary: Path) -> tuple[list[FleetDevice], bool]:
    """:func:`resolve_devices`, plus whether every GPU the engine listed was refused.

    One function because both answers come from one ``--list-devices`` run, and
    that run costs a subprocess against a driver that may be wedged. Asking twice
    would pay it twice.

    The binary's ``--list-devices`` is authoritative, including when it lists
    nothing: it prints every non-CPU device it can use, so an empty list means
    the engine has no usable GPU rather than that we failed to look. The Vulkan
    VRAM probe is consulted only when the binary produced no output at all, and
    it reports the same index space. A
    probe that times out raises instead (a wedged GPU driver); falling through
    to the in-process Vulkan probe there could hang this thread unkillably.
    """
    from lilbee.providers.fleet.cuda_runtime import assert_gpu_devices_usable
    from lilbee.providers.fleet.gpu_select import enumerate_gpu_vram

    probe = probe_devices(binary)
    devices = probe.devices
    # A CUDA build that links a runtime it cannot init a GPU with must fail loud,
    # not silently fall back to CPU (the Vulkan VRAM probe below would mask it).
    # Only when the engine actually answered, though: a binary that does not
    # support --list-devices enumerated nothing because it was never asked, and
    # accusing its driver of failing to initialize would be both wrong and fatal.
    if probe.spoke_protocol:
        assert_gpu_devices_usable(binary, devices, probe.output)
    if not devices and probe.spoke_protocol:
        _warn_gpu_present_but_unenumerated(binary)
    if not devices and not probe.spoke_protocol:
        # Only when the binary never answered the question. An engine that ran
        # and listed nothing is reporting a fact, not a gap: believing the host
        # loader instead invents devices the engine cannot see. A CPU-only build
        # on a desktop with mesa is the clearest case, and the cost is not merely
        # a wrong device list. The fleet is planned onto GPUs, the pins are
        # no-ops, the shared-RAM guard is off because devices looked non-empty,
        # and every role then loads its full weights into system RAM while
        # running on the CPU anyway.
        #
        # Keyed on the exit code and the header rather than on there being no
        # output at all: the probe merges stderr into stdout, so a build that
        # predates --list-devices prints usage text and would otherwise be read
        # as an authoritative "no GPUs here".
        from lilbee.providers.fleet.gpu_select import integrated_vulkan_indices

        integrated = integrated_vulkan_indices()
        devices = [
            FleetDevice(
                VULKAN_BACKEND, idx, "", vram, free, unified=idx in integrated, from_loader=True
            )
            for idx, vram, free in (enumerate_gpu_vram() or [])
        ]
        if devices:
            log.warning(
                "The engine's device probe returned nothing, so placement is using "
                "the host's Vulkan loader instead and found %d device(s). If the "
                "engine has no Vulkan backend it will run on CPU regardless; set %s "
                "to override the engine location if that is wrong.",
                len(devices),
                "LILBEE_ENGINE_DIR",
            )
    return devices, probe.refused_all


_DEVICE_PROBE_TTL_S = 2.0
# A failed probe is cached much longer than a good one: each retry against a
# wedged GPU driver costs a full probe timeout, so a per-poll retry would stall
# every placement read for a minute at a time.
_DEVICE_PROBE_FAILURE_TTL_S = 60.0


class _ReadDeviceCache:
    """Short-TTL device-probe cache for the read/view path.

    Not a ``cachetools.TTLCache``: it caches the *failure* too, under its own
    longer TTL, and re-raises it. A memoizing cache stores return values only,
    so a failing probe would re-spawn the subprocess on every placement read.

    Inspecting placement (GET placement/gpus, preview, ``placement show``)
    resolves devices on every call, which spawns a ``llama-server --list-devices``
    subprocess; a brief TTL collapses a burst of reads onto one probe. A probe
    failure is cached too (with its own TTL) and re-raised to every read in the
    window. The launch path is never served from here -- it sizes against the
    clean-box plan snapshot below (captured after stale-server reaping).
    """

    def __init__(self, ttl_s: float, failure_ttl_s: float) -> None:
        self._ttl_s = ttl_s
        self._failure_ttl_s = failure_ttl_s
        self._lock = threading.Lock()
        self._at: float | None = None
        self._devices: list[FleetDevice] | None = None
        self._failure: ProviderError | None = None

    def get(self, binary: Path) -> list[FleetDevice]:
        with self._lock:
            ttl = self._ttl_s if self._failure is None else self._failure_ttl_s
            fresh = self._at is not None and time.monotonic() - self._at < ttl
            if fresh and self._failure is not None:
                raise self._failure
            if self._devices is None or not fresh:
                self._at = time.monotonic()
                try:
                    self._devices = resolve_devices(binary)
                except ProviderError as exc:
                    self._devices = None
                    self._failure = exc
                    raise
                self._failure = None
            return self._devices

    def clear(self) -> None:
        with self._lock:
            self._at = None
            self._devices = None
            self._failure = None


_read_device_cache = _ReadDeviceCache(_DEVICE_PROBE_TTL_S, _DEVICE_PROBE_FAILURE_TTL_S)


def clear_read_device_cache() -> None:
    """Drop the read-path device probe cache (e.g. after the fleet is reconfigured).

    Also drops what the host's Vulkan loader told us about device types, which is
    otherwise held for the process lifetime and would survive a driver reload or
    an eGPU being plugged in.
    """
    from lilbee.providers.fleet.gpu_select import (
        integrated_vulkan_indices,
        vulkan_device_types_by_name,
    )

    _read_device_cache.clear()
    vulkan_device_types_by_name.cache_clear()
    integrated_vulkan_indices.cache_clear()


@dataclass(frozen=True)
class _PlanProbe:
    """Clean-box memory snapshot every plan is sized against.

    Captured once, right after stale-server reaping and before the first build,
    when nothing lilbee owns is loaded. Reloads re-plan against this same
    snapshot instead of re-probing: a live probe under a loaded fleet reports
    our own residency as unavailable, which would shrink chat context and slot
    counts, widen splits, and (on a unified-memory host) evict roles outright.
    Launches stay a pure function of config + hardware + this snapshot, so the
    reload diff restarts only real changes. Cleared on full fleet teardown so
    the next boot probes the clean box afresh.
    """

    devices: tuple[FleetDevice, ...]
    # What one role may size its ctx and slots against, already scaled by
    # cfg.gpu_memory_fraction. System memory only on a host with no GPU.
    sizing_budget: int
    free_system: int
    # The engine listed GPUs and lilbee rejected all of them, so the plan is
    # CPU-shaped while the engine would still choose one of those devices.
    engine_devices_all_refused: bool = False


class _PlanProbeStore:
    """Holds the captured plan snapshot; a single instance below (no bare global)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._probe: _PlanProbe | None = None

    def set(self, probe: _PlanProbe) -> None:
        with self._lock:
            self._probe = probe

    def get(self) -> _PlanProbe | None:
        with self._lock:
            return self._probe

    def clear(self) -> None:
        with self._lock:
            self._probe = None


_plan_probe_store = _PlanProbeStore()


# Where the ladder stops. Sized for chat, below which the answers are too short
# to be useful, so a role that still will not load here has a real problem the
# planner cannot size its way out of and the failure should surface. Roles whose
# window already sits under it (a small embedding context) are left alone rather
# than raised to meet it, so for them the ladder is a no-op and the failure
# surfaces after the one retry.
MIN_DOWNSHIFT_CTX = 4096


class _CtxDownshiftStore:
    """How many halvings each role's auto context has taken after a load OOM.

    An estimate that was too optimistic is only recoverable if the retry asks
    for something different. Halving the auto context does that, and keeping the
    count here rather than in the launch means the whole plan is re-predicted
    against the smaller number, including the placement it implies.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._steps: dict[WorkerRole, int] = {}
        # The last unshifted context each role was sized from, recorded as it is
        # applied. Deciding whether another halving would change anything needs
        # the number being halved, and this is the only place that sees it.
        self._base: dict[WorkerRole, int] = {}

    def steps(self, role: WorkerRole) -> int:
        with self._lock:
            return self._steps.get(role, 0)

    def note_base(self, role: WorkerRole, ctx: int) -> None:
        with self._lock:
            self._base[role] = ctx

    def base(self, role: WorkerRole) -> int | None:
        with self._lock:
            return self._base.get(role)

    def step(self, role: WorkerRole) -> int:
        with self._lock:
            taken = self._steps.get(role, 0) + 1
            self._steps[role] = taken
            return taken

    def clear(self, role: WorkerRole | None = None) -> None:
        with self._lock:
            if role is None:
                self._steps.clear()
                self._base.clear()
                return
            self._steps.pop(role, None)
            self._base.pop(role, None)


_ctx_downshift_store = _CtxDownshiftStore()


def apply_ctx_downshift(role: WorkerRole, ctx: int) -> int:
    """*ctx* halved once per downshift step recorded for *role*, floored.

    Never more than *ctx*. The floor is a stopping point, not a target: applied
    to a context already below it (a small embedding window, a model trained for
    2048 tokens) a bare floor would hand back a larger number, and the retry
    after a load OOM would ask for more memory than the launch that just ran out
    of it. Such a role simply has nothing to give back, and its failure surfaces
    after the one retry instead.

    A user's ``cfg.num_ctx`` pin is returned untouched: serving a window smaller
    than the one that was asked for, without being asked, is worse than failing
    to load and saying so.
    """
    from lilbee.core.config import cfg

    if role is WorkerRole.CHAT and cfg.num_ctx is not None:
        return ctx
    _ctx_downshift_store.note_base(role, ctx)
    return _shifted(ctx, _ctx_downshift_store.steps(role))


def _shifted(ctx: int, steps: int) -> int:
    """*ctx* halved *steps* times, never below the floor and never above *ctx*."""
    return min(ctx, max(MIN_DOWNSHIFT_CTX, ctx >> steps)) if steps else ctx


def record_ctx_downshift(role: WorkerRole) -> bool:
    """Take one downshift step for *role*; False when there is none left to take.

    False means the retry would ask for the same thing again, so the caller must
    surface the load failure instead of respawning an identical launch.
    """
    from lilbee.core.config import cfg

    if role is WorkerRole.CHAT and cfg.num_ctx is not None:
        return False
    base = _ctx_downshift_store.base(role)
    if base is None:
        # Nothing has been sized for this role yet, so there is no number to
        # decide against. Allow one step rather than trusting that a plan always
        # runs first: an unbounded grant here would let a caller that never
        # sizes anything loop forever.
        if _ctx_downshift_store.steps(role):
            return False
        _ctx_downshift_store.step(role)
        return True
    steps = _ctx_downshift_store.steps(role)
    if _shifted(base, steps + 1) == _shifted(base, steps):
        return False
    _ctx_downshift_store.step(role)
    return True


def clear_ctx_downshift(role: WorkerRole | None = None) -> None:
    """Forget *role*'s recorded downshift, or every role's, back to full size.

    Called when a role's engine reports ready, which is proof the reduced plan
    loaded: keeping the reduction after that would carry a shrunken window into
    a machine that has since freed memory, or into a smaller model the user
    switched to, and would then refuse on its first failure with a budget it
    had already spent.
    """
    _ctx_downshift_store.clear(role)


def _probe_engine_devices() -> tuple[list[FleetDevice], bool]:
    """Apply the fleet GPU/CUDA env, resolve the binary, and enumerate devices.

    This is the wedge point: a missing binary raises NOT_FOUND, and a CUDA build
    that cannot init a GPU (a broken-runtime host) raises loud from resolve_devices
    rather than silently degrading. Device enumeration reads no residency, so it is
    safe to run while an incumbent engine is still up.
    """
    from lilbee.providers.fleet.cuda_runtime import apply_cuda_runtime_env
    from lilbee.providers.fleet.gpu_env import apply_fleet_gpu_env

    apply_fleet_gpu_env()
    binary = resolve_llama_server()
    apply_cuda_runtime_env(binary)
    devices, refused = _resolve_devices_and_refusal(binary)
    if devices:
        return devices, refused
    return _reprobe_while_a_gpu_is_installed(binary, refused)


def _reprobe_while_a_gpu_is_installed(
    binary: Path, refused: bool
) -> tuple[list[FleetDevice], bool]:
    """Ask again when the host has a GPU the engine did not list.

    The plan snapshot is taken once, on a clean box, and is not retaken until a
    full teardown, so an empty first answer decides the whole run. A GPU driver
    that is still initializing when the daemon starts, which is ordinary under
    systemd or right after a container gains a device, would leave a GPU host
    serving on CPU until someone noticed and restarted it.

    Only where a card is actually installed. A host with no GPU answers empty
    every time and must not pay a retry for it on every start.
    """
    from lilbee.providers.fleet.gpu_hardware import installed_gpu_vendor_ids

    if not installed_gpu_vendor_ids():
        return [], refused
    for attempt in range(1, _PROBE_RETRIES + 1):
        log.info(
            "The engine listed no GPU on a host that has one; asking again in %.1fs "
            "(attempt %d of %d) in case the driver is still initializing.",
            _PROBE_RETRY_DELAY_S,
            attempt,
            _PROBE_RETRIES,
        )
        time.sleep(_PROBE_RETRY_DELAY_S)
        clear_read_device_cache()
        devices, refused = _resolve_devices_and_refusal(binary)
        if devices:
            return devices, refused
    return [], refused


def assert_engine_probeable() -> None:
    """Raise if the engine cannot be probed; capture no snapshot.

    A build precondition that must run BEFORE stopping a replaceable incumbent:
    it surfaces a wedged GPU probe or an unusable CUDA runtime without taking the
    residency-dependent memory snapshot (that belongs on the clean box, after the
    stop, in capture_plan_probe). resolve_devices caches within its TTL, so the
    follow-up capture reuses this enumeration rather than re-probing the hardware.
    """
    _probe_engine_devices()


def capture_plan_probe() -> None:
    """Snapshot devices and memory for planning; call only on a clean box."""
    devices, refused_all = _probe_engine_devices()
    _plan_probe_store.set(
        _PlanProbe(
            devices=tuple(devices),
            sizing_budget=_device_sizing_budget(devices),
            free_system=model_cache.free_system_memory(),
            engine_devices_all_refused=refused_all,
        )
    )


def refresh_plan_devices() -> None:
    """Re-read which devices exist, keeping the clean-box memory figures.

    The snapshot is captured once and only a full teardown clears it, so an eGPU
    unplugged, a driver reset, or a VM hot-remove left the fleet pinning a device
    that is no longer there and every rebuild replanning onto it.

    Only the structural half is restated. The memory figures are what make a
    reload plan the way the boot did, and re-taking them while the fleet is
    resident would charge it against itself, which is the whole reason the
    snapshot exists.

    A probe that cannot run leaves the snapshot alone: the last known device list
    is a better answer than none, and the loud paths for an unreachable engine
    live in the build, not here.
    """
    probe = _plan_probe_store.get()
    if probe is None:
        return
    clear_read_device_cache()
    try:
        devices, refused_all = _probe_engine_devices()
    except (ProviderError, OSError) as exc:
        log.debug("Device rediscovery could not run, keeping the previous list: %s", exc)
        return
    if tuple(devices) == probe.devices:
        return
    log.info(
        "The set of GPUs changed since this fleet was planned (%d device(s) now, %d before); "
        "replanning against the ones that are here.",
        len(devices),
        len(probe.devices),
    )
    _plan_probe_store.set(
        _PlanProbe(
            devices=tuple(devices),
            sizing_budget=_device_sizing_budget(devices),
            free_system=probe.free_system,
            engine_devices_all_refused=refused_all,
        )
    )


def clear_plan_probe() -> None:
    """Drop the plan snapshot (full fleet teardown); the next build re-captures."""
    _plan_probe_store.clear()


def _cpu_pin_when_every_device_was_refused() -> tuple[str, ...]:
    """``("none",)`` when the engine offered GPUs that lilbee refused, else empty.

    Dropping a device from lilbee's view does not stop the engine using it. With
    no pin at all, ggml applies its own selection, and its fallback takes the
    first non-CPU adapter, which is exactly the paravirtual device just refused;
    with every layer offloaded by default the model then runs on it while
    placement budgeted against system RAM. Naming no device keeps the engine on
    the CPU the plan was shaped for.
    """
    probe = _plan_probe_store.get()
    if probe is None or not probe.engine_devices_all_refused:
        return ()
    log.warning(
        "The engine listed GPU devices that lilbee will not plan onto, so it is being "
        "run on the CPU. Serving from one of them would be slower than the CPU or fail "
        "outright, and placement has been sized for system RAM."
    )
    return (_NO_DEVICE,)


def _plan_devices(binary: Path) -> list[FleetDevice]:
    """Devices the plan paths size against: the snapshot, else a live probe."""
    probe = _plan_probe_store.get()
    return list(probe.devices) if probe is not None else resolve_devices(binary)


def plan_sizing_budget(device: FleetDevice | None = None) -> int:
    """Usable memory for ctx/slot sizing: *device*'s own, else the snapshot, else live."""
    from lilbee.core.config import cfg

    if device is not None:
        return int(device.total_bytes * cfg.gpu_memory_fraction)
    probe = _plan_probe_store.get()
    if probe is not None:
        return probe.sizing_budget
    return _device_sizing_budget(_live_sizing_devices())


def _device_sizing_budget(devices: Sequence[FleetDevice]) -> int:
    """Memory one role may size its ctx and slots against, in bytes.

    Read from the engine's own device report, which ran under the environment the
    servers will run under and states each device's memory whatever the backend.
    A host-memory read answers with system RAM on every host without an NVIDIA
    card, which gave a 24 GiB AMD card a budget the size of the machine, and on
    Apple Silicon it ignores that Metal will not allocate past
    ``recommendedMaxWorkingSetSize``, which is the figure the probe carries.

    The smallest device, since this is asked before placement has picked one;
    :func:`_launch_for` re-sizes against the card the role actually landed on.
    System memory only when the engine reports no device at all, where the fleet
    runs on the CPU and system memory is the budget.
    """
    from lilbee.core.config import cfg

    if devices:
        return int(min(d.total_bytes for d in devices) * cfg.gpu_memory_fraction)
    return int(model_cache.total_system_memory() * cfg.gpu_memory_fraction)


def _live_sizing_devices() -> list[FleetDevice]:
    """Devices to size against with no plan snapshot; empty when none can be read."""
    try:
        return _read_device_cache.get(resolve_llama_server())
    except (ProviderError, OSError):
        return []


def _plan_free_system_memory() -> int:
    """Free system RAM for the unified-memory budget: the snapshot, else live."""
    probe = _plan_probe_store.get()
    return probe.free_system if probe is not None else model_cache.free_system_memory()


def _unreported_bytes(role: WorkerRole, mmproj: Path | None) -> int:
    """Estimated bytes the engine allocates without printing a buffer line.

    A vision projector's weights: llama.cpp allocates them in clip's own loader,
    which prints a size but not the "buffer size = N MiB" shape the readback
    reads, so the report is short by exactly this and the self-check would warn
    on a load that was sized correctly.
    """
    if role is not WorkerRole.VISION or mmproj is None:
        return 0
    try:
        return mmproj.stat().st_size
    except OSError:
        return 0


def _chat_no_mmap(weights_bytes: int, *, on_network_fs: bool = False) -> bool:
    """Whether the chat server should malloc its weights instead of mmapping them.

    Local disk mmaps: lazy page-fault paging gives a faster first token on a cold
    cache -- the common desktop first launch -- matching mmap-by-default engines.
    ``--no-mmap``'s buffered full read only wins on an already-hot cache and it
    pessimizes cold start, so it is not worth defaulting on for local disk. A
    network filesystem still prefers the buffered read whenever the host copy
    fits, because mmap page faults served over the wire can wedge the loader in
    uninterruptible I/O (see ``_NO_MMAP_NETWORK_RAM_FRACTION``).
    """
    if not on_network_fs:
        return False
    return weights_bytes <= model_cache.total_system_memory() * _NO_MMAP_NETWORK_RAM_FRACTION


def _device_names(devices: tuple[FleetDevice, ...]) -> tuple[str, ...]:
    """``--device`` names for *devices*, empty when the backend pins through env.

    Vulkan and SYCL, because neither one's environment variable speaks the space
    the probe enumerated. Vulkan's indexes the raw loader enumeration while the
    names come from the engine's filtered list, so the two disagree wherever ggml
    drops or merges a device. SYCL's is not an index list at all but a selector
    over a backend runtime, so a device the engine calls ``SYCL1`` need not be
    Level Zero ordinal 1: OpenCL devices interleave, discarded devices shift the
    numbering, and multi-tile cards appear as sub-devices.

    ``--device`` sidesteps both by naming devices exactly as ``--list-devices``
    printed them, which is where these indices were read from. CUDA and ROCm
    keep composing their variables, which do share the probe's space.
    """
    if not devices or devices[0].backend not in _NAME_PINNED_BACKENDS:
        return ()
    if any(d.from_loader for d in devices):
        # These indices are raw loader ordinals, and --device speaks the engine's
        # own post-filter naming, so Vulkan1 here can name Vulkan0 there or
        # nothing at all. Sizing against them is still worth doing; pinning by
        # them is not. Left unpinned, ggml applies its own device selection,
        # which is the filtering lilbee is trying to agree with in the first
        # place. The env pin is not the answer either: it takes raw ordinals but
        # switches off the type filter, the support check and the dedup with them.
        return ()
    return tuple(f"{d.backend}{d.index}" for d in devices)


def _unified_memory_budget(devices: list[FleetDevice]) -> int | None:
    """Shared-RAM placement budget (free RAM minus the OS floor), or ``None``.

    ``None`` once any device has memory of its own, since dedicated VRAM is the
    constraint there rather than system RAM. A host whose only devices are
    integrated, and a host with no devices at all, both stay inside the system
    budget: their GPU memory is the system's memory.
    """
    # Only a device with memory of its own lifts the system-RAM constraint. An
    # integrated GPU or an Apple Silicon Mac reports a slice of the same RAM the
    # OS is using, so treating its total as headroom over-commits the machine by
    # roughly the whole system footprint.
    if any(not device.unified for device in devices):
        return None
    return _capped_by_device_memory(
        max(0, _plan_free_system_memory() - _system_memory_floor()), devices
    )


def _unified_admission_budget(devices: list[FleetDevice]) -> int | None:
    """Shared-RAM pool a role set is *admitted* against, or ``None`` if dedicated.

    Total installed RAM minus the OS floor, not what happens to be free. Sizing
    asks a different question and keeps using free RAM: how much context can be
    backed right now. Admission asks whether the machine can host this fleet at
    all, and the plan defines the whole intended residency, so charging it
    against a live figure refuses a 600 MB model on a box that is merely busy at
    the moment, which is what happened. The GPU path already charges total
    capacity for exactly this reason.
    """
    if _unified_memory_budget(devices) is None:
        return None
    return _capped_by_device_memory(
        max(0, model_cache.total_system_memory() - _system_memory_floor()), devices
    )


def _system_memory_floor() -> int:
    """RAM held back for the OS when placing against system memory.

    ``cfg.system_memory_reserve_gb``, still capped at a quarter of installed RAM:
    a fixed reserve leaves a 7-8 GB host with no budget at all and refuses even
    tiny models, so the proportional cap holds however the reserve is set.
    """
    from lilbee.core.config import cfg

    total = model_cache.total_system_memory()
    return min(int(cfg.system_memory_reserve_gb * 1024**3), total // _SYSTEM_MEMORY_FLOOR_DIVISOR)


def _capped_by_device_memory(budget: int, devices: Sequence[FleetDevice]) -> int:
    """*budget*, never above what the devices can address between them.

    A shared-memory device still has a ceiling of its own: an integrated GPU
    addresses a fixed aperture of system RAM, and Metal will not allocate past
    ``recommendedMaxWorkingSetSize``. Both report that ceiling as their total, so
    a host budget derived from installed RAM promises memory the devices cannot
    reach. Unchanged where the engine reports no device, since the fleet is then
    running on the CPU and the host figure is the true one.
    """
    if not devices:
        return budget
    return min(budget, sum(d.total_bytes for d in devices))


def _device_capacity(devices: list[FleetDevice], charge_against_free: bool) -> dict[int, int]:
    """Per-device memory placement may charge against, keyed by device index.

    A card's total is what it holds, not what is going spare. A compositor, a
    browser, or a training job sitting on VRAM is invisible in the total, and the
    usable fraction placement applies covers fragmentation and driver overhead
    rather than other tenants, so a plan fits on paper and OOMs at load.

    Free bytes answer that, but only where they mean "everyone else's residency":
    that is the clean-box snapshot, taken after stale servers are reaped and
    before anything is built. Read live on a warm box they also exclude the
    fleet's own models, and since a plan always describes the complete intended
    residency, charging them there would count the fleet against itself and
    report a running plan as unplaceable. Those callers keep the total.

    Placement applies its usable fraction to whatever this returns, so a card
    with a tenant keeps a proportional margin rather than being packed to its
    last free byte, where fragmentation is worst.
    """
    packable = _packable_devices(devices)
    if not charge_against_free:
        return {d.index: d.total_bytes for d in packable}
    return {d.index: min(d.total_bytes, d.free_bytes) for d in packable}


def _packable_devices(devices: list[FleetDevice]) -> list[FleetDevice]:
    """The devices bin-packing may charge against.

    An integrated GPU's memory is the host's. Packing it beside a dedicated card
    promises the same RAM twice, once to its own budget and once to everything
    else on the machine, and its heap is often the larger number, so the packer
    prefers it: a 32 GiB shared heap outbids a 24 GiB card that actually has the
    memory. Where a dedicated device exists it is the one to serve from, and the
    integrated one is left to the shared-memory budget.

    A host with nothing but integrated devices keeps them. There is nothing else
    to serve from, and that path is governed by the system budget rather than by
    per-device packing.
    """
    dedicated = [d for d in devices if not d.unified]
    return dedicated or devices


def _resolve_placement(
    placement: PlacementSpec | None,
    inputs: list[ModelPlacementInput],
    model_refs: dict[WorkerRole, str],
    devices: list[FleetDevice],
    *,
    unified_budget: int | None,
    charge_against_free: bool = False,
) -> Placement:
    """Resolve a Placement from the manual spec when set, else the auto planner."""
    estimate_peak = _peak_estimator(model_refs)
    capacity = _device_capacity(devices, charge_against_free)
    if placement is not None:
        return placement_from_spec(
            placement,
            tuple(model_refs),
            capacity,
            estimate_peak=estimate_peak,
        )
    # The chat split's card count is decided against the snapshot's free VRAM (what the
    # launch sizes its context against) so placement and launch agree. A split needs
    # >=2 GPUs, so skip the chat model's gguf read entirely below that.
    chat_ctx_fit, chat_ctx_target = (
        _chat_split_ctx_objective(model_refs) if len(capacity) >= _MIN_SPLIT_GPUS else (None, 0)
    )
    return plan_placement(
        inputs,
        [(idx, budget) for idx, budget in capacity.items()],
        estimate_peak=estimate_peak,
        unified_budget=unified_budget,
        chat_ctx_fit=chat_ctx_fit,
        chat_ctx_target=chat_ctx_target,
        free_headroom={d.index: d.free_bytes for d in devices},
    )


def _placement_or_auto(
    placement: PlacementSpec | None,
    inputs: list[ModelPlacementInput],
    model_refs: dict[WorkerRole, str],
    devices: list[FleetDevice],
    *,
    unified_budget: int | None,
    charge_against_free: bool = False,
) -> tuple[Placement, bool]:
    """Resolve a saved spec, falling back to auto when it no longer fits the hardware.

    Returns the placement and whether the spec was the one applied. Hardware moves
    under a saved placement: a card is removed, a driver stops enumerating a GPU, a
    container starts without one. Refusing to plan there takes chat, embed and
    ingest down over a pin set on hardware the host no longer has, so the fleet
    degrades to automatic placement and logs why. An interactive apply still fails
    loud (:func:`lilbee.app.placement.set_placement`), where the pin is what the
    caller just asked for and a silent substitution would be the surprise.
    """
    if placement is None:
        return _resolve_placement(
            None,
            inputs,
            model_refs,
            devices,
            unified_budget=unified_budget,
            charge_against_free=charge_against_free,
        ), False
    try:
        return _resolve_placement(
            placement,
            inputs,
            model_refs,
            devices,
            unified_budget=unified_budget,
            charge_against_free=charge_against_free,
        ), True
    except PlacementError as exc:
        log.warning(
            "The saved GPU placement does not fit this hardware (%s); using automatic "
            "placement instead. Set a new placement to replace it.",
            exc,
        )
    return _resolve_placement(
        None,
        inputs,
        model_refs,
        devices,
        unified_budget=unified_budget,
        charge_against_free=charge_against_free,
    ), False


@dataclass(frozen=True)
class ResolvedPlacement:
    """Devices + resolved instance plans + model refs for the placement view."""

    devices: tuple[FleetDevice, ...]
    instances: tuple[InstancePlan, ...]
    unplaceable_roles: tuple[WorkerRole, ...]
    model_refs: dict[WorkerRole, str]
    # Roles placed anyway despite not fitting, with the shortfall in bytes. The
    # planner has always known this and only logged it, so a surface showed a
    # tight role as comfortably placed.
    tight_roles: dict[WorkerRole, int] = field(default_factory=dict)
    co_tenants: frozenset[WorkerRole] = frozenset()
    # False when a spec was given but did not fit the hardware, so these instances
    # are the auto planner's and a surface must not present them as the manual plan.
    spec_applied: bool = True
    # Roles configured but skipped because their model isn't installed (role -> ref).
    # Distinct from unplaceable_roles (installed but won't fit); lets a surface show
    # "not downloaded" instead of an empty table on a fresh install.
    skipped_not_installed: dict[WorkerRole, str] = field(default_factory=dict)


def resolve_placement_plan(
    placement: PlacementSpec | None, *, fall_back_to_auto: bool = False
) -> ResolvedPlacement:
    """Probe devices and resolve the auto-or-manual placement, without launching.

    ``fall_back_to_auto`` reads *placement* as a saved setting rather than a
    request: one that no longer fits the hardware resolves to the auto plan with
    ``spec_applied`` False instead of raising.
    """
    from lilbee.providers.fleet.cuda_runtime import apply_cuda_runtime_env
    from lilbee.providers.fleet.gpu_env import apply_fleet_gpu_env

    apply_fleet_gpu_env()
    binary = resolve_llama_server()
    apply_cuda_runtime_env(binary)
    devices = _read_device_cache.get(binary)
    unified_budget = _unified_memory_budget(devices)
    inputs, model_refs, _, skipped_not_installed = _server_model_inputs(
        None, unified_budget=unified_budget, total_vram=sum(d.total_bytes for d in devices)
    )
    admission_budget = _unified_admission_budget(devices)
    if fall_back_to_auto:
        resolved, spec_applied = _placement_or_auto(
            placement, inputs, model_refs, devices, unified_budget=admission_budget
        )
    else:
        resolved = _resolve_placement(
            placement, inputs, model_refs, devices, unified_budget=admission_budget
        )
        spec_applied = placement is not None
    return ResolvedPlacement(
        devices=tuple(devices),
        instances=resolved.instances,
        unplaceable_roles=resolved.unplaceable_roles,
        model_refs=model_refs,
        co_tenants=resolved.co_tenants,
        skipped_not_installed=skipped_not_installed,
        spec_applied=spec_applied,
        tight_roles=dict(resolved.tight_roles),
    )


@dataclass(frozen=True)
class FleetPlan:
    """The servers to start, and the roles that share one swap group."""

    launches: tuple[InstanceLaunch, ...]
    co_tenants: frozenset[WorkerRole] = frozenset()
    # Configured roles left unplaced because their model isn't installed (role ->
    # ref), so the warm path can fail a not-installed chat with a named reason
    # instead of spinning the warm line forever.
    skipped_not_installed: dict[WorkerRole, str] = field(default_factory=dict)


def _log_placement_findings(placement: Placement, model_refs: dict[WorkerRole, str]) -> None:
    """Warn about placements that exceed the memory budget.

    Shared-memory roles that fit nowhere get no server (loading them would OOM the
    host). GPU roles are never refused: one whose estimate exceeds the free VRAM
    still loads on demand, with a warning carrying the shortfall.
    """
    for role in placement.unplaceable_roles:
        log.warning(
            "%s model %s does not fit available memory and will not be served; "
            "free up memory or use a smaller model.",
            role.value,
            model_refs[role],
        )
    for role, shortfall in placement.tight_roles.items():
        log.warning(
            "Memory is tight for the %s model %s: it is estimated to need %.1f GiB more "
            "GPU memory than is available. It will still load on demand, keeping the "
            "layers that fit on the GPU and the rest in system memory; if it runs "
            "slowly, free up GPU memory or use a smaller model.",
            role.value,
            model_refs[role],
            # A sub-0.05 GiB shortfall would render as "0.0 GiB more".
            max(shortfall / 1024**3, 0.1),
        )
    if placement.co_tenants:
        log.info(
            "%s share GPU memory and load on demand; only one is resident at a time.",
            ", ".join(sorted(role.value for role in placement.co_tenants)),
        )


def plan_launches(
    roles: tuple[WorkerRole, ...] | None,
    binary: Path,
    by_index: dict[int, FleetDevice],
    devices: list[FleetDevice],
) -> FleetPlan:
    """Plan placement for *roles* (``None`` = all configured) and build their launches."""
    from lilbee.core.config import cfg

    unified_budget = _unified_memory_budget(devices)
    inputs, model_refs, reservation, skipped_not_installed = _server_model_inputs(
        roles,
        unified_budget=unified_budget,
        device_count=len(devices),
        total_vram=sum(d.total_bytes for d in devices),
    )
    spec = PlacementSpec.from_json(cfg.placement) if cfg.placement else None
    placement, _spec_applied = _placement_or_auto(
        spec,
        inputs,
        model_refs,
        devices,
        unified_budget=_unified_admission_budget(devices),
        # Only the clean-box snapshot's free bytes mean "what other tenants hold";
        # a live probe here would also be missing the fleet's own residency.
        charge_against_free=_plan_probe_store.get() is not None,
    )
    _log_placement_findings(placement, model_refs)
    reserved_by_device = _non_chat_reservation(placement.instances, inputs, placement.co_tenants)
    charged = {inp.role: inp.est_vram_bytes for inp in inputs}
    return FleetPlan(
        launches=tuple(
            _launch_for(
                plan,
                model_refs[plan.role],
                binary,
                by_index,
                unified_budget=unified_budget,
                chat_reservation=reservation,
                reserved_by_device=reserved_by_device,
                est_vram_bytes=charged.get(plan.role, 0),
            )
            for plan in placement.instances
        ),
        co_tenants=placement.co_tenants,
        skipped_not_installed=skipped_not_installed,
    )


def plan_all_launches() -> FleetPlan:
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
    devices = _plan_devices(binary)
    by_index = {d.index: d for d in devices}
    return plan_launches(None, binary, by_index, devices)
