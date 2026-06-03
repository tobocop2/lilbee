"""Launch planning for the fleet: device probe, VRAM estimate, placement, argv."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lilbee.core.config.enums import KV_CACHE_TYPE_BYTES, KvCacheType
from lilbee.providers.fleet.adapters import ROLE_SPECS, build_server_argv
from lilbee.providers.fleet.binary import llama_server_runtime_env, resolve_llama_server_binary
from lilbee.providers.fleet.devices import FleetDevice, probe_devices, visible_env
from lilbee.providers.fleet.fleet import Fleet, InstanceLaunch
from lilbee.providers.fleet.placement import (
    InstancePlan,
    ModelPlacementInput,
    estimate_model_vram,
    plan_placement,
)
from lilbee.providers.roles import WorkerRole

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

# Fleet-only concurrency: continuous-batching slots (--parallel) per server.
_CHAT_SLOTS = 4
_AUX_SLOTS = 1
_EMBED_ROLES = (WorkerRole.EMBED, WorkerRole.RERANK)
# Roles whose loaders offload every layer regardless of cfg.n_gpu_layers; only
# chat honors cfg.n_gpu_layers.
_ALL_LAYER_ROLES = (WorkerRole.EMBED, WorkerRole.RERANK, WorkerRole.VISION)
_FLASH_ON = "on"
_FLASH_OFF = "off"
_DEFAULT_THREADS = 4

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

# RAM kept free for the OS when placing against system memory (no discrete GPU).
_SYSTEM_MEMORY_FLOOR_BYTES = 4 * 1024**3


def _slots_for(
    role: WorkerRole,
    weights: int,
    meta: dict[str, str] | None,
    ctx: int,
    *,
    unified_budget: int | None = None,
) -> int:
    """Continuous-batching slots (--parallel) for a role's server.

    Chat batches concurrent turns; vision batches concurrent OCR pages since a
    one-page decode underutilizes the GPU; embed/rerank are single-slot (their
    batching is request-side). Chat and vision are VRAM-aware, so a small or
    shared GPU drops toward 1 instead of OOMing on the configured ceiling.
    ``unified_budget`` (free system RAM on a host with no discrete GPU) caps the
    sizing so a model steps down to fit free memory rather than being refused.
    """
    if role is WorkerRole.CHAT:
        return _resolve_chat_slots(weights, meta, ctx, unified_budget=unified_budget)
    if role is WorkerRole.VISION:
        return _resolve_vision_slots(weights, meta, ctx, unified_budget=unified_budget)
    return _AUX_SLOTS


def _resolve_chat_slots(
    weights: int, meta: dict[str, str] | None, ctx: int, *, unified_budget: int | None = None
) -> int:
    """Largest chat slot count (<= ``_CHAT_SLOTS``) whose weights + per-slot KV fit
    the memory budget; steps to 1 when none fit. KV is sized at ``cfg.kv_cache_type``
    to match the launch."""
    from lilbee.core.config import cfg

    return _fit_slots(
        _CHAT_SLOTS,
        weights,
        meta,
        ctx,
        kv_elem=KV_CACHE_TYPE_BYTES[cfg.kv_cache_type],
        budget=_slot_budget(_CHAT_VRAM_FRACTION, unified_budget),
    )


def _resolve_vision_slots(
    weights: int, meta: dict[str, str] | None, ctx: int, *, unified_budget: int | None = None
) -> int:
    """Largest OCR batching slot count (<= ``cfg.vision_ocr_concurrency``) that fits
    the memory budget; 1 when the ceiling is 1 or nothing larger fits. KV sized at f16."""
    from lilbee.core.config import cfg

    ceiling = max(1, cfg.vision_ocr_concurrency)
    if ceiling == 1:
        return 1
    return _fit_slots(
        ceiling,
        weights,
        meta,
        ctx,
        kv_elem=KV_CACHE_TYPE_BYTES[KvCacheType.F16],
        budget=_slot_budget(_VISION_VRAM_FRACTION, unified_budget),
    )


def _slot_budget(vram_fraction: float, unified_budget: int | None) -> int:
    """Memory budget for slot sizing: *vram_fraction* of usable VRAM, capped by
    ``unified_budget`` (free system RAM) when there is no discrete GPU so the count
    steps down to fit free memory instead of overcommitting."""
    from lilbee.core.config import cfg
    from lilbee.providers.model_cache import get_available_memory

    budget = int(get_available_memory(cfg.gpu_memory_fraction) * vram_fraction)
    if unified_budget is not None:
        budget = min(budget, unified_budget)
    return budget


def _fit_slots(
    ceiling: int,
    weights: int,
    meta: dict[str, str] | None,
    ctx: int,
    *,
    kv_elem: int,
    budget: int,
) -> int:
    """Largest slot count in ``1..ceiling`` whose weights + per-slot KV fit *budget*;
    1 when none larger fit."""
    for slots in range(ceiling, 1, -1):
        if (
            estimate_model_vram(weights, meta, ctx=ctx, slots=slots, kv_elem_bytes=kv_elem)
            <= budget
        ):
            return slots
    return 1


def _role_ctx(
    role: WorkerRole,
    model_path: Path,
    meta: dict[str, str] | None,
    chat_available_bytes: int | None = None,
    chat_slots: int = 1,
) -> int:
    """Per-slot context for a role, derived as the in-process loader does.

    Embed/rerank use the embedding model's training context; vision uses the
    vision loader's training-context picker; chat honors ``cfg.num_ctx`` then
    falls back to the dynamic chat-ctx picker. *chat_available_bytes* / *chat_slots*
    (set by the launch path for a tensor-split chat model) are the combined free
    VRAM of its devices and its ``--parallel`` count, so its per-slot context is
    sized against the whole split, divided across the batching slots.
    """
    from lilbee.core.config import cfg
    from lilbee.providers.engine_params import (
        EMBED_FALLBACK_CTX,
        resolve_chat_ctx,
        resolve_vision_ctx,
    )

    if role in _EMBED_ROLES:
        from lilbee.providers.gguf_meta import train_ctx_from_meta

        return train_ctx_from_meta(meta, fallback=EMBED_FALLBACK_CTX, model_path=model_path)
    if role is WorkerRole.VISION:
        return resolve_vision_ctx(model_path)
    if cfg.num_ctx is not None:
        return cfg.num_ctx
    return resolve_chat_ctx(model_path, meta, chat_available_bytes, chat_slots)


def _role_gpu_layers(role: WorkerRole) -> int:
    """GPU-layer offload: chat honors ``cfg.n_gpu_layers``, others offload all layers."""
    from lilbee.providers.engine_params import resolve_n_gpu_layers

    return resolve_n_gpu_layers(embedding=role in _ALL_LAYER_ROLES)


def _flash_attn_flag() -> str:
    """``--flash-attn`` for chat and vision: on unless ``cfg.flash_attention`` is ``False``."""
    from lilbee.core.config import cfg

    return _FLASH_OFF if cfg.flash_attention is False else _FLASH_ON


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
    role: WorkerRole, model_ref: str, *, slots: int | None = None, unified_budget: int | None = None
) -> ModelPlacementInput:
    """Estimate one role-model's VRAM from its GGUF on disk (+ mmproj for vision).

    ``slots`` defaults to the role's resolved batching slots (chat and vision are
    VRAM-aware); callers/tests may pass an explicit count.
    """
    from lilbee.core.config import cfg
    from lilbee.providers.engine_params import resolve_model_path
    from lilbee.providers.gguf_meta import read_gguf_metadata

    path = resolve_model_path(model_ref)
    weights = path.stat().st_size
    if role is WorkerRole.VISION:
        mmproj = _vision_mmproj(model_ref)
        if mmproj is not None:
            weights += mmproj.stat().st_size
    meta = read_gguf_metadata(path)
    ctx = _role_ctx(role, path, meta)
    if slots is None:
        slots = _slots_for(role, weights, meta, ctx, unified_budget=unified_budget)
    # Chat passes --cache-type; embed/rerank/vision run f16 KV, so estimate their
    # KV at f16 to match the runtime rather than the chat-tuned cfg.kv_cache_type.
    kv_type = cfg.kv_cache_type if role is WorkerRole.CHAT else KvCacheType.F16
    est = estimate_model_vram(
        weights, meta, ctx=ctx, slots=slots, kv_elem_bytes=KV_CACHE_TYPE_BYTES[kv_type]
    )
    return ModelPlacementInput(role=role, est_vram_bytes=est)


def _server_model_inputs(
    roles: tuple[WorkerRole, ...] | None = None,
    *,
    unified_budget: int | None = None,
) -> tuple[list[ModelPlacementInput], dict[WorkerRole, str]]:
    """Build placement inputs for the configured server roles.

    When *roles* is given, only those roles are considered. Skips an unconfigured
    optional role, a vision model with no resolvable mmproj projector, and a role
    whose configured model is not installed on disk.
    """
    from lilbee.core.config import cfg
    from lilbee.providers.base import ProviderError

    inputs: list[ModelPlacementInput] = []
    model_refs: dict[WorkerRole, str] = {}
    for role, accessor in _SERVER_ROLE_PARAMS.items():
        if roles is not None and role not in roles:
            continue
        ref = accessor(cfg)
        if not ref:
            continue  # unconfigured optional role -> no server
        if role is WorkerRole.VISION and _vision_mmproj(ref) is None:
            continue  # no projector -> vision can't run on a server
        try:
            estimate = _estimate_role(role, ref, unified_budget=unified_budget)
        except (ProviderError, OSError):
            # The configured model is not installed/resolvable. Skip this role
            # rather than failing the whole fleet build: search-only indexing
            # must not require an installed chat model, and a genuinely-needed
            # role surfaces a clear per-role error on first use instead of a
            # build-time traceback.
            log.warning("Skipping %s server: model %r is not installed.", role.value, ref)
            continue
        inputs.append(estimate)
        model_refs[role] = ref
    return inputs, model_refs


def _launch_for(
    plan: InstancePlan,
    model_ref: str,
    binary: Path,
    data_dir: Path,
    by_index: dict[int, FleetDevice],
    *,
    unified_budget: int | None = None,
) -> InstanceLaunch:
    """Build the launch spec (argv + device-pinning env) for one planned instance."""
    from lilbee.providers.engine_params import resolve_model_path
    from lilbee.providers.gguf_meta import read_gguf_metadata

    model_path = resolve_model_path(model_ref)
    weights_bytes = model_path.stat().st_size
    meta = read_gguf_metadata(model_path)
    chosen = tuple(by_index[i] for i in plan.devices)
    is_chat = plan.role is WorkerRole.CHAT
    is_vision = plan.role is WorkerRole.VISION
    # A tensor-split chat model's KV budget spans every device it occupies; sizing
    # its context against one GPU collapses it to the floor (see resolve_chat_ctx).
    # The slots count divides the shared --ctx-size, so pass it too.
    split_chat = is_chat and len(chosen) > 1
    chat_available = sum(d.free_bytes for d in chosen) if split_chat else None
    chat_slots = _slots_for(WorkerRole.CHAT, weights_bytes, meta, 0) if split_chat else 1
    ctx = _role_ctx(plan.role, model_path, meta, chat_available, chat_slots)
    mmproj = _vision_mmproj(model_ref) if is_vision else None
    # Size slots from the same weights the estimator used (vision counts mmproj) so
    # the launched --parallel matches the placement estimate.
    mmproj_bytes = mmproj.stat().st_size if mmproj is not None and mmproj.exists() else 0
    slots = _slots_for(
        plan.role, weights_bytes + mmproj_bytes, meta, ctx, unified_budget=unified_budget
    )
    argv = build_server_argv(
        binary=binary,
        spec=ROLE_SPECS[plan.role],
        model_path=model_path,
        devices=plan.devices,
        n_gpu_layers=_role_gpu_layers(plan.role),
        slots=slots,
        ctx_per_slot=ctx,
        tensor_split=plan.tensor_split,
        mmproj=mmproj,
        flash_attn=_flash_attn_flag() if (is_chat or is_vision) else None,
        cache_type=_cache_type_flag() if is_chat else None,
        batch_size=ctx if plan.role in _EMBED_ROLES else None,
        threads=(os.cpu_count() or _DEFAULT_THREADS) if is_vision else None,
    )
    return InstanceLaunch(
        role=plan.role,
        argv=argv,
        env_overrides={**visible_env(chosen), **llama_server_runtime_env()},
        model=model_ref,
        # Stamp the owning lilbee pid so a concurrent instance's reaper won't
        # touch this server (only a dead parent's orphans get reaped).
        port_file=data_dir / f"llama-server-{plan.role.value}-{os.getpid()}.port",
        # Embed/rerank truncate oversize inputs to the per-slot context.
        token_cap=ctx if plan.role in _EMBED_ROLES else None,
        # Weights size scales the cold-load ready timeout (larger model = longer).
        weights_bytes=weights_bytes,
        # Slots is the chat concurrency the gate admits; ctx is what a client fits to.
        slots=slots,
        ctx=ctx,
    )


def resolve_devices(binary: Path) -> list[FleetDevice]:
    """Enumerate devices in the binary's index space, or the Vulkan VRAM probe.

    The binary's ``--list-devices`` is authoritative; when it enumerates nothing,
    fall back to the Vulkan VRAM probe, which reports the same index space.
    """
    from lilbee.providers.fleet.gpu_select import enumerate_gpu_vram

    devices = probe_devices(binary)
    if not devices:
        devices = [
            FleetDevice("Vulkan", idx, "", vram, vram) for idx, vram in (enumerate_gpu_vram() or [])
        ]
    return devices


def _unified_memory_budget(devices: list[FleetDevice]) -> int | None:
    """Shared-RAM placement budget (free RAM minus the OS floor) when there is no
    discrete GPU, else ``None``. Discrete GPUs load into dedicated VRAM, so system
    RAM is not the constraint there."""
    if devices:
        return None
    from lilbee.providers.model_cache import free_system_memory

    return max(0, free_system_memory() - _SYSTEM_MEMORY_FLOOR_BYTES)


def plan_launches(
    roles: tuple[WorkerRole, ...] | None,
    binary: Path,
    by_index: dict[int, FleetDevice],
    devices: list[FleetDevice],
) -> list[InstanceLaunch]:
    """Plan placement for *roles* (``None`` = all configured) and build their launches."""
    from lilbee.core.config import cfg

    unified_budget = _unified_memory_budget(devices)
    inputs, model_refs = _server_model_inputs(roles, unified_budget=unified_budget)
    placement = plan_placement(
        inputs,
        [(d.index, d.free_bytes) for d in devices],
        unified_budget=unified_budget,
    )
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
            cfg.data_dir,
            by_index,
            unified_budget=unified_budget,
        )
        for plan in placement.instances
    ]


def build_fleet(
    on_spawning: Callable[[WorkerRole], None] | None = None,
    on_spawned: Callable[[WorkerRole], None] | None = None,
) -> Fleet:
    """Resolve devices via the binary, plan placement, spawn and monitor the fleet."""
    from lilbee.core.config import cfg
    from lilbee.providers.fleet.gpu_env import apply_fleet_gpu_env

    # Disable crash-prone Vulkan layers / dual-vendor ICDs and apply any
    # cfg.gpu_devices pin before the device probe and spawn (both inherit env).
    apply_fleet_gpu_env()
    binary = resolve_llama_server_binary()
    devices = resolve_devices(binary)
    by_index = {d.index: d for d in devices}
    launches = plan_launches(None, binary, by_index, devices)
    fleet = Fleet(data_dir=cfg.data_dir, on_spawning=on_spawning, on_spawned=on_spawned)
    # Plan joint placement for every role, but spawn none: the provider brings up
    # each role on first use (warm_up_pool brings up all).
    fleet.start(launches, eager_roles=frozenset())
    return fleet
