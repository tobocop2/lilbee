"""Launch planning for the fleet: device probe, VRAM estimate, placement, argv."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lilbee.core.config.enums import KvCacheType
from lilbee.providers.fleet.adapters import ROLE_SPECS, build_server_argv
from lilbee.providers.fleet.binary import llama_server_runtime_env, resolve_llama_server
from lilbee.providers.fleet.devices import FleetDevice, probe_devices, visible_env
from lilbee.providers.fleet.fleet import Fleet, InstanceLaunch
from lilbee.providers.fleet.placement import InstancePlan, ModelPlacementInput, plan_placement
from lilbee.providers.fleet.vram import estimate_instance_footprint
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
# Nominal context for bootstrapping a tensor-split chat's slot count before its real
# per-slot context is known; small enough that weights dominate the footprint.
_SPLIT_BOOTSTRAP_CTX = 512
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

# RAM kept free for the OS when placing against system memory (no discrete GPU).
_SYSTEM_MEMORY_FLOOR_BYTES = 4 * 1024**3


def _slots_for(
    role: WorkerRole,
    model_path: Path,
    ctx: int,
    *,
    mmproj_path: Path | None = None,
    unified_budget: int | None = None,
    chat_reservation: int = 0,
) -> int:
    """Continuous-batching slots (--parallel) for a role's server.

    Chat batches concurrent turns; vision batches concurrent OCR pages since a
    one-page decode underutilizes the GPU; embed/rerank are single-slot (their
    batching is request-side). Chat and vision are memory-aware (gguf-parser
    footprint), so a small or shared host drops toward 1 instead of overcommitting.
    ``unified_budget`` caps sizing against free system RAM with no discrete GPU;
    ``chat_reservation`` is the search-role footprint held back from chat.
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
) -> ModelPlacementInput:
    """Estimate one role-model's footprint via gguf-parser (+ mmproj for vision).

    ``slots`` defaults to the role's resolved batching slots (chat and vision are
    memory-aware); ``chat_reservation`` shrinks chat to leave room for the search
    roles. Charges the unified footprint with no discrete GPU, else the VRAM one.
    """
    from lilbee.providers.engine_params import resolve_model_path
    from lilbee.providers.gguf_meta import read_gguf_metadata

    path = resolve_model_path(model_ref)
    mmproj = _vision_mmproj(model_ref) if role is WorkerRole.VISION else None
    meta = read_gguf_metadata(path)
    ctx = _role_ctx(role, path, meta)
    if slots is None:
        slots = _slots_for(
            role,
            path,
            ctx,
            mmproj_path=mmproj,
            unified_budget=unified_budget,
            chat_reservation=chat_reservation,
        )
    est = estimate_instance_footprint(
        path,
        ctx=ctx,
        slots=slots,
        gpu_layers=_role_gpu_layers(role),
        flash_attn=_role_flash(role),
        kv_cache_type=_role_kv_cache_type(role),
        mmproj_path=mmproj,
    )
    return ModelPlacementInput(
        role=role, est_vram_bytes=est.footprint(unified=unified_budget is not None)
    )


def _search_reservation(inputs: dict[WorkerRole, ModelPlacementInput]) -> int:
    """Total footprint of the placed search roles, held back ahead of chat."""
    return sum(inputs[role].est_vram_bytes for role in _EMBED_ROLES if role in inputs)


def _server_model_inputs(
    roles: tuple[WorkerRole, ...] | None = None,
    *,
    unified_budget: int | None = None,
) -> tuple[list[ModelPlacementInput], dict[WorkerRole, str], int]:
    """Build placement inputs for the configured server roles.

    The search and vision roles are estimated first; chat is then sized against the
    budget minus the search footprint (the ``reservation``) so a large chat cannot
    starve embed/rerank on a shared-memory host. When *roles* is given, only those
    are considered. Skips an unconfigured optional role, a vision model with no
    resolvable mmproj projector, and a role whose model is not installed on disk.
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
                role, ref, unified_budget=unified_budget, chat_reservation=chat_reservation
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
    data_dir: Path,
    by_index: dict[int, FleetDevice],
    *,
    unified_budget: int | None = None,
    chat_reservation: int = 0,
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
    mmproj = _vision_mmproj(model_ref) if is_vision else None
    # A tensor-split chat model's KV budget spans every device it occupies; sizing
    # its context against one GPU collapses it to the floor (see resolve_chat_ctx).
    # The slots count divides the shared --ctx-size, so bootstrap it first.
    split_chat = is_chat and len(chosen) > 1
    chat_available = sum(d.free_bytes for d in chosen) if split_chat else None
    chat_slots = (
        _slots_for(
            WorkerRole.CHAT, model_path, _SPLIT_BOOTSTRAP_CTX, chat_reservation=chat_reservation
        )
        if split_chat
        else 1
    )
    ctx = _role_ctx(plan.role, model_path, meta, chat_available, chat_slots)
    # Size slots the same way the estimator did so the launched --parallel matches
    # the placement estimate (chat honors the search reservation; mmproj via gguf-parser).
    slots = _slots_for(
        plan.role,
        model_path,
        ctx,
        mmproj_path=mmproj,
        unified_budget=unified_budget,
        chat_reservation=chat_reservation,
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
    inputs, model_refs, reservation = _server_model_inputs(roles, unified_budget=unified_budget)
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
            chat_reservation=reservation,
        )
        for plan in placement.instances
    ]


def plan_all_launches() -> list[InstanceLaunch]:
    """Apply GPU env, probe devices, and plan launches for every configured role.

    Disables crash-prone Vulkan layers / dual-vendor ICDs and applies any
    ``cfg.gpu_devices`` pin before the probe and plan (both inherit the env).
    """
    from lilbee.providers.fleet.gpu_env import apply_fleet_gpu_env

    apply_fleet_gpu_env()
    binary = resolve_llama_server()
    devices = resolve_devices(binary)
    by_index = {d.index: d for d in devices}
    return plan_launches(None, binary, by_index, devices)


def build_fleet(
    on_spawning: Callable[[WorkerRole], None] | None = None,
    on_spawned: Callable[[WorkerRole], None] | None = None,
) -> Fleet:
    """Resolve devices via the binary, plan placement, spawn and monitor the fleet."""
    from lilbee.core.config import cfg

    launches = plan_all_launches()
    fleet = Fleet(data_dir=cfg.data_dir, on_spawning=on_spawning, on_spawned=on_spawned)
    # Plan joint placement for every role, but spawn none: the provider brings up
    # each role on first use (warm_up_pool brings up all).
    fleet.start(launches, eager_roles=frozenset())
    return fleet
