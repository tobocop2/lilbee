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

# Server roles -> (slots, model-ref accessor). chat/embed are always configured;
# reranker_model/vision_model may be "" (unconfigured) -> skipped, so that role
# has no server. Vision additionally needs an mmproj projector.
_SERVER_ROLE_PARAMS: dict[WorkerRole, tuple[int, Callable[[Any], str]]] = {
    WorkerRole.CHAT: (_CHAT_SLOTS, lambda c: str(c.chat_model)),
    WorkerRole.EMBED: (_AUX_SLOTS, lambda c: str(c.embedding_model)),
    WorkerRole.RERANK: (_AUX_SLOTS, lambda c: str(c.reranker_model)),
    WorkerRole.VISION: (_AUX_SLOTS, lambda c: str(c.vision_model)),
}


def _role_ctx(role: WorkerRole, model_path: Path, meta: dict[str, str] | None) -> int:
    """Per-slot context for a role, derived as the in-process loader does.

    Embed/rerank use the embedding model's training context; vision uses the
    vision loader's training-context picker; chat honors ``cfg.num_ctx`` then
    falls back to the dynamic chat-ctx picker.
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
    return resolve_chat_ctx(model_path, meta)


def _role_gpu_layers(role: WorkerRole) -> int:
    """GPU-layer offload: chat honors ``cfg.n_gpu_layers``, others offload all layers."""
    from lilbee.providers.engine_params import resolve_n_gpu_layers

    return resolve_n_gpu_layers(embedding=role in _ALL_LAYER_ROLES)


def _flash_attn_flag() -> str:
    """``--flash-attn`` value for chat; on unless ``cfg.flash_attention`` is ``False``."""
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


def _estimate_role(role: WorkerRole, model_ref: str, *, slots: int) -> ModelPlacementInput:
    """Estimate one role-model's VRAM from its GGUF on disk (+ mmproj for vision)."""
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
    # Chat passes --cache-type; embed/rerank/vision run f16 KV, so estimate their
    # KV at f16 to match the runtime rather than the chat-tuned cfg.kv_cache_type.
    kv_type = cfg.kv_cache_type if role is WorkerRole.CHAT else KvCacheType.F16
    est = estimate_model_vram(
        weights, meta, ctx=ctx, slots=slots, kv_elem_bytes=KV_CACHE_TYPE_BYTES[kv_type]
    )
    return ModelPlacementInput(role=role, est_vram_bytes=est)


def _server_model_inputs(
    roles: tuple[WorkerRole, ...] | None = None,
) -> tuple[list[ModelPlacementInput], dict[WorkerRole, str]]:
    """Build placement inputs for the configured server roles.

    When *roles* is given, only those roles are considered. Skips an unconfigured
    optional role and a vision model with no resolvable mmproj projector.
    """
    from lilbee.core.config import cfg

    inputs: list[ModelPlacementInput] = []
    model_refs: dict[WorkerRole, str] = {}
    for role, (slots, accessor) in _SERVER_ROLE_PARAMS.items():
        if roles is not None and role not in roles:
            continue
        ref = accessor(cfg)
        if not ref:
            continue  # unconfigured optional role -> no server
        if role is WorkerRole.VISION and _vision_mmproj(ref) is None:
            continue  # no projector -> vision can't run on a server
        inputs.append(_estimate_role(role, ref, slots=slots))
        model_refs[role] = ref
    return inputs, model_refs


def _launch_for(
    plan: InstancePlan,
    model_ref: str,
    binary: Path,
    data_dir: Path,
    by_index: dict[int, FleetDevice],
) -> InstanceLaunch:
    """Build the launch spec (argv + device-pinning env) for one planned instance."""
    from lilbee.providers.engine_params import resolve_model_path
    from lilbee.providers.gguf_meta import read_gguf_metadata

    slots, _accessor = _SERVER_ROLE_PARAMS[plan.role]
    model_path = resolve_model_path(model_ref)
    weights_bytes = model_path.stat().st_size
    ctx = _role_ctx(plan.role, model_path, read_gguf_metadata(model_path))
    chosen = tuple(by_index[i] for i in plan.devices)
    is_chat = plan.role is WorkerRole.CHAT
    is_vision = plan.role is WorkerRole.VISION
    mmproj = _vision_mmproj(model_ref) if is_vision else None
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
        flash_attn=_flash_attn_flag() if is_chat else None,
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


def plan_launches(
    roles: tuple[WorkerRole, ...] | None,
    binary: Path,
    by_index: dict[int, FleetDevice],
    devices: list[FleetDevice],
) -> list[InstanceLaunch]:
    """Plan placement for *roles* (``None`` = all configured) and build their launches."""
    from lilbee.core.config import cfg

    inputs, model_refs = _server_model_inputs(roles)
    placement = plan_placement(inputs, [(d.index, d.free_bytes) for d in devices])
    return [
        _launch_for(plan, model_refs[plan.role], binary, cfg.data_dir, by_index)
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
    fleet.start(launches)
    return fleet
