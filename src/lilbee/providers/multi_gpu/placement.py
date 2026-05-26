"""VRAM-aware placement planner for the multi-GPU llama-server fleet.

Estimates each role-model's VRAM footprint from GGUF metadata and bin-packs
instances across GPUs (first-fit-decreasing): a model that fits one GPU runs as a
single pinned instance, small models co-locate on a GPU with spare VRAM, and a
model too big for any single GPU is tensor-split across enough GPUs to fit. A role
that fits nowhere falls back to in-process. See docs/architecture.md for rationale.
"""

from __future__ import annotations

from dataclasses import dataclass

from lilbee.providers.roles import WorkerRole

# Mirrors vLLM's gpu_memory_utilization default: never pack a GPU past 90%.
_VRAM_USABLE_FRACTION = 0.9
# Flat per-instance overhead (CUDA context, compute buffers) beyond weights + KV.
_MODEL_OVERHEAD_BYTES = 1024**3


@dataclass(frozen=True)
class ModelPlacementInput:
    """A role's model and its estimated single-instance VRAM footprint."""

    role: WorkerRole
    est_vram_bytes: int


@dataclass(frozen=True)
class InstancePlan:
    """One planned llama-server instance.

    ``devices`` >1 means the model is split across them; ``tensor_split`` is the
    per-device proportion (free VRAM in GiB) so an unequal pair splits by capacity
    rather than evenly. Empty for a single-device instance.
    """

    role: WorkerRole
    devices: tuple[int, ...]
    tensor_split: tuple[int, ...] = ()


@dataclass(frozen=True)
class Placement:
    """Planner output: server instances plus roles left to run in-process."""

    instances: tuple[InstancePlan, ...]
    in_process_roles: tuple[WorkerRole, ...]


def estimate_model_vram(
    weights_bytes: int,
    meta: dict[str, str] | None,
    *,
    ctx: int,
    slots: int,
    kv_elem_bytes: int,
) -> int:
    """Estimate an instance's VRAM: weights + KV cache + flat overhead.

    Weights use the GGUF file size (the quantized weights map ~1:1 into VRAM). The
    KV cache scales with ``ctx x slots``; missing metadata yields a 0 KV term
    (weights + overhead still counted), which only under-estimates the aux roles
    that barely use KV.
    """
    kv = _estimate_kv_cache_bytes(meta, ctx=ctx, slots=slots, kv_elem_bytes=kv_elem_bytes)
    return weights_bytes + kv + _MODEL_OVERHEAD_BYTES


def _estimate_kv_cache_bytes(
    meta: dict[str, str] | None, *, ctx: int, slots: int, kv_elem_bytes: int
) -> int:
    """Coarse KV-cache size: ``2 x layers x kv_dim x ctx x slots x elem_bytes``."""
    if meta is None:
        return 0
    layers = _int_field(meta, "block_count")
    kv_dim = _int_field(meta, "embedding_length")
    if layers == 0 or kv_dim == 0:
        return 0
    per_token = 2 * layers * kv_dim * kv_elem_bytes
    return per_token * ctx * slots


def _int_field(meta: dict[str, str], key: str) -> int:
    """Parse an int GGUF metadata field, ``0`` when absent or unparseable."""
    try:
        return int(meta.get(key, "0") or "0")
    except ValueError:
        return 0


def plan_placement(
    models: list[ModelPlacementInput],
    devices: list[tuple[int, int]],
) -> Placement:
    """Bin-pack *models* onto *devices* (``[(index, vram_bytes), ...]``).

    First-fit-decreasing by footprint with a 90% headroom per GPU. A model that
    fits one GPU takes a single instance; one too big for any single GPU is
    tensor-split across the fewest GPUs whose combined headroom fits; a model that
    fits nowhere is returned as an in-process role.
    """
    remaining: dict[int, float] = {idx: vram * _VRAM_USABLE_FRACTION for idx, vram in devices}
    instances: list[InstancePlan] = []
    in_process: list[WorkerRole] = []

    for model in sorted(models, key=lambda m: m.est_vram_bytes, reverse=True):
        single = _best_single_device(model.est_vram_bytes, remaining)
        if single is not None:
            remaining[single] -= model.est_vram_bytes
            instances.append(InstancePlan(role=model.role, devices=(single,)))
            continue
        split = _devices_for_split(model.est_vram_bytes, remaining)
        if split is not None:
            ratio = tuple(max(1, int(remaining[idx] / 1024**3)) for idx in split)
            _charge_split(model.est_vram_bytes, split, remaining)
            instances.append(
                InstancePlan(role=model.role, devices=tuple(split), tensor_split=ratio)
            )
            continue
        in_process.append(model.role)

    return Placement(instances=tuple(instances), in_process_roles=tuple(in_process))


def _best_single_device(need: int, remaining: dict[int, float]) -> int | None:
    """Index of the device with the most free VRAM that still fits *need*."""
    candidates = [idx for idx, free in remaining.items() if free >= need]
    if not candidates:
        return None
    return max(candidates, key=lambda idx: remaining[idx])


def _devices_for_split(need: int, remaining: dict[int, float]) -> list[int] | None:
    """Fewest devices (most-free first) whose combined headroom fits *need*."""
    by_free = sorted(remaining, key=lambda idx: remaining[idx], reverse=True)
    chosen: list[int] = []
    total = 0.0
    for idx in by_free:
        chosen.append(idx)
        total += remaining[idx]
        if total >= need:
            return chosen
    return None


def _charge_split(need: int, split: list[int], remaining: dict[int, float]) -> None:
    """Debit *need* across *split* in proportion to each device's free VRAM."""
    total_free = sum(remaining[idx] for idx in split)
    for idx in split:
        remaining[idx] -= need * (remaining[idx] / total_free)
