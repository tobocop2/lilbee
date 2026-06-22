"""VRAM-aware placement planner for the multi-GPU llama-server fleet.

Estimates each role-model's VRAM footprint from GGUF metadata and bin-packs
instances across GPUs (first-fit-decreasing): a model that fits one GPU runs as a
single pinned instance, small models co-locate on a GPU with spare VRAM, and a
model too big for any single GPU is tensor-split across enough GPUs to fit. A role
that fits nowhere gets no server (its calls error). See docs/architecture.md.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from lilbee.providers.fleet.placement_spec import PlacementError, PlacementSpec
from lilbee.providers.fleet.vram import USABLE_VRAM_FRACTION
from lilbee.providers.roles import WorkerRole

# Search-critical roles reserved ahead of the elastic chat model in a shared pool,
# so a large chat can never crowd embed/rerank out (which would 503 every search).
_SEARCH_ROLES = (WorkerRole.EMBED, WorkerRole.RERANK)

# (role, per-device tensor-split ratio) -> the instance's per-device VRAM footprint
# vector aligned to that ratio. A split is accepted only when every card's entry
# fits its own headroom, and each card is charged its own entry, not the sum.
PeakEstimator = Callable[[WorkerRole, tuple[int, ...]], tuple[int, ...]]


@dataclass(frozen=True)
class ModelPlacementInput:
    """A role's model, its estimated single-instance footprint, and replica count.

    ``replicas`` > 1 requests N data-parallel instances (one per GPU) for the role,
    each charged ``est_vram_bytes``; capped at runtime by the GPUs with room.
    """

    role: WorkerRole
    est_vram_bytes: int
    replicas: int = 1


@dataclass(frozen=True)
class InstancePlan:
    """One planned llama-server instance.

    ``devices`` >1 means the model is split across them; ``tensor_split`` is the
    per-device proportion (free VRAM in GiB) so an unequal pair splits by capacity
    rather than evenly. Empty for a single-device instance. ``replica`` is the
    instance's index within its role's data-parallel pool (0 for a single server).
    """

    role: WorkerRole
    devices: tuple[int, ...]
    tensor_split: tuple[int, ...] = ()
    replica: int = 0


@dataclass(frozen=True)
class Placement:
    """Planner output: server instances plus roles that fit on no device.

    ``unplaceable_roles`` get no server, so a call to them surfaces a
    ``ProviderError`` (there is no in-process fallback).
    """

    instances: tuple[InstancePlan, ...]
    unplaceable_roles: tuple[WorkerRole, ...]


def plan_placement(
    models: list[ModelPlacementInput],
    devices: list[tuple[int, int]],
    *,
    estimate_peak: PeakEstimator,
    unified_budget: int | None = None,
) -> Placement:
    """Bin-pack *models* onto *devices* (``[(index, vram_bytes), ...]``).

    First-fit-decreasing by footprint with a 90% headroom per GPU. A model that
    fits one GPU takes a single instance; one too big for any single GPU is
    tensor-split across the fewest GPUs whose per-device share (from
    *estimate_peak*) each fits; a model that fits nowhere is an unplaceable role.

    No GPU devices is the CPU/unified-memory case (a GPU-less host, or an Apple
    Silicon box where the probe found nothing): roles run as single un-pinned
    instances. ``unified_budget`` (free system RAM, bytes) gates them against one
    shared pool so an oversize model is unplaceable instead of OOM-livelocking the
    host; ``None`` keeps the legacy ungated behavior.
    """
    if not devices:
        if unified_budget is None:
            return Placement(
                instances=tuple(
                    InstancePlan(role=m.role, devices=(), replica=r)
                    for m in models
                    for r in range(m.replicas)
                ),
                unplaceable_roles=(),
            )
        return _place_shared_memory(models, unified_budget)
    remaining: dict[int, float] = {idx: vram * USABLE_VRAM_FRACTION for idx, vram in devices}
    instances: list[InstancePlan] = []
    unplaceable: list[WorkerRole] = []

    # Single-instance roles first; then data-parallel replicas fill the remaining
    # headroom. Within the singles, place the search-critical roles (embed/rerank)
    # before chat: chat tensor-splits across cards and, placed first, can claim
    # them all and leave an essential search role unplaceable. Search-first here
    # mirrors the shared-memory path's reservation.
    singles = [m for m in models if m.replicas <= 1]
    replicated = [m for m in models if m.replicas > 1]
    for model in sorted(singles, key=_shared_pool_order):
        plan = _place_single(model, remaining, estimate_peak)
        if plan is None:
            unplaceable.append(model.role)
        else:
            instances.append(plan)
    for model in replicated:
        replica_plans = _place_replicas(model, remaining)
        if replica_plans:
            instances.extend(replica_plans)
        else:
            unplaceable.append(model.role)

    return Placement(instances=tuple(instances), unplaceable_roles=tuple(unplaceable))


def _place_single(
    model: ModelPlacementInput,
    remaining: dict[int, float],
    estimate_peak: PeakEstimator,
) -> InstancePlan | None:
    """Place one instance: a single GPU when it fits, else a tensor-split, else None."""
    single = _best_single_device(model.est_vram_bytes, remaining)
    if single is not None:
        remaining[single] -= model.est_vram_bytes
        return InstancePlan(role=model.role, devices=(single,))
    return _place_split(model, remaining, estimate_peak)


def _place_split(
    model: ModelPlacementInput,
    remaining: dict[int, float],
    estimate_peak: PeakEstimator,
) -> InstancePlan | None:
    """Tensor-split across the fewest most-free GPUs whose per-device share each fits.

    Charges each chosen card its own entry from *estimate_peak*'s vector, so the
    busiest card (which OOMs first) is what gates the split, not the summed pool.
    """
    by_free = sorted(remaining, key=lambda idx: remaining[idx], reverse=True)
    for count in range(2, len(by_free) + 1):
        chosen = by_free[:count]
        ratio = tuple(max(1, int(remaining[idx] / 1024**3)) for idx in chosen)
        per_device = estimate_peak(model.role, ratio)
        if len(per_device) == count and all(
            peak <= remaining[idx] for idx, peak in zip(chosen, per_device, strict=True)
        ):
            for idx, peak in zip(chosen, per_device, strict=True):
                remaining[idx] -= peak
            return InstancePlan(role=model.role, devices=tuple(chosen), tensor_split=ratio)
    return None


def _place_replicas(model: ModelPlacementInput, remaining: dict[int, float]) -> list[InstancePlan]:
    """Place up to ``model.replicas`` instances, one per distinct GPU (most-free first).

    Spreads for throughput: each replica lands on a card not yet hosting one of this
    role's replicas, only co-locating a second round once every card has one. Stops
    early when no card has room, so the pool shrinks to what fits.
    """
    plans: list[InstancePlan] = []
    used: set[int] = set()
    for replica in range(model.replicas):
        candidates = [idx for idx, free in remaining.items() if free >= model.est_vram_bytes]
        if not candidates:
            break
        fresh = [idx for idx in candidates if idx not in used]
        pick = max(fresh or candidates, key=lambda idx: remaining[idx])
        remaining[pick] -= model.est_vram_bytes
        used.add(pick)
        if len(used) == len(remaining):
            used = set()
        plans.append(InstancePlan(role=model.role, devices=(pick,), replica=replica))
    return plans


def _place_shared_memory(models: list[ModelPlacementInput], budget: int) -> Placement:
    """Fit un-pinned roles into one shared RAM *budget*.

    Search-critical roles (embed/rerank) are reserved first so the elastic chat
    model can never crowd them out; the rest pack largest-first. Replicas run as N
    co-resident processes against the shared pool (no per-GPU spread without GPUs).
    A role with no instance placed is unplaceable (gets no server, its calls error).
    """
    remaining = budget
    instances: list[InstancePlan] = []
    unplaceable: list[WorkerRole] = []
    for model in sorted(models, key=_shared_pool_order):
        placed = 0
        for _ in range(model.replicas):
            if model.est_vram_bytes > remaining:
                break
            remaining -= model.est_vram_bytes
            instances.append(InstancePlan(role=model.role, devices=(), replica=placed))
            placed += 1
        if placed == 0:
            unplaceable.append(model.role)
    return Placement(instances=tuple(instances), unplaceable_roles=tuple(unplaceable))


def _shared_pool_order(model: ModelPlacementInput) -> tuple[int, int]:
    """Sort key: search roles first, then everything else largest-first."""
    is_search = 0 if model.role in _SEARCH_ROLES else 1
    return (is_search, -model.est_vram_bytes)


def _best_single_device(need: int, remaining: dict[int, float]) -> int | None:
    """Index of the device with the most free VRAM that still fits *need*."""
    candidates = [idx for idx, free in remaining.items() if free >= need]
    if not candidates:
        return None
    return max(candidates, key=lambda idx: remaining[idx])


def placement_from_spec(
    spec: PlacementSpec,
    active_roles: tuple[WorkerRole, ...],
    device_free: dict[int, int],
    *,
    estimate_peak: PeakEstimator,
) -> Placement:
    """Build a Placement from a manual *spec*, charging each card and failing loud.

    Every active role must have an entry; every device must exist; each card must
    fit the sum of the per-device peaks charged to it. Mirrors the auto planner's
    USABLE_VRAM_FRACTION headroom and busiest-card accounting.
    """
    remaining = {idx: free * USABLE_VRAM_FRACTION for idx, free in device_free.items()}
    instances: list[InstancePlan] = []
    for role in active_roles:
        rp = spec.roles.get(role)
        if rp is None:
            raise PlacementError(f"{role.value} has a model but no placement entry in placement")
        for idx in rp.devices:
            if idx not in device_free:
                raise PlacementError(
                    f"{role.value} pinned to device {idx} but only "
                    f"{len(device_free)} GPU(s) detected"
                )
        ratio = rp.tensor_split or tuple(1 for _ in rp.devices)
        per_device = estimate_peak(role, ratio)
        for replica in range(rp.replicas):
            for idx, peak in zip(rp.devices, per_device, strict=True):
                if peak > remaining[idx]:
                    raise PlacementError(
                        f"{role.value} pinned to device {idx} needs {peak / 1024**3:.1f} GiB but "
                        f"device {idx} has {remaining[idx] / 1024**3:.1f} GiB usable "
                        f"({device_free[idx] / 1024**3:.1f} GiB free, 90% headroom)"
                    )
            for idx, peak in zip(rp.devices, per_device, strict=True):
                remaining[idx] -= peak
            instances.append(
                InstancePlan(
                    role=role,
                    devices=tuple(rp.devices),
                    tensor_split=ratio if len(rp.devices) > 1 else (),
                    replica=replica,
                )
            )
    return Placement(instances=tuple(instances), unplaceable_roles=())
