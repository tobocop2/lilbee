"""VRAM-aware placement planner for the multi-GPU llama-server fleet.

Estimates each role-model's VRAM footprint from GGUF metadata and bin-packs
instances across GPUs in ``placement_rank`` order, largest-first within a rank: a
model that fits one GPU runs as a single pinned instance, small models co-locate
on a GPU with spare VRAM, and a model too big for any single GPU is tensor-split
across enough GPUs to fit. Roles that never run in the same phase (ingest OCR vs a
query) share one swap group when they cannot all co-reside, so only the phase in
use is charged; a role that fits nowhere gets no server (its calls error). See
docs/architecture.md.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from lilbee.providers.fleet.placement_spec import PlacementError, PlacementSpec, RolePlacement
from lilbee.providers.fleet.vram import USABLE_VRAM_FRACTION
from lilbee.providers.roles import ROLE_REGISTRY, WorkerRole

# (role, per-device tensor-split ratio) -> the instance's per-device VRAM footprint
# vector aligned to that ratio. A split is accepted only when every card's entry
# fits its own headroom, and each card is charged its own entry, not the sum.
PeakEstimator = Callable[[WorkerRole, tuple[int, ...]], tuple[int, ...]]

# (per-device tensor-split ratio, chosen cards' snapshot free-VRAM bytes) -> the per-slot
# context the launch would serve on that chat shard. Lets the planner widen a chat
# split onto idle cards when a tighter shard would starve KV below the target.
SplitCtxFitter = Callable[[tuple[int, ...], Sequence[int]], int]


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
    """Planner output: server instances, swap tenants, and roles that fit on no device.

    ``unplaceable_roles`` get no server, so a call to them surfaces a
    ``ProviderError`` (there is no in-process fallback). ``co_tenants`` share one
    llama-swap group and evict each other on demand, so only one is resident at a
    time; each runs a single instance.
    """

    instances: tuple[InstancePlan, ...]
    unplaceable_roles: tuple[WorkerRole, ...]
    co_tenants: frozenset[WorkerRole] = frozenset()


def plan_placement(
    models: list[ModelPlacementInput],
    devices: list[tuple[int, int]],
    *,
    estimate_peak: PeakEstimator,
    unified_budget: int | None = None,
    chat_ctx_fit: SplitCtxFitter | None = None,
    chat_ctx_target: int = 0,
    free_headroom: dict[int, int] | None = None,
) -> Placement:
    """Bin-pack *models* onto *devices* (``[(index, vram_bytes), ...]``).

    Roles are charged in ``placement_rank`` order, largest-first within a rank, with
    a 90% headroom per GPU. A model that fits one GPU takes a single instance; one
    too big for any single GPU is tensor-split. Chat is charged last: when it fits
    only if vision's VRAM is refunded, the two become ``co_tenants`` of one swap
    group instead of either becoming unplaceable. A model that fits nowhere, even
    alone beside the pinned tier, is an unplaceable role.

    A chat split widens past the fewest fitting cards when ``chat_ctx_fit`` shows a
    tighter shard would starve its served context below ``chat_ctx_target``; the
    fitter is sized against ``free_headroom`` (the plan snapshot's free VRAM per
    device index; see ``planning.capture_plan_probe``). See
    docs/architecture.md (Placement). Other splits keep the fewest-cards behavior.

    No GPU devices is the CPU/unified-memory case (a GPU-less host, or an Apple
    Silicon box where the probe found nothing): roles run as single un-pinned
    instances. ``unified_budget`` (free system RAM, bytes) gates them against one
    shared pool so an oversize model is unplaceable instead of OOM-livelocking the
    host; ``None`` keeps the legacy ungated behavior.
    """
    if not devices:
        return (
            _place_ungated(models)
            if unified_budget is None
            else _place_shared_memory(models, unified_budget)
        )
    remaining: dict[int, float] = {idx: vram * USABLE_VRAM_FRACTION for idx, vram in devices}

    # The persistent singles are every replicas<=1 role plus replica 0 of each
    # replicated role. They are charged before the elastic replicas, and the chat
    # model is charged last of all (``placement_rank``).
    replicated = [m for m in models if m.replicas > 1]
    persistent_singles = [m for m in models if m.replicas <= 1] + [
        _persistent_single(m) for m in replicated
    ]

    def place(model: ModelPlacementInput) -> _Placed | None:
        return _place_single(
            model,
            remaining,
            estimate_peak,
            chat_ctx_fit=chat_ctx_fit,
            chat_ctx_target=chat_ctx_target,
            free_headroom=free_headroom,
        )

    instances, unplaceable, co_tenants = _place_persistent(persistent_singles, place, remaining)
    instances.extend(_place_elastic(replicated, instances, remaining, co_tenants))

    return Placement(
        instances=tuple(instances),
        unplaceable_roles=tuple(unplaceable),
        co_tenants=co_tenants,
    )


def _place_ungated(models: list[ModelPlacementInput]) -> Placement:
    """Every role as a single un-pinned instance: no GPU and no measured RAM budget."""
    return Placement(
        instances=tuple(
            InstancePlan(role=m.role, devices=(), replica=r)
            for m in models
            for r in range(m.replicas)
        ),
        unplaceable_roles=(),
    )


def _place_persistent(
    persistent_singles: list[ModelPlacementInput],
    place: Callable[[ModelPlacementInput], _Placed | None],
    remaining: dict[int, float],
) -> tuple[list[InstancePlan], list[WorkerRole], frozenset[WorkerRole]]:
    """Charge every persistent single in ``placement_rank`` order.

    A role that does not fit refunds every already-charged role it is phase-disjoint
    from (never co-resident with) and retries; the roles that let it in become one
    swap group. This restores the in-process pool's behavior, where a role loaded
    only when its phase ran, so ingest never paid for the query models or vice versa.
    """
    instances: list[InstancePlan] = []
    unplaceable: list[WorkerRole] = []
    charges: dict[WorkerRole, dict[int, float]] = {}
    co_tenants: set[WorkerRole] = set()
    for model in sorted(persistent_singles, key=_shared_pool_order):
        placed = place(model)
        if placed is None:
            placed, group = _place_beside_disjoint(model, place, remaining, charges)
            co_tenants |= group
        if placed is None:
            unplaceable.append(model.role)
        else:
            instances.append(placed.plan)
            charges[model.role] = placed.charges
    return instances, unplaceable, frozenset(co_tenants)


def _place_beside_disjoint(
    model: ModelPlacementInput,
    place: Callable[[ModelPlacementInput], _Placed | None],
    remaining: dict[int, float],
    charges: dict[WorkerRole, dict[int, float]],
) -> tuple[_Placed | None, frozenset[WorkerRole]]:
    """Retry *model* with every phase-disjoint charged role's VRAM refunded.

    Phase-disjoint roles are never resident together, so they can share one swap
    group: only the resident one is charged. On success *model* and the refunded
    roles form the group and the refunded roles drop out of ``charges`` (their VRAM
    is now the group's shared budget). On failure every refund is rolled back and
    *model* is genuinely oversize.
    """
    refunds = {r: c for r, c in charges.items() if _phase_disjoint(r, model.role)}
    if not refunds:
        return None, frozenset()
    for charged in refunds.values():
        for idx, held in charged.items():
            remaining[idx] += held
    placed = place(model)
    if placed is not None:
        for role in refunds:
            del charges[role]
        return placed, frozenset(refunds) | {model.role}
    for charged in refunds.values():
        for idx, held in charged.items():
            remaining[idx] -= held
    return None, frozenset()


def _phase_disjoint(a: WorkerRole, b: WorkerRole) -> bool:
    """True when *a* and *b* share no run phase, so they are never co-resident."""
    return ROLE_REGISTRY[a].phases.isdisjoint(ROLE_REGISTRY[b].phases)


def _place_elastic(
    replicated: list[ModelPlacementInput],
    instances: list[InstancePlan],
    remaining: dict[int, float],
    co_tenants: frozenset[WorkerRole],
) -> list[InstancePlan]:
    """Fill the residual VRAM with replicas 1..N-1 of each placed, non-co-tenant role."""
    placed_roles = {plan.role for plan in instances}
    elastic: list[InstancePlan] = []
    for model in replicated:
        if model.role not in placed_roles or model.role in co_tenants:
            continue  # unplaceable, or a co-tenant capped to its single instance
        elastic.extend(_place_replicas(model, remaining, start=1))
    return elastic


def _persistent_single(model: ModelPlacementInput) -> ModelPlacementInput:
    """The replica-0 persistent instance of a replicated role, sized as one server."""
    return ModelPlacementInput(role=model.role, est_vram_bytes=model.est_vram_bytes, replicas=1)


@dataclass(frozen=True)
class _Placed:
    """One placed instance and the per-device VRAM it was charged."""

    plan: InstancePlan
    charges: dict[int, float]


def _place_single(
    model: ModelPlacementInput,
    remaining: dict[int, float],
    estimate_peak: PeakEstimator,
    *,
    chat_ctx_fit: SplitCtxFitter | None = None,
    chat_ctx_target: int = 0,
    free_headroom: dict[int, int] | None = None,
) -> _Placed | None:
    """Place one instance: a single GPU when it fits, else a tensor-split, else None."""
    single = _best_single_device(model.est_vram_bytes, remaining)
    if single is not None:
        remaining[single] -= model.est_vram_bytes
        return _Placed(
            plan=InstancePlan(role=model.role, devices=(single,)),
            charges={single: float(model.est_vram_bytes)},
        )
    return _place_split(
        model,
        remaining,
        estimate_peak,
        chat_ctx_fit=chat_ctx_fit,
        chat_ctx_target=chat_ctx_target,
        free_headroom=free_headroom,
    )


def _place_split(
    model: ModelPlacementInput,
    remaining: dict[int, float],
    estimate_peak: PeakEstimator,
    *,
    chat_ctx_fit: SplitCtxFitter | None = None,
    chat_ctx_target: int = 0,
    free_headroom: dict[int, int] | None = None,
) -> _Placed | None:
    """Tensor-split across the most-free GPUs whose per-device share each fits.

    Charges each chosen card its own entry from *estimate_peak*'s vector, so the
    busiest card (which OOMs first) gates the split, not the summed pool. A chat
    split widens past the fewest fitting cards via *chat_ctx_fit* (see
    :func:`plan_placement`); every other split takes the fewest that fit.
    """
    by_free = sorted(remaining, key=lambda idx: remaining[idx], reverse=True)
    best: tuple[int, list[int], tuple[int, ...], tuple[int, ...]] | None = None
    for count in range(2, len(by_free) + 1):
        chosen = by_free[:count]
        ratio = _vram_proportional_split(chosen, remaining)
        per_device = estimate_peak(model.role, ratio)
        if len(per_device) != count or not all(
            peak <= remaining[idx] for idx, peak in zip(chosen, per_device, strict=True)
        ):
            continue
        # Only chat is widened past the fewest fitting cards; everything else (and
        # the no-fitter generic path) takes the first shard that fits.
        if model.role is not WorkerRole.CHAT or chat_ctx_fit is None or free_headroom is None:
            return _charge_split(model, chosen, ratio, per_device, remaining)
        served = chat_ctx_fit(ratio, [free_headroom[idx] for idx in chosen])
        if served >= chat_ctx_target:
            return _charge_split(model, chosen, ratio, per_device, remaining)
        if best is None or served > best[0]:
            best = (served, chosen, ratio, per_device)
    if best is not None:
        _served, chosen, ratio, per_device = best
        return _charge_split(model, chosen, ratio, per_device, remaining)
    return None


def _charge_split(
    model: ModelPlacementInput,
    chosen: list[int],
    ratio: tuple[int, ...],
    per_device: tuple[int, ...],
    remaining: dict[int, float],
) -> _Placed:
    """Debit each chosen card its own per-device share and return the split plan."""
    for idx, peak in zip(chosen, per_device, strict=True):
        remaining[idx] -= peak
    return _Placed(
        plan=InstancePlan(role=model.role, devices=tuple(chosen), tensor_split=ratio),
        charges={idx: float(peak) for idx, peak in zip(chosen, per_device, strict=True)},
    )


def _place_replicas(
    model: ModelPlacementInput, remaining: dict[int, float], *, start: int = 0
) -> list[InstancePlan]:
    """Place the elastic replicas ``start..model.replicas-1``, one per distinct GPU
    (most-free first).

    Spreads for throughput: each replica lands on a card not yet hosting one of this
    role's replicas, only co-locating a second round once every card has one. Stops
    early when no card has room, so the pool shrinks to the residual VRAM. ``start``
    skips the indices already placed as persistent singles (1 for the elastic batch).
    """
    plans: list[InstancePlan] = []
    used: set[int] = set()
    for replica in range(start, model.replicas):
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

    Roles pack in ``placement_rank`` order (the elastic chat model last). Replicas
    run as N co-resident processes against the shared pool (no per-GPU spread without
    GPUs). A role that does not fit refunds every already-charged phase-disjoint role
    (never resident with it), which caps those roles to a single instance and makes
    the set one swap group; a role that still fits nowhere is unplaceable.
    """
    remaining = budget
    instances: list[InstancePlan] = []
    unplaceable: list[WorkerRole] = []
    charged: dict[WorkerRole, int] = {}
    co_tenants: set[WorkerRole] = set()
    for model in sorted(models, key=_shared_pool_order):
        placed = 0
        for _ in range(model.replicas):
            if model.est_vram_bytes > remaining:
                break
            remaining -= model.est_vram_bytes
            instances.append(InstancePlan(role=model.role, devices=(), replica=placed))
            placed += 1
        if placed:
            charged[model.role] = placed * model.est_vram_bytes
            continue
        remaining, group = _shared_beside_disjoint(model, remaining, charged, instances)
        if group:
            instances.append(InstancePlan(role=model.role, devices=()))
            charged[model.role] = model.est_vram_bytes
            co_tenants |= group
        else:
            unplaceable.append(model.role)
    return Placement(
        instances=tuple(instances),
        unplaceable_roles=tuple(unplaceable),
        co_tenants=frozenset(co_tenants),
    )


def _shared_beside_disjoint(
    model: ModelPlacementInput,
    remaining: int,
    charged: dict[WorkerRole, int],
    instances: list[InstancePlan],
) -> tuple[int, frozenset[WorkerRole]]:
    """Refund the shared-pool roles phase-disjoint from *model* and cap them to one instance.

    The refunded roles and *model* become one swap group: only one is resident, so
    each keeps a single instance and only *model*'s footprint is charged. Returns the
    budget after reclaiming their VRAM and charging *model*, plus the group; on a
    genuinely oversize *model* the budget is unchanged and the group empty.
    """
    refunds = [r for r in list(charged) if _phase_disjoint(r, model.role)]
    reclaim = sum(charged[r] for r in refunds)
    if not refunds or model.est_vram_bytes > remaining + reclaim:
        return remaining, frozenset()
    for role in refunds:
        del charged[role]
        instances[:] = [plan for plan in instances if plan.role is not role]
        instances.append(InstancePlan(role=role, devices=(), replica=0))
    return remaining + reclaim - model.est_vram_bytes, frozenset(refunds) | {model.role}


def _shared_pool_order(model: ModelPlacementInput) -> tuple[int, int]:
    """Sort key: by the role's placement rank, then largest-first within a rank."""
    return (ROLE_REGISTRY[model.role].placement_rank, -model.est_vram_bytes)


def _best_single_device(need: int, remaining: dict[int, float]) -> int | None:
    """Index of the device with the most free VRAM that still fits *need*."""
    candidates = [idx for idx, free in remaining.items() if free >= need]
    if not candidates:
        return None
    return max(candidates, key=lambda idx: remaining[idx])


def placement_from_spec(
    spec: PlacementSpec,
    active_roles: tuple[WorkerRole, ...],
    device_capacity: dict[int, int],
    *,
    estimate_peak: PeakEstimator,
) -> Placement:
    """Build a Placement from a manual *spec*, charging each card and failing loud.

    ``device_capacity`` is each card's total VRAM (not instantaneous free): the
    plan defines the fleet's full intended residency, so charging it against live
    free VRAM would double-count models already loaded. Every active role must
    have an entry; every device must exist; each card must fit the sum of the
    per-device peaks charged to it, within the USABLE_VRAM_FRACTION headroom.
    """
    remaining = {idx: total * USABLE_VRAM_FRACTION for idx, total in device_capacity.items()}
    instances: list[InstancePlan] = []
    for role in active_roles:
        rp = _required_entry(spec, role, device_capacity)
        ratio = rp.tensor_split or _vram_proportional_split(rp.devices, remaining)
        per_device = estimate_peak(role, ratio)
        split = ratio if len(rp.devices) > 1 else ()
        for replica in range(rp.replicas):
            _charge_devices(role, rp.devices, per_device, remaining, device_capacity)
            instances.append(
                InstancePlan(
                    role=role, devices=tuple(rp.devices), tensor_split=split, replica=replica
                )
            )
    return Placement(instances=tuple(instances), unplaceable_roles=())


def _vram_proportional_split(
    devices: Sequence[int], remaining: dict[int, float]
) -> tuple[int, ...]:
    """Tensor-split ratio proportional to each card's remaining usable VRAM.

    Each card's shard tracks its remaining VRAM (whole GiB, min 1), so a card
    already carrying other roles takes a smaller share. This is the single source
    of the proportion for both placement paths: the auto planner (:func:`_place_split`)
    and a manual spec entry with no explicit ``tensor_split`` (:func:`placement_from_spec`).
    Keeping it in one place is what guarantees a manually-applied layout is charged
    the same way the planner charges the identical layout, instead of an even split
    that would falsely reject a fit the planner itself serves.
    """
    return tuple(max(1, int(remaining[idx] / 1024**3)) for idx in devices)


def _required_entry(
    spec: PlacementSpec, role: WorkerRole, device_capacity: dict[int, int]
) -> RolePlacement:
    """Return *role*'s placement entry, failing loud if absent or pinned off-hardware."""
    rp = spec.roles.get(role)
    if rp is None:
        raise PlacementError(f"{role.value} has a model but no placement entry in placement")
    for idx in rp.devices:
        if idx not in device_capacity:
            raise PlacementError(
                f"{role.value} pinned to device {idx} but only "
                f"{len(device_capacity)} GPU(s) detected"
            )
    return rp


def _charge_devices(
    role: WorkerRole,
    devices: tuple[int, ...],
    per_device: tuple[int, ...],
    remaining: dict[int, float],
    device_capacity: dict[int, int],
) -> None:
    """Subtract one instance's per-device peaks from *remaining*; fail loud if a card overflows."""
    for idx, peak in zip(devices, per_device, strict=True):
        if peak > remaining[idx]:
            raise PlacementError(
                f"{role.value} pinned to device {idx} needs {peak / 1024**3:.1f} GiB but "
                f"device {idx} has {remaining[idx] / 1024**3:.1f} GiB usable "
                f"({device_capacity[idx] / 1024**3:.1f} GiB total, 90% headroom)"
            )
    for idx, peak in zip(devices, per_device, strict=True):
        remaining[idx] -= peak
