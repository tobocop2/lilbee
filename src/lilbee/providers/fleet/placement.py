"""VRAM-aware placement planner for the multi-GPU llama-server fleet.

Estimates each role-model's VRAM footprint from GGUF metadata and bin-packs
instances across GPUs in ``placement_rank`` order, largest-first within a rank: a
model that fits one GPU runs as a single pinned instance, small models co-locate
on a GPU with spare VRAM, and a model too big for any single GPU is tensor-split
across enough GPUs to fit. Roles that never run in the same phase (ingest OCR vs a
query) share one swap group when they cannot all co-reside, so only the phase in
use is charged. On GPUs the estimate advises but never refuses: a role that fits
nowhere is placed tight (best-effort, with a warning). See docs/architecture.md.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import reduce
from math import gcd

from lilbee.providers.fleet.placement_spec import PlacementError, PlacementSpec, RolePlacement
from lilbee.providers.fleet.vram import usable_vram_fraction
from lilbee.providers.roles import ROLE_REGISTRY, WorkerRole

log = logging.getLogger(__name__)

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

    ``est_ram_bytes`` is what the same estimate puts in system memory, which is
    non-zero only when something offloads. Placement divides GPUs and so does not
    read it; admission does, because system memory is a bound too.
    """

    role: WorkerRole
    est_vram_bytes: int
    replicas: int = 1
    est_ram_bytes: int = 0


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
    """Planner output: server instances, swap tenants, and best-effort placements.

    ``unplaceable_roles`` get no server, so a call to them surfaces a
    ``ProviderError`` (there is no in-process fallback); only the shared-memory
    path produces them (an oversize load there OOM-livelocks the host). On GPUs
    a role that fits nowhere is placed anyway and listed in ``tight_roles`` with
    its estimated shortfall in bytes. ``co_tenants`` share one llama-swap group
    and evict each other on demand, so only one is resident at a time; each runs
    a single instance.
    """

    instances: tuple[InstancePlan, ...]
    unplaceable_roles: tuple[WorkerRole, ...]
    co_tenants: frozenset[WorkerRole] = frozenset()
    tight_roles: dict[WorkerRole, int] = field(default_factory=dict)


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
    alone beside the pinned tier, is still placed tight and reported in
    ``tight_roles``.

    A chat split widens past the fewest fitting cards when ``chat_ctx_fit`` shows a
    tighter shard would starve its served context below ``chat_ctx_target``; the
    fitter is sized against ``free_headroom`` (the plan snapshot's free VRAM per
    device index; see ``planning.capture_plan_probe``). See
    docs/architecture.md (Placement). Other splits keep the fewest-cards behavior.

    A ``unified_budget`` (free system RAM, bytes) means every device this host has
    shares the host's memory, or it has none: an integrated GPU, an Apple Silicon
    Mac, a GPU-less box. Those go through the shared pool whether or not a device
    enumerated, because the constraint is the same RAM either way, and only that
    path can refuse a role. Bin-packing them per device instead reads one pool as
    several and never refuses anything, so a role set that cannot fit is admitted,
    loads, and swap-livelocks the machine, which is what the budget exists to
    prevent. ``None`` means at least one device has memory of its own, and the
    per-GPU packing below applies.
    """
    if unified_budget is not None:
        return _place_shared_memory(models, unified_budget)
    if not devices:
        return _place_ungated(models)
    usable = usable_vram_fraction()
    remaining: dict[int, float] = {idx: vram * usable for idx, vram in devices}

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

    instances, co_tenants, tight = _place_persistent(persistent_singles, place, remaining)
    instances.extend(_place_elastic(replicated, instances, remaining, co_tenants))

    return Placement(
        instances=tuple(instances),
        unplaceable_roles=(),
        co_tenants=co_tenants,
        tight_roles=tight,
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
) -> tuple[list[InstancePlan], frozenset[WorkerRole], dict[WorkerRole, int]]:
    """Charge every persistent single in ``placement_rank`` order.

    A role that does not fit refunds every already-charged role it is phase-disjoint
    from (never co-resident with) and retries; the roles that let it in become one
    swap group. This restores the in-process pool's behavior, where a role loaded
    only when its phase ran, so ingest never paid for the query models or vice versa.
    A role that still fits nowhere is placed tight (:func:`_place_tight`) instead
    of refused.
    """
    instances: list[InstancePlan] = []
    charges: dict[WorkerRole, dict[int, float]] = {}
    co_tenants: set[WorkerRole] = set()
    tight: dict[WorkerRole, int] = {}
    for model in sorted(persistent_singles, key=_shared_pool_order):
        placed = place(model)
        if placed is None:
            placed, group = _place_beside_disjoint(model, place, remaining, charges)
            co_tenants |= group
        if placed is None:
            placed, group, shortfall = _place_tight(model, remaining, charges)
            co_tenants |= group
            tight[model.role] = shortfall
        instances.append(placed.plan)
        charges[model.role] = placed.charges
    return instances, frozenset(co_tenants), tight


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
        slot = _group_slot(placed.charges, refunds, remaining)
        for role in refunds:
            del charges[role]
        return _Placed(plan=placed.plan, charges=slot), frozenset(refunds) | {model.role}
    for charged in refunds.values():
        for idx, held in charged.items():
            remaining[idx] -= held
    return None, frozenset()


def _tight_device_group(needed: int, remaining: dict[int, float]) -> tuple[int, ...]:
    """Cards to give a model that fits nowhere else.

    The most-free card alone when that one card is enough, otherwise every card
    with headroom, ordered most-free first so the split's main device is the
    roomiest. Not a minimal subset: finding the smallest group that fits would
    need an estimate per candidate group, and this path is reached only after
    the estimating search has already failed.
    """
    by_room = sorted(remaining, key=lambda idx: remaining[idx], reverse=True)
    if not by_room:
        return ()
    if remaining[by_room[0]] >= needed:
        return (by_room[0],)
    usable = [idx for idx in by_room if remaining[idx] > 0]
    return tuple(usable or by_room[:1])


def _place_tight(
    model: ModelPlacementInput,
    remaining: dict[int, float],
    charges: dict[WorkerRole, dict[int, float]],
) -> tuple[_Placed, frozenset[WorkerRole], int]:
    """Place an oversize *model* best-effort instead of refusing it.

    Refunds every phase-disjoint charged role (they become *model*'s swap group),
    then gives *model* the widest set of cards that helps, and drains them so the
    elastic tier places nothing there. Returns the estimated shortfall in bytes.

    Widest rather than the single most-free card. Pinned to one card with the
    others excluded by that pin, a model too big for it has nowhere to go and the
    tight placement is only a slower refusal; given the group, llama-server can
    split across them. One card is still used when one card is enough, since a
    split it does not need costs it interconnect bandwidth.
    """
    refunds = {r: c for r, c in charges.items() if _phase_disjoint(r, model.role)}
    for charged in refunds.values():
        for idx, held in charged.items():
            remaining[idx] += held
    devices = _tight_device_group(model.est_vram_bytes, remaining)
    claimed = {idx: remaining[idx] for idx in devices}
    available = sum(claimed.values())
    for idx in devices:
        remaining[idx] = 0.0
    slot = _group_slot(claimed, refunds, remaining)
    for role in refunds:
        del charges[role]
    placed = _Placed(
        plan=InstancePlan(role=model.role, devices=devices),
        charges=slot,
    )
    group = frozenset(refunds) | {model.role} if refunds else frozenset()
    return placed, group, int(model.est_vram_bytes - available)


def _group_slot(
    trigger: dict[int, float],
    refunds: dict[WorkerRole, dict[int, float]],
    remaining: dict[int, float],
) -> dict[int, float]:
    """Charge each device the swap group's largest member, not just the trigger.

    Only one member is resident at a time, but every evicted member must be able to
    swap back in, so the group's per-device slot is the max across members. The
    shortfall beyond the trigger's own charge is deducted from *remaining* here so
    the elastic replica tier cannot claim VRAM an evicted member needs to return to.
    The returned slot is recorded as the trigger's charge, so a later trigger that
    refunds this group reclaims the whole slot.
    """
    slot = dict(trigger)
    for charged in refunds.values():
        for idx, held in charged.items():
            need = max(slot.get(idx, 0.0), held)
            remaining[idx] -= need - slot.get(idx, 0.0)
            slot[idx] = need
    return slot


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

    Each card count is tried at several proportions (:func:`_split_ratio_candidates`),
    because a footprint that does not scale with the shard can overflow the
    smallest card at the proportional ratio and fit at a shifted one. The sweep
    stops after ``_MAX_SPLIT_ESTIMATES`` estimator calls and says so, since each
    call is a subprocess.
    """
    from lilbee.providers.base import ProviderError

    by_free = sorted(remaining, key=lambda idx: remaining[idx], reverse=True)
    best: tuple[int, list[int], tuple[int, ...], tuple[int, ...]] | None = None
    spent = 0
    for count in range(2, len(by_free) + 1):
        chosen = by_free[:count]
        for ratio in _split_ratio_candidates(chosen, remaining):
            if spent >= _MAX_SPLIT_ESTIMATES:
                log.info(
                    "Stopped looking for a tensor split for %s after %d estimates; "
                    "wider layouts and finer proportions were not tried.",
                    model.role.value,
                    spent,
                )
                return _best_or_none(best, model, remaining)
            spent += 1
            try:
                per_device = estimate_peak(model.role, ratio)
            except (ProviderError, OSError):
                # An unsizable model cannot evaluate a split; the tight single-card
                # path downstream still places it.
                continue
            if len(per_device) != count or not all(
                peak <= remaining[idx] for idx, peak in zip(chosen, per_device, strict=True)
            ):
                continue
            # Only chat is widened past the fewest fitting cards; everything else (and
            # the no-fitter generic path) takes the first shard that fits.
            if model.role is not WorkerRole.CHAT or chat_ctx_fit is None or free_headroom is None:
                return _charge_split(model, chosen, ratio, per_device, remaining)
            # The context fit bisects, and every probe is another gguf-parser
            # run, so it costs far more than the estimate that preceded it and
            # has to be charged against the same budget. Uncounted, a wide box
            # ran roughly 190 subprocesses against a documented cap of 24, all
            # while holding the cross-process build lock that every other lilbee
            # start waits on for 90 seconds before failing.
            spent += _CTX_FIT_ESTIMATE_COST
            served = chat_ctx_fit(ratio, [free_headroom[idx] for idx in chosen])
            if served >= chat_ctx_target:
                return _charge_split(model, chosen, ratio, per_device, remaining)
            if best is None or served > best[0]:
                best = (served, chosen, ratio, per_device)
    return _best_or_none(best, model, remaining)


def _best_or_none(
    best: tuple[int, list[int], tuple[int, ...], tuple[int, ...]] | None,
    model: ModelPlacementInput,
    remaining: dict[int, float],
) -> _Placed | None:
    """Charge the widest chat split found short of the target, if there was one."""
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
    # A charged role's swap-back need: its single-instance footprint, or, for the
    # role holding a swap group's budget, the whole group slot.
    swap_need = {m.role: m.est_vram_bytes for m in models}
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
        remaining, group = _shared_beside_disjoint(model, remaining, charged, swap_need, instances)
        if group:
            instances.append(InstancePlan(role=model.role, devices=()))
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
    swap_need: dict[WorkerRole, int],
    instances: list[InstancePlan],
) -> tuple[int, frozenset[WorkerRole]]:
    """Refund the shared-pool roles phase-disjoint from *model* and cap them to one instance.

    The refunded roles and *model* become one swap group: only one member is resident
    at a time, but every member must be able to swap back in, so the group is charged
    its largest member's footprint (the slot), recorded under *model*'s role so a
    later trigger reclaims the whole slot. Returns the budget after reclaiming the
    refunded VRAM and charging the slot, plus the group; when even the slot does not
    fit the reclaimed budget the group is empty and the budget unchanged.
    """
    refunds = [r for r in list(charged) if _phase_disjoint(r, model.role)]
    if not refunds:
        return remaining, frozenset()
    reclaim = sum(charged[r] for r in refunds)
    slot = max(model.est_vram_bytes, *(swap_need[r] for r in refunds))
    if slot > remaining + reclaim:
        return remaining, frozenset()
    for role in refunds:
        del charged[role]
        instances[:] = [plan for plan in instances if plan.role is not role]
        instances.append(InstancePlan(role=role, devices=(), replica=0))
    charged[model.role] = slot
    swap_need[model.role] = slot
    return remaining + reclaim - slot, frozenset(refunds) | {model.role}


def _shared_pool_order(model: ModelPlacementInput) -> tuple[int, int, int]:
    """Sort key: placement rank, then most-phases-first, then largest-first.

    A role in more phases is co-resident with more of the fleet and can never make
    room by swapping (nothing is phase-disjoint from it), so it charges before a
    same-rank single-phase sibling: a large LLM reranker must not crowd out the
    embedder that both ingest and query need.
    """
    info = ROLE_REGISTRY[model.role]
    return (info.placement_rank, -len(info.phases), -model.est_vram_bytes)


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
    per-device peaks charged to it, within the cfg.usable_vram_fraction headroom.
    """
    usable = usable_vram_fraction()
    remaining = {idx: total * usable for idx, total in device_capacity.items()}
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
    devices: Sequence[int], remaining: dict[int, float], *, divisor: int = 1
) -> tuple[int, ...]:
    """Tensor-split ratio proportional to each card's remaining usable VRAM.

    *divisor* sets the resolution: 1 is whole GiB, 4 is quarter-GiB shares.

    Each card's shard tracks its remaining VRAM (whole GiB, min 1), so a card
    already carrying other roles takes a smaller share. This is the single source
    of the proportion for both placement paths: the auto planner (:func:`_place_split`)
    and a manual spec entry with no explicit ``tensor_split`` (:func:`placement_from_spec`).
    Keeping it in one place is what guarantees a manually-applied layout is charged
    the same way the planner charges the identical layout, instead of an even split
    that would falsely reject a fit the planner itself serves.
    """
    return tuple(max(1, int(remaining[idx] * divisor / 1024**3)) for idx in devices)


# How many proportions the sweep will try per card count, and how many estimator
# calls the whole sweep may spend. The estimator shells out to gguf-parser, so an
# unbounded ladder on a wide box turns a plan into a minute of subprocesses; the
# cap is what keeps the search's cost linear in cards rather than in cards times
# candidates.
# Rungs on the ladder below. Not a cap applied to it: the ladder builds exactly
# this many, and a slice pretending to enforce a bound it cannot reach would be
# decoration. Named so the estimator memo can be sized against a whole plan.
_MAX_RATIO_CANDIDATES = 3
_MAX_SPLIT_ESTIMATES = 24
# What one context fit costs in estimator runs. It bisects the servable window,
# so it is not one call but a handful, and charging it as one understated the
# sweep by roughly an order of magnitude.
_CTX_FIT_ESTIMATE_COST = 8
# Sub-GiB resolution for the shifted candidates. Whole GiB is coarse enough that
# two cards 700 MiB apart quantize to the same share.
_RATIO_QUANTUM_DIVISOR = 4


def _split_ratio_candidates(
    devices: Sequence[int], remaining: dict[int, float]
) -> tuple[tuple[int, ...], ...]:
    """Proportions worth trying for a split across *devices*, best-first.

    The VRAM-proportional ratio leads, because it is right whenever the footprint
    scales with the shard. It is not always right: KV, compute buffers and a
    fixed per-device overhead do not scale, so the proportional shard can
    overflow the smallest card while the group has room. The rest of the ladder
    shifts load toward the roomiest card at finer resolution, which is the
    direction that helps when it does not.

    Deduplicated by proportion rather than by tuple, so equal cards cost one
    estimate rather than three: (24, 24) and (96, 96) are the same split asked
    twice, and each ask is a subprocess.
    """
    quantum = _vram_proportional_split(devices, remaining, divisor=_RATIO_QUANTUM_DIVISOR)
    candidates = [
        _vram_proportional_split(devices, remaining),
        quantum,
        _shifted_toward_roomiest(devices, remaining, quantum),
    ]
    seen: dict[tuple[int, ...], tuple[int, ...]] = {}
    for candidate in candidates:
        seen.setdefault(_normalized(candidate), candidate)
    return tuple(seen.values())


def _normalized(ratio: tuple[int, ...]) -> tuple[int, ...]:
    """*ratio* in lowest terms, so the same proportion compares equal."""
    divisor = reduce(gcd, ratio)
    return tuple(part // divisor for part in ratio)


def _shifted_toward_roomiest(
    devices: Sequence[int], remaining: dict[int, float], base: tuple[int, ...]
) -> tuple[int, ...]:
    """*base* with a share moved from the tightest card to the roomiest.

    A tenth of the tightest card's share, which is enough to clear a fixed
    per-device overhead without distorting a proportion that was nearly right.
    Cards with identical room have nowhere to shift to, so *base* is returned
    unchanged rather than making one of them worse. A lone card takes the same
    path, being trivially equal to itself.
    """
    order = sorted(range(len(devices)), key=lambda pos: remaining[devices[pos]])
    tightest, roomiest = order[0], order[-1]
    if remaining[devices[tightest]] == remaining[devices[roomiest]]:
        return base
    moved = max(1, base[tightest] // 10)
    shifted = list(base)
    shifted[tightest] = max(1, shifted[tightest] - moved)
    shifted[roomiest] += moved
    return tuple(shifted)


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
    """Subtract one instance's per-device peaks from *remaining*; fail loud if a card overflows.

    An estimate that does not cover every pinned device is a PlacementError rather
    than a zip mismatch: gguf-parser returns no per-device breakdown for some
    models, and the auto planner skips such a candidate (see :func:`_place_split`),
    so the manual path must refuse in the currency callers already handle.
    """
    if len(per_device) != len(devices):
        raise PlacementError(
            f"{role.value} is pinned to {len(devices)} device(s) but its memory estimate "
            f"covers {len(per_device)}; clear the placement to place it automatically"
        )
    for idx, peak in zip(devices, per_device, strict=True):
        if peak > remaining[idx]:
            raise PlacementError(
                f"{role.value} pinned to device {idx} needs {peak / 1024**3:.1f} GiB but "
                f"device {idx} has {remaining[idx] / 1024**3:.1f} GiB usable "
                f"({device_capacity[idx] / 1024**3:.1f} GiB total, 90% headroom)"
            )
    for idx, peak in zip(devices, per_device, strict=True):
        remaining[idx] -= peak
