"""VRAM-aware placement planner for the multi-GPU llama-server fleet.

Estimates each role-model's VRAM footprint from GGUF metadata and bin-packs
instances across GPUs in ``placement_rank`` order, largest-first within a rank: a
model that fits one GPU runs as a single pinned instance, small models co-locate
on a GPU with spare VRAM, and a model too big for any single GPU is tensor-split
across enough GPUs to fit. Roles that never run in the same phase (ingest OCR vs a
query) share one swap group when they cannot all co-reside, so only the phase in
use is charged. On GPUs a plan with one engine places a role that fits nowhere
tight (best-effort, with a warning); a plan with several engines refuses that role.
See docs/internals/engine.md.
"""

from __future__ import annotations

import functools
import itertools
import logging
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import reduce
from math import ceil, gcd
from types import MappingProxyType

from lilbee.providers.fleet.groups import class_roles, residency_classes
from lilbee.providers.fleet.memory_types import MemoryKind
from lilbee.providers.fleet.placement_spec import (
    PlacementError,
    PlacementSpec,
    RolePlacement,
    TargetRefusedError,
)
from lilbee.providers.fleet.vram import usable_vram_fraction
from lilbee.providers.roles import ROLE_REGISTRY, RoleTarget, WorkerRole

log = logging.getLogger(__name__)

# The one device key of the shared memory pool, where a plan has no cards.
_POOL = 0
# Float slack when a merged class is tested against the room left: charges are whole bytes
# summed as floats, so an exact fit can read a rounding error short.
_ROUNDING_BYTES = 1.0

# (role, per-device tensor-split ratio) -> the instance's per-device VRAM footprint
# vector aligned to that ratio. A split is accepted only when every card's entry
# fits its own headroom, and each card is charged its own entry, not the sum.
PeakEstimator = Callable[[WorkerRole, tuple[int, ...]], tuple[int, ...]]

# (per-device tensor-split ratio, chosen cards' snapshot free-VRAM bytes) -> the per-slot
# context the launch would serve on that chat shard. Lets the planner widen a chat
# split onto idle cards when a tighter shard would starve KV below the target.
SplitCtxFitter = Callable[[tuple[int, ...], Sequence[int]], int]
# Plans the auto roles of a spec on the cards the other roles leave (index to
# total bytes) and the roles that were charged to cards.
AutoPlanner = Callable[[dict[int, int], frozenset[WorkerRole]], "Placement"]
# Words a role pinned to devices the machine lacks, from the role and those devices.
LostCardWording = Callable[[WorkerRole, Sequence[int]], str]


@dataclass(frozen=True)
class ModelPlacementInput:
    """A role's model, its estimated single-instance footprint, and replica count.

    ``replicas`` > 1 requests N data-parallel instances (one per GPU) for the role,
    each charged ``est_vram_bytes``; capped at runtime by the GPUs with room.

    ``est_ram_bytes`` is what the same estimate puts in system memory, which is
    non-zero only when something offloads. Placement divides GPUs and so does not
    read it; admission does, because system memory is a bound too.

    ``kind`` says whether the engine sized the figures or declared a fixed
    reserve, which no later step corrects against the weights on disk.
    """

    role: WorkerRole
    est_vram_bytes: int
    replicas: int = 1
    est_ram_bytes: int = 0
    kind: MemoryKind = MemoryKind.SIZED
    size_known: bool = True
    """False when the figures are the unknown-size reserve, not a measured size."""
    equal_split_only: bool = False
    """True when the engine launches a split only as an equal share on each card."""


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
    target: RoleTarget = RoleTarget.GPU


@dataclass(frozen=True)
class RoomHolder:
    """A role charged to a card ahead of the role that a refusal names."""

    role: WorkerRole
    model: str
    charged_bytes: float


@dataclass(frozen=True)
class LowFreeMemory:
    """System memory a CPU role brings its plan to, and the memory free when it was read."""

    wanted: int
    free: int


@dataclass(frozen=True)
class SplitMiss:
    """The nearest a split came to fitting: its card count and the card it overshot most."""

    count: int
    device: int
    need_bytes: float
    room_bytes: float


@dataclass(frozen=True)
class Placement:
    """Planner output: server instances, swap tenants, and best-effort placements.

    ``unplaceable_roles`` get no server, so a call to them surfaces a
    ``ProviderError`` (there is no in-process fallback); the shared-memory path
    and a plan with several engines produce them. On GPUs a plan with one engine
    places a role that fits nowhere anyway and lists it in ``tight_roles`` with
    its estimated shortfall in bytes. ``swap_classes`` split the co-tenants of one
    swap process: roles in one class stay loaded together, and loading a
    role of another class evicts them. Each co-tenant runs a single instance.
    ``low_free_memory`` lists each CPU role that fits the machine but
    not the memory free now; the launch path logs it.
    """

    instances: tuple[InstancePlan, ...]
    unplaceable_roles: tuple[WorkerRole, ...]
    swap_classes: tuple[frozenset[WorkerRole], ...] = ()
    tight_roles: dict[WorkerRole, int] = field(default_factory=dict)
    low_free_memory: dict[WorkerRole, LowFreeMemory] = field(default_factory=dict)
    split_misses: dict[WorkerRole, SplitMiss] = field(default_factory=dict, compare=False)
    """Per role, the nearest split the planner tried, from the first pack that tried one."""

    @property
    def co_tenants(self) -> frozenset[WorkerRole]:
        """Every role that runs in the shared swap process."""
        return class_roles(self.swap_classes)


def plan_placement(
    models: list[ModelPlacementInput],
    devices: list[tuple[int, int]],
    *,
    estimate_peak: PeakEstimator,
    unified_budget: int | None = None,
    chat_ctx_fit: SplitCtxFitter | None = None,
    chat_ctx_target: int = 0,
    free_headroom: dict[int, int] | None = None,
    refuse_oversize: bool = False,
) -> Placement:
    """Bin-pack *models* onto *devices* (``[(index, vram_bytes), ...]``).

    Roles are charged in ``placement_rank`` order, largest-first within a rank, with
    a 90% headroom per GPU. A model that fits one GPU takes a single instance; one
    too big for any single GPU is tensor-split. Chat is charged last: when it fits
    only if vision's VRAM is refunded, the two become ``co_tenants`` of one swap
    process instead of either becoming unplaceable; ``swap_classes`` splits them into
    the classes that stay loaded together. A model that fits nowhere, even
    alone beside the pinned tier, is still placed tight and reported in
    ``tight_roles``, unless ``refuse_oversize`` is set: it is then refused and takes
    no card.

    A chat split widens past the fewest fitting cards when ``chat_ctx_fit`` shows a
    tighter shard would starve its served context below ``chat_ctx_target``; the
    fitter is sized against ``free_headroom`` (the plan snapshot's free VRAM per
    device index; see ``planning.capture_plan_probe``). See
    docs/internals/engine.md (Placement). Other splits keep the fewest-cards behavior.

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
    cards: dict[int, float] = {idx: vram * usable for idx, vram in devices}
    misses: dict[WorkerRole, SplitMiss] = {}

    def placer(remaining: dict[int, float]) -> Callable[[ModelPlacementInput], _Placed | None]:
        def place(model: ModelPlacementInput) -> _Placed | None:
            return _place_single(
                model,
                remaining,
                estimate_peak,
                chat_ctx_fit=chat_ctx_fit,
                chat_ctx_target=chat_ctx_target,
                free_headroom=free_headroom,
                misses=misses,
            )

        return place

    if refuse_oversize:
        # The search repeats passes over the same roles and splits; one estimate each.
        estimate_peak = functools.cache(estimate_peak)
        return replace(_place_refusing(models, cards, placer), split_misses=misses)

    remaining = dict(cards)
    # The persistent singles are every replicas<=1 role plus replica 0 of each
    # replicated role. They are charged before the elastic replicas, and the chat
    # model is charged last of all (``placement_rank``).
    replicated = [m for m in models if m.replicas > 1]
    persistent_singles = [m for m in models if m.replicas <= 1] + [
        _persistent_single(m) for m in replicated
    ]

    instances, swap_classes, tight = _place_persistent(
        persistent_singles, placer(remaining), remaining
    )
    instances.extend(_place_elastic(replicated, instances, remaining, class_roles(swap_classes)))

    return Placement(
        instances=tuple(instances),
        unplaceable_roles=(),
        swap_classes=swap_classes,
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


@dataclass(frozen=True)
class _RefusingPack:
    """One pass of the refusing packer over fresh cards."""

    instances: list[InstancePlan]
    swap: _SwapGroup
    refused: list[WorkerRole]
    remaining: dict[int, float]


def _place_refusing(
    models: list[ModelPlacementInput],
    cards: dict[int, float],
    placer: Callable[[dict[int, float]], Callable[[ModelPlacementInput], _Placed | None]],
) -> Placement:
    """Give each role a card or a refusal; a role with no device charge takes no card.

    The plan refuses the fewest roles. It tries to refuse no role, then each single
    role, then each pair, and stops at the first set whose removal lets every other
    role fit. Among sets of the same size it refuses the roles of lowest rank first.
    With ``n`` roles it makes at most ``2 * 2**n`` packing passes, two orders per set.
    """
    on_cards = [m for m in models if m.est_vram_bytes > 0]
    singles = [_persistent_single(m) for m in on_cards]
    by_priority = sorted(singles, key=_shared_pool_order)
    by_size = sorted(singles, key=lambda m: -m.est_vram_bytes)
    packed = _fewest_refusals(by_priority, by_size, cards, placer)
    replicated = [m for m in on_cards if m.replicas > 1]
    packed.swap.merge(packed.remaining)
    classes = packed.swap.classes
    elastic = _place_elastic(replicated, packed.instances, packed.remaining, class_roles(classes))
    unpinned = _place_ungated([m for m in models if m.est_vram_bytes == 0])
    return Placement(
        instances=(*packed.instances, *elastic, *unpinned.instances),
        unplaceable_roles=tuple(packed.refused),
        swap_classes=classes,
    )


def _fewest_refusals(
    by_priority: list[ModelPlacementInput],
    by_size: list[ModelPlacementInput],
    cards: dict[int, float],
    placer: Callable[[dict[int, float]], Callable[[ModelPlacementInput], _Placed | None]],
) -> _RefusingPack:
    """The pack that refuses the fewest roles, the lowest rank first among equals."""
    lowest_first = list(reversed(by_priority))
    for count in range(len(by_priority) + 1):
        for dropped in itertools.combinations(lowest_first, count):
            for order in (by_priority, by_size):
                kept = [m for m in order if m not in dropped]
                packed = _pack_refusing(kept, cards, placer)
                if not packed.refused:
                    return replace(packed, refused=[m.role for m in dropped])
    raise AssertionError("refusing every role always packs")  # pragma: no cover


def _pack_refusing(
    singles: list[ModelPlacementInput],
    cards: dict[int, float],
    placer: Callable[[dict[int, float]], Callable[[ModelPlacementInput], _Placed | None]],
) -> _RefusingPack:
    """Charge *singles* in the order given on fresh cards; a role with no room charges nothing."""
    remaining = dict(cards)
    place = placer(remaining)
    instances: list[InstancePlan] = []
    charges: dict[WorkerRole, dict[int, float]] = {}
    swap = _SwapGroup()
    refused: list[WorkerRole] = []
    for model in singles:
        placed = place(model)
        if placed is not None:
            charges[model.role] = placed.charges
        else:
            placed = _place_beside_disjoint(model, place, remaining, charges, swap)
        if placed is None:
            refused.append(model.role)
            continue
        instances.append(placed.plan)
    return _RefusingPack(instances, swap, refused, remaining)


def _place_persistent(
    persistent_singles: list[ModelPlacementInput],
    place: Callable[[ModelPlacementInput], _Placed | None],
    remaining: dict[int, float],
) -> tuple[list[InstancePlan], tuple[frozenset[WorkerRole], ...], dict[WorkerRole, int]]:
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
    swap = _SwapGroup()
    tight: dict[WorkerRole, int] = {}
    for model in sorted(persistent_singles, key=_shared_pool_order):
        placed = place(model)
        if placed is not None:
            charges[model.role] = placed.charges
        else:
            placed = _place_beside_disjoint(model, place, remaining, charges, swap)
        if placed is None:
            placed, tight[model.role] = _place_tight(model, remaining, charges, swap)
        instances.append(placed.plan)
    swap.merge(remaining)
    return instances, swap.classes, tight


def _class_slot(
    own: Mapping[WorkerRole, Mapping[int, float]],
    classes: tuple[frozenset[WorkerRole], ...],
) -> dict[int, float]:
    """Per device, the most VRAM one class holds: its members' charges added up."""
    slot: dict[int, float] = {}
    for members in classes:
        held: dict[int, float] = {}
        for role in members:
            for idx, charge in own[role].items():
                held[idx] = held.get(idx, 0.0) + charge
        for idx, total in held.items():
            slot[idx] = max(slot.get(idx, 0.0), total)
    return slot


@dataclass
class _SwapGroup:
    """The swap group a pack builds: each member's own charge, the holder and the classes.

    Roles of one class stay loaded together; loading a role of another class evicts
    them. The group holds, per device, the largest class sum. ``holder`` is the member
    placed last: a later role takes the group's room back when the holder never runs
    with it. ``tight`` roles hold what was left of a card, so they join no class.
    """

    own: dict[WorkerRole, dict[int, float]] = field(default_factory=dict)
    holder: WorkerRole | None = None
    tight: set[WorkerRole] = field(default_factory=set)
    classes: tuple[frozenset[WorkerRole], ...] = ()

    def slot(self) -> dict[int, float]:
        """The VRAM the group holds on each device."""
        return _class_slot(self.own, self.classes)

    def frees_for(self, role: WorkerRole) -> bool:
        """True when the group gives *role* its room: the holder never runs with *role*."""
        return self.holder is not None and _phase_disjoint(self.holder, role)

    def form(
        self,
        members: dict[WorkerRole, dict[int, float]],
        holder: WorkerRole,
        remaining: dict[int, float],
    ) -> None:
        """Make *members* the group, one class each, and charge its slot beyond the holder's own."""
        self.own = members
        self.holder = holder
        self.classes = _by_first_role(frozenset({role}) for role in members)
        for idx, amount in self.slot().items():
            remaining[idx] -= amount - members[holder].get(idx, 0.0)

    def merge(self, remaining: dict[int, float]) -> None:
        """Merge each set of phase-sharing members whose extra room fits what is left.

        Runs once the pack has placed every role, so a merge never takes room the
        base plan gives a later role. A tight member holds what its card had left and
        joins no class.
        """
        held = self.slot()
        for component in residency_classes(role for role in self.own if role not in self.tight):
            candidate = (*(c for c in self.classes if not c <= component), component)
            grown = _class_slot(self.own, candidate)
            growth = {idx: amount - held.get(idx, 0.0) for idx, amount in grown.items()}
            if all(remaining[idx] - extra >= -_ROUNDING_BYTES for idx, extra in growth.items()):
                for idx, extra in growth.items():
                    remaining[idx] -= extra
                self.classes, held = candidate, grown
        self.classes = _by_first_role(self.classes)


def _by_first_role(
    classes: Iterable[frozenset[WorkerRole]],
) -> tuple[frozenset[WorkerRole], ...]:
    """The classes in a fixed order: by the smallest role value in each."""
    return tuple(sorted(classes, key=lambda c: min(r.value for r in c)))


def _refundable(
    role: WorkerRole,
    charges: dict[WorkerRole, dict[int, float]],
    swap: _SwapGroup,
) -> tuple[dict[WorkerRole, dict[int, float]], bool]:
    """The charged roles phase-disjoint from *role*, and whether the swap group yields too."""
    refunds = {r: c for r, c in charges.items() if _phase_disjoint(r, role)}
    return refunds, swap.frees_for(role)


def _free_room(
    remaining: dict[int, float],
    refunds: dict[WorkerRole, dict[int, float]],
    swap: _SwapGroup,
    group_yields: bool,
) -> dict[WorkerRole, dict[int, float]]:
    """Add the refunds' and a yielding group's VRAM back to *remaining*; the roles freed."""
    returned = [*refunds.values(), *([swap.slot()] if group_yields else [])]
    for charged in returned:
        for idx, held in charged.items():
            remaining[idx] += held
    return {**(swap.own if group_yields else {}), **refunds}


def _regroup(
    model: WorkerRole,
    charge: dict[int, float],
    freed: dict[WorkerRole, dict[int, float]],
    remaining: dict[int, float],
    charges: dict[WorkerRole, dict[int, float]],
    swap: _SwapGroup,
) -> None:
    """Make *model* and the *freed* roles the swap group; they leave the standalone charges."""
    for role in freed:
        charges.pop(role, None)
    swap.form({**freed, model: charge}, model, remaining)


def _place_beside_disjoint(
    model: ModelPlacementInput,
    place: Callable[[ModelPlacementInput], _Placed | None],
    remaining: dict[int, float],
    charges: dict[WorkerRole, dict[int, float]],
    swap: _SwapGroup,
) -> _Placed | None:
    """Retry *model* with every phase-disjoint charged role's VRAM refunded.

    Phase-disjoint roles are never resident together, so they can share one swap
    group: only the resident class is charged. On success *model* and the refunded
    roles form the group and leave ``charges`` (their VRAM is now the group's slot).
    On failure every refund is rolled back and *model* is genuinely oversize.
    """
    refunds, group_yields = _refundable(model.role, charges, swap)
    if not refunds and not group_yields:
        return None
    before = dict(remaining)
    freed = _free_room(remaining, refunds, swap, group_yields)
    placed = place(model)
    if placed is None:
        remaining.update(before)
        return None
    _regroup(model.role, placed.charges, freed, remaining, charges, swap)
    return placed


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


def _holders_sentence(holders: Sequence[RoomHolder]) -> str:
    """The sentence that names each role charged to a card ahead of the refused one."""
    named = [
        f"the {h.role.value} model {h.model} ({h.charged_bytes / 1024**3:.1f} GiB)" for h in holders
    ]
    verb = "holds" if len(named) == 1 else "hold"
    joined = " and ".join(named)
    return f"{joined[0].upper()}{joined[1:]} {verb} the rest."


def card_refusal(
    subject: str,
    device: int,
    needed: float,
    room: float,
    capacity: int,
    *,
    fix: bool,
    holders: Sequence[RoomHolder] = (),
) -> str | None:
    """The refusal of *needed* bytes on a card with *room* usable bytes; None when they fit."""
    if needed <= room:
        return None
    figures = (
        f"{subject} needs {needed / 1024**3:.1f} GiB but device {device} has "
        f"{room / 1024**3:.1f} GiB usable ({capacity / 1024**3:.1f} GiB total, "
        f"{usable_vram_fraction():.0%} headroom)"
    )
    held = _holders_sentence(holders) if holders else ""
    if not fix:
        return f"{figures}. {held}" if held else figures
    if held and needed <= capacity * usable_vram_fraction():
        roles = " or ".join(h.role.value for h in holders)
        return (
            f"{figures}. {held} Use a smaller {roles} model, or set the target of one of "
            "these models to cpu or another card."
        )
    return f"{figures}. Use a smaller model, or set its target to cpu. {held}".rstrip()


def _place_tight(
    model: ModelPlacementInput,
    remaining: dict[int, float],
    charges: dict[WorkerRole, dict[int, float]],
    swap: _SwapGroup,
) -> tuple[_Placed, int]:
    """Place an oversize *model* best-effort instead of refusing it.

    Refunds every phase-disjoint charged role (they become *model*'s swap group),
    then gives *model* the widest set of cards that helps, and drains them so the
    elastic tier places nothing there. Returns the estimated shortfall in bytes.

    Widest rather than the single most-free card. Pinned to one card with the
    others excluded by that pin, a model too big for it has nowhere to go and the
    tight placement is only a slower refusal; given the group, llama-server can
    split across them. One card is still used when one card is enough, since a
    split it does not need costs it interconnect bandwidth. A fixed reserve takes the
    roomiest card alone: it has no share per card to check before the launch.
    """
    refunds, group_yields = _refundable(model.role, charges, swap)
    freed = _free_room(remaining, refunds, swap, group_yields)
    devices = _tight_device_group(model.est_vram_bytes, remaining)
    if model.kind is MemoryKind.FIXED_RESERVE:
        devices = devices[:1]
    claimed = {idx: remaining[idx] for idx in devices}
    available = sum(claimed.values())
    for idx in devices:
        remaining[idx] = 0.0
    swap.tight.add(model.role)
    if freed:
        _regroup(model.role, claimed, freed, remaining, charges, swap)
    else:
        charges[model.role] = claimed
    placed = _Placed(plan=InstancePlan(role=model.role, devices=devices), charges=claimed)
    return placed, ceil(model.est_vram_bytes - available)


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
    return ModelPlacementInput(
        role=model.role,
        est_vram_bytes=model.est_vram_bytes,
        replicas=1,
        equal_split_only=model.equal_split_only,
    )


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
    misses: dict[WorkerRole, SplitMiss],
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
        misses=misses,
    )


def _place_split(
    model: ModelPlacementInput,
    remaining: dict[int, float],
    estimate_peak: PeakEstimator,
    *,
    chat_ctx_fit: SplitCtxFitter | None = None,
    chat_ctx_target: int = 0,
    free_headroom: dict[int, int] | None = None,
    misses: dict[WorkerRole, SplitMiss],
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
    call is a subprocess. A split the engine refuses outright runs no estimate and is not counted.

    When no split is placed, the estimated split whose busiest card overshoots its room
    least goes to *misses* for the role, unless an earlier pack already recorded one.
    """
    from lilbee.providers.base import ProviderError
    from lilbee.providers.fleet.engines.base import SplitRefusedError

    by_free = sorted(remaining, key=lambda idx: remaining[idx], reverse=True)
    best: tuple[int, list[int], tuple[int, ...], tuple[int, ...]] | None = None
    nearest: SplitMiss | None = None
    spent = 0
    refusals: set[str] = set()
    for count in range(2, len(by_free) + 1):
        chosen = by_free[:count]
        for ratio in _split_ratio_candidates(chosen, remaining, equal_only=model.equal_split_only):
            if spent >= _MAX_SPLIT_ESTIMATES:
                log.info(
                    "Stopped looking for a tensor split for %s after %d estimates; "
                    "wider layouts and finer proportions were not tried.",
                    model.role.value,
                    spent,
                )
                return _noting_miss(_best_or_none(best, model, remaining), model, nearest, misses)
            spent += 1
            try:
                per_device = estimate_peak(model.role, ratio)
            except SplitRefusedError as exc:
                # The engine's launch rule refuses this split before any estimate runs,
                # so it spends none of the budget.
                spent -= 1
                _log_first_refusal(refusals, count, model.role, exc)
                continue
            except (ProviderError, OSError):
                # An unsizable model cannot evaluate a split; the tight single-card
                # path downstream places it or, when refusing, refuses it.
                continue
            if not _split_fits(chosen, per_device, remaining):
                nearest = _nearer_miss(nearest, chosen, per_device, remaining)
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
    return _noting_miss(_best_or_none(best, model, remaining), model, nearest, misses)


def _split_fits(
    chosen: list[int], per_device: tuple[int, ...], remaining: dict[int, float]
) -> bool:
    """Whether *per_device* gives one figure per chosen card and each card holds its own."""
    return len(per_device) == len(chosen) and all(
        peak <= remaining[idx] for idx, peak in zip(chosen, per_device, strict=True)
    )


def _nearer_miss(
    nearest: SplitMiss | None,
    chosen: list[int],
    per_device: tuple[int, ...],
    remaining: dict[int, float],
) -> SplitMiss | None:
    """*nearest*, or the split of *chosen* when its busiest card overshoots its room less.

    An estimate that does not give one figure per chosen card is no split and changes nothing.
    """
    if len(per_device) != len(chosen):
        return nearest
    candidate = max(
        (
            SplitMiss(len(chosen), idx, float(peak), remaining[idx])
            for idx, peak in zip(chosen, per_device, strict=True)
        ),
        key=_overshoot,
    )
    if nearest is None or _overshoot(candidate) < _overshoot(nearest):
        return candidate
    return nearest


def _overshoot(miss: SplitMiss) -> float:
    """How far the busiest card of *miss* is over its room, in bytes."""
    return miss.need_bytes - miss.room_bytes


def _noting_miss(
    placed: _Placed | None,
    model: ModelPlacementInput,
    nearest: SplitMiss | None,
    misses: dict[WorkerRole, SplitMiss],
) -> _Placed | None:
    """Return *placed*; when it is None, keep *nearest* for the role unless one is kept."""
    if placed is None and nearest is not None:
        misses.setdefault(model.role, nearest)
    return placed


def _log_first_refusal(seen: set[str], count: int, role: WorkerRole, refusal: Exception) -> None:
    """Log *refusal* once per message, so a sweep names each rule without repeating it."""
    if str(refusal) not in seen:
        seen.add(str(refusal))
        log.info("No %d-card split for %s: %s", count, role.value, refusal)


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
    footprints = {m.role: m.est_vram_bytes for m in models}
    swap = _SwapGroup()
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
        remaining, joined = _shared_beside_disjoint(
            model, remaining, charged, footprints, swap, instances
        )
        if joined:
            instances.append(InstancePlan(role=model.role, devices=()))
        else:
            unplaceable.append(model.role)
    room = {_POOL: float(remaining)}
    swap.merge(room)
    return Placement(
        instances=tuple(instances),
        unplaceable_roles=tuple(unplaceable),
        swap_classes=swap.classes,
    )


def _shared_beside_disjoint(
    model: ModelPlacementInput,
    remaining: int,
    charged: dict[WorkerRole, int],
    footprints: dict[WorkerRole, int],
    swap: _SwapGroup,
    instances: list[InstancePlan],
) -> tuple[int, bool]:
    """Refund the shared-pool roles phase-disjoint from *model* and cap them to one instance.

    The refunded roles and *model* become one swap group on the one pool. Every member
    must be able to swap back in, so the group holds its largest class sum (see
    :class:`_SwapGroup`). Returns the budget after reclaiming the refunded memory and
    charging the group, and whether *model* joined; when even the group's slot does
    not fit the reclaimed budget the budget is unchanged.
    """
    refunds = [r for r in charged if _phase_disjoint(r, model.role)]
    group_yields = swap.frees_for(model.role)
    if not refunds and not group_yields:
        return remaining, False
    members = {r: {_POOL: float(footprints[r])} for r in refunds}
    members = {**(swap.own if group_yields else {}), **members}
    members[model.role] = {_POOL: float(model.est_vram_bytes)}
    reclaim = sum(charged[r] for r in refunds) + (int(swap.slot()[_POOL]) if group_yields else 0)
    if max(charge[_POOL] for charge in members.values()) > remaining + reclaim:
        return remaining, False
    for role in refunds:
        del charged[role]
        instances[:] = [plan for plan in instances if plan.role is not role]
        instances.append(InstancePlan(role=role, devices=(), replica=0))
    room = {_POOL: float(remaining + reclaim - model.est_vram_bytes)}
    swap.form(members, model.role, room)
    return int(room[_POOL]), True


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
    plan_auto: AutoPlanner | None = None,
    model_refs: Mapping[WorkerRole, str] = MappingProxyType({}),
    lost_card: LostCardWording | None = None,
    equal_split_roles: frozenset[WorkerRole] = frozenset(),
) -> Placement:
    """Build a Placement from a manual *spec*, charging each card and failing loud.

    A card role of *equal_split_roles* with no explicit ``tensor_split`` takes the
    first ratio of the equal-only ladder (:func:`_split_ratio_candidates`).

    ``device_capacity`` is each card's total VRAM (not instantaneous free): the
    plan defines the fleet's full intended residency, so charging it against live
    free VRAM would double-count models already loaded. Every active role must
    have an entry; every device must exist (*lost_card* words the refusal for a
    device the machine lacks); each card must fit the sum of the
    per-device peaks charged to it, within the cfg.usable_vram_fraction headroom.

    A role with the auto target is planned last by *plan_auto*, on the capacity
    the other roles leave. It never names a card, so a card the machine lacks
    cannot refuse it.

    A role with no entry, or one its card cannot hold, raises a plain error when the
    automatic plan can replace the spec. Otherwise it raises
    :class:`TargetRefusedError`, naming the role.
    """
    from lilbee.providers.fleet.engines.base import SplitRefusedError

    usable = usable_vram_fraction()
    remaining = {idx: total * usable for idx, total in device_capacity.items()}
    instances: list[InstancePlan] = []
    auto_roles: list[WorkerRole] = []
    card_roles: list[WorkerRole] = []
    held: dict[int, list[RoomHolder]] = {}
    for role in active_roles:
        rp = _required_entry(
            spec, role, device_capacity, lost_card, refuse_alone=refuses_role_alone(spec)
        )
        if rp.device_target is RoleTarget.AUTO:
            auto_roles.append(role)
            continue
        if rp.device_target is not RoleTarget.GPU:
            instances.extend(
                InstancePlan(role=role, devices=(), replica=replica, target=rp.device_target)
                for replica in range(rp.replicas)
            )
            continue
        card_roles.append(role)
        ratio = (
            rp.tensor_split
            or _split_ratio_candidates(rp.devices, remaining, equal_only=role in equal_split_roles)[
                0
            ]
        )
        try:
            per_device = estimate_peak(role, ratio)
        except SplitRefusedError as exc:
            raise _spec_refusal(role, str(exc), refuse_alone=refuses_role_alone(spec)) from exc
        split = ratio if len(rp.devices) > 1 else ()
        for replica in range(rp.replicas):
            _charge_devices(
                role,
                rp.devices,
                per_device,
                remaining,
                device_capacity,
                refuse_alone=refuses_role_alone(spec),
                held=held,
            )
            instances.append(
                InstancePlan(
                    role=role, devices=tuple(rp.devices), tensor_split=split, replica=replica
                )
            )
        model = model_refs.get(role, role.value)
        for idx, peak in zip(rp.devices, per_device, strict=True):
            held.setdefault(idx, []).append(RoomHolder(role, model, peak * rp.replicas))
    if not auto_roles:
        return Placement(instances=tuple(instances), unplaceable_roles=())
    if plan_auto is None:
        names = ", ".join(role.value for role in auto_roles)
        raise PlacementError(f"{names}: the auto target needs a planner to place it")
    left = {idx: round(room / usable) for idx, room in remaining.items()}
    placed = plan_auto(left, frozenset(card_roles))
    return replace(placed, instances=(*instances, *placed.instances))


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
    devices: Sequence[int], remaining: dict[int, float], *, equal_only: bool = False
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

    An engine that launches only equal shares (*equal_only*) gets the proportions
    above that are already equal, then the equal ratio itself. On cards of different
    sizes the equal ratio is the one split it accepts, and each card then holds an
    equal share of the footprint, so the smallest card gates it.
    """
    quantum = _vram_proportional_split(devices, remaining, divisor=_RATIO_QUANTUM_DIVISOR)
    candidates = [
        _vram_proportional_split(devices, remaining),
        quantum,
        _shifted_toward_roomiest(devices, remaining, quantum),
    ]
    if equal_only:
        candidates = [c for c in candidates if len(set(c)) == 1]
        candidates.append((1,) * len(devices))
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


def _device_phrase(indices: Sequence[int]) -> str:
    """The device indices as a phrase: "device 0", "devices 0 and 1"."""
    labels = [str(index) for index in sorted(indices)]
    if len(labels) == 1:
        return f"device {labels[0]}"
    return f"devices {', '.join(labels[:-1])} and {labels[-1]}"


def dropped_pins_sentence(
    spec: PlacementSpec, placed: Placement, roles: Collection[WorkerRole]
) -> str:
    """The sentence that says where each role of *spec* on a card runs in *placed*.

    Empty when the spec pins none of *roles* to a card.
    """
    parts: list[str] = []
    for role in roles:
        entry = spec.roles.get(role)
        if entry is None or entry.device_target is not RoleTarget.GPU:
            continue
        pinned = _device_phrase(entry.devices)
        now = {idx for inst in placed.instances if inst.role is role for idx in inst.devices}
        if not now:
            parts.append(f"{role.value} was pinned to {pinned} and does not run")
        elif now == set(entry.devices):
            parts.append(f"{role.value} was pinned to {pinned} and still runs on {pinned}")
        else:
            moved = _device_phrase(sorted(now))
            parts.append(f"{role.value} was pinned to {pinned} and now runs on {moved}")
    return f"The pins are dropped: {'; '.join(parts)}." if parts else ""


def lost_card_reason(
    role: WorkerRole,
    lost: Sequence[int],
    listed: Sequence[int],
    *,
    on_host: int = 0,
    floor: bool = False,
    hint: str = "",
) -> str:
    """The sentence for *role* pinned to the *lost* devices that the machine does not list.

    *floor* says the *on_host* count is a lower bound, because a card table did not read.
    """
    present = f"only {_device_phrase(listed)}" if listed else "no devices"
    noun = "card" if on_host == 1 else "cards"
    bound = "at least " if floor else ""
    count = f" ({bound}{on_host} {noun} on this host)" if on_host > len(listed) else ""
    fix = f"Set the {role.value} devices to a card that exists, or set its target to cpu."
    found = f"{role.value} is pinned to {_device_phrase(lost)}, but this machine lists {present}"
    return " ".join(part for part in (f"{found}{count}.", hint, fix) if part)


def unseen_card_reason(
    role: WorkerRole, engine: str, unseen: Sequence[int], listed: Sequence[int]
) -> str:
    """The sentence for *role* placed on *unseen* devices that the *engine* does not list.

    A device whose index the engine lists, but for another card, is named as that.
    """
    other = [index for index in unseen if index in listed]
    missing = [index for index in unseen if index not in listed]
    if other:
        found = f"lists a different card at {_device_phrase(other)}"
        if missing:
            found += f" and no card at {_device_phrase(missing)}"
    else:
        found = f"lists only {_device_phrase(listed)}"
    return (
        f"{role.value} is placed on {_device_phrase(unseen)}, but the {engine} engine "
        f"{found}. Set the {role.value} target to cpu, or choose an engine that lists "
        "that card."
    )


def _required_entry(
    spec: PlacementSpec,
    role: WorkerRole,
    device_capacity: dict[int, int],
    lost_card: LostCardWording | None = None,
    *,
    refuse_alone: bool,
) -> RolePlacement:
    """Return *role*'s placement entry, failing loud if absent or pinned off-hardware.

    With *refuse_alone* a missing entry refuses that role alone.
    """
    rp = spec.roles.get(role)
    if rp is None:
        reason = (
            f"{role.value} has a model but the placement has no entry for it. "
            f"Add an entry for {role.value} with devices or a target, or clear the placement."
        )
        raise _spec_refusal(role, reason, refuse_alone=refuse_alone)
    lost = [idx for idx in rp.devices if idx not in device_capacity]
    if lost:
        word = lost_card or functools.partial(lost_card_reason, listed=list(device_capacity))
        raise PlacementError(word(role, lost))
    return rp


def _spec_refusal(role: WorkerRole, reason: str, *, refuse_alone: bool) -> PlacementError:
    """A manual spec's refusal: it names *role* when the launch drops that role alone."""
    return TargetRefusedError({role: reason}) if refuse_alone else PlacementError(reason)


def refuses_role_alone(spec: PlacementSpec) -> bool:
    """Whether *spec* drops a role it cannot place alone, rather than falling back to auto."""
    return not spec.all_on_cards


def refuse_role_without_entry(spec: PlacementSpec, role: WorkerRole) -> None:
    """Raise serving's refusal when *spec* has no entry for *role* and drops it alone."""
    if role not in spec.roles and refuses_role_alone(spec):
        _required_entry(spec, role, {}, refuse_alone=True)


def _charge_devices(
    role: WorkerRole,
    devices: tuple[int, ...],
    per_device: tuple[int, ...],
    remaining: dict[int, float],
    device_capacity: dict[int, int],
    *,
    refuse_alone: bool,
    held: Mapping[int, Sequence[RoomHolder]],
) -> None:
    """Subtract one instance's per-device peaks from *remaining*; fail loud if a card overflows.

    An estimate that does not cover every pinned device is a PlacementError rather
    than a zip mismatch: an engine can state no per-device breakdown, and the auto
    planner skips such a candidate (see :func:`_place_split`), so the manual path
    must refuse in the currency callers already handle. With *refuse_alone* the
    error names the role, for the launch to drop only that role.
    """

    def refusal(reason: str) -> PlacementError:
        return _spec_refusal(role, reason, refuse_alone=refuse_alone)

    if not per_device and len(devices) > 1:
        raise refusal(
            f"{role.value} has no per-card memory figure, so it cannot be pinned to "
            f"{len(devices)} cards. Pin one card, or set its target to auto."
        )
    if len(per_device) != len(devices):
        raise refusal(
            f"{role.value} is pinned to {len(devices)} device(s) but its memory estimate "
            f"covers {len(per_device)}; clear the placement to place it automatically"
        )
    for idx, peak in zip(devices, per_device, strict=True):
        reason = card_refusal(
            f"{role.value} pinned to device {idx}",
            idx,
            peak,
            remaining[idx],
            device_capacity[idx],
            fix=refuse_alone,
            holders=held.get(idx, ()),
        )
        if reason is not None:
            raise refusal(reason)
    for idx, peak in zip(devices, per_device, strict=True):
        remaining[idx] -= peak
