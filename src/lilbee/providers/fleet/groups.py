"""Swap-group identity for the llama-swap processes that front the fleet."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum

from lilbee.providers.roles import ROLE_REGISTRY, WorkerRole


class SwapGroup(StrEnum):
    """One swap process's group: a role's own, or the shared co-tenant process."""

    CHAT = "chat"
    EMBED = "embed"
    RERANK = "rerank"
    VISION = "vision"
    CO_TENANT = "co-tenant"

    @property
    def swaps(self) -> bool:
        """Whether the process evicts between residency classes (one exclusive group each)."""
        return self is SwapGroup.CO_TENANT


def group_for(role: WorkerRole, co_tenants: frozenset[WorkerRole]) -> SwapGroup:
    """The swap group *role* runs in: the shared co-tenant group, or its own."""
    return SwapGroup.CO_TENANT if role in co_tenants else SwapGroup(role.value)


def class_roles(classes: tuple[frozenset[WorkerRole], ...]) -> frozenset[WorkerRole]:
    """Every role in *classes*."""
    return frozenset().union(*classes)


def class_label(role: WorkerRole, classes: tuple[frozenset[WorkerRole], ...]) -> str:
    """The name of the class of *role* (its index in *classes*); empty when it has none."""
    return next((str(i) for i, members in enumerate(classes) if role in members), "")


def residency_classes(roles: Iterable[WorkerRole]) -> tuple[frozenset[WorkerRole], ...]:
    """Split *roles* into the sets that share a run phase, directly or through a third role."""
    classes: list[frozenset[WorkerRole]] = []
    for role in sorted(roles, key=lambda r: r.value):
        phases = ROLE_REGISTRY[role].phases
        touching = [c for c in classes if any(ROLE_REGISTRY[m].phases & phases for m in c)]
        merged = frozenset({role}).union(*touching)
        classes = [c for c in classes if c not in touching] + [merged]
    return tuple(sorted(classes, key=lambda c: min(r.value for r in c)))


def swap_summary(classes: tuple[frozenset[WorkerRole], ...]) -> str:
    """The text for the co-tenant roles: ``+`` joins roles that stay loaded together."""
    parts = [" + ".join(sorted(r.value for r in c)) for c in classes]
    if all(len(c) == 1 for c in classes):
        return f"{', '.join(parts)}: share memory, one loaded at a time"
    return f"{', '.join(parts)}: share memory, one group loaded at a time"
