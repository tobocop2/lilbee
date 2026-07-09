"""Swap-group identity for the llama-swap processes that front the fleet."""

from __future__ import annotations

from enum import StrEnum

from lilbee.providers.roles import WorkerRole


class SwapGroup(StrEnum):
    """One llama-swap process's group: a role's own, or the shared chat/vision pair."""

    CHAT = "chat"
    EMBED = "embed"
    RERANK = "rerank"
    VISION = "vision"
    CO_TENANT = "co-tenant"

    @property
    def swaps(self) -> bool:
        """Whether loading a member evicts its siblings (llama-swap ``swap: true``)."""
        return self is SwapGroup.CO_TENANT


def group_for(role: WorkerRole, co_tenants: frozenset[WorkerRole]) -> SwapGroup:
    """The swap group *role* runs in: the shared co-tenant group, or its own."""
    return SwapGroup.CO_TENANT if role in co_tenants else SwapGroup(role.value)
