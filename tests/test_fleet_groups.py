"""Tests for swap-group identity and the role-to-group mapping."""

from __future__ import annotations

from lilbee.providers.fleet.groups import SwapGroup, group_for
from lilbee.providers.roles import WorkerRole


class TestSwapGroupPolicy:
    def test_only_the_co_tenant_group_evicts_its_members(self) -> None:
        assert SwapGroup.CO_TENANT.swaps is True
        for group in (SwapGroup.CHAT, SwapGroup.EMBED, SwapGroup.RERANK, SwapGroup.VISION):
            assert group.swaps is False

    def test_every_role_has_a_group_of_its_own(self) -> None:
        # group_for falls back to the role's own group, so the name must resolve.
        for role in WorkerRole:
            assert SwapGroup(role.value).value == role.value


class TestGroupFor:
    def test_a_role_outside_the_co_tenants_keeps_its_own_group(self) -> None:
        co_tenants = frozenset({WorkerRole.CHAT, WorkerRole.VISION})
        assert group_for(WorkerRole.EMBED, co_tenants) is SwapGroup.EMBED
        assert group_for(WorkerRole.RERANK, co_tenants) is SwapGroup.RERANK

    def test_co_tenants_share_one_group(self) -> None:
        co_tenants = frozenset({WorkerRole.CHAT, WorkerRole.VISION})
        assert group_for(WorkerRole.CHAT, co_tenants) is SwapGroup.CO_TENANT
        assert group_for(WorkerRole.VISION, co_tenants) is SwapGroup.CO_TENANT

    def test_no_co_tenancy_leaves_chat_and_vision_pinned_apart(self) -> None:
        assert group_for(WorkerRole.CHAT, frozenset()) is SwapGroup.CHAT
        assert group_for(WorkerRole.VISION, frozenset()) is SwapGroup.VISION
