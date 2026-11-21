"""Tests for swap-group identity and the role-to-group mapping."""

from __future__ import annotations

from lilbee.providers.fleet.groups import (
    SwapGroup,
    class_label,
    class_roles,
    group_for,
    residency_classes,
    swap_summary,
)
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


class TestResidencyClasses:
    def test_chat_and_rerank_share_the_query_phase_and_vision_stands_alone(self) -> None:
        classes = residency_classes({WorkerRole.CHAT, WorkerRole.RERANK, WorkerRole.VISION})
        assert classes == (
            frozenset({WorkerRole.CHAT, WorkerRole.RERANK}),
            frozenset({WorkerRole.VISION}),
        )

    def test_a_role_in_both_phases_joins_every_role_through_the_shared_phase(self) -> None:
        classes = residency_classes(set(WorkerRole))
        assert classes == (frozenset(WorkerRole),)

    def test_a_single_role_is_its_own_class(self) -> None:
        assert residency_classes({WorkerRole.VISION}) == (frozenset({WorkerRole.VISION}),)

    def test_no_roles_make_no_classes(self) -> None:
        assert residency_classes(set()) == ()


class TestClassHelpers:
    classes = (frozenset({WorkerRole.CHAT, WorkerRole.RERANK}), frozenset({WorkerRole.VISION}))

    def test_class_roles_is_the_union(self) -> None:
        assert class_roles(self.classes) == {
            WorkerRole.CHAT,
            WorkerRole.RERANK,
            WorkerRole.VISION,
        }
        assert class_roles(()) == frozenset()

    def test_class_label_is_the_index_of_the_class(self) -> None:
        assert class_label(WorkerRole.RERANK, self.classes) == "0"
        assert class_label(WorkerRole.VISION, self.classes) == "1"

    def test_a_role_outside_every_class_has_no_label(self) -> None:
        assert class_label(WorkerRole.EMBED, self.classes) == ""


class TestSwapSummary:
    def test_classes_of_one_role_take_turns(self) -> None:
        classes = (frozenset({WorkerRole.CHAT}), frozenset({WorkerRole.VISION}))
        assert swap_summary(classes) == "chat, vision: share memory, one loaded at a time"

    def test_roles_of_one_class_are_joined_by_a_plus(self) -> None:
        classes = (frozenset({WorkerRole.CHAT, WorkerRole.RERANK}), frozenset({WorkerRole.VISION}))
        assert swap_summary(classes) == (
            "chat + rerank, vision: share memory, one group loaded at a time"
        )
