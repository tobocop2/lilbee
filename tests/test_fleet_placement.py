"""Tests for the multi-GPU placement planner."""

from __future__ import annotations

from lilbee.providers.fleet.placement import (
    InstancePlan,
    ModelPlacementInput,
    Placement,
    plan_placement,
)
from lilbee.providers.roles import WorkerRole

_GB = 1024**3


class TestPlanPlacement:
    def test_single_model_fits_one_gpu(self) -> None:
        plan = plan_placement(
            [ModelPlacementInput(WorkerRole.CHAT, 10 * _GB)],
            [(0, 24 * _GB)],
        )
        assert plan == Placement(
            instances=(InstancePlan(WorkerRole.CHAT, (0,)),),
            unplaceable_roles=(),
        )

    def test_colocates_small_models_on_one_gpu(self) -> None:
        plan = plan_placement(
            [
                ModelPlacementInput(WorkerRole.EMBED, 1 * _GB),
                ModelPlacementInput(WorkerRole.RERANK, 1 * _GB),
            ],
            [(0, 24 * _GB)],
        )
        assert plan.unplaceable_roles == ()
        assert {i.role for i in plan.instances} == {WorkerRole.EMBED, WorkerRole.RERANK}
        assert all(i.devices == (0,) for i in plan.instances)

    def test_tensor_splits_model_too_big_for_one_gpu(self) -> None:
        # 30 GB > 24*0.9 per GPU, but < combined headroom -> split across both.
        plan = plan_placement(
            [ModelPlacementInput(WorkerRole.CHAT, 30 * _GB)],
            [(0, 24 * _GB), (1, 24 * _GB)],
        )
        # Equal cards -> equal proportion (int(24*0.9 GiB) = 21 each).
        assert plan.instances == (InstancePlan(WorkerRole.CHAT, (0, 1), (21, 21)),)
        assert plan.unplaceable_roles == ()

    def test_tensor_split_is_proportional_on_unequal_gpus(self) -> None:
        # 28 GB splits across a 24 GB + 16 GB pair; the ratio must follow free VRAM
        # (int(24*0.9)=21, int(16*0.9)=14), not an even 1:1 that would OOM the small card.
        plan = plan_placement(
            [ModelPlacementInput(WorkerRole.CHAT, 28 * _GB)],
            [(0, 24 * _GB), (1, 16 * _GB)],
        )
        assert plan.instances[0].devices == (0, 1)
        assert plan.instances[0].tensor_split == (21, 14)

    def test_unplaceable_when_model_fits_nowhere(self) -> None:
        plan = plan_placement(
            [ModelPlacementInput(WorkerRole.CHAT, 100 * _GB)],
            [(0, 24 * _GB), (1, 24 * _GB)],
        )
        assert plan.instances == ()
        assert plan.unplaceable_roles == (WorkerRole.CHAT,)

    def test_no_gpu_devices_places_every_role_on_cpu(self) -> None:
        # A GPU-less host (or a probe that found nothing): each role runs as a
        # single un-pinned CPU instance so a fleet-of-one works without a GPU.
        plan = plan_placement(
            [
                ModelPlacementInput(WorkerRole.CHAT, 5 * _GB),
                ModelPlacementInput(WorkerRole.EMBED, 1 * _GB),
            ],
            [],
        )
        assert {i.role for i in plan.instances} == {WorkerRole.CHAT, WorkerRole.EMBED}
        assert all(i.devices == () and i.tensor_split == () for i in plan.instances)
        assert plan.unplaceable_roles == ()

    def test_unified_budget_places_roles_that_fit_unpinned(self) -> None:
        # No discrete GPU but a measured shared-RAM budget (Apple Silicon / CPU):
        # roles that fit run un-pinned; system RAM is the shared pool.
        plan = plan_placement(
            [
                ModelPlacementInput(WorkerRole.CHAT, 8 * _GB),
                ModelPlacementInput(WorkerRole.EMBED, 1 * _GB),
            ],
            [],
            unified_budget=12 * _GB,
        )
        assert {i.role for i in plan.instances} == {WorkerRole.CHAT, WorkerRole.EMBED}
        assert all(i.devices == () for i in plan.instances)
        assert plan.unplaceable_roles == ()

    def test_unified_budget_marks_oversize_role_unplaceable(self) -> None:
        # A model larger than free system RAM must NOT load (it would force the OS
        # into an OOM livelock): it is unplaceable, gets no server, calls error.
        plan = plan_placement(
            [ModelPlacementInput(WorkerRole.CHAT, 20 * _GB)],
            [],
            unified_budget=11 * _GB,
        )
        assert plan.instances == ()
        assert plan.unplaceable_roles == (WorkerRole.CHAT,)

    def test_shared_pool_reserves_search_roles_before_chat(self) -> None:
        # Search-first: embed is reserved before the elastic chat, so a chat that
        # would consume the whole 12 GB pool is dropped instead of starving search
        # (the proven embed-starvation bug on Apple Silicon).
        plan = plan_placement(
            [
                ModelPlacementInput(WorkerRole.CHAT, 10 * _GB),
                ModelPlacementInput(WorkerRole.EMBED, 4 * _GB),
            ],
            [],
            unified_budget=12 * _GB,
        )
        assert {i.role for i in plan.instances} == {WorkerRole.EMBED}
        assert plan.unplaceable_roles == (WorkerRole.CHAT,)

    def test_first_fit_decreasing_places_largest_first(self) -> None:
        plan = plan_placement(
            [
                ModelPlacementInput(WorkerRole.EMBED, 5 * _GB),
                ModelPlacementInput(WorkerRole.CHAT, 20 * _GB),
            ],
            [(0, 24 * _GB), (1, 24 * _GB)],
        )
        by_role = {i.role: i.devices for i in plan.instances}
        # Largest (chat) placed first on device 0; embed then lands on the
        # now-emptier device 1. Each is a single-GPU instance.
        assert by_role == {WorkerRole.CHAT: (0,), WorkerRole.EMBED: (1,)}
        assert plan.unplaceable_roles == ()


class TestReplicas:
    """Data-parallel replicas: N instances of a role spread one-per-GPU for throughput."""

    def _embeds(self, plan: Placement) -> list[InstancePlan]:
        return [i for i in plan.instances if i.role is WorkerRole.EMBED]

    def test_replicas_spread_one_per_distinct_gpu(self) -> None:
        plan = plan_placement(
            [ModelPlacementInput(WorkerRole.EMBED, 1 * _GB, replicas=3)],
            [(0, 24 * _GB), (1, 24 * _GB), (2, 24 * _GB)],
        )
        embeds = self._embeds(plan)
        assert len(embeds) == 3
        assert {i.devices[0] for i in embeds} == {0, 1, 2}  # one per distinct card
        assert {i.replica for i in embeds} == {0, 1, 2}  # distinct replica indices
        assert plan.unplaceable_roles == ()

    def test_replicas_wrap_to_colocate_when_more_than_gpus(self) -> None:
        # embed ~10GB, 2x24GB (21.6 usable) fits 2 per card -> 4 of 8 requested place.
        plan = plan_placement(
            [ModelPlacementInput(WorkerRole.EMBED, 10 * _GB, replicas=8)],
            [(0, 24 * _GB), (1, 24 * _GB)],
        )
        embeds = self._embeds(plan)
        assert len(embeds) == 4  # two per card, then no room
        assert sorted(i.devices[0] for i in embeds) == [0, 0, 1, 1]
        assert {i.replica for i in embeds} == {0, 1, 2, 3}

    def test_replicated_role_unplaceable_when_no_gpu_fits_one(self) -> None:
        plan = plan_placement(
            [ModelPlacementInput(WorkerRole.EMBED, 100 * _GB, replicas=2)],
            [(0, 24 * _GB)],
        )
        assert plan.instances == ()
        assert plan.unplaceable_roles == (WorkerRole.EMBED,)

    def test_chat_placed_before_replicas_then_replicas_fill_remaining(self) -> None:
        plan = plan_placement(
            [
                ModelPlacementInput(WorkerRole.CHAT, 20 * _GB),  # single, claims a card first
                ModelPlacementInput(WorkerRole.EMBED, 1 * _GB, replicas=2),
            ],
            [(0, 24 * _GB), (1, 24 * _GB)],
        )
        assert len([i for i in plan.instances if i.role is WorkerRole.CHAT]) == 1
        assert len(self._embeds(plan)) == 2
        assert plan.unplaceable_roles == ()

    def test_unified_budget_runs_replicas_as_coresident_processes(self) -> None:
        plan = plan_placement(
            [ModelPlacementInput(WorkerRole.EMBED, 2 * _GB, replicas=3)],
            [],
            unified_budget=10 * _GB,
        )
        embeds = self._embeds(plan)
        assert len(embeds) == 3  # 3x2GB <= 10GB
        assert all(i.devices == () for i in embeds)
        assert {i.replica for i in embeds} == {0, 1, 2}

    def test_unified_budget_caps_replicas_to_what_fits(self) -> None:
        plan = plan_placement(
            [ModelPlacementInput(WorkerRole.EMBED, 4 * _GB, replicas=5)],
            [],
            unified_budget=10 * _GB,
        )
        assert len(self._embeds(plan)) == 2  # 2x4=8<=10; a third (12) overruns

    def test_no_gpu_no_budget_expands_every_replica(self) -> None:
        # GPU-less + ungated legacy path: each replica becomes an un-pinned instance.
        plan = plan_placement(
            [ModelPlacementInput(WorkerRole.EMBED, 1 * _GB, replicas=2)],
            [],
        )
        embeds = self._embeds(plan)
        assert len(embeds) == 2
        assert {i.replica for i in embeds} == {0, 1}
