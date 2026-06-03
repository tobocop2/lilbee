"""Tests for the multi-GPU placement planner."""

from __future__ import annotations

from lilbee.providers.fleet.placement import (
    InstancePlan,
    ModelPlacementInput,
    Placement,
    estimate_model_vram,
    plan_placement,
)
from lilbee.providers.roles import WorkerRole

_GB = 1024**3


class TestEstimateModelVram:
    def test_includes_weights_kv_and_overhead(self) -> None:
        meta = {"block_count": "32", "embedding_length": "4096"}
        est = estimate_model_vram(_GB, meta, ctx=4096, slots=4, kv_elem_bytes=2)
        kv = 2 * 32 * 4096 * 2 * 4096 * 4
        assert est == _GB + kv + _GB  # weights + kv + overhead

    def test_zero_kv_when_meta_none(self) -> None:
        assert estimate_model_vram(_GB, None, ctx=4096, slots=4, kv_elem_bytes=2) == _GB + _GB

    def test_zero_kv_when_shape_fields_missing(self) -> None:
        assert estimate_model_vram(_GB, {}, ctx=4096, slots=4, kv_elem_bytes=2) == _GB + _GB

    def test_unparseable_field_treated_as_zero(self) -> None:
        meta = {"block_count": "garbage", "embedding_length": "4096"}
        assert estimate_model_vram(_GB, meta, ctx=4096, slots=4, kv_elem_bytes=2) == _GB + _GB

    def test_gqa_kv_scales_by_kv_heads_not_embedding_length(self) -> None:
        # GQA: kv_dim = head_count_kv * head_dim (head_dim = embedding_length /
        # head_count), far smaller than the full embedding width. Mirrors the
        # Qwen3-Coder-30B shape (8x fewer KV heads than query heads).
        meta = {
            "block_count": "48",
            "embedding_length": "2048",
            "head_count": "32",
            "head_count_kv": "4",
        }
        est = estimate_model_vram(_GB, meta, ctx=4096, slots=1, kv_elem_bytes=1)
        head_dim = 2048 // 32
        kv = 2 * 48 * (4 * head_dim) * 4096 * 1 * 1
        assert est == _GB + kv + _GB

    def test_kv_falls_back_to_embedding_length_without_head_metadata(self) -> None:
        # No GQA head fields -> assume full attention (kv_dim = embedding_length).
        meta = {"block_count": "32", "embedding_length": "4096"}
        est = estimate_model_vram(_GB, meta, ctx=4096, slots=1, kv_elem_bytes=1)
        kv = 2 * 32 * 4096 * 4096 * 1 * 1
        assert est == _GB + kv + _GB


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

    def test_unified_budget_is_one_shared_pool_across_roles(self) -> None:
        # Largest-first: chat fits the 12 GB pool, leaving too little for embed.
        plan = plan_placement(
            [
                ModelPlacementInput(WorkerRole.CHAT, 10 * _GB),
                ModelPlacementInput(WorkerRole.EMBED, 4 * _GB),
            ],
            [],
            unified_budget=12 * _GB,
        )
        assert {i.role for i in plan.instances} == {WorkerRole.CHAT}
        assert plan.unplaceable_roles == (WorkerRole.EMBED,)

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
