"""Tests for the multi-GPU placement planner."""

from __future__ import annotations

from lilbee.providers.multi_gpu.placement import (
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


class TestPlanPlacement:
    def test_single_model_fits_one_gpu(self) -> None:
        plan = plan_placement(
            [ModelPlacementInput(WorkerRole.CHAT, 10 * _GB)],
            [(0, 24 * _GB)],
        )
        assert plan == Placement(
            instances=(InstancePlan(WorkerRole.CHAT, (0,)),),
            in_process_roles=(),
        )

    def test_colocates_small_models_on_one_gpu(self) -> None:
        plan = plan_placement(
            [
                ModelPlacementInput(WorkerRole.EMBED, 1 * _GB),
                ModelPlacementInput(WorkerRole.RERANK, 1 * _GB),
            ],
            [(0, 24 * _GB)],
        )
        assert plan.in_process_roles == ()
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
        assert plan.in_process_roles == ()

    def test_tensor_split_is_proportional_on_unequal_gpus(self) -> None:
        # 28 GB splits across a 24 GB + 16 GB pair; the ratio must follow free VRAM
        # (int(24*0.9)=21, int(16*0.9)=14), not an even 1:1 that would OOM the small card.
        plan = plan_placement(
            [ModelPlacementInput(WorkerRole.CHAT, 28 * _GB)],
            [(0, 24 * _GB), (1, 16 * _GB)],
        )
        assert plan.instances[0].devices == (0, 1)
        assert plan.instances[0].tensor_split == (21, 14)

    def test_in_process_when_model_fits_nowhere(self) -> None:
        plan = plan_placement(
            [ModelPlacementInput(WorkerRole.CHAT, 100 * _GB)],
            [(0, 24 * _GB), (1, 24 * _GB)],
        )
        assert plan.instances == ()
        assert plan.in_process_roles == (WorkerRole.CHAT,)

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
        assert plan.in_process_roles == ()
