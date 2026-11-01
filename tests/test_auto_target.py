"""The auto target plans a role on the cards the named roles leave."""

from __future__ import annotations

from pathlib import Path

import pytest

from lilbee.app import placement as app_placement
from lilbee.app.engine_choice import BookReading
from lilbee.core.config import cfg
from lilbee.providers.fleet import planning
from lilbee.providers.fleet.placement import (
    InstancePlan,
    ModelPlacementInput,
    Placement,
    placement_from_spec,
)
from lilbee.providers.fleet.placement_spec import (
    PlacementError,
    PlacementSpec,
    TargetRefusedError,
)
from lilbee.providers.roles import RoleTarget, WorkerRole
from tests._fleet_engines import (
    CardEngine,
    Role,
    cards_of,
    engines_for_all_roles,
    prepare_host,
)

GIB = 1 << 30
ROLES = (Role(WorkerRole.CHAT, 20), Role(WorkerRole.EMBED, 6), Role(WorkerRole.VISION, 8))
SPEC = '{"chat": {"devices": [0]}, "vision": {"target": "auto"}, "embed": {"target": "cpu"}}'
VISION_REFUSED = (
    "vision: needs 8.0 GiB, which does not fit in the memory left beside chat. "
    "Use a smaller model, or set its target to cpu."
)
LOST_CHAT = (
    "chat: GPU 1 is not on this machine (1 found). "
    "Set the chat devices to a GPU that exists, or set its target to cpu."
)


def _host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cards: tuple[int, ...], spec: str
) -> CardEngine:
    prepare_host(monkeypatch, tmp_path, ROLES, 64 * GIB)
    monkeypatch.setattr(cfg, "placement", spec)
    engine = CardEngine("engine-a", tmp_path / "engine-server", ROLES, cards_of(cards), spans=True)
    reading = BookReading(adapters=engines_for_all_roles(engine), notices={})
    monkeypatch.setattr(app_placement, "_planning_engines", lambda: reading)
    return engine


def _where(spec: str, engine: CardEngine) -> dict[str, tuple[tuple[int, ...], str]]:
    resolved = planning.resolve_placement_plan(
        PlacementSpec.from_json(spec), engines=engines_for_all_roles(engine), fall_back_to_auto=True
    )
    return {p.role.value: (p.devices, p.target.value) for p in resolved.instances}


class TestAutoPlansOnTheRemainingCards:
    """Row auto_plans_on_remaining_cards."""

    def test_chat_on_card_0_vision_auto_embed_cpu_on_two_cards(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        engine = _host(monkeypatch, tmp_path, (24, 24), SPEC)
        assert _where(SPEC, engine) == {
            "chat": ((0,), "gpu"),
            "embed": ((), "cpu"),
            "vision": ((1,), "gpu"),
        }

    def test_the_launch_runs_the_same_plan(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _host(monkeypatch, tmp_path, (24, 24), SPEC)
        engine = CardEngine(
            "engine-a", tmp_path / "engine-server", ROLES, cards_of((24, 24)), spans=True
        )
        plan = planning.plan_all_launches(engines=engines_for_all_roles(engine))
        assert sorted(launch.role.value for launch in plan.launches) == ["chat", "embed", "vision"]
        assert plan.refused_roles == {}

    def test_a_pinned_card_that_is_gone_refuses_that_role_and_auto_takes_its_card(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        spec = (
            '{"chat": {"devices": [1]}, "vision": {"target": "auto"}, "embed": {"target": "cpu"}}'
        )
        engine = _host(monkeypatch, tmp_path, (24,), spec)
        plan = planning.plan_all_launches(engines=engines_for_all_roles(engine))
        assert sorted(launch.role.value for launch in plan.launches) == ["embed", "vision"]
        assert plan.refused_roles == {WorkerRole.CHAT: LOST_CHAT}
        assert _where(spec, engine) == {"embed": ((), "cpu"), "vision": ((0,), "gpu")}

    def test_the_view_reports_the_roles_that_run_and_the_reason_for_the_rest(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        spec = (
            '{"chat": {"devices": [1]}, "vision": {"target": "auto"}, "embed": {"target": "cpu"}}'
        )
        _host(monkeypatch, tmp_path, (24,), spec)
        view = app_placement.get_placement()
        assert {r.role.value: (r.devices, r.target.value) for r in view.roles} == {
            "embed": ((), "cpu"),
            "vision": ((0,), "gpu"),
        }
        saved = PlacementSpec.from_json(spec).to_json()
        assert (view.manual, view.refusal, view.rejected_spec_json) == (False, LOST_CHAT, saved)

    def test_an_auto_role_with_no_room_beside_a_named_role_is_refused_alone(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        engine = _host(monkeypatch, tmp_path, (24,), SPEC)
        plan = planning.plan_all_launches(engines=engines_for_all_roles(engine))
        assert sorted(launch.role.value for launch in plan.launches) == ["chat", "embed"]
        assert plan.refused_roles == {WorkerRole.VISION: VISION_REFUSED}
        assert plan.co_tenants == frozenset()

    def test_the_view_names_the_auto_role_it_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _host(monkeypatch, tmp_path, (24,), SPEC)
        view = app_placement.get_placement()
        assert {r.role.value: (r.devices, r.target.value) for r in view.roles} == {
            "chat": ((0,), "gpu"),
            "embed": ((), "cpu"),
        }
        assert view.tight == ()
        assert (view.manual, view.refusal) == (False, VISION_REFUSED)

    def test_setting_the_placement_refuses_with_the_same_reason(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        engine = _host(monkeypatch, tmp_path, (24,), SPEC)
        with pytest.raises(TargetRefusedError, match=r"vision: needs 8\.0 GiB") as refused:
            planning.resolve_placement_plan(
                PlacementSpec.from_json(SPEC), engines=engines_for_all_roles(engine)
            )
        assert refused.value.reasons == {WorkerRole.VISION: VISION_REFUSED}

    def test_a_role_on_an_unusable_target_and_an_auto_role_with_no_room_are_both_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # The target refusal comes first and the auto refusal only on the plan after it.
        spec = (
            '{"chat": {"devices": [0]}, "vision": {"target": "auto"}, "embed": {"target": "npu"}}'
        )
        engine = _host(monkeypatch, tmp_path, (24,), spec)
        plan = planning.plan_all_launches(engines=engines_for_all_roles(engine))
        assert [launch.role.value for launch in plan.launches] == ["chat"]
        assert set(plan.refused_roles) == {WorkerRole.EMBED, WorkerRole.VISION}
        assert plan.refused_roles[WorkerRole.VISION] == VISION_REFUSED

    def test_an_auto_role_that_fits_beside_a_named_role_is_not_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        spec = (
            '{"chat": {"devices": [0]}, "vision": {"target": "auto"}, "embed": {"target": "auto"}}'
        )
        engine = _host(monkeypatch, tmp_path, (48,), spec)
        plan = planning.plan_all_launches(engines=engines_for_all_roles(engine))
        assert sorted(launch.role.value for launch in plan.launches) == ["chat", "embed", "vision"]
        assert plan.refused_roles == {}

    def test_auto_roles_beside_only_a_cpu_role_keep_their_swap_group(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # No role is named on a card, so the planner's own answer stands.
        spec = (
            '{"chat": {"target": "auto"}, "vision": {"target": "auto"}, "embed": {"target": "cpu"}}'
        )
        engine = _host(monkeypatch, tmp_path, (24,), spec)
        plan = planning.plan_all_launches(engines=engines_for_all_roles(engine))
        assert plan.co_tenants == frozenset({WorkerRole.CHAT, WorkerRole.VISION})
        assert plan.refused_roles == {}

    def test_an_oversize_auto_role_beside_only_a_cpu_role_stays_tight(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        spec = (
            '{"chat": {"target": "auto"}, "vision": {"target": "cpu"}, "embed": {"target": "cpu"}}'
        )
        engine = _host(monkeypatch, tmp_path, (16,), spec)
        resolved = planning.resolve_placement_plan(
            PlacementSpec.from_json(spec),
            engines=engines_for_all_roles(engine),
            fall_back_to_auto=True,
        )
        assert set(resolved.tight_roles) == {WorkerRole.CHAT}
        assert resolved.refused_roles == {}

    def test_a_mixed_plan_beside_no_named_role_keeps_the_planners_refusal(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        spec = (
            '{"chat": {"target": "auto"}, "vision": {"target": "cpu"}, "embed": {"target": "cpu"}}'
        )
        engine = _host(monkeypatch, tmp_path, (16,), spec)
        other = CardEngine("engine-b", tmp_path / "other", ROLES, cards_of((16,)), spans=True)
        engines = {**engines_for_all_roles(engine), WorkerRole.EMBED: other}
        resolved = planning.resolve_placement_plan(
            PlacementSpec.from_json(spec), engines=engines, fall_back_to_auto=True
        )
        assert resolved.unplaceable_roles == (WorkerRole.CHAT,)
        assert resolved.tight_roles == {}
        assert resolved.refused_roles == {}

    def test_a_role_no_pin_reaches_keeps_the_planners_answer_beside_a_named_role(self) -> None:
        placed = Placement(instances=(), unplaceable_roles=(WorkerRole.VISION, WorkerRole.EMBED))
        inputs = [
            ModelPlacementInput(WorkerRole.VISION, 8 * GIB),
            ModelPlacementInput(WorkerRole.EMBED, 6 * GIB),
        ]
        assert planning._auto_roles_past_named(
            placed, inputs, (WorkerRole.VISION,), frozenset({WorkerRole.CHAT})
        ) == {
            WorkerRole.EMBED: (
                "embed: needs 6.0 GiB, which does not fit in the memory left beside chat. "
                "Use a smaller model, or set its target to cpu."
            )
        }


def _peak(_role: WorkerRole, ratio: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(10 * GIB for _ in ratio)


class TestPlacementFromSpecWithAuto:
    def test_the_planner_gets_the_capacity_the_named_roles_leave(self) -> None:
        spec = PlacementSpec.from_json('{"chat": {"devices": [0]}, "embed": {"target": "auto"}}')
        seen: list[tuple[dict[int, int], frozenset[WorkerRole]]] = []

        def plan_auto(cards: dict[int, int], on_cards: frozenset[WorkerRole]) -> Placement:
            seen.append((cards, on_cards))
            return Placement(
                instances=(InstancePlan(role=WorkerRole.EMBED, devices=(1,)),),
                unplaceable_roles=(),
            )

        placed = placement_from_spec(
            spec,
            (WorkerRole.CHAT, WorkerRole.EMBED),
            {0: 24 * GIB, 1: 24 * GIB},
            estimate_peak=_peak,
            plan_auto=plan_auto,
        )
        # Card 0 keeps (24 * 0.9 - 10) GiB usable, handed back as total bytes (divided by 0.9).
        assert seen == [({0: 13839339065, 1: 24 * GIB}, frozenset({WorkerRole.CHAT}))]
        assert [(p.role.value, p.devices) for p in placed.instances] == [
            ("chat", (0,)),
            ("embed", (1,)),
        ]

    def test_an_auto_role_with_no_planner_is_refused_by_name(self) -> None:
        spec = PlacementSpec.from_json('{"embed": {"target": "auto"}}')
        with pytest.raises(PlacementError, match="embed: the auto target needs a planner"):
            placement_from_spec(spec, (WorkerRole.EMBED,), {0: GIB}, estimate_peak=_peak)

    def test_a_unified_budget_loses_the_bytes_of_the_roles_on_cards(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: dict[str, object] = {}

        def fake_plan(models: list[ModelPlacementInput], cards: object, **kw: object) -> Placement:
            captured.update(models=[m.role for m in models], budget=kw["unified_budget"])
            return Placement(instances=(), unplaceable_roles=())

        monkeypatch.setattr(planning, "plan_placement", fake_plan)
        monkeypatch.setattr(planning, "_peak_estimator", lambda refs, engines: _peak)
        spec = PlacementSpec.from_json('{"chat": {"devices": [0]}, "embed": {"target": "auto"}}')
        inputs = [
            ModelPlacementInput(WorkerRole.CHAT, 30 * GIB),
            ModelPlacementInput(WorkerRole.EMBED, 6 * GIB),
        ]
        engine = CardEngine("e", tmp_path / "e", ROLES, cards_of((24,)), spans=True)
        planning._resolve_placement(
            spec,
            inputs,
            {WorkerRole.CHAT: "chat.gguf", WorkerRole.EMBED: "embed.gguf"},
            cards_of((24,)),
            engines=engines_for_all_roles(engine),
            unified_budget=40 * GIB,
        )
        assert captured == {"models": [WorkerRole.EMBED], "budget": 10 * GIB}


def test_the_auto_target_reads_back_from_json() -> None:
    spec = PlacementSpec.from_json(SPEC)
    assert spec.roles[WorkerRole.VISION].target is RoleTarget.AUTO
