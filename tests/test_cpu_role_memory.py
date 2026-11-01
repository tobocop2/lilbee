"""A CPU role is charged to system memory and not to a card, and every surface refuses alike."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock

import pytest
from litestar import Litestar
from litestar.testing import TestClient
from typer.testing import CliRunner

from lilbee.app import placement as app_placement
from lilbee.app import services as services_mod
from lilbee.app.engine_choice import BookReading
from lilbee.app.services import build_services
from lilbee.cli.commands import setup as setup_cmd
from lilbee.cli.placement import placement_app
from lilbee.core.config import cfg
from lilbee.providers.fleet import planning
from lilbee.providers.fleet import provider as fleet_provider
from lilbee.providers.fleet.devices import FleetDevice
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.memory_types import MemoryEstimate, MemoryKind
from lilbee.providers.fleet.placement import InstancePlan, ModelPlacementInput, Placement
from lilbee.providers.fleet.placement_spec import (
    PlacementError,
    PlacementSpec,
    TargetRefusedError,
)
from lilbee.providers.fleet.readback import check_launch
from lilbee.providers.roles import RoleTarget, WorkerRole
from lilbee.server import auth as auth_mod
from lilbee.server.chat_completions_api.routes import completions_router
from lilbee.server.routes.placement import placement_preview_route, placement_route
from tests._fleet_engines import (
    CardEngine,
    Role,
    cards_of,
    engines_for_all_roles,
    prepare_host,
)
from tests._litestar_test_client import create_test_client
from tests.conformance.fake_engine import FakeEngine
from tests.test_engine_real_fleet import GGUF_CHAT, GGUF_EMBED, _Gguf, _StubSwap

GIB = 1 << 30
CPU_EMBED = '{"chat": {"devices": [0]}, "embed": {"target": "cpu"}}'
GPU_EMBED = '{"chat": {"devices": [0]}, "embed": {"devices": [0]}}'
TOO_BIG = (
    "embed: the CPU needs 6.0 GiB of system memory, 5.0 GiB of 5.0 GiB is left. "
    "Choose a smaller model or a GPU."
)
LOST_CARD = (
    "chat: GPU 3 is not on this machine (1 found). "
    "Set the chat devices to a GPU that exists, or set its target to cpu."
)
ROLES = (Role(WorkerRole.CHAT, 10), Role(WorkerRole.EMBED, 6))


def _plan(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, spec: str, total_gib: int):
    prepare_host(monkeypatch, tmp_path, ROLES, total_gib * GIB)
    monkeypatch.setattr(cfg, "placement", spec)
    engine = CardEngine("engine-a", tmp_path / "engine-server", ROLES, cards_of((24,)), spans=True)
    return planning.plan_all_launches(engines=engines_for_all_roles(engine)), engine


def _instances(inputs: list[ModelPlacementInput]) -> list[InstancePlan]:
    return [InstancePlan(role=model.role, devices=()) for model in inputs]


def _charges(plan) -> dict[WorkerRole, int]:
    return {launch.role: launch.est_vram_bytes for launch in plan.launches}


class TestTheChargeGoesToSystemMemory:
    """Rows cpu_role_charged_to_system_memory and cpu_role_not_charged_to_card."""

    def test_a_cpu_embed_beside_chat_leaves_the_card_to_chat(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with_gpu, _ = _plan(monkeypatch, tmp_path, GPU_EMBED, 64)
        with_cpu, _ = _plan(monkeypatch, tmp_path, CPU_EMBED, 64)
        assert _charges(with_gpu) == {WorkerRole.CHAT: 10 * GIB, WorkerRole.EMBED: 6 * GIB}
        assert _charges(with_cpu) == {WorkerRole.CHAT: 10 * GIB, WorkerRole.EMBED: 0}
        assert sum(_charges(with_gpu).values()) == 16 * GIB
        assert sum(_charges(with_cpu).values()) == 10 * GIB

    def test_the_cpu_role_is_sized_with_no_layer_on_a_card_and_held_by_the_host(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[int | None] = []

        def sized(role, ref, *, engine, gpu_layers=None, **_kw):
            seen.append(gpu_layers)
            return ModelPlacementInput(role, 3 * GIB, est_ram_bytes=1 * GIB)

        monkeypatch.setattr(planning, "_estimate_role", sized)
        monkeypatch.setattr(cfg, "embedding_model", "org/repo/embed.gguf")
        monkeypatch.setattr(cfg, "reranker_model", "")
        monkeypatch.setattr(cfg, "vision_model", "")
        monkeypatch.setattr(cfg, "chat_model", "")
        inputs, refs, _, _ = planning._server_model_inputs(
            engines={WorkerRole.EMBED: MagicMock()}, cpu_roles=frozenset({WorkerRole.EMBED})
        )
        assert seen == [0]
        assert inputs == [ModelPlacementInput(WorkerRole.EMBED, 0, est_ram_bytes=3 * GIB)]
        assert refs == {WorkerRole.EMBED: "org/repo/embed.gguf"}

    def test_a_cpu_role_past_system_memory_is_refused_and_chat_still_launches(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        plan, _ = _plan(monkeypatch, tmp_path, CPU_EMBED, 5)
        assert [launch.role for launch in plan.launches] == [WorkerRole.CHAT]
        assert plan.refused_roles == {WorkerRole.EMBED: TOO_BIG}

    def test_roles_are_taken_in_order_and_the_first_past_the_total_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(planning, "total_system_memory", lambda: 8 * GIB)
        spec = PlacementSpec.from_json(
            '{"chat": {"devices": [0]}, "embed": {"target": "cpu"}, "rerank": {"target": "cpu"}}'
        )
        inputs = [
            ModelPlacementInput(WorkerRole.CHAT, 1 * GIB, est_ram_bytes=3 * GIB),
            ModelPlacementInput(WorkerRole.EMBED, 0, est_ram_bytes=4 * GIB),
            ModelPlacementInput(WorkerRole.RERANK, 0, est_ram_bytes=2 * GIB),
        ]
        refused = planning._cpu_roles_past_memory(spec, inputs, _instances(inputs), unified=False)
        assert list(refused) == [WorkerRole.RERANK]
        assert refused[WorkerRole.RERANK].startswith(
            "rerank: the CPU needs 2.0 GiB of system memory, 1.0 GiB of 8.0 GiB is left."
        )

    def test_replicas_multiply_the_charge(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(planning, "total_system_memory", lambda: 8 * GIB)
        spec = PlacementSpec.from_json('{"embed": {"target": "cpu", "replicas": 3}}')
        inputs = [ModelPlacementInput(WorkerRole.EMBED, 0, est_ram_bytes=3 * GIB)]
        refused = planning._cpu_roles_past_memory(spec, inputs, _instances(inputs), unified=False)
        assert list(refused) == [WorkerRole.EMBED]

    def test_a_cpu_model_that_is_not_installed_is_skipped_and_recorded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lilbee.providers.base import ProviderError, ProviderErrorKind

        def missing(*_a, **_k):
            raise ProviderError("gone", kind=ProviderErrorKind.NOT_FOUND)

        monkeypatch.setattr(planning, "_estimate_role", missing)
        skipped: dict[WorkerRole, str] = {}
        sized = planning._estimate_cpu_role(
            WorkerRole.EMBED, "org/embed.gguf", engine=MagicMock(), skipped_not_installed=skipped
        )
        assert sized is None
        assert skipped == {WorkerRole.EMBED: "org/embed.gguf"}

    def test_a_cpu_model_the_estimator_cannot_size_is_charged_its_weights(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def unreadable(*_a, **_k):
            raise OSError("no header")

        monkeypatch.setattr(planning, "_estimate_role", unreadable)
        monkeypatch.setattr(planning, "_role_weights_bytes", lambda _r, _ref: 2 * GIB)
        sized = planning._estimate_cpu_role(
            WorkerRole.EMBED, "org/embed.gguf", engine=MagicMock(), skipped_not_installed={}
        )
        assert sized == ModelPlacementInput(WorkerRole.EMBED, 0, est_ram_bytes=2 * GIB)

    def test_a_card_readback_is_skipped_for_a_cpu_launch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        plan, _ = _plan(monkeypatch, tmp_path, CPU_EMBED, 64)
        (embed,) = [launch for launch in plan.launches if launch.role is WorkerRole.EMBED]
        assert embed.est_vram_bytes == 0
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        warning = check_launch(
            log_dir,
            "model",
            embed.role,
            embed.model,
            embed.est_vram_bytes,
            embed.est_vram_by_device,
            embed.est_unreported_bytes,
            embed.est_host_bytes,
        )
        assert warning is None


@pytest.fixture
def cpu_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> CardEngine:
    prepare_host(monkeypatch, tmp_path, ROLES, 5 * GIB)
    engine = CardEngine("engine-a", tmp_path / "engine-server", ROLES, cards_of((24,)), spans=True)
    reading = BookReading(adapters=engines_for_all_roles(engine), notices={})
    monkeypatch.setattr(app_placement, "_planning_engines", lambda: reading)
    return engine


ALL_CPU = (
    '{"chat": {"target": "cpu"}, "embed": {"target": "cpu"},'
    ' "vision": {"target": "cpu"}, "rerank": {"target": "cpu"}}'
)
HUGE = (
    Role(WorkerRole.CHAT, 40),
    Role(WorkerRole.EMBED, 40),
    Role(WorkerRole.VISION, 40),
    Role(WorkerRole.RERANK, 40),
)
PAST_HUGE = (
    "the CPU needs 40.0 GiB of system memory, 24.0 GiB of 64.0 GiB is left. "
    "Choose a smaller model or a GPU."
)
FLOOR_REASON = "a grounded answer needs about"
# A grounded prompt larger than the 4096 tokens that the test models train on.
PAST_THE_TRAINED_WINDOW = 5000
NAMED_CHAT_CPU_EMBED = '{"chat": {"devices": [0]}, "embed": {"target": "cpu"}}'
AUTO_CHAT_CPU_EMBED = '{"chat": {"target": "auto"}, "embed": {"target": "cpu"}}'
UNIFIED_PAST = (
    "embed: the CPU needs 20.0 GiB of system memory, 12.0 GiB of 32.0 GiB is left. "
    "Choose a smaller model or a GPU."
)


def _unified_card(gib: int) -> list[FleetDevice]:
    return [FleetDevice("Metal", 0, "Apple", gib * GIB, gib * GIB, unified=True)]


def _host(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    roles: tuple[Role, ...],
    cards: list[FleetDevice],
    total_gib: int,
    spec: str,
    host_gib: float = 0,
) -> CardEngine:
    prepare_host(monkeypatch, tmp_path, roles, total_gib * GIB)
    monkeypatch.setattr(cfg, "placement", spec)
    return CardEngine(
        "engine-a", tmp_path / "engine-server", roles, cards, spans=True, host_gib=host_gib
    )


def _launched(plan) -> list[WorkerRole]:
    return [launch.role for launch in plan.launches]


def test_a_cpu_chat_past_the_card_launches_in_system_memory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine = _host(monkeypatch, tmp_path, HUGE, cards_of((24,)), 64, ALL_CPU)
    plan = planning.plan_all_launches(engines=engines_for_all_roles(engine))
    assert WorkerRole.CHAT in _launched(plan)
    assert WorkerRole.CHAT not in plan.refused_roles


def test_a_fixed_reserve_past_its_card_is_refused_with_both_figures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    roles = (Role(WorkerRole.CHAT, 40), Role(WorkerRole.EMBED, 6))
    engine = _host(monkeypatch, tmp_path, roles, cards_of((24,)), 64, "")
    monkeypatch.setattr(type(engine), "memory_kind", MemoryKind.FIXED_RESERVE)

    def reserve(_self, request) -> MemoryEstimate:
        size = 40 * GIB if request.model_path.name == "chat.gguf" else 6 * GIB
        return MemoryEstimate(
            kind=MemoryKind.FIXED_RESERVE,
            device_bytes=size,
            host_bytes=2 * GIB,
            shared_bytes=size,
        )

    monkeypatch.setattr(type(engine), "estimate_memory", reserve)
    plan = planning.plan_all_launches(engines=engines_for_all_roles(engine))
    assert _launched(plan) == [WorkerRole.EMBED]
    assert plan.refused_roles == {
        WorkerRole.CHAT: "The chat model org/repo/chat.gguf needs 40.0 GiB but device 0 has "
        "15.6 GiB usable (24.0 GiB total, 90% headroom). "
        "Use a smaller model, or set its target to cpu. "
        "The embed model org/repo/embed.gguf (6.0 GiB) holds the rest."
    }


def test_a_sized_chat_past_its_card_keeps_the_window_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine = _host(monkeypatch, tmp_path, HUGE, cards_of((24,)), 64, "")
    monkeypatch.setattr(planning, "_role_ctx", lambda *_args, **_kwargs: 512)
    plan = planning.plan_all_launches(engines=engines_for_all_roles(engine))
    reason = plan.refused_roles[WorkerRole.CHAT]
    assert FLOOR_REASON in reason
    assert "GiB usable" not in reason


class TestARoleThatDoesNotLaunchChargesNothing:
    """A chat below the prompt floor leaves the bound, so the other cpu roles can launch."""

    def test_a_cpu_chat_below_the_floor_leaves_its_memory_to_the_next_role(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        engine = _host(monkeypatch, tmp_path, HUGE, cards_of((24,)), 64, ALL_CPU)
        monkeypatch.setattr(cfg, "chunk_size", PAST_THE_TRAINED_WINDOW)
        plan = planning.plan_all_launches(engines=engines_for_all_roles(engine))
        assert _launched(plan) == [WorkerRole.EMBED]
        assert FLOOR_REASON in plan.refused_roles[WorkerRole.CHAT]
        assert plan.refused_roles[WorkerRole.VISION] == f"vision: {PAST_HUGE}"
        assert plan.refused_roles[WorkerRole.RERANK] == f"rerank: {PAST_HUGE}"

    def test_the_saved_view_shows_the_role_the_launch_starts(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        engine = _host(monkeypatch, tmp_path, HUGE, cards_of((24,)), 64, ALL_CPU)
        monkeypatch.setattr(cfg, "chunk_size", PAST_THE_TRAINED_WINDOW)
        view = planning.resolve_placement_plan(
            PlacementSpec.from_json(ALL_CPU),
            engines=engines_for_all_roles(engine),
            fall_back_to_auto=True,
        )
        assert sorted(role.value for role in view.refused_roles) == ["rerank", "vision"]
        assert {inst.role for inst in view.instances} == {WorkerRole.CHAT, WorkerRole.EMBED}

    def test_an_interactive_set_refuses_the_same_roles(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        engine = _host(monkeypatch, tmp_path, HUGE, cards_of((24,)), 64, ALL_CPU)
        monkeypatch.setattr(cfg, "chunk_size", PAST_THE_TRAINED_WINDOW)
        with pytest.raises(TargetRefusedError, match="vision: the CPU needs") as caught:
            planning.resolve_placement_plan(
                PlacementSpec.from_json(ALL_CPU), engines=engines_for_all_roles(engine)
            )
        assert sorted(role.value for role in caught.value.reasons) == ["rerank", "vision"]

    def test_a_chat_past_the_bound_keeps_the_bound_reason_on_every_surface(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        engine = _host(monkeypatch, tmp_path, HUGE, _unified_card(32), 32, ALL_CPU)
        plan = planning.plan_all_launches(engines=engines_for_all_roles(engine))
        assert plan.refused_roles[WorkerRole.CHAT].startswith(
            "chat: the CPU needs 40.0 GiB of system memory, 32.0 GiB of 32.0 GiB is left."
        )
        with pytest.raises(TargetRefusedError, match=r"chat: the CPU needs 40\.0 GiB"):
            planning.resolve_placement_plan(
                PlacementSpec.from_json(ALL_CPU), engines=engines_for_all_roles(engine)
            )

    def test_a_card_chat_below_the_floor_holds_no_host_bytes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        roles = (Role(WorkerRole.CHAT, 20), Role(WorkerRole.EMBED, 6))
        engine = _host(monkeypatch, tmp_path, roles, cards_of((24,)), 8, NAMED_CHAT_CPU_EMBED)
        monkeypatch.setattr(planning, "_unusable_chat_ctx_reason", lambda launch: "below the floor")
        held = ModelPlacementInput(WorkerRole.CHAT, 20 * GIB, est_ram_bytes=4 * GIB)
        embed = ModelPlacementInput(WorkerRole.EMBED, 0, est_ram_bytes=6 * GIB)
        placed = Placement(
            instances=(
                InstancePlan(role=WorkerRole.CHAT, devices=(0,)),
                InstancePlan(role=WorkerRole.EMBED, devices=(), target=RoleTarget.CPU),
            ),
            unplaceable_roles=(),
        )
        launching = planning._launching_instances(
            placed,
            [held, embed],
            {WorkerRole.CHAT: "org/repo/chat.gguf", WorkerRole.EMBED: "org/repo/embed.gguf"},
            cards_of((24,)),
            engines=engines_for_all_roles(engine),
        )
        assert [inst.role for inst in launching] == [WorkerRole.EMBED]
        spec = PlacementSpec.from_json(NAMED_CHAT_CPU_EMBED)
        assert planning._cpu_roles_past_memory(spec, [held, embed], launching, unified=False) == {}
        charged = planning._cpu_roles_past_memory(
            spec, [held, embed], placed.instances, unified=False
        )
        assert list(charged) == [WorkerRole.EMBED]

    def test_a_spec_with_no_cpu_role_builds_no_launch_for_the_bound(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        roles = (Role(WorkerRole.CHAT, 10), Role(WorkerRole.EMBED, 6))
        engine = _host(monkeypatch, tmp_path, roles, cards_of((24,)), 64, GPU_EMBED)
        monkeypatch.setattr(
            planning, "_launching_instances", lambda *_a, **_k: pytest.fail("floor read")
        )
        planning.resolve_placement_plan(
            PlacementSpec.from_json(GPU_EMBED), engines=engines_for_all_roles(engine)
        )


class TestTheBoundChargesWhatTheCardsCommit:
    """On a unified host a card role holds the memory the CPU reads; on a card it does not."""

    ROLES: ClassVar[tuple[Role, ...]] = (Role(WorkerRole.CHAT, 20), Role(WorkerRole.EMBED, 20))

    def test_a_named_role_on_a_unified_host_leaves_less_for_the_cpu(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        engine = _host(
            monkeypatch, tmp_path, self.ROLES, _unified_card(32), 32, NAMED_CHAT_CPU_EMBED
        )
        plan = planning.plan_all_launches(engines=engines_for_all_roles(engine))
        assert _launched(plan) == [WorkerRole.CHAT]
        assert plan.refused_roles == {WorkerRole.EMBED: UNIFIED_PAST}

    def test_an_auto_role_on_a_unified_host_leaves_less_for_the_cpu(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        engine = _host(
            monkeypatch, tmp_path, self.ROLES, _unified_card(32), 32, AUTO_CHAT_CPU_EMBED
        )
        plan = planning.plan_all_launches(engines=engines_for_all_roles(engine))
        assert _launched(plan) == [WorkerRole.CHAT]
        assert plan.refused_roles == {WorkerRole.EMBED: UNIFIED_PAST}

    @pytest.mark.parametrize("spec", [NAMED_CHAT_CPU_EMBED, AUTO_CHAT_CPU_EMBED])
    def test_a_unified_role_charges_its_host_bytes_once(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, spec: str
    ) -> None:
        roles = (Role(WorkerRole.CHAT, 12), Role(WorkerRole.EMBED, 20))
        engine = _host(monkeypatch, tmp_path, roles, _unified_card(32), 32, spec, host_gib=2)
        plan = planning.plan_all_launches(engines=engines_for_all_roles(engine))
        assert set(_launched(plan)) == {WorkerRole.CHAT, WorkerRole.EMBED}
        assert plan.refused_roles == {}

    def test_a_card_with_its_own_memory_leaves_the_cpu_all_of_it(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        engine = _host(monkeypatch, tmp_path, self.ROLES, cards_of((24,)), 24, NAMED_CHAT_CPU_EMBED)
        plan = planning.plan_all_launches(engines=engines_for_all_roles(engine))
        assert set(_launched(plan)) == {WorkerRole.CHAT, WorkerRole.EMBED}
        assert plan.refused_roles == {}

    def test_a_swap_group_commits_its_largest_member(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(planning, "total_system_memory", lambda: 32 * GIB)
        spec = PlacementSpec.from_json(
            '{"chat": {"target": "auto"}, "vision": {"target": "auto"}, "embed": {"target": "cpu"}}'
        )
        inputs = [
            ModelPlacementInput(WorkerRole.CHAT, 20 * GIB),
            ModelPlacementInput(WorkerRole.VISION, 10 * GIB),
            ModelPlacementInput(WorkerRole.EMBED, 0, est_ram_bytes=12 * GIB),
        ]
        instances = [
            InstancePlan(role=WorkerRole.CHAT, devices=(0,)),
            InstancePlan(role=WorkerRole.VISION, devices=(0,)),
            InstancePlan(role=WorkerRole.EMBED, devices=(), target=RoleTarget.CPU),
        ]
        swapped = frozenset({WorkerRole.CHAT, WorkerRole.VISION})
        assert (
            planning._cpu_roles_past_memory(
                spec, inputs, instances, unified=True, co_tenants=swapped
            )
            == {}
        )
        refused = planning._cpu_roles_past_memory(
            spec, inputs, instances, unified=True, co_tenants=frozenset()
        )
        assert list(refused) == [WorkerRole.EMBED]

    def test_a_planned_swap_group_leaves_the_cpu_all_but_its_largest_member(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        raw = (
            '{"chat": {"target": "auto"}, "vision": {"target": "auto"}, "embed": {"target": "cpu"}}'
        )
        roles = (
            Role(WorkerRole.CHAT, 20),
            Role(WorkerRole.VISION, 16),
            Role(WorkerRole.EMBED, 10),
        )
        engine = _host(monkeypatch, tmp_path, roles, _unified_card(32), 32, raw)
        view = planning.resolve_placement_plan(
            PlacementSpec.from_json(raw), engines=engines_for_all_roles(engine)
        )
        assert view.co_tenants == {WorkerRole.CHAT, WorkerRole.VISION}
        assert WorkerRole.EMBED in {inst.role for inst in view.instances}

    def test_each_replica_of_a_card_role_commits_its_figure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(planning, "total_system_memory", lambda: 32 * GIB)
        spec = PlacementSpec.from_json(NAMED_CHAT_CPU_EMBED)
        inputs = [
            ModelPlacementInput(WorkerRole.CHAT, 12 * GIB, est_ram_bytes=1 * GIB),
            ModelPlacementInput(WorkerRole.EMBED, 0, est_ram_bytes=11 * GIB),
        ]
        chat = InstancePlan(role=WorkerRole.CHAT, devices=(0,))
        embed = InstancePlan(role=WorkerRole.EMBED, devices=(), target=RoleTarget.CPU)
        one = planning._cpu_roles_past_memory(spec, inputs, [chat, embed], unified=True)
        two = planning._cpu_roles_past_memory(spec, inputs, [chat, chat, embed], unified=True)
        assert one == {}
        assert two[WorkerRole.EMBED].startswith(
            "embed: the CPU needs 11.0 GiB of system memory, 8.0 GiB of 32.0 GiB is left."
        )


class TestEveryEntryPointGivesTheSameRefusal:
    """Row entry_points_same_refusal: library, REST, CLI and MCP read one message."""

    SPEC: ClassVar[dict] = {"chat": {"devices": [0]}, "embed": {"target": "cpu"}}

    def test_library(self, cpu_host: CardEngine) -> None:
        spec = PlacementSpec.from_json(json.dumps(self.SPEC))
        with pytest.raises(PlacementError) as caught:
            app_placement.preview_placement(spec)
        assert str(caught.value) == TOO_BIG

    def test_rest(self, cpu_host: CardEngine) -> None:
        with create_test_client([placement_preview_route]) as client:
            response = client.post("/api/placement/preview", json={"spec": self.SPEC})
        assert response.status_code == 422
        assert response.json()["detail"] == TOO_BIG

    def test_cli(self, cpu_host: CardEngine) -> None:
        result = CliRunner().invoke(placement_app, ["preview", "--spec", json.dumps(self.SPEC)])
        assert result.exit_code == 1
        assert " ".join(result.output.split()) == TOO_BIG

    def test_mcp(self, cpu_host: CardEngine) -> None:
        from lilbee import mcp_server

        assert mcp_server.preview_placement_tool(self.SPEC) == {"error": TOO_BIG}

    def test_set_does_not_persist_a_refused_spec(
        self, cpu_host: CardEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        written = MagicMock()
        monkeypatch.setattr(app_placement.settings, "update_values", written)
        with pytest.raises(PlacementError, match="Choose a smaller model or a GPU"):
            app_placement.set_placement(PlacementSpec.from_json(json.dumps(self.SPEC)))
        written.assert_not_called()


class TestASavedRefusedSpecIsReadable:
    """Row placement_view_refused_spec: GET answers with the spec and the refusal."""

    SAVED: ClassVar[str] = '{"chat": {"devices": [0]}, "embed": {"target": "cpu"}}'

    def test_view_carries_the_spec_the_refusal_and_the_roles_that_run(
        self, cpu_host: CardEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cfg, "placement", self.SAVED)
        view = app_placement.get_placement()
        assert view.refusal == TOO_BIG
        assert view.rejected_spec_json == self.SAVED
        assert view.manual is False
        assert [g.index for g in view.gpus] == [0]
        # The fleet launches chat alone, so the view shows chat alone and no embed.
        assert [(r.role, r.devices, r.target) for r in view.roles] == [
            (WorkerRole.CHAT, (0,), RoleTarget.GPU)
        ]

    def test_an_unreadable_host_refuses_a_cpu_role_in_plain_words_and_chat_still_runs(
        self, cpu_host: CardEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def unreadable() -> int:
            raise RuntimeError("no memory reading")

        monkeypatch.setattr(planning, "total_system_memory", unreadable)
        monkeypatch.setattr(cfg, "placement", self.SAVED)
        view = app_placement.get_placement()
        assert view.refusal == (
            "embed: the system memory of this machine cannot be read, so the CPU "
            "cannot be sized. Choose a GPU."
        )
        assert [r.role for r in view.roles] == [WorkerRole.CHAT]

    def test_an_unreadable_host_does_not_matter_to_a_spec_with_no_cpu_role(
        self, cpu_host: CardEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def unreadable() -> int:
            raise RuntimeError("no memory reading")

        monkeypatch.setattr(planning, "total_system_memory", unreadable)
        spec = PlacementSpec.from_json('{"chat": {"devices": [0]}, "embed": {"devices": [0]}}')
        assert planning._cpu_roles_past_memory(spec, [], [], unified=False) == {}

    def test_get_placement_and_gpus_are_200_with_the_refusal(
        self, cpu_host: CardEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lilbee.server.routes.placement import gpus_route

        monkeypatch.setattr(cfg, "placement", self.SAVED)
        with create_test_client([placement_route, gpus_route]) as client:
            placed = client.get("/api/placement")
            gpus = client.get("/api/gpus")
        assert placed.status_code == 200
        assert placed.json()["refusal"] == TOO_BIG
        assert placed.json()["rejected_spec_json"] == self.SAVED
        assert gpus.status_code == 200
        assert [g["index"] for g in gpus.json()["gpus"]] == [0]

    def test_a_working_cpu_role_shows_its_target_and_no_card(
        self, cpu_host: CardEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prepare_host(monkeypatch, cpu_host.binary().parent, ROLES, 64 * GIB)
        monkeypatch.setattr(cfg, "placement", self.SAVED)
        view = app_placement.get_placement()
        (embed,) = [r for r in view.roles if r.role is WorkerRole.EMBED]
        assert (embed.target, embed.devices) == (RoleTarget.CPU, ())
        assert view.refusal is None

    def test_the_cli_shows_the_refusal_of_a_saved_spec(
        self, cpu_host: CardEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cfg, "placement", self.SAVED)
        result = CliRunner().invoke(placement_app, ["show"])
        assert result.exit_code == 0
        assert TOO_BIG in " ".join(result.output.split())


class TestTheSelfCheckPrintsTheMessage:
    def test_a_refusal_prints_its_text_and_no_class_name(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def refuse(_path: Path) -> None:
            raise PlacementError(TOO_BIG)

        with pytest.raises(Exception, match="1") as caught:
            setup_cmd._self_check_leg(Path("m.gguf"), "r", "f", "embed", refuse)
        assert type(caught.value).__name__ == "Exit"
        out = " ".join(capsys.readouterr().out.split())
        assert out.endswith(f"SELF-CHECK FAILED: {TOO_BIG}")
        assert "PlacementError" not in out

    def test_another_failure_keeps_its_repr(self, capsys: pytest.CaptureFixture[str]) -> None:
        def broken(_path: Path) -> None:
            raise OSError("disk")

        with pytest.raises(Exception, match="1"):
            setup_cmd._self_check_leg(Path("m.gguf"), "r", "f", "embed", broken)
        assert "OSError('disk')" in capsys.readouterr().out


@pytest.fixture
def lost_card_fleet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[services_mod.Services, FakeEngine]]:
    """The real fleet provider with chat pinned to a card this machine lacks."""
    from lilbee.providers import engine_params

    monkeypatch.delitem(choice.ADAPTERS, EngineName.MLX, raising=False)
    monkeypatch.setitem(choice.ADAPTERS, EngineName.LLAMA, _Gguf())
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x" * 4096)
    monkeypatch.setattr(engine_params, "resolve_model_path", lambda ref: model)
    planning.clear_read_device_cache()
    for module in (planning, planning.model_cache):
        monkeypatch.setattr(module, "total_system_memory", lambda: 64 * GIB)
        monkeypatch.setattr(module, "free_system_memory", lambda: 64 * GIB)
    monkeypatch.setattr(cfg, "system_memory_reserve_gb", 0)
    monkeypatch.setattr(cfg, "gpu_memory_fraction", 1.0)
    monkeypatch.setattr(cfg, "num_ctx", None)
    monkeypatch.setattr(cfg, "placement", '{"chat": {"devices": [3]}, "embed": {"target": "cpu"}}')
    monkeypatch.setattr(cfg, "reranker_model", "")
    monkeypatch.setattr(cfg, "vision_model", "")
    monkeypatch.setattr(cfg, "chat_model", GGUF_CHAT)
    monkeypatch.setattr(cfg, "embedding_model", GGUF_EMBED)
    monkeypatch.setattr(cfg, "data_root", tmp_path)
    monkeypatch.setattr(cfg, "documents_dir", tmp_path / "documents")
    monkeypatch.setattr(cfg, "data_dir", tmp_path / "data")
    monkeypatch.setattr(cfg, "lancedb_dir", tmp_path / "data" / "lancedb")
    monkeypatch.setattr(cfg, "worker_pool_eager_start", False)
    cfg.documents_dir.mkdir(parents=True, exist_ok=True)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(fleet_provider, "SwapManager", _StubSwap)
    _StubSwap.started = []
    with FakeEngine() as engine:
        _StubSwap.url = engine.url
        services = build_services(cfg)
        monkeypatch.setattr(services_mod._state, "singleton", services)
        yield services, engine
        services.store.close()


class TestAChatWithChatRefusedAnswersAReadable4xx:
    """The route is reached with the real fleet provider and a chat model that is installed."""

    def test_v1_chat_completions_is_a_400_that_carries_the_card_message(
        self, lost_card_fleet: tuple[services_mod.Services, FakeEngine], monkeypatch
    ) -> None:
        services, _engine = lost_card_fleet
        manifest = SimpleNamespace(
            ref=GGUF_CHAT, task="chat", downloaded_at="2026-05-15T00:00:00+00:00"
        )
        monkeypatch.setattr(services.registry, "list_installed", lambda *_a, **_k: [manifest])
        app = Litestar(route_handlers=[completions_router], logging_config=None)
        monkeypatch.setattr(auth_mod.session_manager, "token", "test-token-" + "x" * 40)
        monkeypatch.setattr(auth_mod.session_manager, "_initialized", True)
        headers = {"Authorization": f"Bearer {auth_mod.session_manager.token}"}
        body = {"model": GGUF_CHAT, "messages": [{"role": "user", "content": "hi"}]}
        with TestClient(app) as client:
            response = client.post("/v1/chat/completions", json=body, headers=headers)
        assert (response.status_code, response.json()) == (
            400,
            {
                "error": {
                    "message": LOST_CARD,
                    "type": "invalid_request_error",
                    "code": "invalid_request",
                }
            },
        )
        assert {role for batch in _StubSwap.started for role, _m in batch} == {WorkerRole.EMBED}
