"""A named role too big for its card is refused alone; a cpu role is charged its single pool."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lilbee.app import placement as app_placement
from lilbee.app.engine_choice import BookReading
from lilbee.core.config import cfg
from lilbee.providers.fleet import planning
from lilbee.providers.fleet.memory_types import MemoryEstimate, MemoryKind
from lilbee.providers.fleet.placement import InstancePlan, placement_from_spec
from lilbee.providers.fleet.placement_spec import (
    PlacementError,
    PlacementSpec,
    TargetRefusedError,
)
from lilbee.providers.roles import WorkerRole
from tests._fleet_engines import CardEngine, Role, cards_of, engines_for_all_roles, prepare_host

GIB = 1 << 30
TOO_BIG_ROLES = (Role(WorkerRole.CHAT, 30), Role(WorkerRole.EMBED, 6))
TOO_BIG = (
    "chat pinned to device 0 needs 30.0 GiB but device 0 has 21.6 GiB usable "
    "(24.0 GiB total, 90% headroom). Use a smaller model, or set its target to cpu."
)
CHAT_NAMED_EMBED_AUTO = '{"chat": {"devices": [0]}, "embed": {"target": "auto"}}'
CHAT_NAMED_EMBED_CPU = '{"chat": {"devices": [0]}, "embed": {"target": "cpu"}}'
NO_TARGET = '{"chat": {"devices": [0]}, "embed": {"devices": [0]}}'
ALL_GPU_TARGET = (
    '{"chat": {"target": "gpu", "devices": [0]}, "embed": {"target": "gpu", "devices": [0]}}'
)


def _engine(tmp_path: Path) -> CardEngine:
    return CardEngine(
        "engine-a", tmp_path / "engine-server", TOO_BIG_ROLES, cards_of((24,)), spans=True
    )


def _plan(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, spec: str):
    prepare_host(monkeypatch, tmp_path, TOO_BIG_ROLES, 64 * GIB)
    monkeypatch.setattr(cfg, "placement", spec)
    return planning.plan_all_launches(engines=engines_for_all_roles(_engine(tmp_path)))


class TestARoleTooBigForItsCardIsRefusedAlone:
    @pytest.mark.parametrize("spec", [CHAT_NAMED_EMBED_AUTO, CHAT_NAMED_EMBED_CPU])
    def test_the_launch_drops_the_role_and_starts_the_others(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, spec: str
    ) -> None:
        plan = _plan(monkeypatch, tmp_path, spec)
        assert [launch.role for launch in plan.launches] == [WorkerRole.EMBED]
        assert plan.refused_roles == {WorkerRole.CHAT: TOO_BIG}

    @pytest.mark.parametrize("spec", [NO_TARGET, ALL_GPU_TARGET])
    def test_a_spec_with_every_role_on_a_card_keeps_its_fallback_to_auto(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, spec: str
    ) -> None:
        plan = _plan(monkeypatch, tmp_path, spec)
        assert "pinned to device" not in plan.refused_roles.get(WorkerRole.CHAT, "")
        assert WorkerRole.EMBED in {launch.role for launch in plan.launches}

    def test_the_saved_view_names_the_role_and_shows_what_runs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        prepare_host(monkeypatch, tmp_path, TOO_BIG_ROLES, 64 * GIB)
        monkeypatch.setattr(cfg, "placement", CHAT_NAMED_EMBED_AUTO)
        reading = BookReading(adapters=engines_for_all_roles(_engine(tmp_path)), notices={})
        monkeypatch.setattr(app_placement, "_planning_engines", lambda: reading)
        view = app_placement.get_placement()
        assert view.refusal == TOO_BIG
        assert [row.role for row in view.roles] == [WorkerRole.EMBED]

    def test_an_interactive_set_raises_the_reason_and_writes_nothing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        prepare_host(monkeypatch, tmp_path, TOO_BIG_ROLES, 64 * GIB)
        reading = BookReading(adapters=engines_for_all_roles(_engine(tmp_path)), notices={})
        monkeypatch.setattr(app_placement, "_planning_engines", lambda: reading)
        written = MagicMock()
        monkeypatch.setattr(app_placement.settings, "update_values", written)
        spec = PlacementSpec.from_json(CHAT_NAMED_EMBED_AUTO)
        with pytest.raises(TargetRefusedError) as caught:
            app_placement.set_placement(spec)
        assert str(caught.value) == TOO_BIG
        written.assert_not_called()

    def test_an_estimate_with_no_share_for_each_card_refuses_the_role_alone(self) -> None:
        spec = PlacementSpec.from_json(
            '{"chat": {"devices": [0, 1]}, "embed": {"target": "cpu"}}',
        )
        with pytest.raises(TargetRefusedError, match="chat is pinned to 2 device") as caught:
            placement_from_spec(
                spec,
                (WorkerRole.CHAT, WorkerRole.EMBED),
                {0: 24 * GIB, 1: 24 * GIB},
                estimate_peak=lambda _role, _ratio: (GIB,),
            )
        assert list(caught.value.reasons) == [WorkerRole.CHAT]

    def test_with_every_role_on_a_card_the_same_failure_is_a_plain_error(self) -> None:
        spec = PlacementSpec.from_json('{"chat": {"devices": [0, 1]}}')
        with pytest.raises(PlacementError, match="chat is pinned to 2 device") as caught:
            placement_from_spec(
                spec,
                (WorkerRole.CHAT,),
                {0: 24 * GIB, 1: 24 * GIB},
                estimate_peak=lambda _role, _ratio: (GIB,),
            )
        assert not isinstance(caught.value, TargetRefusedError)


class _MappedEngine(CardEngine):
    """Reports the shape measured on a mapped file at zero layers: a phantom card, a small pool."""

    def estimate_memory(self, request) -> MemoryEstimate:
        if request.gpu_layers != 0:
            return super().estimate_memory(request)
        return MemoryEstimate(
            kind=MemoryKind.SIZED,
            device_bytes=6 * GIB,
            host_bytes=1 * GIB,
            shared_bytes=self.pool_bytes,
        )

    pool_bytes = 2 * GIB


class TestACpuRoleIsChargedItsSinglePool:
    ROLES = (Role(WorkerRole.CHAT, 10), Role(WorkerRole.EMBED, 6))

    def _plan(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, total: int, pool: int):
        prepare_host(monkeypatch, tmp_path, self.ROLES, total * GIB)
        monkeypatch.setattr(cfg, "placement", CHAT_NAMED_EMBED_CPU)
        engine = _MappedEngine(
            "engine-a", tmp_path / "engine-server", self.ROLES, cards_of((24,)), spans=True
        )
        engine.pool_bytes = pool * GIB
        return planning.plan_all_launches(engines=engines_for_all_roles(engine))

    def test_a_mapped_model_that_fits_is_not_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        plan = self._plan(monkeypatch, tmp_path, total=5, pool=2)
        assert {launch.role for launch in plan.launches} == {WorkerRole.CHAT, WorkerRole.EMBED}
        assert plan.refused_roles == {}

    def test_a_role_past_the_bound_is_still_refused_with_no_fallback(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        plan = self._plan(monkeypatch, tmp_path, total=5, pool=6)
        assert [launch.role for launch in plan.launches] == [WorkerRole.CHAT]
        assert plan.refused_roles[WorkerRole.EMBED].startswith(
            "embed: the CPU needs 6.0 GiB of system memory, 5.0 GiB of 5.0 GiB is left."
        )

    def test_the_bound_never_reads_live_free_memory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        prepare_host(monkeypatch, tmp_path, self.ROLES, 5 * GIB)
        monkeypatch.setattr(planning, "free_system_memory", lambda: pytest.fail("live free read"))
        spec = PlacementSpec.from_json(CHAT_NAMED_EMBED_CPU)
        inputs = [planning.ModelPlacementInput(WorkerRole.EMBED, 0, est_ram_bytes=2 * GIB)]
        launching = [InstancePlan(role=WorkerRole.EMBED, devices=())]
        assert planning._cpu_roles_past_memory(spec, inputs, launching, unified=False) == {}

    def test_a_chat_role_is_not_scaled_by_the_card_serve_budget(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        roles = (Role(WorkerRole.CHAT, 3),)
        prepare_host(monkeypatch, tmp_path, roles, 64 * GIB)
        monkeypatch.setattr(cfg, "gpu_memory_fraction", 0.45)
        engine = _MappedEngine("engine-a", tmp_path / "engine-server", roles, cards_of((24,)))
        sized = planning._estimate_cpu_role(
            WorkerRole.CHAT, cfg.chat_model, engine=engine, skipped_not_installed={}
        )
        assert sized is not None
        assert sized.est_vram_bytes == 0
        assert sized.est_ram_bytes == 2 * GIB

    def test_a_chat_model_loaded_without_a_map_is_also_charged_its_weights(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        roles = (Role(WorkerRole.CHAT, 3),)
        prepare_host(monkeypatch, tmp_path, roles, 64 * GIB)
        engine = _MappedEngine("engine-a", tmp_path / "engine-server", roles, cards_of((24,)))
        monkeypatch.setattr(planning, "_host_bytes_must_be_resident", lambda _r, _ref: True)
        monkeypatch.setattr(planning, "_role_weights_bytes", lambda _r, _ref: 7 * GIB)
        sized = planning._estimate_cpu_role(
            WorkerRole.CHAT, cfg.chat_model, engine=engine, skipped_not_installed={}
        )
        assert sized is not None
        assert sized.est_ram_bytes == 9 * GIB
