"""An auto plan never gives a role a target that its engine build cannot run."""

from __future__ import annotations

from pathlib import Path

import pytest

from lilbee.app import placement as app_placement
from lilbee.app.engine_choice import BookReading
from lilbee.core.config import cfg
from lilbee.providers.fleet import planning
from lilbee.providers.fleet.engines.base import LaunchRequest
from lilbee.providers.fleet.engines.ovms import OVMS
from lilbee.providers.fleet.placement_spec import PlacementSpec, TargetRefusedError
from lilbee.providers.roles import EngineBackend, RoleTarget, WorkerRole
from tests._fleet_engines import (
    CardEngine,
    Role,
    cards_of,
    engines_for_all_roles,
    model_file,
    prepare_host,
)

GIB = 1 << 30
ROLES = (Role(WorkerRole.CHAT, 4), Role(WorkerRole.EMBED, 1))


class TargetsEngine(CardEngine):
    """An engine whose installed build lists a stated set of targets and counts the asks."""

    def __init__(self, *args, targets: frozenset[RoleTarget], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._targets = targets
        self.asked = 0
        self.requested: list[RoleTarget] = []

    def device_targets(self, binary: Path) -> frozenset[RoleTarget]:
        self.asked += 1
        return self._targets

    def launch_argv(self, spec, request) -> list[str]:
        self.requested.append(request.target)
        return super().launch_argv(spec, request)


CPU_ONLY = frozenset({RoleTarget.CPU})
BOTH = frozenset({RoleTarget.CPU, RoleTarget.GPU})


def _engine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cards: tuple[int, ...],
    targets: frozenset[RoleTarget],
) -> TargetsEngine:
    prepare_host(monkeypatch, tmp_path, ROLES, 64 * GIB)
    engine = TargetsEngine(
        "engine-a", tmp_path / "engine-server", ROLES, cards_of(cards), targets=targets
    )
    reading = BookReading(adapters=engines_for_all_roles(engine), notices={})
    monkeypatch.setattr(app_placement, "_planning_engines", lambda: reading)
    monkeypatch.setattr(cfg, "placement", "")
    return engine


def _targets(engine: TargetsEngine, spec: PlacementSpec | None = None) -> dict[str, str]:
    resolved = planning.resolve_placement_plan(spec, engines=engines_for_all_roles(engine))
    return {p.role.value: p.target.value for p in resolved.instances}


def test_auto_plan_without_gpu_target_runs_on_cpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine = _engine(monkeypatch, tmp_path, (), CPU_ONLY)
    assert _targets(engine) == {"chat": "cpu", "embed": "cpu"}


def test_auto_plan_with_card_keeps_gpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    engine = _engine(monkeypatch, tmp_path, (24,), BOTH)
    assert _targets(engine) == {"chat": "gpu", "embed": "gpu"}


def test_auto_plan_with_gpu_target_and_no_card_is_not_rewritten(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine = _engine(monkeypatch, tmp_path, (), BOTH)
    assert set(_targets(engine).values()) == {"gpu"}


def test_npu_only_build_keeps_its_plan(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    engine = _engine(monkeypatch, tmp_path, (), frozenset({RoleTarget.NPU}))
    assert set(_targets(engine).values()) == {"gpu"}


def test_engine_is_asked_once_per_plan(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    engine = _engine(monkeypatch, tmp_path, (), CPU_ONLY)
    _targets(engine)
    assert engine.asked == 1


def test_auto_entry_in_a_spec_runs_on_cpu_without_gpu_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine = _engine(monkeypatch, tmp_path, (), CPU_ONLY)
    spec = PlacementSpec.from_json('{"chat": {"target": "auto"}, "embed": {"target": "cpu"}}')
    assert _targets(engine, spec) == {"chat": "cpu", "embed": "cpu"}


def test_named_gpu_target_on_a_build_without_gpu_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine = _engine(monkeypatch, tmp_path, (24,), CPU_ONLY)
    spec = PlacementSpec.from_json('{"chat": {"devices": [0], "target": "gpu"}}')
    with pytest.raises(TargetRefusedError):
        _targets(engine, spec)


def test_pinned_device_is_not_rewritten(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    engine = _engine(monkeypatch, tmp_path, (24,), CPU_ONLY)
    spec = PlacementSpec.from_json('{"chat": {"devices": [0]}, "embed": {"target": "cpu"}}')
    assert _targets(engine, spec)["chat"] == "gpu"


def test_engine_is_asked_once_when_a_spec_falls_back_to_auto(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine = _engine(monkeypatch, tmp_path, (), CPU_ONLY)
    spec = PlacementSpec.from_json('{"chat": {"devices": [3]}, "embed": {"devices": [3]}}')
    planning.resolve_placement_plan(
        spec, engines=engines_for_all_roles(engine), fall_back_to_auto=True
    )
    assert engine.asked == 1


def test_engine_is_asked_once_for_a_named_target_and_the_auto_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine = _engine(monkeypatch, tmp_path, (), CPU_ONLY)
    spec = PlacementSpec.from_json('{"chat": {"target": "cpu"}, "embed": {"target": "auto"}}')
    planning.resolve_placement_plan(spec, engines=engines_for_all_roles(engine))
    assert engine.asked == 1


def _self_check_target(engine: TargetsEngine, tmp_path: Path) -> RoleTarget:
    planning.build_single_role_launch(
        WorkerRole.EMBED, tmp_path / model_file(WorkerRole.EMBED), engine=engine
    )
    assert len(engine.requested) == 1
    return engine.requested[0]


def test_self_check_without_gpu_target_launches_on_cpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine = _engine(monkeypatch, tmp_path, (), CPU_ONLY)
    assert _self_check_target(engine, tmp_path) is RoleTarget.CPU


def test_self_check_with_card_launches_on_gpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine = _engine(monkeypatch, tmp_path, (24,), BOTH)
    assert _self_check_target(engine, tmp_path) is RoleTarget.GPU


def test_self_check_with_a_named_target_keeps_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine = _engine(monkeypatch, tmp_path, (24,), BOTH)
    monkeypatch.setattr(cfg, "placement", '{"embed": {"target": "cpu"}}')
    assert _self_check_target(engine, tmp_path) is RoleTarget.CPU


AUTO_BOTH = '{"chat": {"target": "auto"}, "embed": {"target": "auto"}}'


@pytest.mark.parametrize(
    ("cards", "targets", "expected"),
    [
        pytest.param((), CPU_ONLY, RoleTarget.CPU, id="no-card"),
        pytest.param((24,), BOTH, RoleTarget.GPU, id="one-card"),
    ],
)
def test_self_check_of_an_auto_entry_matches_serving(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cards: tuple[int, ...],
    targets: frozenset[RoleTarget],
    expected: RoleTarget,
) -> None:
    engine = _engine(monkeypatch, tmp_path, cards, targets)
    monkeypatch.setattr(cfg, "placement", AUTO_BOTH)
    serving = _targets(engine, PlacementSpec.from_json(AUTO_BOTH))["embed"]
    assert serving == expected.value
    engine.requested.clear()
    assert _self_check_target(engine, tmp_path) is expected


class TestMovedRoleKeepsItsLowFreeMemoryWarning:
    """A role the auto plan moves to the CPU still warns when the free memory is short."""

    @pytest.mark.parametrize("backend", [EngineBackend.CPU, EngineBackend.CUDA])
    @pytest.mark.parametrize("spec", [None, AUTO_BOTH])
    def test_the_launch_warns_once_that_the_plan_passes_the_free_memory(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        spec: str | None,
        backend: EngineBackend,
    ) -> None:
        small = (Role(WorkerRole.CHAT, 1), Role(WorkerRole.EMBED, 1))
        prepare_host(monkeypatch, tmp_path, small, 64 * GIB)
        monkeypatch.setattr(cfg, "placement", spec)
        monkeypatch.setattr(cfg, "n_gpu_layers", None)
        monkeypatch.setattr(planning, "_plan_free_system_memory", lambda _e: int(1.5 * GIB))
        engine = TargetsEngine(
            "engine-a",
            tmp_path / "engine-server",
            small,
            [],
            targets=CPU_ONLY,
            backend=backend,
            cpu_only=backend is EngineBackend.CPU,
        )
        with caplog.at_level("WARNING"):
            plan = planning.plan_all_launches(engines=engines_for_all_roles(engine))
        assert {launch.role for launch in plan.launches} == {WorkerRole.CHAT, WorkerRole.EMBED}
        warned = [r.getMessage() for r in caplog.records if "is free" in r.getMessage()]
        assert len(warned) == 1
        assert "only 1.5 GiB is free" in warned[0]


def test_ovms_argv_for_auto_plan_without_gpu_names_cpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine = _engine(monkeypatch, tmp_path, (), CPU_ONLY)
    resolved = planning.resolve_placement_plan(None, engines=engines_for_all_roles(engine))
    argv = OVMS.launch_argv(
        OVMS.role_spec(WorkerRole.CHAT, None, None),
        LaunchRequest(
            binary=Path("ovms"),
            model_path=Path("m"),
            devices=(),
            n_gpu_layers=0,
            slots=1,
            ctx_per_slot=1024,
            target=resolved.instances[0].target,
        ),
    )
    assert argv[argv.index("--target_device") + 1] == "CPU"
