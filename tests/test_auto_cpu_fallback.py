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
from lilbee.providers.roles import RoleTarget, WorkerRole
from tests._fleet_engines import (
    CardEngine,
    Role,
    cards_of,
    engines_for_all_roles,
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

    def device_targets(self, binary: Path) -> frozenset[RoleTarget]:
        self.asked += 1
        return self._targets


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


def test_named_gpu_target_is_not_rewritten(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    engine = _engine(monkeypatch, tmp_path, (24,), CPU_ONLY)
    spec = PlacementSpec.from_json('{"chat": {"devices": [0], "target": "gpu"}}')
    with pytest.raises(TargetRefusedError):
        _targets(engine, spec)


def test_pinned_device_is_not_rewritten(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    engine = _engine(monkeypatch, tmp_path, (24,), CPU_ONLY)
    spec = PlacementSpec.from_json('{"chat": {"devices": [0]}, "embed": {"target": "cpu"}}')
    assert _targets(engine, spec)["chat"] == "gpu"


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
