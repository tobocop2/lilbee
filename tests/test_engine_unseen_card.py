"""A role is not launched on a card that its own engine's device read does not list."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import pytest

from lilbee.core.config import cfg
from lilbee.providers.fleet import gpu_hardware, planning
from lilbee.providers.fleet.placement_spec import PlacementSpec
from lilbee.providers.roles import WorkerRole
from tests._fleet_engines import CardEngine, Role, cards_of, prepare_host

ROLES = (Role(WorkerRole.CHAT, 1), Role(WorkerRole.EMBED, 1))
CHAT_ON_CARD_ZERO = '"chat": {"devices": [0], "target": "gpu"}'
EMBED_ON_CARD_ONE = "{" + CHAT_ON_CARD_ZERO + ', "embed": {"devices": [1], "target": "gpu"}}'
EMBED_ON_CARD_ZERO = "{" + CHAT_ON_CARD_ZERO + ', "embed": {"devices": [0], "target": "gpu"}}'
REFUSAL = (
    "embed is placed on device 1, but the narrow engine lists only device 0. "
    "Set the embed target to cpu, or choose an engine that lists that card."
)


@pytest.fixture(autouse=True)
def _clean_snapshot() -> Iterator[None]:
    planning.clear_plan_probe()
    yield
    planning.clear_plan_probe()


def _engines(tmp_path: Path, narrow: CardEngine | None = None) -> dict[WorkerRole, CardEngine]:
    """Chat on an engine that reads two small cards, embed on *narrow* (one big card by default).

    The two-card engine reads the least free memory, so it is the plan's snapshot engine.
    """
    wide = CardEngine("wide", tmp_path / "wide-server", ROLES, cards_of((4, 4)))
    narrow = narrow or CardEngine("narrow", tmp_path / "narrow-server", ROLES, cards_of((16,)))
    return {WorkerRole.CHAT: wide, WorkerRole.EMBED: narrow}


def _host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, placement: str | None) -> None:
    prepare_host(monkeypatch, tmp_path, ROLES, 64 << 30)
    monkeypatch.setattr(cfg, "placement", placement)


def test_a_role_on_a_card_its_engine_does_not_list_is_refused_with_the_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _host(tmp_path, monkeypatch, EMBED_ON_CARD_ONE)
    engines = _engines(tmp_path)
    planning.capture_plan_probe(engines)
    plan = planning.plan_all_launches(engines=engines)
    assert [launch.role for launch in plan.launches] == [WorkerRole.CHAT]
    assert plan.refused_roles == {WorkerRole.EMBED: REFUSAL}


def test_a_role_on_a_card_its_engine_lists_is_launched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _host(tmp_path, monkeypatch, EMBED_ON_CARD_ZERO)
    engines = _engines(tmp_path)
    planning.capture_plan_probe(engines)
    plan = planning.plan_all_launches(engines=engines)
    assert {launch.role for launch in plan.launches} == {WorkerRole.CHAT, WorkerRole.EMBED}
    assert plan.refused_roles == {}


def test_the_placement_view_reports_the_same_refusal_as_the_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _host(tmp_path, monkeypatch, EMBED_ON_CARD_ONE)
    engines = _engines(tmp_path)
    planning.capture_plan_probe(engines)
    resolved = planning.resolve_placement_plan(
        PlacementSpec.from_json(EMBED_ON_CARD_ONE), engines=engines
    )
    assert resolved.unlaunched_roles == {WorkerRole.EMBED: REFUSAL}
    assert [inst.role for inst in resolved.instances] == [WorkerRole.CHAT]


def test_a_cpu_only_engine_takes_no_card_and_is_not_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _host(tmp_path, monkeypatch, EMBED_ON_CARD_ONE)
    cpu_build = CardEngine("cpu-build", tmp_path / "cpu-server", ROLES, [], cpu_only=True)
    engines = _engines(tmp_path, cpu_build)
    planning.capture_plan_probe(engines)
    plan = planning.plan_all_launches(engines=engines)
    assert {launch.role for launch in plan.launches} == {WorkerRole.CHAT, WorkerRole.EMBED}
    assert plan.refused_roles == {}


def test_an_engine_that_lists_no_card_keeps_its_placement_on_a_one_card_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _host(tmp_path, monkeypatch, None)
    monkeypatch.setattr(planning, "nvidia_card_rows", lambda: 0)
    monkeypatch.setattr(gpu_hardware, "installed_gpu_counts", Counter)
    wide = CardEngine("wide", tmp_path / "wide-server", ROLES, cards_of((4,)))
    blind = CardEngine("blind", tmp_path / "blind-server", ROLES, [])
    engines = {WorkerRole.CHAT: wide, WorkerRole.EMBED: blind}
    planning.capture_plan_probe(engines)
    plan = planning.plan_all_launches(engines=engines)
    assert {launch.role for launch in plan.launches} == {WorkerRole.CHAT, WorkerRole.EMBED}
    assert plan.refused_roles == {}


def test_a_plan_with_no_snapshot_has_no_engine_read_to_refuse_against(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _host(tmp_path, monkeypatch, EMBED_ON_CARD_ONE)
    engines = _engines(tmp_path)
    devices = cards_of((4, 4))
    plan = planning.plan_launches(None, {d.index: d for d in devices}, devices, engines=engines)
    assert {launch.role for launch in plan.launches} == {WorkerRole.CHAT, WorkerRole.EMBED}
    assert plan.refused_roles == {}
