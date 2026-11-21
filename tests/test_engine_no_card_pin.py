"""A role on an engine that lists no card is left unpinned; the launch says it runs on the CPU."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import pytest

from lilbee.providers.fleet import gpu_hardware, planning
from lilbee.providers.fleet.devices import FleetDevice
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.placement import InstancePlan, Placement
from lilbee.providers.fleet.placement_spec import RoleTarget
from lilbee.providers.roles import WorkerRole
from tests._fleet_engines import CardEngine, Role, cards_of, prepare_host

CHAT = WorkerRole.CHAT
EMBED = WorkerRole.EMBED
ROLES = (Role(CHAT, 1), Role(EMBED, 1))
NOTICE = (
    "The blind engine lists no GPU, so the embed role runs on the CPU. "
    "Install a GPU build of the engine to run the role on a card."
)
_PLANNING_LOG = "lilbee.providers.fleet.planning"


@pytest.fixture(autouse=True)
def _clean_snapshot() -> Iterator[None]:
    planning.clear_plan_probe()
    yield
    planning.clear_plan_probe()


def _host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    backend: str = "CUDA",
    embed_engine_lists_cards: bool = False,
) -> dict[WorkerRole, CardEngine]:
    """Chat on an engine that reads one card; embed on one that reads none, unless told."""
    prepare_host(monkeypatch, tmp_path, ROLES, 64 << 30)
    monkeypatch.setattr(planning, "nvidia_card_rows", lambda: 0)
    monkeypatch.setattr(gpu_hardware, "installed_gpu_counts", Counter)
    cards = cards_of((4,), backend)
    wide = CardEngine("wide", tmp_path / "wide-server", ROLES, cards, takes_device_names=True)
    blind = CardEngine(
        "blind",
        tmp_path / "blind-server",
        ROLES,
        cards if embed_engine_lists_cards else [],
        takes_device_names=True,
    )
    engines = {CHAT: wide, EMBED: blind}
    planning.capture_plan_probe(engines)
    return engines


def _launches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    backend: str = "CUDA",
    embed_engine_lists_cards: bool = False,
) -> dict[WorkerRole, InstanceLaunch]:
    engines = _host(
        tmp_path, monkeypatch, backend=backend, embed_engine_lists_cards=embed_engine_lists_cards
    )
    plan = planning.plan_all_launches(engines=engines)
    assert {launch.role for launch in plan.launches} == {CHAT, EMBED}
    return {launch.role: launch for launch in plan.launches}


def _env_pins(launch: InstanceLaunch) -> list[str]:
    return [key for key in launch.env_overrides if key.endswith("VISIBLE_DEVICES")]


def test_an_engine_that_lists_no_card_gets_no_env_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    by_role = _launches(tmp_path, monkeypatch)
    assert by_role[CHAT].env_overrides["CUDA_VISIBLE_DEVICES"] == "0"
    assert _env_pins(by_role[EMBED]) == []


def test_an_engine_that_lists_no_card_gets_no_device_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    by_role = _launches(tmp_path, monkeypatch, backend="Vulkan")
    assert "--device" in by_role[CHAT].argv
    assert "--device" not in by_role[EMBED].argv


def test_the_user_is_told_the_role_runs_on_the_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger=_PLANNING_LOG):
        _launches(tmp_path, monkeypatch)
    messages = [record.getMessage() for record in caplog.records]
    assert NOTICE in messages
    assert not any("lists no GPU, so the chat role" in message for message in messages)


def test_an_engine_that_lists_the_card_keeps_its_pin_and_gets_no_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger=_PLANNING_LOG):
        by_role = _launches(tmp_path, monkeypatch, embed_engine_lists_cards=True)
    assert by_role[EMBED].env_overrides["CUDA_VISIBLE_DEVICES"] == "0"
    assert not any("lists no GPU" in record.getMessage() for record in caplog.records)


def _notices(caplog: pytest.LogCaptureFixture) -> int:
    return sum(record.getMessage() == NOTICE for record in caplog.records)


def test_the_notice_is_logged_once_by_the_launch_and_never_by_the_placement_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    engines = _host(tmp_path, monkeypatch)
    with caplog.at_level(logging.WARNING, logger=_PLANNING_LOG):
        planning.resolve_placement_plan(None, engines=engines)
        planning.resolve_placement_plan(None, engines=engines)
        assert _notices(caplog) == 0
        planning.plan_all_launches(engines=engines)
        assert _notices(caplog) == 1
        planning.resolve_placement_plan(None, engines=engines)
        assert _notices(caplog) == 1


class _Probe:
    """The snapshot fields the pin reads."""

    def __init__(self, devices: tuple[FleetDevice, ...], *, refused_all: bool = False) -> None:
        self.devices = devices
        self.engine_devices_all_refused = refused_all


def _pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    probe: _Probe | None,
    target: RoleTarget = RoleTarget.GPU,
) -> tuple[FleetDevice, ...]:
    """The pin of one embed instance, and the notice the launch logs for it."""
    cards = tuple(cards_of((4,)))
    monkeypatch.setattr(planning, "_current_plan_probe", lambda _engine: probe)
    plan = InstancePlan(role=EMBED, devices=(0,), target=target)
    engine = CardEngine("blind", tmp_path / "blind-server", ROLES, [])
    planning._log_roles_left_unpinned(Placement((plan,), ()), {0: cards[0]}, {EMBED: engine})
    return planning._pinned_devices(cards, plan, engine)


def test_a_plan_with_no_snapshot_keeps_the_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger=_PLANNING_LOG):
        kept = _pin(tmp_path, monkeypatch, None)
    assert kept == tuple(cards_of((4,)))
    assert caplog.records == []


def test_a_role_that_lists_no_card_is_unpinned_and_told(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger=_PLANNING_LOG):
        kept = _pin(tmp_path, monkeypatch, _Probe(()))
    assert kept == ()
    assert _notices(caplog) == 1


def test_a_role_with_two_replicas_is_told_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(planning, "_current_plan_probe", lambda _engine: _Probe(()))
    cards = tuple(cards_of((4,)))
    first = InstancePlan(role=EMBED, devices=(0,))
    second = InstancePlan(role=EMBED, devices=(0,), replica=1)
    engine = CardEngine("blind", tmp_path / "blind-server", ROLES, [])
    with caplog.at_level(logging.WARNING, logger=_PLANNING_LOG):
        planning._log_roles_left_unpinned(
            Placement((first, second), ()), {0: cards[0]}, {EMBED: engine}
        )
    assert _notices(caplog) == 1


def test_a_cpu_target_is_not_told_it_runs_on_the_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger=_PLANNING_LOG):
        kept = _pin(tmp_path, monkeypatch, _Probe(()), RoleTarget.CPU)
    assert kept == tuple(cards_of((4,)))
    assert caplog.records == []


def test_an_engine_whose_cards_were_all_refused_is_unpinned_and_gets_no_second_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The refusal notice already says the engine runs on the CPU; this adds none."""
    with caplog.at_level(logging.WARNING, logger=_PLANNING_LOG):
        kept = _pin(tmp_path, monkeypatch, _Probe((), refused_all=True))
    assert kept == ()
    assert caplog.records == []
