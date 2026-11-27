"""A role is not launched on a card that its own engine's device read does not list."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import pytest

from lilbee.app import placement as app_placement
from lilbee.app.engine_choice import BookReading
from lilbee.core.config import cfg
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import gpu_hardware, planning
from lilbee.providers.fleet.placement import unseen_card_reason
from lilbee.providers.fleet.placement_spec import PlacementSpec
from lilbee.providers.roles import WorkerRole
from tests._fleet_engines import CardEngine, Role, cards_of, engines_for_all_roles, prepare_host

ROLES = (Role(WorkerRole.CHAT, 1), Role(WorkerRole.EMBED, 1))
CHAT_ON_CARD_ZERO = '"chat": {"devices": [0], "target": "gpu"}'
EMBED_ON_CARD_ONE = "{" + CHAT_ON_CARD_ZERO + ', "embed": {"devices": [1], "target": "gpu"}}'
EMBED_ON_CARD_ZERO = "{" + CHAT_ON_CARD_ZERO + ', "embed": {"devices": [0], "target": "gpu"}}'
REFUSAL = (
    "embed is placed on device 1, but the narrow engine lists only device 0. "
    "Set the embed target to cpu, or choose an engine that lists that card."
)
CHAT_ON_CARD_ZERO_EMBED_ON_CPU = "{" + CHAT_ON_CARD_ZERO + ', "embed": {"target": "cpu"}}'
CHAT_OTHER_CARD_REFUSAL = (
    "chat is placed on device 0, but the narrow engine lists a different card at device 0. "
    "Set the chat target to cpu, or choose an engine that lists that card."
)
OTHER_CARD_REFUSAL = (
    "embed is placed on device 0, but the narrow engine lists a different card at device 0. "
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
    # The narrow engine reads the same card 0 as the wide engine: a launch maps cards by
    # identity, so a different card under the same index is a card the engine does not list.
    same_card = CardEngine("narrow", tmp_path / "narrow-server", ROLES, cards_of((4,)))
    engines = _engines(tmp_path, same_card)
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


def test_a_role_on_a_card_its_engine_lists_as_another_card_is_refused_with_the_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The narrow engine lists a 16 GiB card 0; the plan's card 0 is a 4 GiB card."""
    _host(tmp_path, monkeypatch, EMBED_ON_CARD_ZERO)
    engines = _engines(tmp_path)
    planning.capture_plan_probe(engines)
    plan = planning.plan_all_launches(engines=engines)
    assert [launch.role for launch in plan.launches] == [WorkerRole.CHAT]
    assert plan.refused_roles == {WorkerRole.EMBED: OTHER_CARD_REFUSAL}


def test_a_chat_on_a_card_its_engine_lists_as_another_card_is_refused_and_embed_is_planned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _host(tmp_path, monkeypatch, EMBED_ON_CARD_ZERO)
    wide = CardEngine("wide", tmp_path / "wide-server", ROLES, cards_of((4, 4)))
    narrow = CardEngine("narrow", tmp_path / "narrow-server", ROLES, cards_of((16,)))
    engines = {WorkerRole.CHAT: narrow, WorkerRole.EMBED: wide}
    planning.capture_plan_probe(engines)
    plan = planning.plan_all_launches(engines=engines)
    assert [launch.role for launch in plan.launches] == [WorkerRole.EMBED]
    assert list(plan.refused_roles) == [WorkerRole.CHAT]


def test_the_placement_view_reports_the_same_refusal_for_another_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _host(tmp_path, monkeypatch, EMBED_ON_CARD_ZERO)
    engines = _engines(tmp_path)
    planning.capture_plan_probe(engines)
    resolved = planning.resolve_placement_plan(
        PlacementSpec.from_json(EMBED_ON_CARD_ZERO), engines=engines
    )
    assert resolved.unlaunched_roles == {WorkerRole.EMBED: OTHER_CARD_REFUSAL}
    assert [inst.role for inst in resolved.instances] == [WorkerRole.CHAT]


def test_the_placement_view_lists_a_role_on_another_card_as_refused_and_keeps_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _host(tmp_path, monkeypatch, EMBED_ON_CARD_ZERO)
    engines = _engines(tmp_path)
    planning.capture_plan_probe(engines)
    reading = BookReading(adapters=engines, notices={})
    monkeypatch.setattr(app_placement, "_planning_engines", lambda: reading)
    view = app_placement.get_placement()
    assert view.refusal == OTHER_CARD_REFUSAL
    assert view.refusal_kind is app_placement.PlacementRefusal.REFUSED
    assert [role.role for role in view.roles] == [WorkerRole.CHAT]


def _chat_on_another_card_embed_on_cpu(tmp_path: Path) -> dict[WorkerRole, CardEngine]:
    """Chat's engine lists one 16 GiB card 0; the plan's card 0 is a 4 GiB card."""
    wide = CardEngine("wide", tmp_path / "wide-server", ROLES, cards_of((4, 4)))
    narrow = CardEngine("narrow", tmp_path / "narrow-server", ROLES, cards_of((16,)))
    return {WorkerRole.CHAT: narrow, WorkerRole.EMBED: wide}


def test_a_chat_on_another_card_with_embed_on_the_cpu_is_refused_and_embed_is_planned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _host(tmp_path, monkeypatch, CHAT_ON_CARD_ZERO_EMBED_ON_CPU)
    engines = _chat_on_another_card_embed_on_cpu(tmp_path)
    planning.capture_plan_probe(engines)
    plan = planning.plan_all_launches(engines=engines)
    assert [launch.role for launch in plan.launches] == [WorkerRole.EMBED]
    assert plan.refused_roles == {WorkerRole.CHAT: CHAT_OTHER_CARD_REFUSAL}


def test_a_chat_on_another_card_commits_no_memory_before_the_cpu_embed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The chat holds 1 GiB of host memory when it launches; the CPU embed needs 1 GiB too."""
    _host(tmp_path, monkeypatch, CHAT_ON_CARD_ZERO_EMBED_ON_CPU)
    for module in (planning, planning.model_cache):
        monkeypatch.setattr(module, "total_system_memory", lambda: 3 << 29)
    other_card = _chat_on_another_card_embed_on_cpu(tmp_path)
    other_card[WorkerRole.CHAT] = CardEngine(
        "narrow", tmp_path / "narrow-server", ROLES, cards_of((16,)), host_gib=1
    )
    planning.capture_plan_probe(other_card)
    plan = planning.plan_all_launches(engines=other_card)
    assert [launch.role for launch in plan.launches] == [WorkerRole.EMBED]
    assert plan.refused_roles == {WorkerRole.CHAT: CHAT_OTHER_CARD_REFUSAL}


def test_a_chat_on_a_listed_card_commits_its_memory_before_the_cpu_embed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same memory as the test above, with the chat's card listed: the embed no longer fits."""
    _host(tmp_path, monkeypatch, CHAT_ON_CARD_ZERO_EMBED_ON_CPU)
    for module in (planning, planning.model_cache):
        monkeypatch.setattr(module, "total_system_memory", lambda: 3 << 29)
    wide = CardEngine("wide", tmp_path / "wide-server", ROLES, cards_of((4, 4)), host_gib=1)
    engines = {WorkerRole.CHAT: wide, WorkerRole.EMBED: wide}
    planning.capture_plan_probe(engines)
    plan = planning.plan_all_launches(engines=engines)
    assert [launch.role for launch in plan.launches] == [WorkerRole.CHAT]
    assert list(plan.refused_roles) == [WorkerRole.EMBED]


def test_the_placement_view_refuses_a_chat_on_another_card_with_embed_on_the_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _host(tmp_path, monkeypatch, CHAT_ON_CARD_ZERO_EMBED_ON_CPU)
    engines = _chat_on_another_card_embed_on_cpu(tmp_path)
    planning.capture_plan_probe(engines)
    resolved = planning.resolve_placement_plan(
        PlacementSpec.from_json(CHAT_ON_CARD_ZERO_EMBED_ON_CPU), engines=engines
    )
    assert resolved.unlaunched_roles == {WorkerRole.CHAT: CHAT_OTHER_CARD_REFUSAL}
    assert [inst.role for inst in resolved.instances] == [WorkerRole.EMBED]


def test_the_reason_names_both_a_different_card_and_a_missing_card() -> None:
    reason = unseen_card_reason(WorkerRole.EMBED, "narrow", [0, 2], [0, 1])
    assert reason == (
        "embed is placed on devices 0 and 2, but the narrow engine lists a different card "
        "at device 0 and no card at device 2. Set the embed target to cpu, or choose an "
        "engine that lists that card."
    )


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


class _BrokenLaunchEngine(CardEngine):
    """An engine whose launch build fails where a plain provider error is not a refusal."""

    def launch_argv(self, spec, request) -> list[str]:
        raise ProviderError("the launch cannot build its command line")


def test_a_plain_provider_error_inside_the_launch_is_not_swallowed_as_a_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the card and rank-pool refusals are caught; any other launch error escapes the plan."""
    _host(tmp_path, monkeypatch, EMBED_ON_CARD_ZERO)
    engine = _BrokenLaunchEngine("wide", tmp_path / "wide-server", ROLES, cards_of((4, 4)))
    with pytest.raises(ProviderError, match="the launch cannot build its command line"):
        planning.plan_all_launches(engines=engines_for_all_roles(engine))
