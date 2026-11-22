"""A chat role the memory budget leaves no room for fails its warm with the planner's text."""

from __future__ import annotations

from pathlib import Path

import pytest

from lilbee.core.config import cfg
from lilbee.providers.fleet import planning
from lilbee.providers.fleet.provider import FleetProvider
from lilbee.providers.roles import WorkerRole
from lilbee.providers.warm_progress import WarmPhase
from tests._fleet_engines import GIB, engine_pair, engines_for_all_roles, one_machine


def _budget_plan(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> planning.FleetPlan:
    """A unified-memory host of 8 GiB with a 20 GiB chat model: chat has no room."""
    device = one_machine(monkeypatch, tmp_path, unified=True, total=8 * GIB)
    chat, _ = engine_pair(tmp_path, device, 20, 1)
    return planning.plan_all_launches(engines=engines_for_all_roles(chat))


def _reason() -> str:
    return (
        f"chat model {cfg.chat_model} does not fit available memory and will not be "
        "served; free up memory or use a smaller model."
    )


def test_the_launch_plan_refuses_a_budget_unplaceable_role_with_the_planner_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = _budget_plan(monkeypatch, tmp_path)
    assert WorkerRole.CHAT not in {launch.role for launch in plan.launches}
    assert plan.refused_roles[WorkerRole.CHAT] == _reason()
    assert plan.refused_roles[WorkerRole.EMBED] == (
        f"embed model {cfg.embedding_model} does not fit available memory and will not be "
        "served; free up memory or use a smaller model."
    )


def test_the_warm_of_a_budget_refused_chat_role_fails_with_that_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = _budget_plan(monkeypatch, tmp_path)
    provider = FleetProvider()

    def _ensure_fleet(**_: object) -> None:
        provider._refused_roles = dict(plan.refused_roles)

    monkeypatch.setattr(provider, "_ensure_fleet", _ensure_fleet)
    monkeypatch.setattr(provider, "_preload_roles", lambda roles=None: None)
    provider._warm_up_blocking()
    snapshot = provider.warm_progress()
    assert snapshot is not None
    assert snapshot.phase is WarmPhase.ERROR
    assert snapshot.error == _reason()
