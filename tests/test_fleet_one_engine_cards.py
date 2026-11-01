"""One engine on two and three cards plans as the recorded golden says."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from lilbee.core.config import cfg
from lilbee.providers.fleet import planning
from lilbee.providers.fleet.planning import FleetPlan
from lilbee.providers.roles import WorkerRole
from tests._fleet_engines import (
    GIB,
    CardEngine,
    Role,
    cards_of,
    engines_for_all_roles,
    prepare_host,
)

GOLDEN = Path(__file__).parent / "fleet_one_engine_cards_golden.json"
C, E, R, V = WorkerRole.CHAT, WorkerRole.EMBED, WorkerRole.RERANK, WorkerRole.VISION


@dataclass(frozen=True)
class Scenario:
    """One engine, its cards in GiB and its roles."""

    cards: tuple[int, ...]
    roles: tuple[Role, ...]
    spans: bool = False
    num_ctx: int | None = None


SCENARIOS = {
    "two-tight-launches": Scenario((8, 8), (Role(C, 12), Role(E, 2)), num_ctx=4096),
    "three-tight-launches": Scenario((8, 8, 8), (Role(C, 18), Role(E, 2)), num_ctx=4096),
    "two-fit": Scenario((8, 8), (Role(C, 7), Role(E, 2))),
    "two-split": Scenario((8, 8), (Role(C, 12), Role(E, 2)), spans=True),
    "two-tight-widen": Scenario((8, 8), (Role(C, 20), Role(E, 2))),
    "two-tight-no-span": Scenario((8, 8), (Role(C, 12), Role(E, 2))),
    "two-replicas": Scenario((8, 8), (Role(C, 4), Role(E, 2, replicas=2))),
    "two-swap-group": Scenario((8, 8), (Role(C, 7), Role(E, 7), Role(V, 7))),
    "three-fit": Scenario((8, 8, 8), (Role(C, 7), Role(E, 7), Role(V, 7))),
    "three-split": Scenario((8, 8, 8), (Role(C, 18), Role(E, 2)), spans=True),
    "three-tight-widen": Scenario((8, 8, 8), (Role(C, 40), Role(E, 2))),
    "three-replicas": Scenario((8, 8, 8), (Role(C, 4), Role(E, 2, replicas=3))),
    "three-rerank": Scenario((8, 8, 8), (Role(C, 7), Role(E, 2), Role(R, 2))),
    "unequal-tight": Scenario((24, 8), (Role(C, 30), Role(E, 2))),
}


def plan_one_engine(engine: CardEngine):
    """Plan every configured role on *engine*."""
    return planning.plan_all_launches(engines=engines_for_all_roles(engine))


def run_scenario(
    monkeypatch, tmp_path: Path, scenario: Scenario, plan: Callable[[CardEngine], FleetPlan]
) -> str:
    """The plan of *scenario*, as a repr with the temp path taken out."""
    prepare_host(monkeypatch, tmp_path, scenario.roles, 64 * GIB)
    monkeypatch.setattr(cfg, "num_ctx", scenario.num_ctx)
    # The default target follows the RAM of the host, and the goldens record the 32 GiB tier.
    monkeypatch.setattr(cfg, "chat_n_ctx_target", 16384)
    engine = CardEngine(
        "engine-a",
        tmp_path / "engine-server",
        scenario.roles,
        cards_of(scenario.cards),
        spans=scenario.spans,
    )
    result = plan(engine)
    group = sorted(role.value for role in result.co_tenants)
    # A frozenset's repr order changes per process, so the group is written sorted.
    text = repr(replace(result, co_tenants=frozenset()))
    return _without_the_temp_path(f"{text} group={group}", tmp_path)


def _without_the_temp_path(text: str, tmp_path: Path) -> str:
    """Replace the temp path by <TMP> and write what follows it with forward slashes."""
    # A repr doubles each backslash of a Windows path, so the path is looked up as repr writes it.
    text = text.replace(str(tmp_path), "<TMP>").replace(repr(str(tmp_path))[1:-1], "<TMP>")
    return re.sub(
        r"<TMP>((?:\\\\[^'\\\\]+)+)", lambda m: "<TMP>" + m.group(1).replace("\\\\", "/"), text
    )


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_one_engine_plan_equals_the_golden(monkeypatch, tmp_path, name) -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert run_scenario(monkeypatch, tmp_path, SCENARIOS[name], plan_one_engine) == golden[name]


def test_every_scenario_has_a_golden() -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert SCENARIOS
    assert set(golden) == set(SCENARIOS)
