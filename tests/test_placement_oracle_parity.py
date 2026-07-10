"""Differential regression: the fleet planner must serve every role the b507
in-process pool would have served on demand.

The in-process ``WorkerPool`` had no VRAM admission and no idle reaping: a role's
worker spawned lazily on first call, so only the roles a single operation used were
ever co-resident. Ingest loaded embed+vision; query loaded embed+chat+rerank. Peak
was ``embed + max(vision, chat+rerank)``, never the sum. This suite feeds low-RAM /
on-the-cusp conditions into the real :func:`plan_placement` and asserts it never
refuses a role that lazy per-phase residency would have served.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from lilbee.providers.fleet.placement import ModelPlacementInput, plan_placement
from lilbee.providers.fleet.vram import USABLE_VRAM_FRACTION
from lilbee.providers.roles import WorkerRole

GB = 1024**3

# Roles co-resident during each operation, per the in-process lazy-load reality.
_WORKING_SETS: tuple[frozenset[WorkerRole], ...] = (
    frozenset({WorkerRole.EMBED, WorkerRole.VISION}),  # ingest: OCR + chunk embed
    frozenset({WorkerRole.CHAT, WorkerRole.EMBED, WorkerRole.RERANK}),  # query
)


@dataclass(frozen=True)
class Scenario:
    name: str
    footprints: dict[WorkerRole, int]  # role -> footprint bytes
    devices: list[tuple[int, int]] = field(default_factory=list)  # (index, total vram)
    unified_budget: int | None = None  # free RAM bytes when devices is empty


def _gb(x: float) -> int:
    return int(x * GB)


def _budget(sc: Scenario) -> int:
    """The usable budget the planner charges against, per path.

    The GPU path applies the 90% ``USABLE_VRAM_FRACTION`` headroom; the unified path
    charges the raw free-RAM budget it is given. The oracle is gated at the same
    basis so the comparison isolates the co-residency policy, not the headroom.
    """
    if sc.devices:
        return int(sum(v for _, v in sc.devices) * USABLE_VRAM_FRACTION)
    assert sc.unified_budget is not None
    return sc.unified_budget


def _estimate(footprints: dict[WorkerRole, int]):
    def estimate(role: WorkerRole, ratio: tuple[int, ...]) -> tuple[int, ...]:
        denom = sum(ratio) or 1
        return tuple(int(footprints[role] * r / denom) for r in ratio)

    return estimate


def _oracle_servable(sc: Scenario) -> set[WorkerRole]:
    """Every role in a working set that fits: what the lazy pool would serve."""
    budget = _budget(sc)
    servable: set[WorkerRole] = set()
    for roles in _WORKING_SETS:
        present = [r for r in roles if r in sc.footprints]
        if sum(sc.footprints[r] for r in present) <= budget:
            servable.update(present)
    return servable


def _planner_servable(sc: Scenario) -> tuple[set[WorkerRole], frozenset[WorkerRole]]:
    models = [
        ModelPlacementInput(role=role, est_vram_bytes=vram) for role, vram in sc.footprints.items()
    ]
    placement = plan_placement(
        models,
        sc.devices,
        estimate_peak=_estimate(sc.footprints),
        unified_budget=sc.unified_budget,
    )
    servable = set(sc.footprints) - set(placement.unplaceable_roles)
    return servable, placement.co_tenants


# Footprints modeled on the friend's stack: a ~4B chat, nomic embed, a small
# cross-encoder rerank, and a ~4B vision OCR model. Cusp scenarios vary only the
# budget so the planner's co-residency policy is the lone variable. The "big vision"
# rows show the regression is not unique to 6GB: a larger OCR model reaches it at 8GB.
_STACK = {
    WorkerRole.CHAT: _gb(4.5),
    WorkerRole.VISION: _gb(4.5),
    WorkerRole.EMBED: _gb(0.6),
    WorkerRole.RERANK: _gb(0.6),
}
_BIG_VISION = {**_STACK, WorkerRole.VISION: _gb(7.0)}
# An LLM reranker larger than the embedder: must never crowd embed out of the plan.
_LLM_RERANK = {
    WorkerRole.CHAT: _gb(3.0),
    WorkerRole.VISION: _gb(2.0),
    WorkerRole.EMBED: _gb(0.6),
    WorkerRole.RERANK: _gb(5.0),
}
# Chat smaller than vision: the swap group must be charged at vision's footprint.
_SMALL_CHAT = {
    WorkerRole.CHAT: _gb(3.0),
    WorkerRole.VISION: _gb(5.0),
    WorkerRole.EMBED: _gb(0.4),
    WorkerRole.RERANK: _gb(0.6),
}

SCENARIOS = [
    Scenario("gpu-24gb-roomy", dict(_STACK), devices=[(0, _gb(24))]),
    Scenario("gpu-12gb-cusp", dict(_STACK), devices=[(0, _gb(12))]),
    Scenario("gpu-8gb-one-big-at-a-time", dict(_STACK), devices=[(0, _gb(8))]),
    Scenario("gpu-6gb-embed-rerank-vision-overflow", dict(_STACK), devices=[(0, _gb(6))]),
    Scenario("gpu-8gb-big-vision", dict(_BIG_VISION), devices=[(0, _gb(8))]),
    Scenario("gpu-6gb-llm-rerank", dict(_LLM_RERANK), devices=[(0, _gb(6))]),
    Scenario("gpu-6gb-small-chat-big-vision", dict(_SMALL_CHAT), devices=[(0, _gb(6))]),
    Scenario("unified-6gb-llm-rerank", dict(_LLM_RERANK), unified_budget=_gb(6)),
    Scenario("unified-6gb-embed-rerank-vision-overflow", dict(_STACK), unified_budget=_gb(6)),
    Scenario("unified-8gb", dict(_STACK), unified_budget=_gb(8)),
    Scenario("unified-12gb", dict(_STACK), unified_budget=_gb(12)),
    Scenario("unified-8gb-big-vision", dict(_BIG_VISION), unified_budget=_gb(8)),
    Scenario("unified-6gb-big-vision", dict(_BIG_VISION), unified_budget=_gb(6)),
]


def _charged_total(sc: Scenario, servable: set[WorkerRole], swap: frozenset[WorkerRole]) -> int:
    """The VRAM the plan reserves: every persistent role, plus one swap-group member.

    Swap-group members never co-reside (llama-swap loads one at a time), so the group
    is charged its largest member; every other served role is charged in full.
    """
    persistent = [r for r in servable if r not in swap]
    total = sum(sc.footprints[r] for r in persistent)
    members = [sc.footprints[r] for r in swap if r in servable]
    if members:
        total += max(members)
    return total


@pytest.mark.parametrize("sc", SCENARIOS, ids=lambda s: s.name)
def test_planner_serves_every_role_the_oracle_would(sc: Scenario) -> None:
    oracle = _oracle_servable(sc)
    servable, _co = _planner_servable(sc)
    refused = {r.value for r in oracle if r not in servable}
    assert not refused, (
        f"{sc.name}: planner refuses {sorted(refused)} that the in-process oracle "
        f"would serve on demand (oracle={sorted(r.value for r in oracle)}, "
        f"planner={sorted(r.value for r in servable)})"
    )


@pytest.mark.parametrize("sc", SCENARIOS, ids=lambda s: s.name)
def test_plan_fits_its_own_budget(sc: Scenario) -> None:
    """Serving a role is not enough: the plan's reservation must fit the budget."""
    servable, swap = _planner_servable(sc)
    reserved = _charged_total(sc, servable, swap)
    budget = _budget(sc)
    assert reserved <= budget, (
        f"{sc.name}: plan reserves {reserved / GB:.2f}GB > budget {budget / GB:.2f}GB "
        f"(served={sorted(r.value for r in servable)}, swap-group={sorted(r.value for r in swap)})"
    )


def _print_diff_table() -> int:
    """Print the oracle-vs-planner diff for every scenario; return the regression count."""
    regressions = 0
    for sc in SCENARIOS:
        oracle = _oracle_servable(sc)
        servable, co = _planner_servable(sc)
        refused = sorted(r.value for r in oracle if r not in servable)
        regressions += bool(refused)
        if sc.devices:
            hw = f"{len(sc.devices)}xGPU {[v // GB for _, v in sc.devices]}GB"
        else:
            assert sc.unified_budget is not None
            hw = f"unified {sc.unified_budget // GB}GB"
        print(f"[{'BUG' if refused else 'OK '}] {sc.name}  ({hw})")
        print(f"        oracle  : {sorted(r.value for r in oracle)}")
        print(
            f"        planner : {sorted(r.value for r in servable)}"
            f"  swap-group={sorted(r.value for r in co)}"
        )
        if refused:
            print(f"        >>> REGRESSION: planner refuses {refused}")
    print(f"\n=== {regressions} scenario(s) regress vs the in-process oracle ===")
    return regressions


if __name__ == "__main__":
    raise SystemExit(1 if _print_diff_table() else 0)
