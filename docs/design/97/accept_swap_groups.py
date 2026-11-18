"""Acceptance script for the swap-group rule: roles that share a run phase never evict each other.

Run from the repository root with the project venv:

    UV_NO_SYNC=1 PYTHONPATH=src .venv/bin/python docs/design/97/accept_swap_groups.py
    ... --record-golden docs/design/97/golden-base.json     (once, at the base)
    ... --table                                              (Expected behaviour table)

Exit 0 when every row passes, 1 when any row fails.

The oracle is not the code under test. Phases come from the in-process pool the planner replaces
(ingest loads embed + vision; a query loads embed + rerank + chat), copied from
tests/test_placement_oracle_parity.py. Eviction comes from llama-swap's documented group
semantics (swap, exclusive, persistent), applied to the config the product renders. The
documented semantics are checked against the real llama-swap binary by probe_llama_swap.sh.

Adaptation point for a later fix lane: ``render_process`` is the only function that knows how
the product turns a placement into a llama-swap config. If the signature of
``build_swap_config`` changes, change that one function; every row reads its output.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import lilbee  # noqa: E402
from lilbee.providers.fleet.groups import SwapGroup, group_for  # noqa: E402
from lilbee.providers.fleet.launch import InstanceLaunch  # noqa: E402
from lilbee.providers.fleet.placement import (  # noqa: E402
    ModelPlacementInput,
    Placement,
    plan_placement,
)
from lilbee.providers.fleet.swap_config import build_swap_config  # noqa: E402
from lilbee.providers.roles import WorkerRole  # noqa: E402

assert str(Path(lilbee.__file__).resolve()).startswith(str(ROOT)), (
    f"lilbee imported from {lilbee.__file__}, not from {ROOT}"
)

GB = 1024**3
USABLE = 0.9  # USABLE_VRAM_FRACTION; the oracle restates it so a change there is visible

_spec = importlib.util.spec_from_file_location(
    "oracle_parity", ROOT / "tests" / "test_placement_oracle_parity.py"
)
assert _spec is not None and _spec.loader is not None
_oracle = importlib.util.module_from_spec(_spec)
sys.modules["oracle_parity"] = _oracle
_spec.loader.exec_module(_oracle)

WORKING_SETS: tuple[frozenset[WorkerRole], ...] = _oracle._WORKING_SETS
assert WORKING_SETS, "the oracle has no working sets"

# ----- scenarios ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    name: str
    footprints: dict[WorkerRole, int]
    devices: tuple[tuple[int, int], ...] = ()
    unified_budget: int | None = None
    refuse_oversize: bool = False
    replicas: tuple[tuple[WorkerRole, int], ...] = ()

    @property
    def key(self) -> str:
        return self.name


def _gb(x: float) -> int:
    return int(x * GB)


def _cases() -> list[Case]:
    cases: list[Case] = []
    for sc in _oracle.SCENARIOS:  # the current oracle scenarios, default mode
        cases.append(
            Case(sc.name, dict(sc.footprints), tuple(sc.devices), sc.unified_budget)
        )
        if sc.devices:  # option mode: refuse_oversize runs a second packer
            cases.append(
                Case(
                    f"{sc.name}|refuse",
                    dict(sc.footprints),
                    tuple(sc.devices),
                    None,
                    refuse_oversize=True,
                )
            )
    # Generator: every combination of the four footprints on every host shape.
    hosts: list[tuple[str, tuple[tuple[int, int], ...], int | None]] = []
    for card in (6, 8, 12, 16, 24):
        hosts.append((f"gpu{card}", ((0, _gb(card)),), None))
    for ram in (6, 8, 12):
        hosts.append((f"uni{ram}", (), _gb(ram)))
    hosts.append(("gpu6x2", ((0, _gb(6)), (1, _gb(6))), None))
    hosts.append(("gpu12+8", ((0, _gb(12)), (1, _gb(8))), None))
    for chat, vision, rerank in itertools.product(
        (2.0, 3.0, 4.5, 6.0), (1.0, 2.0, 4.5, 7.0, 10.0, 18.0), (0.3, 0.6, 2.0, 5.0)
    ):
        fp = {
            WorkerRole.CHAT: _gb(chat),
            WorkerRole.VISION: _gb(vision),
            WorkerRole.EMBED: _gb(0.6),
            WorkerRole.RERANK: _gb(rerank),
        }
        for label, devices, ram in hosts:
            tag = f"gen-{label}-c{chat}-v{vision}-r{rerank}"
            cases.append(Case(tag, fp, devices, ram))
            if devices:
                cases.append(Case(f"{tag}|refuse", fp, devices, None, refuse_oversize=True))
    # Option mode: replicated roles (the elastic tier caps a co-tenant to one instance).
    for sc in _oracle.SCENARIOS:
        cases.append(
            Case(
                f"{sc.name}|replicas",
                dict(sc.footprints),
                tuple(sc.devices),
                sc.unified_budget,
                replicas=((WorkerRole.VISION, 2), (WorkerRole.EMBED, 2)),
            )
        )
    return cases


# ----- running the product ------------------------------------------------------------------


def _estimate(footprints: dict[WorkerRole, int]):
    def estimate(role: WorkerRole, ratio: tuple[int, ...]) -> tuple[int, ...]:
        denom = sum(ratio) or 1
        return tuple(int(footprints[role] * r / denom) for r in ratio)

    return estimate


def run_plan(case: Case) -> Placement:
    reps = dict(case.replicas)
    models = [
        ModelPlacementInput(role=r, est_vram_bytes=v, replicas=reps.get(r, 1))
        for r, v in case.footprints.items()
    ]
    return plan_placement(
        models,
        list(case.devices),
        estimate_peak=_estimate(case.footprints),
        unified_budget=case.unified_budget,
        refuse_oversize=case.refuse_oversize,
    )


def render_process(placement: Placement) -> dict[str, object] | None:
    """The llama-swap config of the co-tenant process, or None when the plan has none.

    HEAD ADAPTATION: at the base a placement carries no residency classes, so every launch
    is unlabeled and ``swap=True`` makes the whole process one evicting group. At the head
    ``Placement.swap_classes`` partitions ``co_tenants`` and each launch carries its class
    in ``InstanceLaunch.swap_class`` (rulings/97.md, "Interface"). Change this function, and
    only this one, when the fix lane names the fields differently.
    """
    classes = getattr(placement, "swap_classes", None)

    def class_of(role: WorkerRole) -> dict[str, object]:
        if classes is None:
            return {}
        return {"swap_class": str(next(i for i, c in enumerate(classes) if role in c))}

    launches = [
        InstanceLaunch(
            role=inst.role,
            argv=["llama-server", "--model", f"/m/{inst.role.value}.gguf"],
            env_overrides={},
            model=f"{inst.role.value}-model",
            replica=inst.replica,
            **class_of(inst.role),
        )
        for inst in placement.instances
        if group_for(inst.role, placement.co_tenants) is SwapGroup.CO_TENANT
    ]
    if not launches:
        return None
    ports = {launch.model_id: 5900 + i for i, launch in enumerate(launches)}
    return json.loads(
        build_swap_config(launches, ports, swap=SwapGroup.CO_TENANT.swaps)
    )


# ----- llama-swap semantics, as documented (docs/config.example.yaml of llama-swap) -----------
#   swap (default true): true, only one member of the group runs at a time; false, all run.
#   exclusive (default true): true, running a member unloads every other group; false, none.
#   persistent (default false): other groups cannot unload this group's members.


def simulate(cfg: dict[str, object], order: list[str]) -> set[str]:
    groups: dict[str, dict] = cfg["groups"]  # type: ignore[assignment]
    where = {m: g for g, spec in groups.items() for m in spec["members"]}
    resident: set[str] = set()
    for model in order:
        spec = groups[where[model]]
        if spec.get("swap", True):
            resident -= set(spec["members"]) - {model}
        if spec.get("exclusive", True):
            resident -= {
                x for x in resident if where[x] != where[model]
                and not groups[where[x]].get("persistent", False)
            }
        resident.add(model)
    return resident


def _mid(role: WorkerRole) -> str:
    return f"{role.value}-0"


# ----- the oracle ---------------------------------------------------------------------------


def share_a_phase(a: WorkerRole, b: WorkerRole) -> bool:
    return any(a in w and b in w for w in WORKING_SETS)


def served(case: Case, p: Placement) -> set[WorkerRole]:
    return set(case.footprints) - set(p.unplaceable_roles)


def budget(case: Case) -> int:
    if case.devices:
        return int(sum(v for _, v in case.devices) * USABLE)
    assert case.unified_budget is not None
    return case.unified_budget


def affordable(case: Case) -> set[WorkerRole]:
    out: set[WorkerRole] = set()
    for ws in WORKING_SETS:
        present = [r for r in ws if r in case.footprints]
        if sum(case.footprints[r] for r in present) <= budget(case):
            out.update(present)
    return out


def oracle_peak(case: Case, members: set[WorkerRole]) -> int:
    """What the in-process pool would hold from *members* at once: the largest phase set."""
    sums = [sum(case.footprints[r] for r in ws if r in members) for ws in WORKING_SETS]
    singles = [case.footprints[r] for r in members]
    return max([*sums, *singles], default=0)


@dataclass
class Row:
    case: str
    check: str
    ok: bool
    detail: str = ""


@dataclass
class Report:
    rows: list[Row] = field(default_factory=list)

    def add(self, case: str, check: str, ok: bool, detail: str = "") -> None:
        self.rows.append(Row(case, check, ok, detail))


def bead_case(p: Placement) -> bool:
    """A plan whose co-tenant set holds two roles that share a phase: the defect's shape."""
    return any(share_a_phase(a, b) for a, b in itertools.combinations(sorted(p.co_tenants), 2))


def fingerprint(p: Placement) -> dict[str, object]:
    return {
        "instances": sorted(
            [i.role.value, list(i.devices), i.replica, list(i.tensor_split or ())]
            for i in p.instances
        ),
        "unplaceable": sorted(r.value for r in p.unplaceable_roles),
        "co_tenants": sorted(r.value for r in p.co_tenants),
        "tight": {r.value: s for r, s in sorted(p.tight_roles.items(), key=lambda kv: kv[0].value)},
    }


def peak_resident(case: Case, cfg: dict[str, object], members: set[WorkerRole]) -> int:
    """Largest byte total llama-swap holds at once from *members*, over every load order."""
    best = 0
    ordered = sorted(members, key=lambda r: r.value)
    for order in itertools.permutations(ordered):
        loaded: list[str] = []
        for role in order:
            loaded.append(_mid(role))
            resident = simulate(cfg, loaded)
            best = max(best, sum(case.footprints[r] for r in members if _mid(r) in resident))
    return best


def single_pool(case: Case) -> bool:
    """One memory pool: a single card or the unified budget (no fragmentation across cards)."""
    return len(case.devices) <= 1


def default_mode(case: Case) -> bool:
    return not case.refuse_oversize and not case.replicas


def check_case(case: Case, p: Placement, golden: dict | None, rep: Report) -> None:
    srv = served(case, p)
    tight = set(p.tight_roles)
    members = {r for r in p.co_tenants if r in srv}
    cfg = render_process(p)
    bead = bead_case(p)
    if default_mode(case):
        # P1 parity: the planner serves every role the in-process pool would.
        scenario = _oracle.Scenario(
            case.name, case.footprints, list(case.devices), case.unified_budget
        )
        oracle = set(case.footprints) if case.devices else _oracle._working_set_servable(scenario)
        refused = sorted(r.value for r in oracle if r not in srv)
        rep.add(case.key, "P1-serves-every-oracle-role", not refused, f"refused {refused}")
        # P2: tight only when the working set exceeds the budget.
        spurious = sorted(r.value for r in tight if r in affordable(case))
        rep.add(case.key, "P2-no-spurious-tight", not spurious, f"spurious {spurious}")
        # P3: what llama-swap can hold at once, beside the standalone roles, fits the budget.
        if cfg is not None and single_pool(case):
            standalone = sum(case.footprints[r] for r in srv - tight if r not in members)
            peak = peak_resident(case, cfg, members - tight)
            rep.add(
                case.key,
                "P3-resident-peak-within-budget",
                standalone + peak <= budget(case),
                f"{(standalone + peak) / GB:.2f} GB resident > {budget(case) / GB:.2f} GB",
            )
    if cfg is not None:
        # R1 (the bead): same-phase roles stay resident together whenever the in-process
        # pool's peak for them fits the budget beside the standalone roles.
        # A tight role is charged what is left of the card, so it is left out of the peak.
        loose = members - tight
        if default_mode(case) and single_pool(case):
            standalone = sum(case.footprints[r] for r in srv - tight if r not in members)
            if standalone + oracle_peak(case, loose) <= budget(case):
                bad: list[str] = []
                for ws in WORKING_SETS:
                    need = sorted(loose & ws, key=lambda r: r.value)
                    for order in itertools.permutations(need):
                        resident = simulate(cfg, [_mid(r) for r in order])
                        lost = [r.value for r in need if _mid(r) not in resident]
                        if lost:
                            bad.append(f"{'>'.join(r.value for r in order)} evicts {lost}")
                if any(len(loose & ws) > 1 for ws in WORKING_SETS):  # a pair to keep together
                    rep.add(
                        case.key, "R1-same-phase-roles-stay-resident", not bad, "; ".join(bad[:2])
                    )
        # R5: roles that share no phase still evict each other, so the slot charge holds.
        stuck: list[str] = []
        for a, b in itertools.permutations(sorted(members, key=lambda r: r.value), 2):
            if not share_a_phase(a, b) and _mid(a) in simulate(cfg, [_mid(a), _mid(b)]):
                stuck.append(f"{a.value} survives loading {b.value}")
        rep.add(case.key, "R5-disjoint-roles-still-evict", not stuck, "; ".join(stuck[:2]))
        # R6: the config holds every launch of the process once.
        ids = {m for spec in cfg["groups"].values() for m in spec["members"]}  # type: ignore[union-attr]
        models = set(cfg["models"])  # type: ignore[arg-type]
        rep.add(
            case.key,
            "R6-config-members-equal-process-launches",
            ids == models,
            f"{sorted(ids)} vs {sorted(models)}",
        )
    # Context rows: the plan of every case equals the base plan; only the grouping may change.
    if golden is not None and case.key in golden:
        base = golden[case.key]["plan"]
        now = fingerprint(p)
        if case.replicas:
            # The merged class holds VRAM the elastic tier could use at the base: replicas may
            # shrink, nothing else moves (an open owner cell, rulings/97.md).
            first = [i for i in now["instances"] if i[2] == 0]
            first_base = [i for i in base["instances"] if i[2] == 0]
            rest = {k: v for k, v in now.items() if k != "instances"}
            rest_base = {k: v for k, v in base.items() if k != "instances"}
            ok = (
                first == first_base
                and all(i in base["instances"] for i in now["instances"])
                and rest == rest_base
            )
        else:
            ok = now == base
        rep.add(case.key, "C1-plan-equals-base", ok, f"base {base} head {now}")
    rep.add(case.key, "BEAD-shape", True, "bead" if bead else "")  # marker row, never fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record-golden")
    ap.add_argument("--golden", default=str(Path(__file__).with_name("golden-base.json")))
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    cases = _cases()
    assert len(cases) > 500, f"generator produced only {len(cases)} cases"
    plans = {c.key: run_plan(c) for c in cases}
    if args.record_golden:
        out = {k: {"plan": fingerprint(p), "bead": bead_case(p)} for k, p in plans.items()}
        Path(args.record_golden).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
        print(f"recorded {len(out)} base plans to {args.record_golden}")
        return 0
    golden = None
    if Path(args.golden).exists():
        golden = json.loads(Path(args.golden).read_text())
    if args.table:
        return table(cases, plans)
    rep = Report()
    for c in cases:
        check_case(c, plans[c.key], golden, rep)
    by_check: dict[str, list[Row]] = defaultdict(list)
    for r in rep.rows:
        by_check[r.check].append(r)
    beads = sorted(r.case for r in by_check["BEAD-shape"] if r.detail == "bead")
    oracle_names = {s.name for s in _oracle.SCENARIOS}
    print(f"lilbee imported from {lilbee.__file__}")
    print(f"inputs: {len(cases)} cases ({len(oracle_names)} oracle scenarios x modes, generated grid)")
    print(f"bead-shaped plans (co-tenant set holds two roles that share a phase): {len(beads)}")
    print("bead-shaped oracle scenarios:", [b for b in beads if b in oracle_names])
    failed = 0
    for check, rows in sorted(by_check.items()):
        if check == "BEAD-shape":
            continue
        bad = [r for r in rows if not r.ok]
        failed += len(bad)
        print(f"{'FAIL' if bad else 'PASS'} {check}: {len(rows) - len(bad)}/{len(rows)} rows")
        if not args.quiet:
            for r in bad[:6]:
                print(f"    {r.case}: {r.detail}")
        oracle_bad = [r.case for r in bad if r.case.split('|')[0] in oracle_names]
        if oracle_bad:
            print(f"    oracle scenarios red: {sorted(set(oracle_bad))}")
    print(f"=== {failed} failing row(s) ===")
    return 1 if failed else 0


def table(cases: list[Case], plans: dict[str, Placement]) -> int:
    print("| scenario | co-tenants (base = head) | tight (base = head) | base llama-swap groups "
          "| expected head groups |")
    print("|---|---|---|---|---|")
    for c in cases:
        if "|" in c.name or c.name.startswith("gen-"):
            continue
        p = plans[c.key]
        cfg = render_process(p)
        base = [sorted(r.value for r in p.co_tenants)] if cfg else []
        print(
            f"| {c.name} | {sorted(r.value for r in p.co_tenants)} "
            f"| {sorted(r.value for r in p.tight_roles)} "
            f"| {base} | {expected_head_groups(c, p)} |"
        )
    return 0


def expected_head_groups(case: Case, p: Placement) -> list[list[str]]:
    """The head's residency classes, from the oracle: merge same-phase roles when it fits."""
    srv = served(case, p)
    members = {r for r in p.co_tenants if r in srv}
    if not members:
        return []
    loose = members - set(p.tight_roles)
    standalone = sum(case.footprints[r] for r in srv - set(p.tight_roles) if r not in members)
    merge = (
        single_pool(case)
        and standalone + oracle_peak(case, loose) <= budget(case)
        and any(len(loose & ws) > 1 for ws in WORKING_SETS)
    )
    if not merge:
        return [[r.value] for r in sorted(members, key=lambda r: r.value)]
    classes: list[set[WorkerRole]] = []
    for role in sorted(members, key=lambda r: r.value):
        joined = [c for c in classes if any(share_a_phase(role, other) for other in c)]
        merged = {role}.union(*joined) if joined else {role}
        classes = [c for c in classes if c not in joined] + [merged]
    return sorted(sorted(r.value for r in c) for c in classes)


if __name__ == "__main__":
    raise SystemExit(main())
