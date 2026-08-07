"""Run the placement matrix, or judge results a previous run wrote.

Sharding and resume are what let the same command run serially on one box or in
parallel across pods: each shard writes one JSON per cell into --out, and the
report merges whatever is there.

    python -m tools.qa.placement_matrix run --out results/          # everything this host can
    python -m tools.qa.placement_matrix run --out results/ --shard 0/4
    python -m tools.qa.placement_matrix report --out results/       # merge and judge
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tools.qa.placement_matrix.cells import DEFAULT_MODELS, build_matrix, iter_pairs
from tools.qa.placement_matrix.observe import gpu_count, run_cell
from tools.qa.placement_matrix.oracles import Failure, Observation, compare, judge_all


def _parse_shard(value: str) -> tuple[int, int]:
    index, _, count = value.partition("/")
    return int(index), int(count)


def _run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    models = tuple(m for m in DEFAULT_MODELS if not args.models or m.key in args.models)
    matrix = build_matrix(models, max_cards=args.max_cards)
    cells = matrix.cells
    if args.shard:
        index, count = _parse_shard(args.shard)
        cells = matrix.shard(index, count)
    cards = gpu_count()
    print(f"{len(cells)} cells, {cards} GPUs visible, writing to {out}")
    for model in models:
        print(f"  {model.key}: {model.probes}")
    for cell in cells:
        target = out / f"{cell.id}.json"
        if args.resume and target.exists():
            print(f"  skip (done)  {cell.id}")
            continue
        print(f"  run          {cell.id}", flush=True)
        try:
            observation = run_cell(cell, workdir=out, available_cards=cards)
        except Exception as exc:  # a crashed cell is a result, not a reason to stop
            print(f"  ERROR        {cell.id}: {type(exc).__name__}: {exc}")
            continue
        target.write_text(json.dumps(observation.to_json(), indent=2))
    return _report(args)


def _report(args: argparse.Namespace) -> int:
    out = Path(args.out)
    observations = [
        Observation.from_json(json.loads(path.read_text()))
        for path in sorted(out.glob("*.json"))
        if not path.name.endswith(".engine.log")
    ]
    by_id = {o.cell_id: o for o in observations}
    failures: list[Failure] = judge_all(observations)

    matrix = build_matrix(DEFAULT_MODELS, max_cards=args.max_cards)
    for low, high in iter_pairs(matrix.cells):
        left, right = by_id.get(low.id), by_id.get(high.id)
        if left is None or right is None:
            continue
        if low.cards != high.cards:
            knob = "cards"
        elif low.usable_fraction != high.usable_fraction:
            knob = "usable VRAM"
        elif low.ballast_gib != high.ballast_gib:
            knob, left, right = "free VRAM", right, left
        else:
            continue
        failures.extend(compare(left, right, knob))

    ran = [o for o in observations if not o.skipped]
    print(f"\n{len(ran)} cells judged, {len(observations) - len(ran)} skipped")
    for failure in sorted(failures, key=lambda f: (f.rule, f.cell_id)):
        print(f"  FAIL [{failure.rule}] {failure.cell_id}: {failure.detail}")
    if not ran:
        # Green on nothing is how a matrix that never ran reads as a pass.
        print("  NOTHING RAN: no cell produced a judgeable result")
        return 1
    if not failures:
        print("  no invariant violations")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="placement_matrix", description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--out", required=True, help="directory of per-cell result JSON")
    parser.add_argument("--shard", help="k/N: run only this shard of the matrix")
    parser.add_argument("--max-cards", type=int, default=4)
    parser.add_argument("--models", nargs="*", default=[], help="model keys to include")
    parser.add_argument("--resume", action="store_true", help="skip cells already recorded")
    args = parser.parse_args(argv)
    return _run(args) if args.command == "run" else _report(args)


if __name__ == "__main__":
    sys.exit(main())
