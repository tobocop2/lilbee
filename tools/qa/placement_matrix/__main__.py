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

from tools.qa.placement_matrix.cells import (
    DEFAULT_MODELS,
    build_matrix,
    iter_pairs,
    pair_by_room,
)
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
            observation = Observation(
                cell_id=cell.id,
                model_key=cell.model.key,
                cards=cell.cards,
                total_free_bytes=0,
                weights_bytes=0,
                planned=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        target.write_text(json.dumps(observation.to_json(), indent=2))
    return _report(args)


def _report(args: argparse.Namespace) -> int:
    out = Path(args.out)
    observations: list[Observation] = []
    unreadable: list[Failure] = []
    for path in sorted(out.glob("*.json")):
        try:
            observations.append(Observation.from_json(json.loads(path.read_text())))
        except (ValueError, TypeError) as exc:
            # Loud, not skipped: a result the merge cannot read is missing coverage,
            # and dropping it quietly shrinks the matrix into a pass.
            unreadable.append(Failure("result-unreadable", path.name, str(exc)))
    by_id = {o.cell_id: o for o in observations}
    failures: list[Failure] = [*unreadable, *judge_all(observations)]

    matrix = build_matrix(DEFAULT_MODELS, max_cards=args.max_cards)
    for left, right in iter_pairs(matrix.cells):
        ordered = pair_by_room(left, right)
        if ordered is None:
            continue
        tighter, roomier, knob = ordered
        observed_tighter, observed_roomier = by_id.get(tighter.id), by_id.get(roomier.id)
        if observed_tighter is None or observed_roomier is None:
            continue
        failures.extend(compare(observed_tighter, observed_roomier, knob))

    ran = [o for o in observations if not o.skipped]
    print(f"\n{len(ran)} cells judged, {len(observations) - len(ran)} skipped")
    for failure in sorted(failures, key=lambda f: (f.rule, f.cell_id)):
        print(f"  FAIL [{failure.rule}] {failure.cell_id}: {failure.detail}")
    if not ran and not unreadable:
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
