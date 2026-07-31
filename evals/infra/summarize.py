"""Whole-run benchmark summary: throughput, GPU and host load, energy, extraction.

Reads the samplers written during the ingest and prints one shareable block.
Percentiles are over the whole run, so a mean is never quoted without the
distribution behind it: mean utilisation reads ~90% even when a third of the
samples are flat zero, which is exactly how extraction starvation hid for days.

Sampler schemas, all written by native9m.sh:
    gpu.csv    ts, util per card (one column per card)      every 2s
    host.csv   ts, loadavg, total threads, MemAvailable MB  every 10s
    sys.csv    ts, total GPU watts, bytes used on volume    every 30s
    rows.csv   ts, merged rows, then one column per shard   every 20s
"""

from __future__ import annotations

import gzip
import pathlib
import re
import statistics as st
import sys

PROF = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/prof")


def rows(name: str, min_fields: int) -> list[list[str]]:
    path = PROF / name
    if not path.exists():
        return []
    out = []
    for line in path.read_text(errors="replace").splitlines():
        parts = line.strip().split(",")
        if len(parts) >= min_fields and parts[0].isdigit():
            out.append(parts)
    return out


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * p), len(ordered) - 1)]


def num(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


print("=" * 66)
print("INGEST BENCHMARK SUMMARY")
print("=" * 66)

gpu = rows("gpu.csv", 2)
if gpu:
    cards = max(len(r) - 1 for r in gpu)
    per_card = [[num(x) for x in r[1 : cards + 1]] for r in gpu]
    flat = [v for card in per_card for v in card if v is not None]
    span = int(gpu[-1][0]) - int(gpu[0][0])
    if flat:
        print(f"\nwindow            {span / 3600:.2f} h over {len(gpu):,} samples (2s)")
        print(f"\nGPU  {cards} cards, {len(flat):,} per-card readings")
        print(
            f"  mean {st.mean(flat):5.1f}%   p10 {pct(flat, 0.10):3.0f}%   "
            f"p50 {pct(flat, 0.50):3.0f}%   p90 {pct(flat, 0.90):3.0f}%   "
            f"p99 {pct(flat, 0.99):3.0f}%"
        )
        # The number that separates a fed fleet from a starved one. Both read
        # ~90% while busy; only this row moves.
        print(f"  readings at exactly 0%   {100 * sum(1 for v in flat if v == 0) / len(flat):.1f}%")
        busy = [r for r in per_card if r and st.mean([v for v in r if v is not None] or [0]) > 10]
        print(f"  busy fraction (>10% mean)  {len(busy) / len(per_card):.3f}")
        if busy:
            while_busy = [
                st.mean([r[i] for r in busy if r[i] is not None]) for i in range(cards)
            ]
            print("  per card while busy      " + "/".join(f"{v:.0f}" for v in while_busy))

host = rows("host.csv", 4)
if host:
    load = [v for r in host if (v := num(r[1])) is not None]
    threads = [v for r in host if (v := num(r[2])) is not None]
    memfree = [v for r in host if (v := num(r[3])) is not None]
    if load:
        print("\nHost")
        print(
            f"  load     mean {st.mean(load):6.1f}   p50 {pct(load, 0.5):6.1f}   "
            f"max {max(load):6.1f}"
        )
    if threads:
        print(f"  threads  p50 {pct(threads, 0.5):,.0f}   max {max(threads):,.0f}")
    if memfree:
        print(f"  mem free min {min(memfree) / 1024:.1f} GB")

sysrows = rows("sys.csv", 2)
watts = [v for r in sysrows if (v := num(r[1])) is not None]
if watts:
    span = int(sysrows[-1][0]) - int(sysrows[0][0])
    print("\nPower")
    print(f"  mean {st.mean(watts):.0f} W   peak {max(watts):.0f} W")
    print(f"  energy {st.mean(watts) * span / 3600 / 1000:.2f} kWh over {span / 3600:.2f} h")

rowsamples = rows("rows.csv", 2)
if len(rowsamples) > 1:
    # Indexed rows are the shards' while the workers run and the merged index's
    # once the merge lands, so the series is the max of the two rather than
    # either alone: reading the merged column alone reports 0 for the whole ingest.
    def indexed(record: list[str]) -> int:
        merged = int(record[1]) if record[1].isdigit() else 0
        shards = sum(int(x) for x in record[2:] if x.isdigit())
        return max(merged, shards)

    t0, n0 = int(rowsamples[0][0]), indexed(rowsamples[0])
    tn, nn = int(rowsamples[-1][0]), indexed(rowsamples[-1])
    if tn > t0:
        print("\nThroughput")
        print(f"  {nn:,} rows in {(tn - t0) / 3600:.2f} h = {(nn - n0) / (tn - t0):.1f} docs/s")
        per_shard = [int(x) for x in rowsamples[-1][2:] if x.isdigit()]
        if per_shard:
            spread = (max(per_shard) - min(per_shard)) / max(max(per_shard), 1)
            print("  per shard  " + "/".join(f"{n:,}" for n in per_shard))
            # An even spread is the result that matters: the failure this
            # replaces reads one shard at everything and the rest at nothing.
            print(f"  shard spread  {100 * spread:.1f}% between the fullest and the emptiest")

traces = sorted(PROF.glob("*.trace.log")) + sorted(PROF.glob("*.trace.log.gz"))
elapsed: list[int] = []
for trace in traces:
    opener = gzip.open if trace.suffix == ".gz" else open
    with opener(trace, "rt", errors="replace") as handle:
        for line in handle:
            match = re.search(r"elapsed_ms=(\d+)", line)
            if match:
                elapsed.append(int(match.group(1)))
if elapsed:
    print(f"\nxberg extraction   {len(elapsed):,} files")
    print(
        f"  mean {st.mean(elapsed):.2f} ms   p50 {pct(elapsed, 0.5):.0f}   "
        f"p90 {pct(elapsed, 0.9):.0f}   p99 {pct(elapsed, 0.99):.0f}   max {max(elapsed)} ms"
    )

folded = {
    p.name: sum(
        int(line.rsplit(" ", 1)[1])
        for line in p.read_text(errors="replace").splitlines()
        if line.rsplit(" ", 1)[-1].isdigit()
    )
    for p in PROF.glob("*.folded")
}
if folded:
    print("\npy-spy samples")
    for name, total in sorted(folded.items()):
        print(f"  {name:<26} {total:,}")
    gil = next((v for k, v in folded.items() if "gil" in k), 0)
    wall = next((v for k, v in folded.items() if "wall" in k), 0)
    if wall:
        # Different workers, so this is a fleet-level ratio rather than one
        # process's: py-spy attaches per pid and profiling both ways on one
        # worker would double its sampling overhead.
        print(f"  GIL-held / wall, across two workers   {gil / wall:.3f}")
print("=" * 66)
