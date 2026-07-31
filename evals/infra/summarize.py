"""Whole-run benchmark summary: throughput, GPU and CPU load, energy, extraction.

Reads the samplers written during the ingest and prints one shareable block.
Percentiles are over the whole run, so a mean is never quoted without the
distribution behind it: mean utilisation reads ~90% even when a third of the
samples are flat zero.
"""

from __future__ import annotations

import pathlib
import statistics as st
import sys

PROF = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/root/prof")


def rows(name: str, min_fields: int) -> list[list[str]]:
    path = PROF / name
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) >= min_fields:
            out.append(parts)
    return out


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * p), len(ordered) - 1)]


print("=" * 62)
print("INGEST BENCHMARK SUMMARY")
print("=" * 62)

sysrows = rows("sys.csv", 5)
if sysrows:
    cpu = [float(r[1]) for r in sysrows]
    gutil = [float(r[2]) for r in sysrows]
    gmem = [float(r[3]) for r in sysrows]
    watts = [float(r[4]) for r in sysrows]
    span = int(sysrows[-1][0]) - int(sysrows[0][0])
    print(f"\nwindow            {span / 3600:.2f} h over {len(sysrows)} samples (5s)")
    print("\nCPU (whole box)")
    print(f"  mean {st.mean(cpu):5.1f}%    p50 {pct(cpu, .5):5.1f}%    "
          f"p95 {pct(cpu, .95):5.1f}%    max {max(cpu):5.1f}%")
    print("\nGPU (mean across 8 cards)")
    print(f"  mean {st.mean(gutil):5.1f}%    p50 {pct(gutil, .5):5.1f}%    "
          f"p95 {pct(gutil, .95):5.1f}%    max {max(gutil):5.1f}%")
    print(f"  busy fraction (>10%)  {sum(1 for g in gutil if g > 10) / len(gutil):.3f}")
    print(f"  vram  mean {st.mean(gmem) / 1024:.1f} GB across all cards")
    print("\nPower")
    print(f"  mean {st.mean(watts):.0f} W    peak {max(watts):.0f} W")
    print(f"  energy {st.mean(watts) * span / 3600 / 1000:.2f} kWh")

gpu = rows("gpu.csv", 9)
if gpu:
    per_card = [[float(x) for x in r[1:9]] for r in gpu if all(x.isdigit() for x in r[1:9])]
    flat = [v for card in per_card for v in card]
    if flat:
        print(f"\nPer-card readings  {len(flat):,} samples x 8 cards")
        print(f"  p10 {pct(flat, .10):.0f}%   p50 {pct(flat, .50):.0f}%   "
              f"p90 {pct(flat, .90):.0f}%   p99 {pct(flat, .99):.0f}%")
        print(f"  readings at exactly 0%   {100 * sum(1 for v in flat if v == 0) / len(flat):.1f}%")

rowsamples = rows("rows.csv", 2)
if len(rowsamples) > 1:
    t0, n0 = int(rowsamples[0][0]), int(rowsamples[0][1])
    tn, nn = int(rowsamples[-1][0]), int(rowsamples[-1][1])
    if tn > t0:
        print(f"\nThroughput")
        print(f"  {nn:,} rows in {(tn - t0) / 3600:.2f} h = {(nn - n0) / (tn - t0):.1f} docs/s")

traces = sorted(PROF.glob("w*.trace.log")) or sorted(PROF.glob("w*.trace.log.gz"))
if traces:
    import gzip
    import re

    elapsed = []
    for t in traces:
        opener = gzip.open if t.suffix == ".gz" else open
        with opener(t, "rt", errors="replace") as fh:
            for line in fh:
                m = re.search(r"elapsed_ms=(\d+)", line)
                if m:
                    elapsed.append(int(m.group(1)))
    if elapsed:
        print(f"\nxberg extraction   {len(elapsed):,} files")
        print(f"  mean {st.mean(elapsed):.2f} ms   p50 {pct(elapsed, .5):.0f}   "
              f"p90 {pct(elapsed, .9):.0f}   p99 {pct(elapsed, .99):.0f}   max {max(elapsed)} ms")

folded = {p.name: sum(int(line.rsplit(" ", 1)[1])
                      for line in p.read_text().splitlines()
                      if line.rsplit(" ", 1)[-1].isdigit())
          for p in PROF.glob("*.folded")}
if folded:
    print("\npy-spy samples")
    for name, total in sorted(folded.items()):
        print(f"  {name:<26} {total:,}")
    gil = next((v for k, v in folded.items() if "gil" in k), 0)
    wall = next((v for k, v in folded.items() if "wall" in k), 0)
    if wall:
        print(f"  GIL-held fraction of sampled wall time   {gil / wall:.3f}")
print("=" * 62)
